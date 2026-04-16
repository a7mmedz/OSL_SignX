"""Stage 1 visualization — generates figures for the report.

Produces the following PNG files in outputs/visualizations/stage1/:

  pose_features_<ID>.png     — per-frame pose feature heatmap for a sample video
  attention_<ID>.png         — self-attention map from PoseFusionEncoder
  predictions_<ID>.png       — top-5 gloss predictions vs ground truth
  method_comparison.png      — bar chart of pi_accuracy per pose backend
  skeleton_<ID>.png          — skeleton overlay on a video frame (MediaPipe)
  feature_norms.png          — mean per-frame feature norm over time

Usage (after Stage 1 training completes):
    python -m signx.inference.visualize_stage1 \\
        --checkpoint outputs/checkpoints/stage1_pose2gloss/best.pt \\
        --config    configs/stage1_pose2gloss.yaml \\
        --n-samples 5
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from ..data import GlossVocab, build_dataset, build_video_transform
from ..models import Stage1Model
from ..pose import PoseAwareFeatureCompiler, build_pose_extractor
from ..training.metrics import compute_pi_accuracy
from ..utils import load_checkpoint, load_config, setup_logging
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

_OUT_ROOT = Path("outputs/visualizations/stage1")


# ─────────────────────────── helpers ──────────────────────────────────────

def _try_import_plt():
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        return plt, gridspec
    except ImportError:
        raise ImportError("matplotlib is required: pip install matplotlib")


def _load_model(cfg, ckpt_path: str | None, device: torch.device) -> Stage1Model:
    extractor = build_pose_extractor(cfg)
    model = Stage1Model(
        pose_input_dim=extractor.output_dim,
        latent_dim=int(cfg.model.latent_dim),
        codebook_size=int(cfg.model.codebook_size),
        codebook_dim=int(cfg.model.codebook_dim),
        decoder_layers=int(cfg.model.decoder_layers),
        num_heads=int(cfg.model.num_fusion_heads),
        vocab_size=int(cfg.vocab.vocab_size),
        num_fusion_layers=int(cfg.model.num_fusion_layers),
        dropout=0.0,
    ).to(device)
    if ckpt_path and Path(ckpt_path).exists():
        load_checkpoint(ckpt_path, model, strict=False)
        logger.info(f"Loaded checkpoint: {ckpt_path}")
    else:
        logger.warning("No checkpoint loaded — model weights are random (for structure demo only).")
    model.eval()
    return model, extractor


def _extract_and_compile(
    extractor, compiler, paths: List[str], device: torch.device
) -> torch.Tensor:
    """Extract pose for a list of video paths → (B, T, D) tensor."""
    feats, max_t = [], 0
    for p in paths:
        f = extractor(p)
        feats.append(f)
        max_t = max(max_t, f.shape[0])
    d = feats[0].shape[-1]
    out = torch.zeros(len(feats), max_t, d)
    for i, f in enumerate(feats):
        out[i, : f.shape[0]] = f
    return compiler(out.to(device))


# ─────────────────────────── figure 1: feature heatmap ────────────────────

def plot_feature_heatmap(pose_feat: torch.Tensor, sample_id: str, vocab_word: str) -> None:
    """pose_feat: (T, D). Saves a temporal × feature heatmap."""
    plt, _ = _try_import_plt()
    data = pose_feat.cpu().float().numpy()          # (T, D)
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(data.T, aspect="auto", cmap="RdBu_r", interpolation="nearest")
    ax.set_xlabel("Frame", fontsize=12)
    ax.set_ylabel("Feature dimension", fontsize=12)
    ax.set_title(f"Pose features — '{vocab_word}' (sample {sample_id})", fontsize=13)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    out = _OUT_ROOT / f"pose_features_{sample_id}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {out}")


# ─────────────────────────── figure 2: attention map ──────────────────────

def plot_attention(model: Stage1Model, pose_feat: torch.Tensor, sample_id: str) -> None:
    """Hook into PoseFusionEncoder to capture the first attention layer's weights."""
    plt, _ = _try_import_plt()
    attn_map: Dict[str, torch.Tensor] = {}

    def _hook(module, input, output):
        # TransformerEncoderLayer doesn't expose attn weights by default;
        # we read the raw QK product from the MultiheadAttention sublayer.
        pass

    # Re-run forward with need_weights=True via manual call
    x = pose_feat.unsqueeze(0).to(next(model.parameters()).device)  # (1, T, D)
    with torch.no_grad():
        x_proj = model.pose_fusion.input_proj(x)
        x_proj = model.pose_fusion.pos_enc(x_proj)
        # First encoder layer's self-attention
        layer = model.pose_fusion.encoder.layers[0]
        _, weights = layer.self_attn(x_proj, x_proj, x_proj, need_weights=True, average_attn_weights=True)
        # weights: (1, T, T)
        attn = weights.squeeze(0).cpu().numpy()   # (T, T)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(attn, cmap="viridis", aspect="auto")
    ax.set_xlabel("Key frame", fontsize=12)
    ax.set_ylabel("Query frame", fontsize=12)
    ax.set_title(f"Self-attention (layer 1) — sample {sample_id}", fontsize=13)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out = _OUT_ROOT / f"attention_{sample_id}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {out}")


# ─────────────────────────── figure 3: predictions ────────────────────────

def plot_predictions(
    model: Stage1Model,
    pose_feat: torch.Tensor,
    gloss_ids: List[int],
    vocab: GlossVocab,
    sample_id: str,
) -> None:
    """Bar chart of top-5 predicted glosses vs ground truth."""
    plt, _ = _try_import_plt()
    device = next(model.parameters()).device
    pose = pose_feat.unsqueeze(0).to(device)        # (1, T, D)

    with torch.no_grad():
        latent = model.encode_pose(pose)             # (1, T, latent_dim)
        bos = torch.tensor([[vocab.blank_id]], device=device)
        out = model(pose, bos)
        logits = out["logits"]                       # (1, 1, V)
        probs = F.softmax(logits[0, 0], dim=-1)     # (V,)

    top5 = torch.topk(probs, k=5)
    top5_ids    = top5.indices.cpu().tolist()
    top5_probs  = top5.values.cpu().tolist()
    top5_labels = [vocab.decode([i]) or f"id={i}" for i in top5_ids]

    gt_label = vocab.decode(gloss_ids) or str(gloss_ids)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["green" if i in gloss_ids else "steelblue" for i in top5_ids]
    bars = ax.barh(top5_labels[::-1], top5_probs[::-1], color=colors[::-1])
    ax.set_xlabel("Probability", fontsize=12)
    ax.set_title(
        f"Top-5 predictions — GT: '{gt_label}'\n(green = correct gloss)",
        fontsize=12,
    )
    ax.set_xlim(0, 1)
    for bar, p in zip(bars, top5_probs[::-1]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{p:.3f}", va="center", fontsize=10)
    fig.tight_layout()
    out = _OUT_ROOT / f"predictions_{sample_id}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {out}")


# ─────────────────────────── figure 4: feature norms ─────────────────────

def plot_feature_norms(pose_feat: torch.Tensor, sample_id: str, vocab_word: str) -> None:
    """Plot the L2 norm of the pose feature vector over time."""
    plt, _ = _try_import_plt()
    norms = pose_feat.norm(dim=-1).cpu().numpy()    # (T,)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(norms, color="steelblue", linewidth=1.5)
    ax.fill_between(range(len(norms)), norms, alpha=0.2, color="steelblue")
    ax.set_xlabel("Frame", fontsize=12)
    ax.set_ylabel("L2 norm", fontsize=12)
    ax.set_title(f"Feature norm over time — '{vocab_word}'", fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = _OUT_ROOT / f"feature_norms_{sample_id}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {out}")


# ─────────────────────────── figure 5: method comparison ─────────────────

def plot_method_comparison(results: Dict[str, float]) -> None:
    """Bar chart of pi_accuracy per pose backend.

    Args:
        results: {"mediapipe": 0.72, "dwpose": 0.75, ...}
    """
    plt, _ = _try_import_plt()
    methods = list(results.keys())
    scores  = [results[m] * 100 for m in methods]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
    bars = ax.bar(methods, scores, color=colors[: len(methods)], width=0.5)
    ax.set_ylabel("P-I Accuracy (%)", fontsize=12)
    ax.set_title("Stage 1 — P-I Accuracy by Pose Extraction Method", fontsize=13)
    ax.set_ylim(0, 100)
    ax.axhline(y=max(scores), color="gray", linestyle="--", linewidth=0.8)
    for bar, s in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{s:.1f}%", ha="center", fontsize=11, fontweight="bold")
    fig.tight_layout()
    out = _OUT_ROOT / "method_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {out}")


# ─────────────────────────── main ─────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Generate Stage 1 report figures.")
    ap.add_argument("--checkpoint", default="outputs/checkpoints/stage1_pose2gloss/best.pt")
    ap.add_argument("--config",     default="configs/stage1_pose2gloss.yaml")
    ap.add_argument("--n-samples",  type=int, default=5,
                    help="Number of test videos to visualise")
    ap.add_argument("--split",      default="test",
                    help="Dataset split to sample from (train/dev/test)")
    ap.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--method-results", nargs="+", default=None,
                    help="Pre-computed results per method: mediapipe=0.72 dwpose=0.75 ...")
    args = ap.parse_args()

    if args.dataset_root:
        import os
        os.environ["DATASET_ROOT"] = args.dataset_root

    setup_logging()
    cfg    = load_config(args.config)
    device = torch.device(args.device)
    vocab  = GlossVocab.from_file(cfg.vocab_file)

    model, extractor = _load_model(cfg, args.checkpoint, device)
    compiler = PoseAwareFeatureCompiler(feature_dim=extractor.output_dim).to(device)
    compiler.eval()

    ds = build_dataset(
        cfg, vocab, split=args.split,
        transform=build_video_transform(cfg, train=False),
    )

    if len(ds) == 0:
        logger.error(f"No samples found in split='{args.split}'. Check your dataset path.")
        return

    n = min(args.n_samples, len(ds))
    indices = random.sample(range(len(ds)), n)

    _OUT_ROOT.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating visualizations for {n} samples → {_OUT_ROOT}")

    refs, hyps = [], []
    for idx in indices:
        sample = ds[idx]
        path   = sample["path"]
        gids   = sample["gloss_ids"].tolist()
        sid    = Path(path).stem
        word   = vocab.decode(gids) if hasattr(vocab, "decode") else str(gids)

        logger.info(f"Processing {sid} — gloss: {word}")

        try:
            raw_feat = extractor(path)                              # (T, D)
            compiled = compiler(raw_feat.unsqueeze(0).to(device))  # (1, T, D)
            pose_2d  = compiled.squeeze(0)                         # (T, D)

            plot_feature_heatmap(pose_2d, sid, word)
            plot_attention(model, pose_2d, sid)
            plot_predictions(model, pose_2d, gids, vocab, sid)
            plot_feature_norms(pose_2d, sid, word)

            # Collect for pi_accuracy
            with torch.no_grad():
                latent = model.encode_pose(pose_2d.unsqueeze(0))
                ys = model.decoder.generate(
                    latent, bos_id=vocab.blank_id, eos_id=vocab.blank_id, max_len=8
                )
            hyp = [int(t) for t in ys[0, 1:].tolist() if t != 0]
            hyps.append(hyp)
            refs.append([g for g in gids if g != 0])
        except Exception as e:
            logger.warning(f"Skipped {sid}: {e}")

    if refs:
        pi = compute_pi_accuracy(refs, hyps)
        logger.info(f"P-I Accuracy on {len(refs)} samples: {pi:.4f} ({pi*100:.1f}%)")

    # Method comparison chart — use provided results or current backend only
    if args.method_results:
        method_scores = {}
        for item in args.method_results:
            k, v = item.split("=")
            method_scores[k] = float(v)
        plot_method_comparison(method_scores)
    elif refs:
        current_backend = str(cfg.pose.backend)
        plot_method_comparison({current_backend: compute_pi_accuracy(refs, hyps)})

    logger.info(f"\nAll figures saved to: {_OUT_ROOT.resolve()}")
    logger.info("Files generated:")
    for f in sorted(_OUT_ROOT.glob("*.png")):
        logger.info(f"  {f.name}")


if __name__ == "__main__":
    main()
