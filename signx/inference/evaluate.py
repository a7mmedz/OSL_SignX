"""Full test-set evaluation script.

Usage:
    python -m signx.inference.evaluate \
        --config configs/stage3_cslr.yaml \
        --stage2-checkpoint outputs/checkpoints/stage2/best.pt \
        --stage3-checkpoint outputs/checkpoints/stage3/best.pt \
        --split test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data import GlossVocab, build_dataset, build_video_transform, collate_video_batch
from ..models.signx_model import Stage2Model, Stage3Model
from ..training.metrics import compute_bleu, compute_pi_accuracy, compute_wer
from ..utils.checkpoint import load_checkpoint
from ..utils.config import load_config
from ..utils.logging_utils import get_logger
from .beam_search import BeamSearchDecoder, ctc_greedy_decode

logger = get_logger(__name__)


def evaluate_dataset(
    stage2: Stage2Model,
    stage3: Stage3Model,
    loader: DataLoader,
    vocab: GlossVocab,
    beam_size: int = 8,
    length_penalty: float = 1.0,
    device: str | torch.device = "cpu",
) -> Dict[str, float]:
    """Evaluate on a DataLoader, return WER / BLEU / P-I metrics."""
    device = torch.device(device)
    stage2.eval().to(device)
    stage3.eval().to(device)

    beam_decoder = BeamSearchDecoder(
        vocab_size=vocab.vocab_size,
        blank_id=vocab.blank_id,
        beam_size=beam_size,
        length_penalty=length_penalty,
    )

    refs: List[List[int]] = []
    hyps: List[List[int]] = []

    with torch.no_grad():
        for batch in loader:
            videos = batch["videos"].to(device)
            gloss_targets = batch["glosses"]
            gloss_lengths = batch["gloss_lengths"]

            latent = stage2.video2pose(videos)
            out = stage3(latent, lengths=batch["video_lengths"].to(device))
            ctc_logits = out["ctc_logits"]                  # (B, T', V)

            for i in range(ctc_logits.shape[0]):
                log_p = F.log_softmax(ctc_logits[i], dim=-1)
                if beam_size > 1:
                    hyp = beam_decoder.decode(log_p)
                else:
                    hyp = ctc_greedy_decode(log_p, blank_id=vocab.blank_id)
                hyps.append(hyp)
                l = int(gloss_lengths[i])
                refs.append(gloss_targets[i, :l].tolist())

    wer = compute_wer(refs, hyps)
    bleu = compute_bleu(refs, hyps)
    pi = compute_pi_accuracy(refs, hyps)
    return {"wer": wer, "bleu": bleu, "pi_accuracy": pi}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage3_cslr.yaml")
    ap.add_argument("--stage2-checkpoint", required=True)
    ap.add_argument("--stage3-checkpoint", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--beam-size", type=int, default=8)
    ap.add_argument("--output", default=None, help="Write results JSON here")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dataset-root", default=None,
                    help="Override dataset_root in paths.yaml (sets DATASET_ROOT env var)")
    args = ap.parse_args()

    if args.dataset_root:
        import os
        os.environ["DATASET_ROOT"] = args.dataset_root

    cfg = load_config(args.config)
    vocab = GlossVocab.from_file(cfg.vocab_file)

    ds = build_dataset(cfg, vocab, split=args.split, transform=build_video_transform(cfg, train=False))
    loader = DataLoader(
        ds, batch_size=int(cfg.train.batch_size), shuffle=False,
        num_workers=int(cfg.num_workers), collate_fn=collate_video_batch,
    )

    stage2_cfg = load_config("configs/stage2_video2pose.yaml")
    stage2 = Stage2Model(
        latent_dim=int(stage2_cfg.model.latent_dim),
        vit_name=str(stage2_cfg.model.vit_name),
        vit_pretrained=False,
    )
    load_checkpoint(args.stage2_checkpoint, stage2, strict=False)

    stage3 = Stage3Model(
        latent_dim=int(cfg.model.latent_dim),
        pruned_dim=int(cfg.model.pruned_dim),
        tconv_channels=list(cfg.model.tconv_channels),
        tconv_kernels=list(cfg.model.tconv_kernels),
        tconv_strides=list(cfg.model.tconv_strides),
        lstm_hidden=int(cfg.model.lstm_hidden),
        lstm_layers=int(cfg.model.lstm_layers),
        bidirectional=bool(cfg.model.lstm_bidirectional),
        transformer_layers=int(cfg.model.transformer_layers),
        transformer_dim=int(cfg.model.transformer_dim),
        transformer_heads=int(cfg.model.transformer_heads),
        transformer_ffn=int(cfg.model.transformer_ffn),
        vocab_size=int(cfg.vocab.vocab_size),
        dropout_attn=float(cfg.model.dropout_attn),
        dropout_relu=float(cfg.model.dropout_relu),
        dropout_res=float(cfg.model.dropout_res),
        max_target_len=int(cfg.decode.max_len),
    )
    load_checkpoint(args.stage3_checkpoint, stage3, strict=False)

    metrics = evaluate_dataset(
        stage2=stage2,
        stage3=stage3,
        loader=loader,
        vocab=vocab,
        beam_size=args.beam_size,
        device=args.device,
    )

    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
