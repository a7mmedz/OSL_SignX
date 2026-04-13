"""Single-video inference pipeline.

Usage:
    python -m signx.inference.predict \
        --video path/to/video.mp4 \
        --stage3-checkpoint outputs/checkpoints/stage3/best.pt \
        --stage2-checkpoint outputs/checkpoints/stage2/best.pt \
        --vocab data/gloss_vocab.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F

from ..data.dataset import _load_video_decord
from ..data.transforms import VideoTransformConfig, VideoTransform
from ..data.vocab import GlossVocab
from ..models.signx_model import Stage2Model, Stage3Model
from ..utils.checkpoint import load_checkpoint
from ..utils.config import load_config
from .beam_search import BeamSearchDecoder, ctc_greedy_decode


def predict_video(
    video_path: str | Path,
    stage2: Stage2Model,
    stage3: Stage3Model,
    vocab: GlossVocab,
    cfg,
    beam_size: int = 8,
    length_penalty: float = 1.0,
    device: str | torch.device = "cpu",
) -> List[str]:
    """Run the full inference pipeline on one video.

    Returns a list of predicted Arabic gloss strings.
    """
    device = torch.device(device)
    transform = VideoTransform(
        VideoTransformConfig(
            image_size=int(cfg.video.image_size),
            mean=tuple(cfg.video.mean),
            std=tuple(cfg.video.std),
            train=False,
        )
    )

    frames = _load_video_decord(video_path, max_frames=int(cfg.video.max_frames))
    video = transform(frames).unsqueeze(0).to(device)  # (1, T, 3, H, W)

    stage2.eval().to(device)
    stage3.eval().to(device)

    with torch.no_grad():
        latent = stage2.video2pose(video)                    # (1, T, D)
        out = stage3(latent)
        ctc_logits = out["ctc_logits"]                       # (1, T', V)
        log_probs = F.log_softmax(ctc_logits[0], dim=-1)    # (T', V)

    if beam_size <= 1:
        ids = ctc_greedy_decode(log_probs, blank_id=vocab.blank_id)
    else:
        decoder = BeamSearchDecoder(
            vocab_size=vocab.vocab_size,
            blank_id=vocab.blank_id,
            beam_size=beam_size,
            length_penalty=length_penalty,
        )
        ids = decoder.decode(log_probs)

    return vocab.decode(ids, strip_blank=True)


def main():
    ap = argparse.ArgumentParser(description="Single-video SignX inference")
    ap.add_argument("--video", required=True, help="Path to .mp4 file")
    ap.add_argument("--stage2-checkpoint", required=True)
    ap.add_argument("--stage3-checkpoint", required=True)
    ap.add_argument("--config", default="configs/stage3_cslr.yaml")
    ap.add_argument("--beam-size", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    vocab = GlossVocab.from_file(cfg.vocab_file)

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

    glosses = predict_video(
        video_path=args.video,
        stage2=stage2,
        stage3=stage3,
        vocab=vocab,
        cfg=cfg,
        beam_size=args.beam_size,
        device=args.device,
    )
    print("Predicted glosses:", " ".join(glosses))


if __name__ == "__main__":
    main()
