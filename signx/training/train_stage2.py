"""Stage 2 training entrypoint: RGB Video -> Latent (matches Stage 1)."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data import GlossVocab, build_dataset, build_video_transform, collate_video_batch
from ..models import Stage1Model, Stage2Model
from ..pose import PoseAwareFeatureCompiler, build_pose_extractor
from ..utils import load_config, load_checkpoint, setup_logging
from .trainer import BaseTrainer


class Stage2Trainer(BaseTrainer):
    """Train Video2Pose with MSE against Stage 1 latents."""

    def __init__(self, model, cfg, train_loader, val_loader, pose_extractor, compiler):
        super().__init__(model, cfg, train_loader, val_loader)
        self.pose_extractor = pose_extractor
        self.compiler = compiler.to(self.device)

    def _gt_latent(self, paths) -> torch.Tensor:
        feats = [self.pose_extractor(p).to(self.device) for p in paths]
        max_t = max(f.shape[0] for f in feats)
        d = feats[0].shape[-1]
        out = torch.zeros(len(feats), max_t, d, device=self.device)
        for i, f in enumerate(feats):
            out[i, : f.shape[0]] = f
        out = self.compiler(out)
        with torch.no_grad():
            return self.model.stage1_model.encode_pose(out)

    def compute_loss(self, batch: Dict) -> Dict[str, torch.Tensor]:
        gt_latent = self._gt_latent(batch["paths"])         # (B, T_pose, D)
        pred_latent = self.model.video2pose(batch["videos"])  # (B, T_vid, D)
        # Match temporal lengths via interpolation along time
        if pred_latent.shape[1] != gt_latent.shape[1]:
            pred_latent = F.interpolate(
                pred_latent.transpose(1, 2),
                size=gt_latent.shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        loss = F.mse_loss(pred_latent, gt_latent)
        return {"loss": loss, "mse": loss}

    @torch.no_grad()
    def evaluate(self, loader) -> Dict[str, float]:
        self.model.eval()
        total = 0.0
        n = 0
        for batch in loader:
            batch = self._to_device(batch)
            d = self.compute_loss(batch)
            total += float(d["loss"]) * batch["videos"].shape[0]
            n += batch["videos"].shape[0]
        mse = total / max(1, n)
        # Negate so the existing "min wer" tracker still picks the best.
        return {"mse": mse, "wer": mse}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage2_video2pose.yaml")
    ap.add_argument("--dataset-root", default=None,
                    help="Override dataset_root in paths.yaml (sets DATASET_ROOT env var)")
    args = ap.parse_args()

    if args.dataset_root:
        import os
        os.environ["DATASET_ROOT"] = args.dataset_root

    cfg = load_config(args.config)
    setup_logging(Path(cfg.log_dir) / cfg.name)

    vocab = GlossVocab.from_file(cfg.vocab_file)
    train_ds = build_dataset(cfg, vocab, split="train", transform=build_video_transform(cfg, train=True))
    val_ds = build_dataset(cfg, vocab, split="dev", transform=build_video_transform(cfg, train=False))

    nw = int(cfg.num_workers)
    train_loader = DataLoader(
        train_ds, batch_size=int(cfg.train.batch_size), shuffle=cfg.data.shuffle,
        num_workers=nw, pin_memory=bool(cfg.pin_memory),
        persistent_workers=(nw > 0),
        collate_fn=collate_video_batch,
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(cfg.train.batch_size), shuffle=False,
        num_workers=nw, pin_memory=bool(cfg.pin_memory),
        persistent_workers=(nw > 0),
        collate_fn=collate_video_batch,
    )

    # Rebuild Stage 1 from its config and load its checkpoint
    stage1_cfg = load_config("configs/stage1_pose2gloss.yaml")
    pose_extractor = build_pose_extractor(stage1_cfg)
    compiler = PoseAwareFeatureCompiler(feature_dim=pose_extractor.output_dim)
    stage1 = Stage1Model(
        pose_input_dim=pose_extractor.output_dim,
        latent_dim=int(stage1_cfg.model.latent_dim),
        codebook_size=int(stage1_cfg.model.codebook_size),
        codebook_dim=int(stage1_cfg.model.codebook_dim),
        decoder_layers=int(stage1_cfg.model.decoder_layers),
        num_heads=int(stage1_cfg.model.num_fusion_heads),
        vocab_size=int(stage1_cfg.vocab.vocab_size),
        num_fusion_layers=int(stage1_cfg.model.num_fusion_layers),
        dropout=float(stage1_cfg.model.dropout),
    )
    ckpt = Path(cfg.model.stage1_checkpoint)
    if ckpt.exists():
        load_checkpoint(ckpt, stage1, strict=False)
    stage1.eval()

    model = Stage2Model(
        latent_dim=int(cfg.model.latent_dim),
        vit_name=str(cfg.model.vit_name),
        vit_pretrained=bool(cfg.model.vit_pretrained),
        stage1_model=stage1,
    )

    trainer = Stage2Trainer(model, cfg, train_loader, val_loader, pose_extractor, compiler)
    trainer.train()


if __name__ == "__main__":
    main()
