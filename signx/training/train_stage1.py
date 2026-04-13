"""Stage 1 training entrypoint: Pose -> Gloss."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from ..data import GlossVocab, build_dataset, build_video_transform, collate_video_batch
from ..models import (
    Stage1Model,
    contrastive_loss,
    word_matching_loss,
)
from ..pose import PoseAwareFeatureCompiler, build_pose_extractor
from ..utils import load_config, setup_logging
from .metrics import compute_pi_accuracy
from .trainer import BaseTrainer


class Stage1Trainer(BaseTrainer):
    """Pose2Gloss trainer with text-CE + word-matching + contrastive losses."""

    def __init__(self, model, cfg, train_loader, val_loader, vocab, pose_extractor, compiler):
        super().__init__(model, cfg, train_loader, val_loader)
        self.vocab = vocab
        self.pose_extractor = pose_extractor
        self.compiler = compiler.to(self.device)

    def _extract_pose_batch(self, paths) -> torch.Tensor:
        feats = []
        max_t = 0
        for p in paths:
            f = self.pose_extractor(p).to(self.device)
            feats.append(f)
            max_t = max(max_t, f.shape[0])
        d = feats[0].shape[-1]
        out = torch.zeros(len(feats), max_t, d, device=self.device)
        for i, f in enumerate(feats):
            out[i, : f.shape[0]] = f
        return self.compiler(out)

    def compute_loss(self, batch: Dict) -> Dict[str, torch.Tensor]:
        pose = self._extract_pose_batch(batch["paths"])
        target = batch["glosses"]                          # (B, L)
        # Teacher-forced: input = target shifted right (prepend blank as BOS)
        bos = torch.full((target.shape[0], 1), self.vocab.blank_id, device=target.device)
        decoder_input = torch.cat([bos, target[:, :-1]], dim=1)
        out = self.model(pose, decoder_input)
        logits = out["logits"]                              # (B, L, V)

        ls = float(self.cfg.loss.label_smoothing)
        l_text = word_matching_loss(logits, target, pad_id=0, label_smoothing=ls)

        # Word-matching loss = same CE here for the simple word-level case;
        # in the full multi-gloss case it acts as an auxiliary token-level CE.
        l_word = l_text.detach() * 0 + l_text

        # Contrastive: pool latent and embedded gloss
        latent_pool = out["latent"].mean(dim=1)
        gloss_emb = self.model.decoder.target_embedding(target).mean(dim=1)
        # Project to matching dim
        latent_pool = latent_pool[:, : gloss_emb.shape[-1]]
        l_contrast = contrastive_loss(
            latent_pool,
            gloss_emb,
            temperature=float(self.cfg.loss.contrastive_temperature),
        )

        commit = self.model.decoder.codebook.last_commit_loss
        loss = (
            float(self.cfg.loss.lambda_text) * l_text
            + float(self.cfg.loss.lambda_word) * l_word
            + float(self.cfg.loss.lambda_contrast) * l_contrast
        )
        if commit is not None:
            loss = loss + 0.1 * commit
        return {
            "loss": loss,
            "text": l_text,
            "word": l_word,
            "contrast": l_contrast,
        }

    @torch.no_grad()
    def evaluate(self, loader) -> Dict[str, float]:
        self.model.eval()
        refs, hyps = [], []
        for batch in loader:
            batch = self._to_device(batch)
            pose = self._extract_pose_batch(batch["paths"])
            latent = self.model.encode_pose(pose)
            ys = self.model.decoder.generate(
                latent, bos_id=self.vocab.blank_id, eos_id=self.vocab.blank_id, max_len=8
            )
            for i in range(ys.shape[0]):
                hyps.append([int(t) for t in ys[i, 1:].tolist() if t != 0])
                refs.append([int(t) for t in batch["glosses"][i].tolist() if t != 0])
        pi = compute_pi_accuracy(refs, hyps)
        return {"pi_accuracy": pi, "wer": 1.0 - pi}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage1_pose2gloss.yaml")
    ap.add_argument("--dataset-root", default=None,
                    help="Override dataset_root in paths.yaml (sets DATASET_ROOT env var)")
    args = ap.parse_args()

    if args.dataset_root:
        import os
        os.environ["DATASET_ROOT"] = args.dataset_root

    cfg = load_config(args.config)
    setup_logging(Path(cfg.log_dir) / cfg.name)

    vocab = GlossVocab.from_file(cfg.vocab_file)
    transform = build_video_transform(cfg, train=True)
    train_ds = build_dataset(cfg, vocab, split="train", transform=transform)
    val_ds = build_dataset(cfg, vocab, split="dev",
                           transform=build_video_transform(cfg, train=False))

    train_loader = DataLoader(
        train_ds, batch_size=int(cfg.train.batch_size), shuffle=cfg.data.shuffle,
        num_workers=int(cfg.num_workers), pin_memory=bool(cfg.pin_memory),
        collate_fn=collate_video_batch,
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(cfg.train.batch_size), shuffle=False,
        num_workers=int(cfg.num_workers), pin_memory=bool(cfg.pin_memory),
        collate_fn=collate_video_batch,
    )

    pose_extractor = build_pose_extractor(cfg)
    compiler = PoseAwareFeatureCompiler(feature_dim=pose_extractor.output_dim)

    model = Stage1Model(
        pose_input_dim=pose_extractor.output_dim,
        latent_dim=int(cfg.model.latent_dim),
        codebook_size=int(cfg.model.codebook_size),
        codebook_dim=int(cfg.model.codebook_dim),
        decoder_layers=int(cfg.model.decoder_layers),
        num_heads=int(cfg.model.num_fusion_heads),
        vocab_size=int(cfg.vocab.vocab_size),
        num_fusion_layers=int(cfg.model.num_fusion_layers),
        dropout=float(cfg.model.dropout),
    )

    trainer = Stage1Trainer(
        model=model, cfg=cfg, train_loader=train_loader, val_loader=val_loader,
        vocab=vocab, pose_extractor=pose_extractor, compiler=compiler,
    )
    trainer.train()


if __name__ == "__main__":
    main()
