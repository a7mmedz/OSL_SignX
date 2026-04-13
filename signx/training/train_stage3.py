"""Stage 3 training entrypoint: Continuous Sign Language Recognition."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data import GlossVocab, build_dataset, build_video_transform, collate_video_batch
from ..models import Stage2Model, Stage3Model, distillation_loss, latent_regularizer
from ..models.losses import word_matching_loss
from ..utils import load_checkpoint, load_config, setup_logging
from .metrics import compute_wer
from .trainer import BaseTrainer


class Stage3Trainer(BaseTrainer):
    """CTC + CE + KD joint trainer for CSLR."""

    def __init__(self, model: Stage3Model, cfg, train_loader, val_loader, vocab, video2pose):
        super().__init__(model, cfg, train_loader, val_loader)
        self.vocab = vocab
        self.video2pose = video2pose.to(self.device)
        self.video2pose.eval()
        for p in self.video2pose.parameters():
            p.requires_grad = False
        self.ctc_loss = torch.nn.CTCLoss(
            blank=vocab.blank_id, reduction="mean", zero_infinity=True
        )

    @torch.no_grad()
    def _get_latent(self, videos: torch.Tensor) -> torch.Tensor:
        return self.video2pose(videos)

    def compute_loss(self, batch: Dict) -> Dict[str, torch.Tensor]:
        videos = batch["videos"]                            # (B, T, C, H, W)
        gloss_targets = batch["glosses"]                    # (B, L)
        gloss_lengths = batch["gloss_lengths"]              # (B,)
        vid_lengths = batch["video_lengths"]                # (B,)

        latent = self._get_latent(videos)                   # (B, T, D)

        bos = torch.full((gloss_targets.shape[0], 1), self.vocab.blank_id, device=self.device)
        dec_input = torch.cat([bos, gloss_targets[:, :-1]], dim=1)

        out = self.model(latent, target=dec_input, lengths=vid_lengths)
        ctc_logits = out["ctc_logits"]                      # (B, T', V)
        dec_logits = out["dec_logits"]                      # (B, L, V)
        new_lengths = out["lengths"]                        # (B,)

        # CTC loss expects (T, B, V), log-probs
        log_probs = F.log_softmax(ctc_logits, dim=-1).permute(1, 0, 2)
        input_lengths = new_lengths.clamp(max=log_probs.shape[0])
        l_ctc = self.ctc_loss(log_probs, gloss_targets, input_lengths, gloss_lengths)

        # CE decoder loss
        ls = float(self.cfg.loss.label_smoothing)
        l_ce = word_matching_loss(dec_logits, gloss_targets, pad_id=0, label_smoothing=ls)

        # Latent regularizer
        l_reg = latent_regularizer(latent, weight=float(self.cfg.loss.lambda_latent_reg))

        # Knowledge distillation (teacher = CTC logits, student = dec_logits)
        # Both over the same sequence length via interpolation
        if dec_logits.shape[1] != ctc_logits.shape[1]:
            teacher = F.interpolate(
                ctc_logits.transpose(1, 2),
                size=dec_logits.shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        else:
            teacher = ctc_logits
        l_kd = distillation_loss(
            dec_logits, teacher.detach(), temperature=float(self.cfg.loss.kd_temperature)
        )

        loss = (
            float(self.cfg.loss.lambda_ctc) * l_ctc
            + float(self.cfg.loss.lambda_ce) * l_ce
            + float(self.cfg.loss.lambda_kd) * l_kd
            + l_reg
        )
        return {"loss": loss, "ctc": l_ctc, "ce": l_ce, "kd": l_kd, "reg": l_reg}

    @torch.no_grad()
    def evaluate(self, loader) -> Dict[str, float]:
        from ..inference.beam_search import ctc_greedy_decode
        self.model.eval()
        refs, hyps = [], []
        for batch in loader:
            batch = self._to_device(batch)
            latent = self._get_latent(batch["videos"])
            out = self.model(latent, lengths=batch["video_lengths"])
            ctc_logits = out["ctc_logits"]
            for i in range(ctc_logits.shape[0]):
                log_p = F.log_softmax(ctc_logits[i], dim=-1)
                hyp = ctc_greedy_decode(log_p, blank_id=self.vocab.blank_id)
                hyps.append(hyp)
                refs.append([int(t) for t in batch["glosses"][i].tolist() if t != 0])
        wer = compute_wer(refs, hyps)
        return {"wer": wer}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage3_cslr.yaml")
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

    train_loader = DataLoader(
        train_ds, batch_size=int(cfg.train.batch_size), shuffle=cfg.data.shuffle,
        num_workers=int(cfg.num_workers), pin_memory=bool(cfg.pin_memory),
        collate_fn=collate_video_batch,
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(cfg.train.batch_size), shuffle=False,
        num_workers=int(cfg.num_workers), collate_fn=collate_video_batch,
    )

    # Load frozen Stage 2 video2pose
    stage2_cfg = load_config("configs/stage2_video2pose.yaml")
    video2pose = Stage2Model(
        latent_dim=int(stage2_cfg.model.latent_dim),
        vit_name=str(stage2_cfg.model.vit_name),
        vit_pretrained=False,
    )
    ckpt = Path(cfg.model.stage2_checkpoint)
    if ckpt.exists():
        load_checkpoint(ckpt, video2pose, strict=False)

    model = Stage3Model(
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

    trainer = Stage3Trainer(model, cfg, train_loader, val_loader, vocab, video2pose.video2pose)
    trainer.train()


if __name__ == "__main__":
    main()
