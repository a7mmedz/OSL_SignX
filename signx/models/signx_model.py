"""Top-level SignX models for each training stage.

  Stage1Model : pose -> latent -> gloss decoder      (Pose2Gloss)
  Stage2Model : RGB  -> ViT     -> latent (matches stage 1)
  Stage3Model : latent -> temporal -> transformer    (CSLR)
  SignXModel  : convenience wrapper exposing all three for end-to-end inference
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .codebook_decoder import CodeBookDecoder
from .pose_fusion import PoseFusionEncoder
from .temporal_model import TemporalModel
from .transformer_decoder import SignXTransformer
from .video2pose import Video2PoseModel


@dataclass
class FeaturePruningState:
    """Holds a learned binary mask over latent dims (Adaptive Feature Pruning)."""
    mask: torch.Tensor          # (latent_dim,) bool/float
    fisher: torch.Tensor        # (latent_dim,) float — running fisher info estimate


class Stage1Model(nn.Module):
    """Pose-to-Gloss model trained in Stage 1."""

    def __init__(
        self,
        pose_input_dim: int,
        latent_dim: int,
        codebook_size: int,
        codebook_dim: int,
        decoder_layers: int,
        num_heads: int,
        vocab_size: int,
        num_fusion_layers: int = 4,
        dropout: float = 0.1,
        max_target_len: int = 64,
    ) -> None:
        super().__init__()
        self.pose_fusion = PoseFusionEncoder(
            pose_input_dim=pose_input_dim,
            latent_dim=latent_dim,
            num_heads=num_heads,
            num_layers=num_fusion_layers,
            dropout=dropout,
        )
        self.decoder = CodeBookDecoder(
            latent_dim=latent_dim,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            num_layers=decoder_layers,
            num_heads=num_heads,
            vocab_size=vocab_size,
            max_target_len=max_target_len,
            dropout=dropout,
        )
        self.latent_dim = latent_dim

    def encode_pose(self, pose: torch.Tensor) -> torch.Tensor:
        return self.pose_fusion(pose)

    def forward(self, pose: torch.Tensor, target: torch.Tensor) -> dict:
        latent = self.encode_pose(pose)
        logits = self.decoder(latent, target)
        return {"latent": latent, "logits": logits}


class Stage2Model(nn.Module):
    """RGB-to-Latent model trained in Stage 2.

    Optionally holds a frozen Stage 1 model so the trainer can compute
    ground-truth pose latents on the fly during training.
    """

    def __init__(
        self,
        latent_dim: int,
        vit_name: str = "vit_base_patch16_224",
        vit_pretrained: bool = True,
        dropout: float = 0.1,
        stage1_model: Optional[Stage1Model] = None,
    ) -> None:
        super().__init__()
        self.video2pose = Video2PoseModel(
            vit_name=vit_name,
            vit_pretrained=vit_pretrained,
            latent_dim=latent_dim,
            dropout=dropout,
        )
        self.stage1_model = stage1_model
        if self.stage1_model is not None:
            for p in self.stage1_model.parameters():
                p.requires_grad = False
            self.stage1_model.eval()

    def forward(self, video: torch.Tensor, pose: Optional[torch.Tensor] = None) -> dict:
        pred_latent = self.video2pose(video)
        out = {"pred_latent": pred_latent}
        if pose is not None and self.stage1_model is not None:
            with torch.no_grad():
                gt_latent = self.stage1_model.encode_pose(pose)
            out["gt_latent"] = gt_latent
        return out


class Stage3Model(nn.Module):
    """CSLR model trained in Stage 3 on latent features."""

    def __init__(
        self,
        latent_dim: int,
        pruned_dim: int,
        tconv_channels: list[int],
        tconv_kernels: list[int],
        tconv_strides: list[int],
        lstm_hidden: int,
        lstm_layers: int,
        bidirectional: bool,
        transformer_layers: int,
        transformer_dim: int,
        transformer_heads: int,
        transformer_ffn: int,
        vocab_size: int,
        dropout_attn: float,
        dropout_relu: float,
        dropout_res: float,
        max_target_len: int = 128,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.pruned_dim = pruned_dim
        # Adaptive pruning state — initialized as all-ones; updated by trainer.
        self.register_buffer("prune_mask", torch.ones(latent_dim))
        self.register_buffer("fisher_info", torch.zeros(latent_dim))
        self.input_norm = nn.LayerNorm(latent_dim)

        self.temporal = TemporalModel(
            in_dim=latent_dim,
            tconv_channels=tconv_channels,
            tconv_kernels=tconv_kernels,
            tconv_strides=tconv_strides,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            bidirectional=bidirectional,
            dropout=dropout_attn,
        )
        self.transformer = SignXTransformer(
            input_dim=self.temporal.output_dim,
            d_model=transformer_dim,
            nhead=transformer_heads,
            num_encoder_layers=transformer_layers,
            num_decoder_layers=transformer_layers,
            dim_feedforward=transformer_ffn,
            vocab_size=vocab_size,
            dropout_attn=dropout_attn,
            dropout_relu=dropout_relu,
            dropout_res=dropout_res,
            max_target_len=max_target_len,
        )

    def apply_pruning(self, latent: torch.Tensor) -> torch.Tensor:
        """Multiply (B, T, D) by the broadcast prune mask."""
        return latent * self.prune_mask.view(1, 1, -1)

    def update_pruning(self, num_keep: int | None = None) -> None:
        """Refresh `prune_mask` from `fisher_info` keeping the top-K dims."""
        k = num_keep if num_keep is not None else self.pruned_dim
        k = min(k, self.latent_dim)
        topk = torch.topk(self.fisher_info, k=k).indices
        new_mask = torch.zeros_like(self.prune_mask)
        new_mask[topk] = 1.0
        self.prune_mask.copy_(new_mask)

    def forward(
        self,
        latent: torch.Tensor,
        target: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> dict:
        """latent: (B, T, latent_dim) — expected to come from Stage 2."""
        x = self.input_norm(self.apply_pruning(latent))
        feats, new_lengths = self.temporal(x, lengths=lengths)
        out = self.transformer(feats, target=target)
        out["lengths"] = new_lengths
        out["features"] = feats
        return out


class SignXModel(nn.Module):
    """Full pipeline wrapping all three stages for end-to-end inference."""

    def __init__(self, stage1: Stage1Model, stage2: Stage2Model, stage3: Stage3Model) -> None:
        super().__init__()
        self.stage1 = stage1
        self.stage2 = stage2
        self.stage3 = stage3

    @torch.no_grad()
    def predict(self, video: torch.Tensor) -> dict:
        """video: (B, T, 3, H, W). Returns Stage 3 outputs."""
        pred_latent = self.stage2.video2pose(video)
        return self.stage3(pred_latent)
