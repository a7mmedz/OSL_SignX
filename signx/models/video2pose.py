"""Video2Pose module (Stage 2).

Takes raw RGB frames and predicts the 2048-dim pose latent that Stage 1
produces from real pose features. Once trained, this lets Stage 3 run on RGB
videos directly without needing any pose extractor at inference time.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .vit_backbone import ViTFrameBackbone


class Video2PoseModel(nn.Module):
    """ViT frame encoder + linear projection to the SignX latent space."""

    def __init__(
        self,
        vit_name: str = "vit_base_patch16_224",
        vit_pretrained: bool = True,
        latent_dim: int = 2048,
        dropout: float = 0.1,
        img_size: int = 224,
    ) -> None:
        super().__init__()
        self.backbone = ViTFrameBackbone(
            model_name=vit_name,
            pretrained=vit_pretrained,
            img_size=img_size,
        )
        feat_dim = self.backbone.feature_dim
        self.proj = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )
        self.latent_dim = latent_dim

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """video: (B, T, 3, H, W). Returns (B, T, latent_dim)."""
        feats = self.backbone(video)        # (B, T, feat_dim)
        return self.proj(feats)              # (B, T, latent_dim)
