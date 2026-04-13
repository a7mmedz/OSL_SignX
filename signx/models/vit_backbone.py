"""Per-frame Vision Transformer backbone used in Stage 2 (Video2Pose).

Wraps a `timm` ViT and exposes a clean interface that:
  - Accepts (B, T, 3, H, W) video tensors.
  - Returns (B, T, vit_feature_dim) per-frame features.

If timm is unavailable (e.g. CI), falls back to a tiny CNN with the same API.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _FallbackBackbone(nn.Module):
    """Minimal CNN backbone for unit tests when timm is missing."""

    def __init__(self, feature_dim: int = 768) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ViTFrameBackbone(nn.Module):
    """Per-frame backbone returning a feature vector per frame."""

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        feature_dim: int = 768,
        img_size: int = 224,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        try:
            import timm  # type: ignore
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,        # global pooled features
                global_pool="avg",
                img_size=img_size,
            )
            # timm reports the actual feature dim
            self.feature_dim = int(self.backbone.num_features)
        except ImportError:
            self.backbone = _FallbackBackbone(feature_dim=feature_dim)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """video: (B, T, 3, H, W). Returns (B, T, feature_dim)."""
        b, t, c, h, w = video.shape
        flat = video.reshape(b * t, c, h, w)
        feats = self.backbone(flat)
        return feats.reshape(b, t, -1)
