"""Video preprocessing and augmentation pipelines.

Inputs throughout the codebase are uint8 tensors of shape (T, H, W, 3).
Outputs are float32 tensors of shape (T, 3, H, W) normalized with ImageNet stats.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F


@dataclass
class VideoTransformConfig:
    image_size: int = 224
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    train: bool = False
    spatial_jitter: float = 0.0
    noise_std: float = 0.0


class VideoTransform:
    """Stateless callable that resizes/normalizes a (T, H, W, 3) uint8 video tensor."""

    def __init__(self, cfg: VideoTransformConfig) -> None:
        self.cfg = cfg
        self.mean = torch.tensor(cfg.mean).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg.std).view(1, 3, 1, 1)

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        if video.dtype != torch.uint8:
            video = video.to(torch.uint8)
        # (T, H, W, 3) -> (T, 3, H, W)
        x = video.permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(
            x,
            size=(self.cfg.image_size, self.cfg.image_size),
            mode="bilinear",
            align_corners=False,
        )
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        if self.cfg.train:
            if self.cfg.spatial_jitter > 0:
                shift = (torch.rand(2) * 2 - 1) * self.cfg.spatial_jitter
                x = torch.roll(
                    x,
                    shifts=(int(shift[0] * x.shape[-2]), int(shift[1] * x.shape[-1])),
                    dims=(-2, -1),
                )
            if self.cfg.noise_std > 0:
                x = x + torch.randn_like(x) * self.cfg.noise_std
        return x


def build_video_transform(cfg, train: bool = False) -> VideoTransform:
    """Construct a `VideoTransform` from a top-level config object."""
    video_cfg = cfg.video if hasattr(cfg, "video") else cfg
    return VideoTransform(
        VideoTransformConfig(
            image_size=int(video_cfg.image_size),
            mean=tuple(video_cfg.mean),
            std=tuple(video_cfg.std),
            train=train,
        )
    )
