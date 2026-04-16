"""Abstract pose extractor interface and factory.

The SignX paper concatenates 5 modalities into a 1959-dim per-frame vector.
For our local development we default to MediaPipe-only (258 dims). The
`build_pose_extractor` factory dispatches based on `cfg.pose.backend`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import torch


class PoseExtractor(ABC):
    """Per-frame pose feature extractor.

    Implementations must return a (T, D) float tensor for an input video.
    """

    output_dim: int = 0

    @abstractmethod
    def extract(self, video_path: str | Path) -> torch.Tensor:
        """Run the extractor on a video file and return (T, output_dim)."""

    def __call__(self, video_path: str | Path) -> torch.Tensor:
        return self.extract(video_path)


class PrecomputedPoseExtractor(PoseExtractor):
    """Loads pose features from a directory of `.pt` files (one per video stem)."""

    def __init__(self, cache_dir: str | Path, output_dim: int) -> None:
        self.cache_dir = Path(cache_dir)
        self.output_dim = output_dim

    def extract(self, video_path: str | Path) -> torch.Tensor:
        stem = Path(video_path).stem
        cache_path = self.cache_dir / f"{stem}.pt"
        if not cache_path.exists():
            raise FileNotFoundError(f"No precomputed pose for {stem} at {cache_path}")
        return torch.load(cache_path, map_location="cpu")


class ZeroPoseExtractor(PoseExtractor):
    """No-op extractor used for unit tests and CI without mediapipe installed."""

    def __init__(self, output_dim: int, num_frames: int = 16) -> None:
        self.output_dim = output_dim
        self.num_frames = num_frames

    def extract(self, video_path: str | Path) -> torch.Tensor:
        return torch.zeros(self.num_frames, self.output_dim)


def build_pose_extractor(cfg) -> PoseExtractor:
    """Construct an extractor from a top-level config object."""
    pose_cfg = cfg.pose
    backend = str(pose_cfg.backend).lower()

    if backend == "mediapipe":
        try:
            from .mediapipe_extractor import MediaPipePoseExtractor
            return MediaPipePoseExtractor(output_dim=int(pose_cfg.mediapipe_dim))
        except ImportError:
            return ZeroPoseExtractor(output_dim=int(pose_cfg.mediapipe_dim))

    if backend == "full5":
        from .full5_extractor import Full5Extractor
        return Full5Extractor()

    if backend == "precomputed":
        if pose_cfg.precomputed_dir is None:
            raise ValueError("pose.precomputed_dir must be set when backend=precomputed")
        return PrecomputedPoseExtractor(
            cache_dir=pose_cfg.precomputed_dir,
            output_dim=int(pose_cfg.full_dim),
        )

    raise ValueError(f"Unknown pose backend: {backend}")
