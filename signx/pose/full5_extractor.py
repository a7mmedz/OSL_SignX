"""Full 5-modality pose extractor matching the SignX paper (Stage 1).

Concatenates per-frame features from all five modalities:

  Modality       Extractor           Dims
  ─────────────────────────────────────────
  1. MediaPipe   mediapipe_extractor  258
  2. DWPose      dwpose_extractor     399
  3. SMPLer-X    smplerx_extractor    432
  4. PrimeDepth  primedepth_extractor 480
  5. Sapiens     sapiens_extractor    390
  ─────────────────────────────────────────
  Total                              1959

All five extractors must be installed. See each module's docstring for
per-modality installation instructions.

Usage in configs/default.yaml:
    pose:
      backend: full5
      full_dim: 1959
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import torch

from .pose_extractor import PoseExtractor
from .mediapipe_extractor import MediaPipePoseExtractor
from .dwpose_extractor import DWPoseExtractor
from .smplerx_extractor import SMPLerXExtractor
from .primedepth_extractor import PrimeDepthExtractor
from .sapiens_extractor import SapiensExtractor

MODALITY_DIMS = [258, 399, 432, 480, 390]
OUTPUT_DIM    = sum(MODALITY_DIMS)   # 1959


def check_full5_available() -> dict[str, bool]:
    """Return availability of each modality without raising."""
    status = {}
    for name, cls in [
        ("mediapipe", MediaPipePoseExtractor),
        ("dwpose",    DWPoseExtractor),
        ("smplerx",   SMPLerXExtractor),
        ("primedepth",PrimeDepthExtractor),
        ("sapiens",   SapiensExtractor),
    ]:
        try:
            cls()
            status[name] = True
        except (ImportError, RuntimeError):
            status[name] = False
    return status


class Full5Extractor(PoseExtractor):
    """Concatenated 5-modality extractor (1959-dim per frame).

    All five modalities must be installed and available.
    Run `from signx.pose.full5_extractor import check_full5_available`
    to check which ones are ready before training.
    """

    output_dim: int = OUTPUT_DIM

    def __init__(self) -> None:
        self._extractors: List[PoseExtractor] = [
            MediaPipePoseExtractor(output_dim=258),
            DWPoseExtractor(),
            SMPLerXExtractor(),
            PrimeDepthExtractor(),
            SapiensExtractor(),
        ]

    def extract(self, video_path: str | Path) -> torch.Tensor:
        """Run all 5 extractors and concatenate along the feature axis.

        Returns:
            (T, 1959) float tensor.
        """
        parts = []
        min_t = None
        for extractor in self._extractors:
            feat = extractor(video_path)   # (T_i, D_i)
            parts.append(feat)
            min_t = feat.shape[0] if min_t is None else min(min_t, feat.shape[0])

        # Align temporal length — truncate to the shortest (rare edge case)
        aligned = [p[:min_t] for p in parts]
        return torch.cat(aligned, dim=-1)  # (T, 1959)
