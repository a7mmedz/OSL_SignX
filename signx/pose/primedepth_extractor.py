"""PrimeDepth monocular depth feature extractor (Stage 1, modality 4).

Runs PrimeDepth to estimate per-frame depth maps, then applies global
average pooling over a 20×24 spatial grid to produce a 480-dim feature
vector per frame (480 = 20 × 24).

Installation (server):
    git clone https://github.com/apple/ml-depth-pro.git   # PrimeDepth uses Depth Pro
    cd ml-depth-pro && pip install -e .
    # Download weights: python -m depth_pro.cli.download

    # Alternative — use the original PrimeDepth repo:
    git clone https://github.com/google-research/google-research.git
    cd google-research/primedepth && pip install -r requirements.txt

References:
    PrimeDepth: https://arxiv.org/abs/2409.09896
    Depth Pro:  https://github.com/apple/ml-depth-pro
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .pose_extractor import PoseExtractor

OUTPUT_DIM  = 480    # 20 × 24 spatial grid after adaptive average pool
_GRID_H     = 20
_GRID_W     = 24


class PrimeDepthExtractor(PoseExtractor):
    """Per-frame depth features extracted by PrimeDepth / Depth Pro.

    The raw depth map (H × W) is resized and pooled to a fixed
    (_GRID_H × _GRID_W) = 480-dim vector, then z-score normalised
    across the frame to remove absolute scale ambiguity.
    """

    output_dim: int = OUTPUT_DIM

    def __init__(self) -> None:
        self._model = None
        self._transform = None
        try:
            import depth_pro  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "PrimeDepth requires Depth Pro.\n"
                "  git clone https://github.com/apple/ml-depth-pro.git\n"
                "  cd ml-depth-pro && pip install -e .\n"
                "  python -m depth_pro.cli.download_models"
            ) from e

    def _load_model(self):
        if self._model is not None:
            return self._model, self._transform
        import depth_pro
        self._model, self._transform = depth_pro.create_model_and_transforms()
        self._model.eval()
        return self._model, self._transform

    def _depth_to_vec(self, depth: np.ndarray) -> np.ndarray:
        """Pool depth map (H, W) to OUTPUT_DIM-dim vector."""
        import torch
        import torch.nn.functional as F

        t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float()          # (1,1,H,W)
        pooled = F.adaptive_avg_pool2d(t, (_GRID_H, _GRID_W))                  # (1,1,20,24)
        vec = pooled.squeeze().numpy().flatten()                                 # (480,)
        # Z-score normalise to remove absolute scale
        mu, std = vec.mean(), vec.std()
        if std > 1e-6:
            vec = (vec - mu) / std
        return vec.astype(np.float32)

    def extract(self, video_path: str | Path) -> torch.Tensor:
        import cv2
        from PIL import Image as PILImage

        model, transform = self._load_model()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        feats = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = PILImage.fromarray(rgb)
                inp = transform(pil_img)

                with torch.no_grad():
                    prediction = model.infer(inp)
                depth = prediction["depth"].squeeze().cpu().numpy()             # (H, W)
                feats.append(self._depth_to_vec(depth))
        finally:
            cap.release()

        if not feats:
            return torch.zeros(1, OUTPUT_DIM, dtype=torch.float32)
        return torch.from_numpy(np.stack(feats, axis=0))
