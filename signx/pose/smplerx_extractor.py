"""SMPLer-X 3-D whole-body mesh extractor (Stage 1, modality 3).

Estimates SMPL-X body parameters and returns the 144 3-D body joints
as a flattened vector: 144 joints × 3 (x, y, z) = 432 dims per frame.

Installation (server):
    # 1. Clone and install SMPLer-X
    git clone https://github.com/caizhongang/SMPLer-X.git
    cd SMPLer-X
    pip install -v -e .

    # 2. Download SMPL-X body model files from https://smpl-x.is.tue.mpg.de/
    #    Place under: SMPLer-X/data/body_models/smplx/

    # 3. Download pretrained checkpoint:
    #    https://github.com/caizhongang/SMPLer-X  (see Releases)
    #    Set env var: SMPLERX_CKPT=/path/to/checkpoint.pth.tar

References:
    SMPLer-X: https://github.com/caizhongang/SMPLer-X
    SMPL-X:   https://smpl-x.is.tue.mpg.de/
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from .pose_extractor import PoseExtractor

OUTPUT_DIM  = 432   # 144 joints × 3
_NUM_JOINTS = 144


class SMPLerXExtractor(PoseExtractor):
    """Per-frame 3-D body joints estimated by SMPLer-X.

    Requires:
      - SMPLer-X installed and importable as `main.inference`
      - SMPL-X body model files present
      - SMPLERX_CKPT environment variable pointing to checkpoint

    Falls back to zeros with an error on missing deps.
    """

    output_dim: int = OUTPUT_DIM

    def __init__(self, checkpoint: str | None = None) -> None:
        self._ckpt = checkpoint or os.environ.get("SMPLERX_CKPT")
        self._model = None

        try:
            import smpler_x  # noqa: F401  (top-level package after install)
        except ImportError as e:
            raise ImportError(
                "SMPLer-X is required for this backend.\n"
                "  git clone https://github.com/caizhongang/SMPLer-X.git\n"
                "  cd SMPLer-X && pip install -v -e .\n"
                "Then set: export SMPLERX_CKPT=/path/to/checkpoint.pth.tar"
            ) from e

    def _load_model(self):
        if self._model is not None:
            return self._model
        if self._ckpt is None:
            raise RuntimeError(
                "Set SMPLERX_CKPT environment variable to the SMPLer-X checkpoint path."
            )
        from smpler_x.inference import SMPLerXInferencer
        self._model = SMPLerXInferencer(self._ckpt)
        return self._model

    def extract(self, video_path: str | Path) -> torch.Tensor:
        import cv2
        model = self._load_model()

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

                output = model.infer(rgb)
                # output["joints_3d"]: (N_persons, 144, 3) — take first person
                if output is not None and len(output.get("joints_3d", [])) > 0:
                    joints = np.array(output["joints_3d"][0], dtype=np.float32)  # (144, 3)
                    joints = joints[:_NUM_JOINTS]                                 # ensure size
                    if joints.shape[0] < _NUM_JOINTS:
                        pad = np.zeros((_NUM_JOINTS - joints.shape[0], 3), dtype=np.float32)
                        joints = np.vstack([joints, pad])
                    vec = joints.flatten()                                        # (432,)
                else:
                    vec = np.zeros(OUTPUT_DIM, dtype=np.float32)

                feats.append(vec)
        finally:
            cap.release()

        if not feats:
            return torch.zeros(1, OUTPUT_DIM, dtype=torch.float32)
        return torch.from_numpy(np.stack(feats, axis=0))
