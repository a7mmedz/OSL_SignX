"""Meta Sapiens whole-body keypoint extractor (Stage 1, modality 5).

Extracts 130 whole-body keypoints using Meta's Sapiens model:
  - 17  body keypoints (COCO)
  - 68  face keypoints
  - 21  left  hand keypoints
  - 21  right hand keypoints
  - 3   extra (pelvis, neck, head-top)
Each keypoint: (x, y, score) -> 130 × 3 = 390 dims per frame.

Installation (server) — two options:

  Option A (HuggingFace, easier):
    pip install transformers accelerate

  Option B (official repo):
    git clone https://github.com/facebookresearch/sapiens.git
    pip install -e sapiens/

References:
    Sapiens: https://arxiv.org/abs/2408.12569
    HF hub:  https://huggingface.co/facebook/sapiens-pose-0.3b-torchscript
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .pose_extractor import PoseExtractor

OUTPUT_DIM  = 390   # 130 keypoints × 3
_NUM_KP     = 130
_HF_MODEL   = "facebook/sapiens-pose-0.3b-torchscript"


class SapiensExtractor(PoseExtractor):
    """Per-frame body keypoints from Meta Sapiens (via HuggingFace).

    Uses the 0.3B torchscript variant — smaller and faster than 1B/2B.
    Downloads ~1.2 GB on first use.
    """

    output_dim: int = OUTPUT_DIM

    def __init__(self, model_id: str = _HF_MODEL) -> None:
        self._model_id = model_id
        self._model    = None
        self._processor = None

        try:
            from transformers import AutoProcessor  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "Sapiens requires the HuggingFace transformers library.\n"
                "  pip install transformers accelerate"
            ) from e

    def _load_model(self):
        if self._model is not None:
            return self._model, self._processor
        import torch
        from transformers import AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self._model_id)
        self._model = torch.jit.load(
            # HF hub caches torchscript files automatically
            self._model_id,
            map_location="cpu",
        )
        self._model.eval()
        return self._model, self._processor

    def _parse_heatmaps(self, heatmaps: np.ndarray) -> np.ndarray:
        """Convert (NUM_KP, H, W) heatmaps to (NUM_KP, 3) keypoints by argmax."""
        num_kp, h, w = heatmaps.shape
        flat = heatmaps.reshape(num_kp, -1)
        idx  = flat.argmax(axis=1)
        ys   = idx // w
        xs   = idx %  w
        scores = flat[np.arange(num_kp), idx]
        # Normalise to [0, 1]
        kps = np.stack([xs / max(w - 1, 1), ys / max(h - 1, 1), scores], axis=1)
        return kps.astype(np.float32)   # (num_kp, 3)

    def extract(self, video_path: str | Path) -> torch.Tensor:
        import cv2
        from PIL import Image as PILImage

        model, processor = self._load_model()

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

                inputs = processor(images=pil_img, return_tensors="pt")
                with torch.no_grad():
                    heatmaps = model(**inputs).squeeze(0).cpu().numpy()         # (NUM_KP, H, W)

                # Pad / truncate to exactly _NUM_KP keypoints
                if heatmaps.shape[0] >= _NUM_KP:
                    heatmaps = heatmaps[:_NUM_KP]
                else:
                    pad = np.zeros((_NUM_KP - heatmaps.shape[0], *heatmaps.shape[1:]))
                    heatmaps = np.concatenate([heatmaps, pad], axis=0)

                kps = self._parse_heatmaps(heatmaps)                            # (130, 3)
                feats.append(kps.flatten())                                      # (390,)
        finally:
            cap.release()

        if not feats:
            return torch.zeros(1, OUTPUT_DIM, dtype=torch.float32)
        return torch.from_numpy(np.stack(feats, axis=0))
