"""MediaPipe-based per-frame pose extractor.

Extracts:
  - 33 pose landmarks (x, y, z, visibility)  -> 132 dims
  - 21 left  hand landmarks (x, y, z)         ->  63 dims
  - 21 right hand landmarks (x, y, z)         ->  63 dims
Total per frame = 258 dims (matches `cfg.pose.mediapipe_dim`).

Face landmarks are intentionally excluded for speed; add them later if needed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .pose_extractor import PoseExtractor


class MediaPipePoseExtractor(PoseExtractor):
    """Single-modality MediaPipe pose extraction."""

    def __init__(self, output_dim: int = 258) -> None:
        try:
            import mediapipe as mp  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "mediapipe is required for the mediapipe pose backend. "
                "Install with: uv pip install mediapipe"
            ) from e
        self.output_dim = output_dim

    @staticmethod
    def _empty_pose() -> np.ndarray:
        return np.zeros(132, dtype=np.float32)

    @staticmethod
    def _empty_hand() -> np.ndarray:
        return np.zeros(63, dtype=np.float32)

    def extract(self, video_path: str | Path) -> torch.Tensor:
        import cv2
        import mediapipe as mp

        holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=False,
        )

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            holistic.close()
            raise RuntimeError(f"Could not open video: {video_path}")

        feats = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)

                if results.pose_landmarks:
                    pose = np.array(
                        [[lm.x, lm.y, lm.z, lm.visibility]
                         for lm in results.pose_landmarks.landmark],
                        dtype=np.float32,
                    ).flatten()
                else:
                    pose = self._empty_pose()

                if results.left_hand_landmarks:
                    lh = np.array(
                        [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark],
                        dtype=np.float32,
                    ).flatten()
                else:
                    lh = self._empty_hand()

                if results.right_hand_landmarks:
                    rh = np.array(
                        [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark],
                        dtype=np.float32,
                    ).flatten()
                else:
                    rh = self._empty_hand()

                feats.append(np.concatenate([pose, lh, rh], axis=0))
        finally:
            cap.release()
            holistic.close()

        if not feats:
            return torch.zeros(1, self.output_dim, dtype=torch.float32)
        return torch.from_numpy(np.stack(feats, axis=0))
