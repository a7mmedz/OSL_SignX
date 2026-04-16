"""DWPose whole-body pose extractor (Stage 1, modality 2).

Extracts 133 whole-body keypoints (COCO-WholeBody format):
  - 17 body keypoints
  - 6  foot keypoints
  - 68 face keypoints
  - 42 hand keypoints (21 left + 21 right)
Each keypoint: (x, y, score) -> 133 × 3 = 399 dims per frame.

Installation (server):
    pip install openmim
    mim install mmengine "mmcv>=2.0.0" mmdet "mmpose>=1.0.0"
    # DWPose config + checkpoint are downloaded on first use via mmpose.

References:
    DWPose: https://github.com/IDEA-Research/DWPose
    RTMPose whole-body: https://github.com/open-mmlab/mmpose
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .pose_extractor import PoseExtractor

OUTPUT_DIM = 399   # 133 keypoints × 3 (x, y, score)
_NUM_KP   = 133


class DWPoseExtractor(PoseExtractor):
    """Whole-body pose via RTMPose (DWPose backbone).

    Downloads the model weights on first call (~200 MB).
    """

    output_dim: int = OUTPUT_DIM

    def __init__(self) -> None:
        try:
            from mmpose.apis import init_model  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "DWPose requires MMPose. Install with:\n"
                "  pip install openmim\n"
                "  mim install mmengine 'mmcv>=2.0.0' mmdet 'mmpose>=1.0.0'"
            ) from e
        self._inferencer = None   # lazy init to avoid loading weights at import time

    def _get_inferencer(self):
        if self._inferencer is not None:
            return self._inferencer
        from mmpose.apis import MMPoseInferencer
        # RTMPose whole-body — downloads ~200 MB on first run
        self._inferencer = MMPoseInferencer(
            pose2d="wholebody",
            device="cpu",    # extractor runs on CPU; GPU used by training loop
        )
        return self._inferencer

    def extract(self, video_path: str | Path) -> torch.Tensor:
        import cv2
        inferencer = self._get_inferencer()

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

                result = next(inferencer(rgb, return_vis=False))
                preds = result.get("predictions", [[]])[0]

                if preds:
                    kps   = np.array(preds[0]["keypoints"],       dtype=np.float32)   # (133, 2)
                    scores= np.array(preds[0]["keypoint_scores"], dtype=np.float32)   # (133,)
                    vec = np.concatenate([kps.flatten(), scores], axis=0)              # (399,)
                else:
                    vec = np.zeros(OUTPUT_DIM, dtype=np.float32)

                feats.append(vec)
        finally:
            cap.release()

        if not feats:
            return torch.zeros(1, OUTPUT_DIM, dtype=torch.float32)
        return torch.from_numpy(np.stack(feats, axis=0))
