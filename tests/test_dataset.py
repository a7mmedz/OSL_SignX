"""Tests for dataset utilities (collate, transforms).

Collate and transform tests run on CUDA when available so we exercise
the device-aware paths (mean/std normalization, interpolation).
"""
import torch
import pytest

from signx.data.collate import collate_video_batch
from signx.data.transforms import VideoTransform, VideoTransformConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_sample(T: int, H: int = 112, W: int = 112, n_glosses: int = 1):
    return {
        "video": torch.randint(0, 255, (T, 3, H, W), dtype=torch.float32),
        "gloss_ids": torch.randint(1, 10, (n_glosses,), dtype=torch.long),
        "video_length": T,
        "gloss_length": n_glosses,
        "item_id": 1,
        "signer_id": 2,
        "take_id": 1,
        "path": "dummy.mp4",
    }


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def test_collate_pads_videos():
    batch = [_make_sample(T=10), _make_sample(T=15), _make_sample(T=8)]
    out = collate_video_batch(batch)
    assert out["videos"].shape == (3, 15, 3, 112, 112)
    assert out["video_lengths"].tolist() == [10, 15, 8]


def test_collate_pads_glosses():
    batch = [_make_sample(T=5, n_glosses=2), _make_sample(T=5, n_glosses=4)]
    out = collate_video_batch(batch)
    assert out["glosses"].shape == (2, 4)
    assert out["gloss_lengths"].tolist() == [2, 4]


# ---------------------------------------------------------------------------
# VideoTransform — run on DEVICE
# ---------------------------------------------------------------------------

def test_video_transform_output_shape():
    cfg = VideoTransformConfig(image_size=224, train=False)
    t = VideoTransform(cfg)
    raw = torch.randint(0, 255, (8, 100, 150, 3), dtype=torch.uint8).to(DEVICE)
    out = t(raw)
    assert out.shape == (8, 3, 224, 224)
    assert out.dtype == torch.float32
    assert out.device.type == DEVICE.type


def test_video_transform_train_noise():
    cfg = VideoTransformConfig(image_size=64, train=True, noise_std=0.01)
    t = VideoTransform(cfg)
    raw = torch.randint(100, 200, (4, 64, 64, 3), dtype=torch.uint8).to(DEVICE)
    out = t(raw)
    assert out.shape == (4, 3, 64, 64)
    assert out.device.type == DEVICE.type


def test_video_transform_spatial_jitter():
    cfg = VideoTransformConfig(image_size=64, train=True, spatial_jitter=0.1)
    t = VideoTransform(cfg)
    raw = torch.randint(0, 255, (6, 64, 64, 3), dtype=torch.uint8).to(DEVICE)
    out = t(raw)
    assert out.shape == (6, 3, 64, 64)
    assert out.device.type == DEVICE.type


def test_video_transform_normalized_range():
    """Output should be approximately zero-mean after normalization."""
    cfg = VideoTransformConfig(image_size=32, train=False)
    t = VideoTransform(cfg)
    # Pixel value 128 ≈ mean of uint8 range → normalized close to 0 with ImageNet stats
    raw = torch.full((4, 32, 32, 3), 128, dtype=torch.uint8).to(DEVICE)
    out = t(raw)
    assert out.abs().mean().item() < 2.0   # sanity: not wildly out of range
