"""Collate functions for variable-length sign language videos."""
from __future__ import annotations

from typing import Dict, List

import torch


def _pad_videos(videos: List[torch.Tensor]) -> torch.Tensor:
    """Pad a list of (T_i, C, H, W) tensors to (B, T_max, C, H, W) with zeros."""
    t_max = max(v.shape[0] for v in videos)
    c, h, w = videos[0].shape[1:]
    out = torch.zeros(len(videos), t_max, c, h, w, dtype=videos[0].dtype)
    for i, v in enumerate(videos):
        out[i, : v.shape[0]] = v
    return out


def _pad_glosses(glosses: List[torch.Tensor], pad_id: int = 0) -> torch.Tensor:
    """Pad gloss id sequences to a (B, L_max) tensor."""
    l_max = max(g.numel() for g in glosses)
    out = torch.full((len(glosses), l_max), pad_id, dtype=torch.long)
    for i, g in enumerate(glosses):
        out[i, : g.numel()] = g
    return out


def collate_video_batch(batch: List[Dict]) -> Dict:
    """Collate a list of dataset items into a batch dict.

    Returned tensors:
        videos:        (B, T_max, C, H, W) float
        video_lengths: (B,) int64 (true frame counts)
        glosses:       (B, L_max) int64 (zero-padded)
        gloss_lengths: (B,) int64
        item_ids:      (B,) int64
        signer_ids:    (B,) int64
        paths:         list[str]
    """
    videos = [b["video"] for b in batch]
    glosses = [b["gloss_ids"] for b in batch]
    return {
        "videos": _pad_videos(videos),
        "video_lengths": torch.tensor([b["video_length"] for b in batch], dtype=torch.long),
        "glosses": _pad_glosses(glosses, pad_id=0),
        "gloss_lengths": torch.tensor([b["gloss_length"] for b in batch], dtype=torch.long),
        "item_ids": torch.tensor([b["item_id"] for b in batch], dtype=torch.long),
        "signer_ids": torch.tensor([b["signer_id"] for b in batch], dtype=torch.long),
        "paths": [b["path"] for b in batch],
    }
