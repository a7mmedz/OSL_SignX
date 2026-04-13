"""Checkpoint save / load helpers, plus top-K checkpoint averaging."""
from __future__ import annotations

import heapq
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    epoch: int = 0,
    metrics: Dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics or {},
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        state["scheduler"] = scheduler.state_dict()
    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> Dict:
    state = torch.load(str(path), map_location=map_location)
    model.load_state_dict(state["model"], strict=strict)
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state and hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(state["scheduler"])
    return state


def average_checkpoints(paths: List[str | Path], output_path: str | Path) -> None:
    """Average the `model` weights of several checkpoints into one."""
    if not paths:
        raise ValueError("No checkpoints to average")
    averaged: Dict[str, torch.Tensor] = {}
    for p in paths:
        state = torch.load(str(p), map_location="cpu")
        msd = state["model"]
        for k, v in msd.items():
            if k not in averaged:
                averaged[k] = v.clone().float()
            else:
                averaged[k] += v.float()
    for k in averaged:
        averaged[k] /= len(paths)
    torch.save({"model": averaged, "averaged_from": [str(p) for p in paths]}, str(output_path))


class TopKCheckpointTracker:
    """Maintains the top-K best checkpoints by a chosen metric."""

    def __init__(self, k: int, mode: str = "min") -> None:
        self.k = k
        self.mode = mode
        self.heap: list[tuple[float, str]] = []  # min-heap

    def maybe_save(self, score: float, path: str | Path, save_fn) -> bool:
        # Convert to "higher is better" so we can keep a min-heap of the top-K.
        signed = -score if self.mode == "min" else score
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, (signed, str(path)))
            save_fn()
            return True
        worst, worst_path = self.heap[0]
        if signed > worst:
            heapq.heapreplace(self.heap, (signed, str(path)))
            try:
                Path(worst_path).unlink(missing_ok=True)
            except OSError:
                pass
            save_fn()
            return True
        return False

    def best_paths(self) -> List[str]:
        return [p for _, p in sorted(self.heap, reverse=True)]
