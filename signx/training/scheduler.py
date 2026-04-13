"""Learning-rate schedulers (Noam, warmup-cosine)."""
from __future__ import annotations

import math
from typing import Any


class NoamScheduler:
    """Noam scheduler from "Attention Is All You Need".

    lr = factor * model_size**-0.5 * min(step**-0.5, step * warmup**-1.5)
    """

    def __init__(
        self,
        optimizer,
        model_size: int,
        warmup: int,
        factor: float = 1.0,
    ) -> None:
        self.optimizer = optimizer
        self.model_size = model_size
        self.warmup = warmup
        self.factor = factor
        self._step = 0

    def step(self) -> None:
        self._step += 1
        lr = self._lr(self._step)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def _lr(self, step: int) -> float:
        return (
            self.factor
            * self.model_size ** -0.5
            * min(step ** -0.5, step * self.warmup ** -1.5)
        )

    def state_dict(self) -> dict:
        return {"_step": self._step}

    def load_state_dict(self, state: dict) -> None:
        self._step = state["_step"]


class WarmupCosineScheduler:
    """Linear warmup followed by cosine decay to zero."""

    def __init__(self, optimizer, base_lr: float, warmup_steps: int, total_steps: int) -> None:
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self._step = 0

    def step(self) -> None:
        self._step += 1
        if self._step <= self.warmup_steps:
            lr = self.base_lr * self._step / max(1, self.warmup_steps)
        else:
            t = (self._step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def state_dict(self) -> dict:
        return {"_step": self._step}

    def load_state_dict(self, state: dict) -> None:
        self._step = state["_step"]


def build_scheduler(optimizer, cfg, total_steps: int | None = None) -> Any:
    """Construct a scheduler from a stage config."""
    name = str(getattr(cfg.train, "scheduler", "warmup_cosine")).lower()
    if name == "noam":
        return NoamScheduler(
            optimizer,
            model_size=int(cfg.train.noam_model_size),
            warmup=int(cfg.train.noam_warmup),
        )
    if name in ("warmup_cosine", "cosine", "default"):
        if total_steps is None:
            total_steps = int(cfg.train.epochs) * 1000  # rough fallback
        warmup_steps = int(getattr(cfg.train, "warmup_epochs", 0)) * 1000
        return WarmupCosineScheduler(
            optimizer,
            base_lr=float(cfg.train.lr),
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )
    raise ValueError(f"Unknown scheduler: {name}")
