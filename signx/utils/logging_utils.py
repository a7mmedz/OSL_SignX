"""Logging setup. Wraps `logging` with optional Weights & Biases integration."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(log_dir: str | Path | None = None, level: int = logging.INFO) -> None:
    """Initialize root logger to stdout and optionally to a file."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "train.log", encoding="utf-8"))
    logging.basicConfig(level=level, format=_LOG_FORMAT, handlers=handlers, force=True)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


class WandbLogger:
    """Thin wrapper that no-ops if wandb is unavailable or disabled."""

    def __init__(self, cfg: Any, run_name: str | None = None) -> None:
        self.enabled = bool(getattr(cfg.logging, "wandb_enabled", False))
        self.run = None
        if not self.enabled:
            return
        try:
            import wandb  # type: ignore
            self.run = wandb.init(
                project=getattr(cfg.logging, "wandb_project", "osl-signx"),
                entity=getattr(cfg.logging, "wandb_entity", None),
                name=run_name,
                config=dict(cfg) if hasattr(cfg, "items") else None,
            )
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"wandb disabled: failed to initialize ({e})"
            )
            self.enabled = False

    def log(self, metrics: dict, step: int | None = None) -> None:
        if not self.enabled or self.run is None:
            return
        import wandb  # type: ignore
        wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self.enabled and self.run is not None:
            import wandb  # type: ignore
            wandb.finish()
