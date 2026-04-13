"""Lightweight visualization helpers (matplotlib-only, optional)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import torch


def save_attention_heatmap(
    attn: torch.Tensor,
    out_path: str | Path,
    title: str | None = None,
) -> None:
    """Save a (T_q, T_k) attention map as a PNG. No-op if matplotlib missing."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(attn.detach().cpu().numpy(), aspect="auto", cmap="viridis")
    if title:
        ax.set_title(title)
    ax.set_xlabel("Key time")
    ax.set_ylabel("Query time")
    fig.colorbar(im, ax=ax)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_training_curves(
    train_history: Dict[str, List[float]],
    val_history: Dict[str, List[float]],
    out_dir: str | Path,
    prefix: str = "",
) -> None:
    """Save per-key training and validation curves as PNG files.

    Args:
        train_history: {metric_name: [value_at_step0, ...]} from the training loop.
        val_history:   {metric_name: [value_at_epoch0, ...]} from evaluation.
        out_dir:       Directory where PNG files are written.
        prefix:        Optional filename prefix (e.g. the stage name).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # One figure per metric, overlaying train and val when available.
    all_keys = set(train_history) | set(val_history)
    for key in sorted(all_keys):
        fig, ax = plt.subplots(figsize=(8, 4))
        if key in train_history and train_history[key]:
            ax.plot(train_history[key], label=f"train/{key}", alpha=0.8)
        if key in val_history and val_history[key]:
            # Val is recorded per eval epoch; x-axis is epoch index.
            ax.plot(val_history[key], label=f"val/{key}", marker="o", markersize=3)
        ax.set_xlabel("Step / Epoch")
        ax.set_ylabel(key)
        ax.set_title(f"{prefix} {key}".strip())
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fname = f"{prefix}_{key}.png" if prefix else f"{key}.png"
        fig.savefig(out_dir / fname, dpi=120)
        plt.close(fig)


def save_alignment_plot(
    alignment: Sequence[int],
    out_path: str | Path,
    title: str | None = None,
) -> None:
    """Save a 1-D alignment trace (e.g. CTC argmax over time)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.plot(list(alignment), drawstyle="steps-mid")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Argmax token id")
    if title:
        ax.set_title(title)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
