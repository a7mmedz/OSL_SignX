"""Config loading with `defaults: [...]` style merging via OmegaConf."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf


def _resolve_defaults(cfg_path: Path) -> DictConfig:
    """Recursively merge `defaults: [<other>]` lists, similar to Hydra."""
    cfg_path = cfg_path.resolve()
    raw = OmegaConf.load(str(cfg_path))
    if not isinstance(raw, DictConfig):
        raise ValueError(f"Top-level config must be a mapping: {cfg_path}")

    base = OmegaConf.create({})
    if "defaults" in raw:
        defaults = raw.pop("defaults")
        if not isinstance(defaults, (list, tuple, ListConfig)):
            defaults = [defaults]
        for d in defaults:
            ref = cfg_path.parent / f"{d}.yaml"
            if not ref.exists():
                raise FileNotFoundError(f"Missing default config referenced by {cfg_path}: {ref}")
            base = OmegaConf.merge(base, _resolve_defaults(ref))
    return OmegaConf.merge(base, raw)


def load_config(path: str | Path) -> DictConfig:
    """Load a YAML config, resolving any `defaults` chain."""
    cfg = _resolve_defaults(Path(path))
    OmegaConf.resolve(cfg)
    return cfg


def save_config(cfg: Any, path: str | Path) -> None:
    """Persist a (possibly resolved) config back to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=str(path))
