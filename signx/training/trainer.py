"""Base trainer providing optimizer setup, AMP, gradient accumulation, etc."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from ..utils.checkpoint import TopKCheckpointTracker, save_checkpoint
from ..utils.logging_utils import WandbLogger, get_logger
from ..utils.visualization import save_training_curves

logger = get_logger(__name__)


class BaseTrainer:
    """Minimal training loop with hooks the per-stage trainers can override.

    Subclasses implement:
        compute_loss(batch) -> dict with at least key "loss"
        evaluate(loader)    -> dict of metrics, must include cfg.checkpoint.metric
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: Any,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device or cfg.device)

        # Validate CUDA availability
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "cfg.device='cuda' but no CUDA GPU is visible. "
                    "Check your CUDA installation or set device: cpu in configs/default.yaml."
                )
            logger.info(
                f"Using GPU: {torch.cuda.get_device_name(self.device)} "
                f"({torch.cuda.get_device_properties(self.device).total_memory // 1024**3} GB)"
            )
            # Seed CUDA RNG for reproducibility
            seed = int(getattr(cfg, "seed", 42))
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = True  # faster conv ops
        else:
            logger.warning("Running on CPU — training will be slow.")

        # Seed CPU RNG
        seed = int(getattr(cfg, "seed", 42))
        torch.manual_seed(seed)

        self.model = model.to(self.device)
        n_gpus = torch.cuda.device_count() if self.device.type == "cuda" else 0
        if getattr(cfg, "multi_gpu", False) and n_gpus > 1:
            logger.info(f"Using {n_gpus} GPUs with DataParallel")
            self.model = nn.DataParallel(self.model, device_ids=list(range(n_gpus)))
        elif n_gpus == 1:
            logger.info("Single GPU mode (multi_gpu=false or only 1 GPU visible)")
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        self.start_epoch = 0
        self.global_step = 0

        self.ckpt_dir = Path(cfg.checkpoint_dir) / cfg.name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = TopKCheckpointTracker(
            k=int(cfg.checkpoint.save_top_k),
            mode=str(cfg.checkpoint.mode),
        )
        self.wandb = WandbLogger(cfg, run_name=cfg.name)

        # Automatic Mixed Precision (fp16) — enabled via cfg.use_amp
        self.use_amp = bool(getattr(cfg, "use_amp", False)) and self.device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None
        if self.use_amp:
            logger.info("AMP (fp16) enabled — using GradScaler")

        # In-memory history for matplotlib curves (key -> list of floats)
        self._train_history: dict[str, list[float]] = {}
        self._val_history: dict[str, list[float]] = {}
        self._curve_dir = Path(cfg.log_dir) / cfg.name / "curves"

    # ----- to be implemented by subclasses -----
    def compute_loss(self, batch: Dict) -> Dict[str, torch.Tensor]:  # pragma: no cover
        raise NotImplementedError

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:      # pragma: no cover
        raise NotImplementedError

    # ----- common machinery -----
    def _build_optimizer(self) -> torch.optim.Optimizer:
        params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            params,
            lr=float(self.cfg.train.lr),
            betas=(float(self.cfg.train.beta1), float(self.cfg.train.beta2)),
            weight_decay=float(self.cfg.train.weight_decay),
        )

    def _build_scheduler(self):
        from .scheduler import build_scheduler
        steps_per_epoch = max(1, len(self.train_loader))
        total_steps = steps_per_epoch * int(self.cfg.train.epochs)
        try:
            return build_scheduler(self.optimizer, self.cfg, total_steps=total_steps)
        except ValueError:
            return None

    def _to_device(self, batch: Dict) -> Dict:
        out = {}
        for k, v in batch.items():
            out[k] = v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
        return out

    def train(self) -> None:
        accum = max(1, int(getattr(self.cfg.train, "grad_accum_steps", 1)))
        clip = float(getattr(self.cfg.train, "clip_grad_norm", 0.0))
        log_interval = int(self.cfg.logging.log_interval)
        for epoch in range(self.start_epoch, int(self.cfg.train.epochs)):
            self.model.train()
            self.optimizer.zero_grad()
            for step, batch in enumerate(self.train_loader):
                batch = self._to_device(batch)

                with autocast(enabled=self.use_amp):
                    loss_dict = self.compute_loss(batch)

                loss = loss_dict["loss"] / accum
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (step + 1) % accum == 0:
                    if self.scaler is not None:
                        if clip > 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        if clip > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                        self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                if step % log_interval == 0:
                    logger.info(
                        f"epoch={epoch} step={step} "
                        + " ".join(f"{k}={float(v):.4f}" for k, v in loss_dict.items())
                    )
                    self.wandb.log(
                        {f"train/{k}": float(v) for k, v in loss_dict.items()},
                        step=self.global_step,
                    )
                    # Track training history for curve plots
                    for k, v in loss_dict.items():
                        self._train_history.setdefault(k, []).append(float(v))

            if self.val_loader is not None and (epoch + 1) % int(self.cfg.logging.eval_interval) == 0:
                metrics = self.evaluate(self.val_loader)
                logger.info(f"[val] epoch={epoch} " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
                self.wandb.log({f"val/{k}": v for k, v in metrics.items()}, step=self.global_step)
                for k, v in metrics.items():
                    self._val_history.setdefault(k, []).append(float(v))
                self._maybe_save(epoch, metrics)

            # Save matplotlib training curves after every epoch
            save_training_curves(
                self._train_history,
                self._val_history,
                self._curve_dir,
                prefix=str(getattr(self.cfg, "name", "")),
            )

        self.wandb.finish()

    def _maybe_save(self, epoch: int, metrics: Dict[str, float]) -> None:
        metric_name = str(self.cfg.checkpoint.metric)
        if metric_name not in metrics:
            return
        score = metrics[metric_name]
        ckpt_path = self.ckpt_dir / f"epoch{epoch:04d}_{metric_name}{score:.4f}.pt"

        def _save():
            save_checkpoint(
                ckpt_path,
                self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                metrics=metrics,
            )

        saved = self.tracker.maybe_save(score, ckpt_path, _save)
        if saved:
            best_link = self.ckpt_dir / "best.pt"
            try:
                best_link.unlink(missing_ok=True)
                best_link.symlink_to(ckpt_path.name)
            except OSError:
                _save()  # fallback: rewrite as a real file
