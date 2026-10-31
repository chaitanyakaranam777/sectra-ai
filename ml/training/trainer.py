"""Training engine: real optimisation loop with validation, early stopping and checkpointing."""
from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ml.architectures.base_model import BaseSentimentModel
from ml.evaluation.metrics import compute_metrics
from ml.evaluation.predictions import collect_predictions, move_batch
from ml.training.callbacks import EarlyStopping, EpochRecord, ProgressWriter, TrainingHistory, write_json_atomic
from ml.training.checkpoint import save_checkpoint
from ml.training.losses import build_loss
from ml.training.optimizer import build_optimizer, build_scheduler, current_lr, select_device


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    label_smoothing: float = 0.0
    grad_clip: float = 1.0
    early_stopping_patience: int = 5
    min_delta: float = 1e-4
    monitor: str = "val_macro_f1"          # val_loss | val_acc | val_macro_f1
    pretrained_lr_scale: float = 1.0
    use_amp: bool = False                    # mixed precision, CUDA only
    device: str = "auto"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.monitor not in {"val_loss", "val_acc", "val_macro_f1"}:
            raise ValueError("monitor must be val_loss, val_acc or val_macro_f1")


class Trainer:
    """Trains one model. Writes ``history.json``, ``status.json``, ``best.pt`` and ``last.pt`` into ``run_dir``.

    ``best.pt`` is chosen on the *validation* split only; the test split is never touched here.
    """

    def __init__(self, model: BaseSentimentModel, train_loader: DataLoader, val_loader: DataLoader,
                 config: TrainConfig, run_dir: str | Path, class_weights: Optional[torch.Tensor] = None,
                 extra: Optional[dict[str, Any]] = None,
                 on_epoch_end: Optional[Callable[[EpochRecord], None]] = None) -> None:
        self.config = config
        self.device = select_device(config.device)
        self.model = model.to(self.device)
        self.train_loader, self.val_loader = train_loader, val_loader
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.extra = extra or {}
        self.on_epoch_end = on_epoch_end
        self.loss_fn = build_loss(class_weights, config.label_smoothing, self.device)
        self.optimizer = build_optimizer(model, config.optimizer, config.lr, config.weight_decay,
                                         config.pretrained_lr_scale)
        self.scheduler, self.scheduler_needs_metric = build_scheduler(self.optimizer, config.scheduler, config.epochs)
        self.amp = config.use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.history = TrainingHistory()
        self.progress = ProgressWriter(self.run_dir / "status.json")
        self.stopper = EarlyStopping(config.early_stopping_patience, config.min_delta,
                                     "min" if config.monitor == "val_loss" else "max")

    def _train_epoch(self, epoch: int) -> tuple[float, float]:
        self.model.train()
        total_loss, correct, n = 0.0, 0, 0
        n_batches = len(self.train_loader)
        for i, batch in enumerate(self.train_loader, 1):
            batch = move_batch(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(self.device.type, enabled=self.amp):
                logits = self.model(**self.model.select_inputs(batch))
                loss = self.loss_fn(logits, batch["label"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, batch {i}; lower the learning rate")
            self.scaler.scale(loss).backward()
            if self.config.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            bs = len(batch["label"])
            total_loss += loss.item() * bs
            correct += int((logits.argmax(1) == batch["label"]).sum())
            n += bs
            if i % 10 == 0 or i == n_batches:
                self.progress.update(epoch=epoch, batch=i, batches_per_epoch=n_batches,
                                     running_train_loss=total_loss / n)
        return total_loss / n, correct / n

    def fit(self) -> dict[str, Any]:
        """Run training; returns a summary dict (also saved as ``summary.json``)."""
        cfg = self.config
        set_seed(cfg.seed)
        start = time.time()
        self.progress.update(state="running", total_epochs=cfg.epochs, device=str(self.device),
                             model=self.model.name, started_at=start, epoch=0)
        try:
            for epoch in range(1, cfg.epochs + 1):
                t0 = time.time()
                tr_loss, tr_acc = self._train_epoch(epoch)
                val = collect_predictions(self.model, self.val_loader, self.device, self.loss_fn)
                vm = compute_metrics(val["y_true"], val["y_pred"])
                if self.scheduler is not None:
                    self.scheduler.step(val["loss"]) if self.scheduler_needs_metric else self.scheduler.step()
                rec = EpochRecord(epoch, tr_loss, tr_acc, val["loss"], vm["accuracy"], vm["f1_macro"],
                                  current_lr(self.optimizer), time.time() - t0)
                self.history.add(rec)
                monitored = {"val_loss": rec.val_loss, "val_acc": rec.val_acc,
                             "val_macro_f1": rec.val_macro_f1}[cfg.monitor]
                if self.stopper.update(monitored, epoch):
                    self.history.best_epoch = epoch
                    save_checkpoint(self.run_dir / "best.pt", self.model, self.optimizer, epoch,
                                    {**asdict(rec), "monitor": cfg.monitor}, self.extra)
                save_checkpoint(self.run_dir / "last.pt", self.model, self.optimizer, epoch, asdict(rec), self.extra)
                self.history.save(self.run_dir / "history.json")
                self.progress.update(state="running", epoch=epoch, last_epoch=asdict(rec),
                                     best_epoch=self.history.best_epoch, best_metric=self.stopper.best)
                if self.on_epoch_end:
                    self.on_epoch_end(rec)
                if self.stopper.should_stop:
                    break
        except BaseException as exc:
            self.progress.update(state="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        duration = time.time() - start
        summary = {"model": self.model.name, "device": str(self.device), "epochs_run": len(self.history.records),
                   "best_epoch": self.history.best_epoch, "monitor": cfg.monitor, "best_metric": self.stopper.best,
                   "stopped_early": self.stopper.should_stop, "duration_s": duration, "config": asdict(cfg),
                   "parameters": self.model.count_parameters()}
        write_json_atomic(self.run_dir / "summary.json", summary)
        self.progress.update(state="completed", duration_s=duration)
        return summary
