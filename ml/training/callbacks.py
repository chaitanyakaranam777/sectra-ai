"""Early stopping, epoch history and atomic progress reporting."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


class EarlyStopping:
    """Stops when ``monitor`` has not improved by ``min_delta`` for ``patience`` epochs."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4, mode: str = "min") -> None:
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.patience, self.min_delta, self.mode = patience, min_delta, mode
        self.best: Optional[float] = None
        self.best_epoch = 0
        self.counter = 0
        self.should_stop = False

    def update(self, value: float, epoch: int) -> bool:
        """Record an epoch's metric; return True if this epoch is the new best."""
        improved = self.best is None or (
            value < self.best - self.min_delta if self.mode == "min" else value > self.best + self.min_delta)
        if improved:
            self.best, self.best_epoch, self.counter = value, epoch, 0
        else:
            self.counter += 1
            self.should_stop = self.counter >= self.patience
        return improved


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    val_macro_f1: float
    lr: float
    duration_s: float


@dataclass
class TrainingHistory:
    """Per-epoch metrics, serialisable for the dashboard's loss/accuracy curves."""
    records: list[EpochRecord] = field(default_factory=list)
    best_epoch: int = 0

    def add(self, record: EpochRecord) -> None:
        self.records.append(record)

    def to_dict(self) -> dict[str, Any]:
        return {"best_epoch": self.best_epoch, "epochs": [asdict(r) for r in self.records]}

    def save(self, path: str | Path) -> None:
        write_json_atomic(path, self.to_dict())


def write_json_atomic(path: str | Path, payload: Any) -> None:
    """Write JSON via temp file + rename so readers (the API) never see a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2, default=float)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


class ProgressWriter:
    """Persists live training state to ``status.json`` (state, epoch, batch progress, last metrics)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.state: dict[str, Any] = {}

    def update(self, **fields: Any) -> None:
        self.state.update(fields)
        write_json_atomic(self.path, self.state)
