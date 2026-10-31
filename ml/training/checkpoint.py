"""Checkpoint saving/loading and run versioning."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn

from ml.architectures import MODEL_REGISTRY, BaseSentimentModel, build_model


def _plain(obj: Any) -> Any:
    """Round-trip through JSON so checkpoints hold only builtin types (safe for weights_only loading)."""
    return json.loads(json.dumps(obj, default=float))


def save_checkpoint(path: str | Path, model: BaseSentimentModel, optimizer: Optional[torch.optim.Optimizer] = None,
                    epoch: int = 0, metrics: Optional[dict[str, Any]] = None,
                    extra: Optional[dict[str, Any]] = None) -> Path:
    """Save weights plus everything needed to rebuild the model (name + constructor config)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": model.name,
        "model_config": _plain(model.config),
        "state_dict": model.state_dict(),
        "epoch": int(epoch),
        "metrics": _plain(metrics or {}),
        "extra": _plain(extra or {}),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint dict (``weights_only=True``: no arbitrary code execution)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location, weights_only=True)


def load_model_from_checkpoint(path: str | Path, device: str | torch.device = "cpu"
                               ) -> tuple[BaseSentimentModel, dict[str, Any]]:
    """Rebuild the architecture from the stored config and restore its weights (eval mode).

    Pretrained-weight downloads are skipped (``pretrained_name`` -> None): the saved
    ``bert_config`` recreates the architecture and the state dict supplies the weights.
    """
    ckpt = load_checkpoint(path, map_location=device)
    name = ckpt["model_name"]
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Checkpoint references unknown model '{name}'")
    cfg = dict(ckpt["model_config"])
    if "pretrained_name" in cfg:
        cfg["pretrained_name"] = None
    if "image_pretrained" in cfg:
        cfg["image_pretrained"] = False
    if "pretrained" in cfg:
        cfg["pretrained"] = False
    model = build_model(name, **cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


def next_run_dir(runs_root: str | Path, model_name: str, dataset: str) -> Path:
    """Versioned run directory ``<root>/<model>_<dataset>_v<N>`` (N = existing versions + 1)."""
    root = Path(runs_root)
    root.mkdir(parents=True, exist_ok=True)
    prefix = f"{model_name}_{dataset}_v"
    versions = [int(p.name[len(prefix):]) for p in root.iterdir()
                if p.is_dir() and p.name.startswith(prefix) and p.name[len(prefix):].isdigit()]
    run_dir = root / f"{prefix}{max(versions, default=0) + 1}"
    run_dir.mkdir()
    return run_dir
