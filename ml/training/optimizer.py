"""Optimizer and learning-rate scheduler factories."""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.optim import Adam, AdamW, Optimizer
from torch.optim import lr_scheduler as sched

_NO_DECAY_KEYS = ("bias", "norm", "embedding")


def build_optimizer(model: nn.Module, name: str = "adamw", lr: float = 1e-3, weight_decay: float = 1e-4,
                    pretrained_lr_scale: float = 1.0) -> Optimizer:
    """Adam/AdamW with two refinements.

    * biases, normalisation layers and embeddings get no weight decay;
    * parameters inside a BERT encoder (``.bert.`` in the name) use ``lr * pretrained_lr_scale``
      so fine-tuning can use a smaller step than the freshly initialised layers.
    """
    if name.lower() not in {"adam", "adamw"}:
        raise ValueError(f"Unsupported optimizer '{name}' (use 'adam' or 'adamw')")
    groups: dict[tuple[bool, bool], list[nn.Parameter]] = {}
    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        decay = not any(k in pname.lower() for k in _NO_DECAY_KEYS) and p.ndim > 1
        pretrained = ".bert." in f".{pname}"
        groups.setdefault((decay, pretrained), []).append(p)
    if not groups:
        raise ValueError("Model has no trainable parameters")
    param_groups = [{"params": ps, "weight_decay": weight_decay if decay else 0.0,
                     "lr": lr * (pretrained_lr_scale if pre else 1.0)}
                    for (decay, pre), ps in groups.items()]
    cls = AdamW if name.lower() == "adamw" else Adam
    return cls(param_groups, lr=lr)


def build_scheduler(optimizer: Optimizer, name: str = "cosine", epochs: int = 10,
                    step_size: int = 5, gamma: float = 0.5, patience: int = 2
                    ) -> tuple[Optional[sched.LRScheduler], bool]:
    """Return ``(scheduler, needs_metric)``. Schedulers step once per epoch; ``plateau`` needs val loss."""
    name = name.lower()
    if name in {"none", ""}:
        return None, False
    if name == "cosine":
        return sched.CosineAnnealingLR(optimizer, T_max=max(1, epochs)), False
    if name == "step":
        return sched.StepLR(optimizer, step_size=step_size, gamma=gamma), False
    if name == "plateau":
        return sched.ReduceLROnPlateau(optimizer, mode="min", factor=gamma, patience=patience), True
    raise ValueError(f"Unsupported scheduler '{name}' (none, cosine, step, plateau)")


def current_lr(optimizer: Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def select_device(preference: str = "auto") -> torch.device:
    """``auto`` -> CUDA when available, else CPU. ``cuda`` raises if no GPU is present."""
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preference.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is available; use device='cpu' or 'auto'")
    return torch.device(preference)
