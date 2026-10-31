"""Loss construction and class weighting."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn

from ml.constants import NUM_CLASSES


def compute_class_weights(labels: Sequence[int], num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Balanced inverse-frequency weights ``n / (C * count_c)``. Weights come from *training* labels only."""
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=num_classes).astype(float)
    if counts.sum() == 0:
        raise ValueError("Cannot compute class weights from an empty label list")
    weights = counts.sum() / (num_classes * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def build_loss(class_weights: Optional[torch.Tensor] = None, label_smoothing: float = 0.0,
               device: torch.device | str = "cpu") -> nn.CrossEntropyLoss:
    """Cross-entropy over logits, optionally class-weighted and label-smoothed."""
    w = class_weights.to(device) if class_weights is not None else None
    return nn.CrossEntropyLoss(weight=w, label_smoothing=label_smoothing)
