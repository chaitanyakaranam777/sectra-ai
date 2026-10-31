"""One shared inference loop used by training (validation) and evaluation (test)."""
from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ml.architectures.base_model import BaseSentimentModel


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def collect_predictions(model: BaseSentimentModel, loader: DataLoader, device: torch.device,
                        loss_fn: Optional[nn.Module] = None) -> dict[str, Any]:
    """Run the model over ``loader`` in eval mode.

    Returns ``y_true``, ``y_pred``, ``y_prob`` (softmax, shape ``(N, C)``) and mean ``loss``
    (``None`` if no ``loss_fn``). The model's previous train/eval mode is restored.
    """
    was_training = model.training
    model.eval()
    probs, labels, total_loss, n = [], [], 0.0, 0
    try:
        for batch in loader:
            batch = move_batch(batch, device)
            logits = model(**model.select_inputs(batch))
            if loss_fn is not None:
                total_loss += float(loss_fn(logits, batch["label"])) * len(batch["label"])
            n += len(batch["label"])
            probs.append(torch.softmax(logits, -1).cpu())
            labels.append(batch["label"].cpu())
    finally:
        model.train(was_training)
    if n == 0:
        raise ValueError("Loader produced no samples")
    y_prob = torch.cat(probs).numpy()
    return {"y_true": torch.cat(labels).numpy(), "y_pred": y_prob.argmax(1), "y_prob": y_prob,
            "loss": total_loss / n if loss_fn is not None else None}
