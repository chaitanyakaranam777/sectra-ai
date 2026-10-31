"""Common interface for every SENTRA AI sentiment classifier."""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from ml.constants import CLASS_NAMES, NUM_CLASSES


class BaseSentimentModel(nn.Module):
    """Base class: models map named input tensors to ``(B, num_classes)`` logits.

    Subclasses set ``name``, ``modality`` ("image" | "text" | "multimodal") and
    ``required_inputs`` (keys of the batch dict passed to ``forward``).
    Softmax is applied only in :meth:`predict_proba`; ``forward`` returns raw
    logits so it can be fed to ``CrossEntropyLoss``.
    """

    name: str = "base"
    modality: str = "image"
    required_inputs: tuple[str, ...] = ()

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.config: dict[str, Any] = {}

    def set_config(self, **kwargs: Any) -> None:
        """Store JSON-serialisable constructor arguments (used to rebuild from a checkpoint)."""
        self.config = {k: (list(v) if isinstance(v, tuple) else v) for k, v in kwargs.items()}

    def select_inputs(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Pick the tensors this model needs from a collated batch."""
        missing = [k for k in self.required_inputs if k not in batch]
        if missing:
            raise KeyError(f"{self.name} requires inputs {missing} that are missing from the batch")
        return {k: batch[k] for k in self.required_inputs}

    @torch.no_grad()
    def predict_proba(self, **inputs: torch.Tensor) -> torch.Tensor:
        """Softmax class probabilities, computed in eval mode (training mode is restored)."""
        was_training = self.training
        self.eval()
        try:
            return torch.softmax(self(**inputs), dim=-1)
        finally:
            self.train(was_training)

    def count_parameters(self, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    def summary(self) -> dict[str, Any]:
        """Architecture description used by the dashboard (real counts, no estimates)."""
        return {
            "name": self.name,
            "modality": self.modality,
            "required_inputs": list(self.required_inputs),
            "classes": list(CLASS_NAMES[: self.num_classes]),
            "parameters_total": self.count_parameters(),
            "parameters_trainable": self.count_parameters(trainable_only=True),
            "components": [
                {"name": n, "type": type(m).__name__,
                 "parameters": sum(p.numel() for p in m.parameters())}
                for n, m in self.named_children()
            ],
            "config": self.config,
        }

    def gradcam_layer(self) -> nn.Module:
        raise NotImplementedError(f"{self.name} has no image encoder, Grad-CAM is unavailable")

    def text_embedding_layer(self) -> nn.Module:
        raise NotImplementedError(f"{self.name} has no text encoder, token attribution is unavailable")
