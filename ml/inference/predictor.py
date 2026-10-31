"""Cached, checkpoint-backed inference."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image

from ml.constants import CLASS_NAMES
from ml.data.image_processing import build_transforms, load_rgb
from ml.data.text_processing import text_encoder_from_state
from ml.training.checkpoint import load_model_from_checkpoint


class Predictor:
    """Loads one checkpoint once and serves predictions for the modalities that model supports.

    A multimodal model needs both text and image; an image model ignores text and vice versa.
    """

    def __init__(self, checkpoint: str | Path, device: str = "cpu") -> None:
        self.checkpoint = str(checkpoint)
        self.device = torch.device(device)
        self.model, ckpt = load_model_from_checkpoint(checkpoint, device)
        extra = ckpt["extra"]
        self.metrics = ckpt.get("metrics", {})
        self.encoder = text_encoder_from_state(extra["text_encoder"]) if extra.get("text_encoder") else None
        self.image_size = int(extra.get("image_size", 224))
        self.transform = build_transforms(self.image_size, train=False)

    @property
    def modality(self) -> str:
        return self.model.modality

    def prepare(self, text: Optional[str] = None, image: Optional[str | Path | Image.Image] = None
                ) -> dict[str, torch.Tensor]:
        """Build a single-sample input dict, validating that required modalities are present."""
        need_text = self.modality in {"text", "multimodal"}
        need_img = self.modality in {"image", "multimodal"}
        if need_text and not (text and text.strip()):
            raise ValueError(f"Model '{self.model.name}' needs non-empty text")
        if need_img and image is None:
            raise ValueError(f"Model '{self.model.name}' needs an image")
        inputs: dict[str, torch.Tensor] = {}
        if need_text:
            ids, mask = self.encoder.encode(text)
            inputs["input_ids"] = torch.tensor([ids], dtype=torch.long)
            inputs["attention_mask"] = torch.tensor([mask], dtype=torch.long)
        if need_img:
            inputs["image"] = self.transform(load_rgb(image)).unsqueeze(0)
        return {k: v.to(self.device) for k, v in inputs.items()}

    def predict(self, text: Optional[str] = None, image: Optional[str | Path | Image.Image] = None) -> dict[str, Any]:
        probs = self.model.predict_proba(**self.prepare(text, image))[0].cpu()
        k = int(probs.argmax())
        return {"label": CLASS_NAMES[k], "confidence": float(probs[k]),
                "probabilities": {c: float(p) for c, p in zip(CLASS_NAMES, probs)},
                "model": self.model.name, "modality": self.modality, "checkpoint": self.checkpoint}


class PredictorCache:
    """Thread-safe cache so a model is loaded once per checkpoint path, not per request."""

    def __init__(self, device: str = "cpu") -> None:
        self.device, self._items, self._lock = device, {}, threading.Lock()

    def get(self, checkpoint: str | Path) -> Predictor:
        key = str(Path(checkpoint).resolve())
        with self._lock:
            if key not in self._items:
                self._items[key] = Predictor(checkpoint, self.device)
            return self._items[key]
