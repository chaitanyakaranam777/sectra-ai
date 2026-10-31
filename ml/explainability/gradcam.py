"""Grad-CAM for the image branch of any model exposing ``gradcam_layer()``."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ml.architectures.base_model import BaseSentimentModel


class GradCAM:
    """Class-discriminative localisation (Selvaraju et al., 2017).

    Heatmap = ReLU(sum_k alpha_k * A_k) where A_k are the target layer's activation maps and
    alpha_k the spatially averaged gradients of the class logit. Hooks are removed by :meth:`close`.
    Limitation: it shows where the *image branch* looked; in multimodal models the text branch also affects the logit.
    """

    def __init__(self, model: BaseSentimentModel, target_layer: Optional[torch.nn.Module] = None) -> None:
        self.model = model
        self.layer = target_layer or model.gradcam_layer()
        self._act: Optional[torch.Tensor] = None
        self._grad: Optional[torch.Tensor] = None
        self._handle = self.layer.register_forward_hook(self._on_forward)

    def _on_forward(self, _m: torch.nn.Module, _i: Any, out: torch.Tensor) -> None:
        self._act = out
        if out.requires_grad:
            out.register_hook(lambda g: setattr(self, "_grad", g))

    def __call__(self, inputs: dict[str, torch.Tensor], class_idx: Optional[int] = None) -> dict[str, Any]:
        """Return ``{"cam": (H,W) float array in [0,1], "class_idx", "probs"}`` for a single-image batch."""
        if inputs["image"].shape[0] != 1:
            raise ValueError("Grad-CAM expects a batch of exactly one image")
        was_training = self.model.training
        self.model.eval()
        self._act = self._grad = None
        try:
            with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):
                x = inputs["image"].clone().requires_grad_(True)
                logits = self.model(**{**inputs, "image": x})
                probs = torch.softmax(logits, -1).detach()[0]
                idx = int(logits.argmax(1)) if class_idx is None else int(class_idx)
                self.model.zero_grad(set_to_none=True)
                logits[0, idx].backward()
        finally:
            self.model.train(was_training)
        if self._act is None or self._grad is None:
            raise RuntimeError("Grad-CAM hooks did not fire; check the target layer")
        weights = self._grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._act.detach()).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=inputs["image"].shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / cam.max() if cam.max() > 0 else cam
        return {"cam": cam.cpu().numpy(), "class_idx": idx, "probs": probs.cpu().numpy()}

    def close(self) -> None:
        self._handle.remove()


def _jet(v: np.ndarray) -> np.ndarray:
    """Piecewise-linear 'jet' colormap, values in [0,1] -> RGB floats."""
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return np.stack([r, g, b], -1)


def overlay_heatmap(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a heatmap over a ``(H,W,3)`` uint8 image; ``cam`` must match its spatial size."""
    if image_rgb.shape[:2] != cam.shape:
        raise ValueError(f"image {image_rgb.shape[:2]} and cam {cam.shape} differ in size")
    blended = (1 - alpha) * image_rgb.astype(np.float32) + alpha * 255 * _jet(cam)
    return blended.clip(0, 255).astype(np.uint8)
