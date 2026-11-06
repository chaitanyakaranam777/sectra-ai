"""Model selection, upload validation and explanation for the prediction endpoints."""
from __future__ import annotations

import base64
import io
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from backend.services.artifacts import ArtifactStore
from backend.settings import Settings
from ml.data.image_processing import denormalize
from ml.explainability.gradcam import GradCAM, overlay_heatmap
from ml.explainability.text_attribution import attribute_tokens
from ml.inference.predictor import Predictor, PredictorCache

ALLOWED_TYPES = {"image/jpeg": "JPEG", "image/png": "PNG"}


class InferenceError(Exception):
    """Client-side problem (bad upload, no model); carries an HTTP status."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def decode_upload(data: bytes, content_type: Optional[str], max_bytes: int) -> Image.Image:
    """Validate declared type, size AND real content (magic bytes via PIL); returns an RGB image in memory."""
    if len(data) > max_bytes:
        raise InferenceError(413, f"Image larger than {max_bytes // 1024} KB")
    if content_type not in ALLOWED_TYPES:
        raise InferenceError(415, "Only JPEG and PNG images are accepted")
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.format != ALLOWED_TYPES[content_type]:
                raise InferenceError(415, f"File content is {im.format}, not {ALLOWED_TYPES[content_type]}")
            im.load()
            rgb = im.convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise InferenceError(422, "File is not a readable image") from exc
    if min(rgb.size) < 16:
        raise InferenceError(422, "Image is too small")
    return rgb


class InferenceService:
    def __init__(self, settings: Settings, store: ArtifactStore) -> None:
        self.s, self.store, self.cache = settings, store, PredictorCache()

    def choose_run(self, mode: str, run_id: Optional[str] = None) -> str:
        """Choose the best eligible completed checkpoint using test evaluation metrics."""
        runs = [
            r for r in self.store.list_runs()
            if r["state"] == "completed"
            and r["model"]
            and (self.store.run_dir(r["run_id"]) / "best.pt").is_file()
        ]

        from ml.architectures import MODEL_REGISTRY

        runs = [
            r for r in runs
            if r["model"] in MODEL_REGISTRY
            and MODEL_REGISTRY[r["model"]].cls.modality == mode
        ]

        if run_id:
            match = [r for r in runs if r["run_id"] == run_id]
            if not match:
                raise InferenceError(404, f"No completed {mode} run '{run_id}'")
            return run_id

        if not runs:
            raise InferenceError(
                409,
                f"No trained {mode} model available. Train one first (POST /api/models/train)."
            )

        available = {r["run_id"] for r in runs}

        # Prefer actual test evaluation metrics over training best_metric.
        try:
            ranking = self.store.comparison().get("ranking", [])
            evaluated = [
                item for item in ranking
                if item.get("run") in available
            ]

            if evaluated:
                best = max(
                    evaluated,
                    key=lambda item: (
                        float(item.get("f1_macro", -1)),
                        float(item.get("accuracy", -1))
                    )
                )
                return best["run"]

        except (ValueError, TypeError, KeyError):
            pass

        # Fallback when evaluation comparison is unavailable.
        return max(
            runs,
            key=lambda r: (
                r["best_metric"] if r["best_metric"] is not None else -1
            )
        )["run_id"]

    def predictor(self, run_id: str) -> Predictor:
        return self.cache.get(self.store.run_dir(run_id) / "best.pt")

    def predict(self, mode: str, text: Optional[str], image: Optional[Image.Image], run_id: Optional[str]) -> dict[str, Any]:
        if text is not None and len(text) > self.s.max_text_chars:
            raise InferenceError(422, f"Text longer than {self.s.max_text_chars} characters")
        rid = self.choose_run(mode, run_id)
        try:
            result = self.predictor(rid).predict(text=text, image=image)
        except ValueError as exc:
            raise InferenceError(422, str(exc)) from exc
        return {**result, "run_id": rid}

    def explain(self, mode: str, text: Optional[str], image: Optional[Image.Image], run_id: Optional[str]) -> dict[str, Any]:
        """Separate visual and textual evidence. Limitations are returned with the payload."""
        rid = self.choose_run(mode, run_id)
        pred = self.predictor(rid)
        try:
            inputs = pred.prepare(text, image)
        except ValueError as exc:
            raise InferenceError(422, str(exc)) from exc
        out: dict[str, Any] = {"run_id": rid, "model": pred.model.name,
                               "prediction": pred.predict(text=text, image=image),
                               "limitations": ["Explanations are first-order approximations of one model's behaviour, not causes.",
                                               "Grad-CAM shows where the image branch responded; in multimodal models the text also affects the score.",
                                               "Token scores can change with small rewordings."]}
        if "image" in inputs:
            cam = GradCAM(pred.model)
            try:
                res = cam({k: v for k, v in inputs.items()})
            finally:
                cam.close()
            base = denormalize(inputs["image"][0])
            buf = io.BytesIO()
            Image.fromarray(overlay_heatmap(base, res["cam"])).save(buf, "PNG")
            out["image_explanation"] = {"method": "grad-cam", "class": int(res["class_idx"]),
                                        "overlay_png_base64": base64.b64encode(buf.getvalue()).decode()}
        if "input_ids" in inputs:
            att = attribute_tokens(pred.model, inputs)
            toks = pred.encoder.tokens(text)
            n = int(inputs["attention_mask"][0].sum())
            out["text_explanation"] = {"method": "gradient x input", "class": int(att["class_idx"]),
                                       "tokens": [{"token": t, "score": float(sc), "weight": float(w)}
                                                  for t, sc, w in zip(toks[:n], att["scores"][:n], att["normalized"][:n])]}
        return out
