"""Gradient x input token attribution for any model with a text branch."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from ml.architectures.base_model import BaseSentimentModel


def attribute_tokens(model: BaseSentimentModel, inputs: dict[str, torch.Tensor],
                     class_idx: Optional[int] = None) -> dict[str, Any]:
    """Per-token influence on a class logit: ``sum_d (d logit / d e_t)[d] * e_t[d]`` over embedding dims.

    Signed scores: positive pushes toward ``class_idx``, negative pushes away. ``normalized`` is
    ``|score| / max|score|`` for highlighting. Padding positions are zeroed.
    Limitation: first-order gradient attributions are local approximations, are not causal effects
    and can change with small input changes; treat them as hints, not proof.
    """
    layer = model.text_embedding_layer()
    store: dict[str, torch.Tensor] = {}

    def hook(_m: torch.nn.Module, _i: Any, out: torch.Tensor) -> None:
        out.retain_grad()
        store["emb"] = out

    handle = layer.register_forward_hook(hook)
    was_training = model.training
    model.eval()
    try:
        with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):
            logits = model(**inputs)
            probs = torch.softmax(logits, -1).detach()[0]
            idx = int(logits.argmax(1)) if class_idx is None else int(class_idx)
            model.zero_grad(set_to_none=True)
            logits[0, idx].backward()
    finally:
        handle.remove()
        model.train(was_training)
    emb = store["emb"]
    if emb.grad is None:
        raise RuntimeError("No gradient reached the embedding layer")
    scores = (emb.grad * emb.detach()).sum(-1)[0] * inputs["attention_mask"][0].to(emb.dtype)
    scores_np = scores.cpu().numpy()
    peak = float(np.abs(scores_np).max())
    return {"scores": scores_np, "normalized": np.abs(scores_np) / peak if peak > 0 else np.zeros_like(scores_np),
            "class_idx": idx, "probs": probs.cpu().numpy()}
