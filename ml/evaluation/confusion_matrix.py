"""Confusion-matrix payloads for the dashboard."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import confusion_matrix

from ml.constants import CLASS_NAMES


def confusion_matrix_payload(y_true: Sequence[int], y_pred: Sequence[int],
                             class_names: Sequence[str] = CLASS_NAMES) -> dict[str, Any]:
    """Counts (rows = true class, columns = predicted) plus row-normalised rates (recall per cell)."""
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sums, out=np.zeros(cm.shape, dtype=float), where=row_sums > 0)
    return {"labels": list(class_names), "matrix": cm.tolist(), "normalized": norm.round(6).tolist()}
