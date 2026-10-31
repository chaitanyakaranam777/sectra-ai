"""Classification metrics computed with scikit-learn from real predictions."""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, f1_score, precision_recall_fscore_support,
                             roc_auc_score)

from ml.constants import CLASS_NAMES
from ml.evaluation.confusion_matrix import confusion_matrix_payload


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int], y_prob: Optional[np.ndarray] = None,
                    class_names: Sequence[str] = CLASS_NAMES) -> dict[str, Any]:
    """Accuracy, macro precision/recall/F1, weighted F1, per-class table, confusion matrix, report.

    ``roc_auc_ovr_macro`` is included only when ``y_prob`` is given and every class occurs in
    ``y_true`` (otherwise AUC is undefined, so it is ``None`` rather than a guess).
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be non-empty and equally long")
    labels = list(range(len(class_names)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    auc = None
    if y_prob is not None and len(np.unique(y_true)) == len(class_names):
        auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=labels))
    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p.mean()),
        "recall_macro": float(r.mean()),
        "f1_macro": float(f.mean()),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "roc_auc_ovr_macro": auc,
        "per_class": {n: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, n in enumerate(class_names)},
        "confusion_matrix": confusion_matrix_payload(y_true, y_pred, class_names),
        "classification_report": classification_report(y_true, y_pred, labels=labels, target_names=list(class_names),
                                                       zero_division=0),
    }
