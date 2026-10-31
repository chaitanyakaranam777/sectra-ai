"""Evaluate trained runs on one shared held-out split and compare them."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import torch

from ml.data.dataset import SentimentDataset, make_loader
from ml.data.text_processing import text_encoder_from_state
from ml.evaluation.metrics import compute_metrics
from ml.evaluation.predictions import collect_predictions
from ml.training.callbacks import write_json_atomic
from ml.training.checkpoint import load_model_from_checkpoint

METRIC_COLUMNS = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted", "roc_auc_ovr_macro"]


def evaluate_run(run_dir: str | Path, samples: pd.DataFrame, split: str = "test", device: str = "cpu",
                 batch_size: int = 32, image_size: Optional[int] = None) -> dict[str, Any]:
    """Load ``<run_dir>/best.pt`` and score it on ``samples[samples.split == split]``.

    The text encoder saved with the run is reused, so no vocabulary is rebuilt from evaluation data.
    """
    run_dir = Path(run_dir)
    model, ckpt = load_model_from_checkpoint(run_dir / "best.pt", device)
    subset = samples[samples["split"] == split]
    if subset.empty:
        raise ValueError(f"No samples in split '{split}'")
    enc_state = ckpt["extra"].get("text_encoder")
    encoder = text_encoder_from_state(enc_state) if enc_state else None
    size = image_size or ckpt["extra"].get("image_size", 224)
    ds = SentimentDataset(subset, model.modality, encoder, size)
    out = collect_predictions(model, make_loader(ds, batch_size), torch.device(device))
    metrics = compute_metrics(out["y_true"], out["y_pred"], out["y_prob"])
    return {"run": run_dir.name, "model": model.name, "split": split, "test_fingerprint": ds.fingerprint(),
            "metrics": metrics}


def compare_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Ranking by macro-F1. Refuses to compare models scored on different sample sets."""
    if not results:
        raise ValueError("No results to compare")
    prints = {r["test_fingerprint"] for r in results}
    if len(prints) != 1:
        raise ValueError("Results come from different evaluation sets; comparison would be unfair")
    rows = [{"run": r["run"], "model": r["model"], **{k: r["metrics"][k] for k in METRIC_COLUMNS}}
            for r in results]
    table = pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)
    return {"split": results[0]["split"], "test_fingerprint": prints.pop(),
            "n_samples": results[0]["metrics"]["n_samples"], "ranking": json.loads(table.to_json(orient="records"))}


def evaluate_and_compare(run_dirs: list[str | Path], samples_csv: str | Path, out_path: str | Path,
                         split: str = "test", device: str = "cpu") -> dict[str, Any]:
    """Evaluate every run, save per-run ``<split>_metrics.json`` and a combined comparison file."""
    samples = pd.read_csv(samples_csv)
    results = [evaluate_run(d, samples, split, device) for d in run_dirs]
    for r, d in zip(results, run_dirs):
        write_json_atomic(Path(d) / f"{split}_metrics.json", r)
    comparison = compare_results(results)
    write_json_atomic(out_path, comparison)
    return comparison
