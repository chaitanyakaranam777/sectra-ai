"""End-to-end data pipeline: ingest -> validate -> clean -> dedupe -> encode -> split -> export.

Raw data is read only; every output goes to ``out_dir`` (``data/processed/<dataset>``).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.constants import CLASS_NAMES, LABEL2ID
from ml.data.text_processing import clean_text, sequence_length_stats
from ml.training.callbacks import write_json_atomic
from ml.training.losses import compute_class_weights
from pipelines.ingestion.mvsa import build_raw_table
from pipelines.preprocessing.split import assert_no_leakage, assign_groups, split_summary, stratified_group_split
from pipelines.validation.checks import add_hashes, duplicate_report, flag_invalid


def _dist(df: pd.DataFrame, col: str = "label") -> dict[str, int]:
    vc = df[col].value_counts()
    return {c: int(vc.get(c, 0)) for c in CLASS_NAMES}


def run_pipeline(raw_root: str | Path, variant: str, out_dir: str | Path, seed: int = 42, val_size: float = 0.1,
                 test_size: float = 0.1, check_images: bool = True, dedupe: bool = True) -> dict[str, Any]:
    """Run all stages and return the report (also written to ``report.json``)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "processing_log.jsonl"
    log_path.write_text("")

    def log(stage: str, **info: Any) -> None:
        with log_path.open("a") as fh:
            fh.write(json.dumps({"t": time.time(), "stage": stage, **info}, default=float) + "\n")

    t0 = time.time()
    df, ingest = build_raw_table(raw_root, variant)
    log("ingestion", rows=len(df), **ingest)

    df = flag_invalid(df, check_images)
    invalid = df["invalid_reason"].notna()
    missing = {k: int(v) for k, v in df.loc[invalid, "invalid_reason"].value_counts().items()}
    log("validation", rows_before=len(df), invalid=int(invalid.sum()), reasons=missing)
    df = df[~invalid]

    unlabeled = df["label"].isna()
    drop_reasons = {k: int(v) for k, v in df.loc[unlabeled, "drop_reason"].value_counts().items()}
    log("label_resolution", rows_before=len(df), dropped=int(unlabeled.sum()), reasons=drop_reasons,
        pair_class_distribution_before=_dist(df.loc[~unlabeled]))
    df = df[~unlabeled].copy()
    if df.empty:
        raise ValueError("No usable samples remain after validation and label resolution")

    df = add_hashes(df)
    dups = duplicate_report(df)
    log("duplicate_analysis", **dups)
    if dedupe:
        before = len(df)
        df = df.drop_duplicates(subset="id").reset_index(drop=True)
        log("dedupe_ids", rows_before=before, rows_after=len(df))

    df["text_clean"] = [clean_text(t) for t in df["text_raw"]]
    df["label_id"] = df["label"].map(LABEL2ID).astype(int)
    df["group_id"] = assign_groups(df)
    df["split"] = stratified_group_split(df, val_size, test_size, seed)
    leak = assert_no_leakage(df)
    log("split", seed=seed, summary=split_summary(df, CLASS_NAMES), leakage=leak)

    cols = ["id", "image_path", "text_raw", "text_clean", "label", "label_id", "text_label", "image_label",
            "split", "group_id", "image_md5", "text_key", "img_width", "img_height"]
    df[cols].to_csv(out / "samples.csv", index=False)

    counts = np.array(list(_dist(df).values()), dtype=float)
    report: dict[str, Any] = {
        "dataset": f"MVSA-{variant.capitalize()}", "variant": variant, "seed": seed,
        "raw_label_rows": ingest["label_rows_parsed"], "malformed_label_rows": ingest["malformed_label_rows"],
        "invalid_records": missing, "dropped_unlabeled": drop_reasons, "duplicates": dups,
        "final_samples": int(len(df)), "class_distribution": _dist(df),
        "imbalance_ratio_max_over_min": float(counts.max() / max(counts.min(), 1.0)),
        "class_weights_train": dict(zip(CLASS_NAMES, compute_class_weights(
            df.loc[df["split"] == "train", "label_id"]).tolist())),
        "splits": split_summary(df, CLASS_NAMES), "leakage_check": leak,
        "text_length_words": sequence_length_stats(df["text_clean"]),
        "image_size_px": {"mean_width": float(df["img_width"].mean()), "mean_height": float(df["img_height"].mean()),
                          "min_side": int(min(df["img_width"].min(), df["img_height"].min()))},
        "duration_s": time.time() - t0, "samples_csv": str(out / "samples.csv"),
    }
    write_json_atomic(out / "report.json", report)
    log("export", **{k: report[k] for k in ("final_samples", "samples_csv")})
    return report
