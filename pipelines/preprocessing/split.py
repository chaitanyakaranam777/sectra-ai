"""Leakage-safe, stratified, group-aware train/validation/test splitting."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def assign_groups(df: pd.DataFrame) -> pd.Series:
    """Connected components over rows that share an image hash or a normalised text.

    Everything in a component is kept in one split, so retweets / re-posted images can never
    appear in both training and evaluation data.
    """
    parent = list(range(len(df)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for col in ("image_md5", "text_key"):
        first: dict[str, int] = {}
        for i, key in enumerate(df[col].tolist()):
            if key in first:
                parent[find(i)] = find(first[key])
            else:
                first[key] = i
    roots = [find(i) for i in range(len(df))]
    return pd.Series(pd.factorize(pd.Series(roots))[0], index=df.index, name="group_id")


def stratified_group_split(df: pd.DataFrame, val_size: float = 0.1, test_size: float = 0.1,
                           seed: int = 42) -> pd.Series:
    """Return a ``split`` Series (train/val/test). Needs ``group_id`` and ``label_id`` columns."""
    if not (0 < val_size < 1 and 0 < test_size < 1 and val_size + test_size < 1):
        raise ValueError("val_size and test_size must be in (0,1) and sum to < 1")
    split = pd.Series("train", index=df.index, name="split")

    def carve(pool: pd.DataFrame, frac: float, name: str, s: int) -> None:
        k = max(2, round(1 / frac))
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=s)
        _, held = next(sgkf.split(pool, pool["label_id"], pool["group_id"]))
        split.loc[pool.index[held]] = name

    carve(df, test_size, "test", seed)
    rest = df[split == "train"]
    carve(rest, val_size / (1 - test_size), "val", seed + 1)
    return split


def assert_no_leakage(df: pd.DataFrame) -> dict[str, Any]:
    """Raise AssertionError if any group, id, image or text crosses splits; returns the check summary."""
    problems = {}
    for col in ("group_id", "id", "image_md5", "text_key"):
        n = int((df.groupby(col)["split"].nunique() > 1).sum())
        if n:
            problems[col] = n
    if problems:
        raise AssertionError(f"Data leakage detected across splits: {problems}")
    return {"checked": ["group_id", "id", "image_md5", "text_key"], "leaking_keys": 0}


def split_summary(df: pd.DataFrame, class_names: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("train", "val", "test"):
        part = df[df["split"] == name]
        counts = part["label"].value_counts()
        out[name] = {"n": int(len(part)), "class_counts": {c: int(counts.get(c, 0)) for c in class_names},
                     "fraction": float(len(part) / max(len(df), 1))}
    return out
