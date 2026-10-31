"""Record validation and duplicate analysis."""
from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from ml.data.image_processing import check_image
from ml.data.text_processing import clean_text


def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def flag_invalid(df: pd.DataFrame, check_images: bool = True) -> pd.DataFrame:
    """Add ``invalid_reason`` (None if the record is usable): missing/empty text, missing/unreadable image."""
    out = df.copy()
    reasons: list[Any] = []
    widths, heights = [], []
    for _, r in out.iterrows():
        reason, w, h = None, 0, 0
        if r["image_path"] is None or pd.isna(r["image_path"]):
            reason = "missing_image"
        elif r["text_raw"] is None or pd.isna(r["text_raw"]):
            reason = "missing_text_file"
        elif not clean_text(r["text_raw"]):
            reason = "empty_text"
        elif check_images:
            chk = check_image(r["image_path"])
            w, h = chk.width, chk.height
            reason = None if chk.valid else f"image_{chk.reason}"
        reasons.append(reason)
        widths.append(w)
        heights.append(h)
    out["invalid_reason"], out["img_width"], out["img_height"] = reasons, widths, heights
    return out


def add_hashes(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``image_md5`` (byte-identical images) and ``text_key`` (normalised text for duplicate detection)."""
    out = df.copy()
    out["image_md5"] = [file_md5(p) for p in out["image_path"]]
    out["text_key"] = [clean_text(t, lowercase=True) for t in out["text_raw"]]
    return out


def duplicate_report(df: pd.DataFrame) -> dict[str, Any]:
    """Counts of repeated ids, repeated images and repeated texts (needs ``add_hashes`` columns)."""
    def extra(col: str) -> int:
        return int(df[col].duplicated(keep="first").sum())
    return {"duplicate_ids": extra("id"), "duplicate_images": extra("image_md5"), "duplicate_texts": extra("text_key"),
            "rows_in_duplicate_image_groups": int(df["image_md5"].duplicated(keep=False).sum()),
            "rows_in_duplicate_text_groups": int(df["text_key"].duplicated(keep=False).sum())}
