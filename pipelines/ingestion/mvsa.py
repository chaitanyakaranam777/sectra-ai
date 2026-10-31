"""MVSA-Single / MVSA-Multiple ingestion: locate files, parse annotations, resolve labels.

Expected raw layout (as distributed by the dataset authors, kept unmodified)::

    <root>/labelResultAll.txt      tab-separated; header line, then ``ID<TAB>text,image[<TAB>text,image...]``
    <root>/data/<ID>.jpg           image
    <root>/data/<ID>.txt           tweet text

MVSA-Single has one annotation per row, MVSA-Multiple has three.
Label resolution (documented standard practice for MVSA):
  1. per modality, take the majority label over annotators (no majority -> dropped);
  2. if text and image labels agree, use it; if one is neutral use the other;
  3. positive-vs-negative conflicts are dropped.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ml.constants import CLASS_NAMES

VARIANTS = {"single": 1, "multiple": 3}
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def _read_text(path: Path) -> Optional[str]:
    """Read a tweet file; MVSA contains some non-UTF-8 files, so fall back to latin-1."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    for enc in ("utf-8", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return None


def locate_dataset(root: str | Path) -> tuple[Path, Path]:
    """Return ``(label_file, data_dir)`` or raise FileNotFoundError with import instructions."""
    root = Path(root)
    label_file = next((p for p in (root / "labelResultAll.txt", root / "labelResultAll.csv") if p.is_file()), None)
    data_dir = next((p for p in (root / "data", root) if p.is_dir() and any(p.glob("*.txt"))), None)
    if label_file is None or data_dir is None:
        raise FileNotFoundError(
            f"MVSA files not found under '{root}'. Expected labelResultAll.txt and a data/ folder with "
            f"<ID>.jpg and <ID>.txt. See docs/DATASET_IMPORT.md for the manual import steps.")
    return label_file, data_dir


def _majority(labels: list[str]) -> Optional[str]:
    top = Counter(labels).most_common()
    if not top or (len(top) > 1 and top[0][1] == top[1][1]):
        return None
    return top[0][0]


def resolve_pair_label(text_label: Optional[str], image_label: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Return ``(label, drop_reason)`` for a text/image label pair."""
    if text_label is None or image_label is None:
        return None, "no_annotator_majority"
    if text_label == image_label:
        return text_label, None
    if "neutral" in (text_label, image_label):
        return (image_label if text_label == "neutral" else text_label), None
    return None, "text_image_conflict"


def parse_labels(label_file: Path, variant: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse annotations into one row per id with resolved text/image/pair labels."""
    n_ann = VARIANTS[variant]
    rows, malformed = [], 0
    for line_no, line in enumerate(label_file.read_text(encoding="utf-8", errors="replace").splitlines()):
        parts = [p.strip() for p in line.split("\t") if p.strip()]
        if line_no == 0 and parts and parts[0].upper() == "ID":
            continue
        if len(parts) != n_ann + 1:
            malformed += 1
            continue
        pairs = [p.lower().split(",") for p in parts[1:]]
        if any(len(p) != 2 or p[0].strip() not in CLASS_NAMES or p[1].strip() not in CLASS_NAMES for p in pairs):
            malformed += 1
            continue
        t_lab = _majority([p[0].strip() for p in pairs])
        i_lab = _majority([p[1].strip() for p in pairs])
        label, reason = resolve_pair_label(t_lab, i_lab)
        rows.append({"id": parts[0], "text_label": t_lab, "image_label": i_lab, "label": label, "drop_reason": reason})
    df = pd.DataFrame(rows, columns=["id", "text_label", "image_label", "label", "drop_reason"])
    return df, {"label_rows_parsed": len(df), "malformed_label_rows": malformed}


def build_raw_table(root: str | Path, variant: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join labels with files on disk. Raw files are only read, never modified."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {sorted(VARIANTS)}")
    label_file, data_dir = locate_dataset(root)
    labels, parse_info = parse_labels(label_file, variant)
    texts, imgs = [], []
    for sid in labels["id"]:
        texts.append(_read_text(data_dir / f"{sid}.txt"))
        imgs.append(next((str(data_dir / f"{sid}{e}") for e in IMAGE_EXTS if (data_dir / f"{sid}{e}").is_file()), None))
    labels["text_raw"], labels["image_path"] = texts, imgs
    return labels, {**parse_info, "label_file": str(label_file), "data_dir": str(data_dir)}
