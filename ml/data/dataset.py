"""PyTorch dataset over the processed samples table."""
from __future__ import annotations

import hashlib
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ml.data.image_processing import build_transforms, load_rgb
from ml.data.text_processing import TextEncoder

MODES = ("image", "text", "multimodal")


class SentimentDataset(Dataset):
    """Rows of the processed table -> dict batches (``image``, ``input_ids``, ``attention_mask``, ``label``).

    Required columns: ``image_path``, ``text_clean`` (or ``text_raw``), ``label_id``, ``id``.
    Text is tokenised once at construction time.
    """

    def __init__(self, df: pd.DataFrame, mode: str, text_encoder: Optional[TextEncoder] = None,
                 image_size: int = 224, train: bool = False) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode != "image" and text_encoder is None:
            raise ValueError(f"mode '{mode}' needs a text_encoder")
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = build_transforms(image_size, train) if mode != "text" else None
        self.labels: np.ndarray = self.df["label_id"].to_numpy(dtype=np.int64)
        self.sample_ids: list[str] = self.df["id"].astype(str).tolist()
        self.image_paths: list[str] = self.df["image_path"].tolist() if mode != "text" else []
        self._ids = self._mask = None
        if mode != "image":
            col = "text_clean" if "text_clean" in self.df else "text_raw"
            enc = [text_encoder.encode(str(t) if isinstance(t, str) else "") for t in self.df[col]]
            self._ids = torch.tensor([e[0] for e in enc], dtype=torch.long)
            self._mask = torch.tensor([e[1] for e in enc], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict[str, Any]:
        item: dict[str, Any] = {"label": torch.tensor(self.labels[i])}
        if self.transform is not None:
            try:
                item["image"] = self.transform(load_rgb(self.image_paths[i]))
            except OSError as exc:
                raise RuntimeError(f"Cannot read image for sample {self.sample_ids[i]}: {exc}") from exc
        if self._ids is not None:
            item["input_ids"], item["attention_mask"] = self._ids[i], self._mask[i]
        return item

    def fingerprint(self) -> str:
        """Hash of the ordered sample ids; identical fingerprints mean an identical evaluation set."""
        return hashlib.sha256("\n".join(sorted(self.sample_ids)).encode()).hexdigest()


def make_loader(dataset: SentimentDataset, batch_size: int = 32, shuffle: bool = False,
                num_workers: int = 0, seed: int = 42) -> DataLoader:
    """DataLoader with a seeded generator so shuffling is reproducible."""
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      generator=g if shuffle else None, drop_last=False)
