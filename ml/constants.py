"""Project-wide constants shared by models, data pipelines and evaluation."""
from __future__ import annotations

#: Class order is ordinal (negative < neutral < positive); index = class id.
CLASS_NAMES: tuple[str, ...] = ("negative", "neutral", "positive")
NUM_CLASSES: int = len(CLASS_NAMES)
LABEL2ID: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}
ID2LABEL: dict[int, str] = {i: name for name, i in LABEL2ID.items()}

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
