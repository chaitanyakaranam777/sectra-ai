"""Model registry: ``build_model("cnn_bert", ...)``."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ml.architectures.base_model import BaseSentimentModel
from ml.architectures.bert import BERTClassifier
from ml.architectures.cnn import CNNClassifier
from ml.architectures.cnn_bert import CNNBERTClassifier
from ml.architectures.cnn_lstm import CNNLSTMClassifier
from ml.architectures.lstm import LSTMClassifier


@dataclass(frozen=True)
class ModelSpec:
    cls: type[BaseSentimentModel]
    text_encoder: str | None  # "word" (LSTM vocabulary) | "bert" (WordPiece) | None (image-only)
    description: str


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "cnn": ModelSpec(CNNClassifier, None, "CNN image-only classifier"),
    "lstm": ModelSpec(LSTMClassifier, "word", "BiLSTM text-only classifier"),
    "bert": ModelSpec(BERTClassifier, "bert", "BERT text-only classifier"),
    "cnn_lstm": ModelSpec(CNNLSTMClassifier, "word", "CNN + BiLSTM multimodal fusion"),
    "cnn_bert": ModelSpec(CNNBERTClassifier, "bert", "CNN + BERT multimodal fusion"),
}


def build_model(name: str, **params: Any) -> BaseSentimentModel:
    """Instantiate a registered architecture; unknown names raise ``KeyError`` listing the options."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name].cls(**params)


__all__ = ["MODEL_REGISTRY", "ModelSpec", "build_model", "BaseSentimentModel", "CNNClassifier",
           "LSTMClassifier", "BERTClassifier", "CNNLSTMClassifier", "CNNBERTClassifier"]
