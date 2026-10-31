"""CNN + BiLSTM multimodal classifier."""
from __future__ import annotations

from typing import Sequence

from ml.architectures.cnn import ImageEncoder
from ml.architectures.fusion import MultimodalSentimentModel
from ml.architectures.lstm import LSTMTextEncoder
from ml.constants import NUM_CLASSES


class CNNLSTMClassifier(MultimodalSentimentModel):
    """Image branch: CNN encoder. Text branch: BiLSTM encoder. Joined by :class:`MultimodalFusion`."""

    name = "cnn_lstm"

    def __init__(self, vocab_size: int, num_classes: int = NUM_CLASSES, image_backbone: str = "scratch",
                 image_channels: Sequence[int] = (32, 64, 128, 256), image_pretrained: bool = False,
                 embed_dim: int = 128, lstm_hidden_dim: int = 128, lstm_layers: int = 1,
                 bidirectional: bool = True, proj_dim: int = 256,
                 fusion_hidden: Sequence[int] = (256, 128), dropout: float = 0.4,
                 use_interaction: bool = False) -> None:
        super().__init__(
            ImageEncoder(image_backbone, image_channels, image_pretrained),
            LSTMTextEncoder(vocab_size, embed_dim, lstm_hidden_dim, lstm_layers, bidirectional, dropout),
            proj_dim, fusion_hidden, dropout, num_classes, use_interaction)
        self.set_config(vocab_size=vocab_size, num_classes=num_classes, image_backbone=image_backbone,
                        image_channels=image_channels, image_pretrained=image_pretrained,
                        embed_dim=embed_dim, lstm_hidden_dim=lstm_hidden_dim, lstm_layers=lstm_layers,
                        bidirectional=bidirectional, proj_dim=proj_dim, fusion_hidden=fusion_hidden,
                        dropout=dropout, use_interaction=use_interaction)
