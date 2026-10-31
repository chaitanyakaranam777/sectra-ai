"""CNN + BERT multimodal classifier."""
from __future__ import annotations

from typing import Any, Optional, Sequence

from ml.architectures.bert import BertTextEncoder
from ml.architectures.cnn import ImageEncoder
from ml.architectures.fusion import MultimodalSentimentModel
from ml.constants import NUM_CLASSES


class CNNBERTClassifier(MultimodalSentimentModel):
    """Image branch: CNN encoder. Text branch: BERT [CLS] vector. Joined by :class:`MultimodalFusion`."""

    name = "cnn_bert"

    def __init__(self, num_classes: int = NUM_CLASSES, image_backbone: str = "scratch",
                 image_channels: Sequence[int] = (32, 64, 128, 256), image_pretrained: bool = False,
                 pretrained_name: Optional[str] = "bert-base-uncased",
                 bert_config: Optional[dict[str, Any]] = None, freeze_encoder: bool = False,
                 proj_dim: int = 256, fusion_hidden: Sequence[int] = (256, 128),
                 dropout: float = 0.3, use_interaction: bool = False) -> None:
        text_encoder = BertTextEncoder(pretrained_name, bert_config, freeze_encoder)
        super().__init__(ImageEncoder(image_backbone, image_channels, image_pretrained), text_encoder,
                         proj_dim, fusion_hidden, dropout, num_classes, use_interaction)
        self.set_config(num_classes=num_classes, image_backbone=image_backbone,
                        image_channels=image_channels, image_pretrained=image_pretrained,
                        pretrained_name=pretrained_name, bert_config=text_encoder.config_dict(),
                        freeze_encoder=freeze_encoder, proj_dim=proj_dim, fusion_hidden=fusion_hidden,
                        dropout=dropout, use_interaction=use_interaction)
