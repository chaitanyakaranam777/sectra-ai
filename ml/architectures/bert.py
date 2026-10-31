"""BERT text encoder and BERT text-only classifier (Hugging Face Transformers)."""
from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn
from transformers import BertConfig, BertModel

from ml.architectures.base_model import BaseSentimentModel
from ml.constants import NUM_CLASSES


class BertTextEncoder(nn.Module):
    """Returns the final-layer ``[CLS]`` hidden state ``(B, hidden_size)``.

    Weight sources, in priority order:
      1. ``bert_config`` dict  -> architecture rebuilt with random weights (used when
         restoring a checkpoint, whose state dict then overwrites the weights);
      2. ``pretrained_name``  -> pretrained weights from the Hugging Face hub / local dir;
      3. neither               -> default ``BertConfig()`` with random weights.
    """

    def __init__(self, pretrained_name: Optional[str] = "bert-base-uncased",
                 bert_config: Optional[dict[str, Any]] = None, freeze: bool = False) -> None:
        super().__init__()
        if bert_config is not None:
            self.bert = BertModel(BertConfig.from_dict(bert_config), add_pooling_layer=False)
        elif pretrained_name:
            self.bert = BertModel.from_pretrained(pretrained_name, add_pooling_layer=False)
        else:
            self.bert = BertModel(BertConfig(), add_pooling_layer=False)
        self.out_dim: int = self.bert.config.hidden_size
        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]

    def embedding_layer(self) -> nn.Module:
        return self.bert.embeddings.word_embeddings

    def config_dict(self) -> dict[str, Any]:
        return self.bert.config.to_dict()


class BERTClassifier(BaseSentimentModel):
    """Fine-tunable BERT text classifier: encoder -> Dropout -> Linear -> GELU -> Dropout -> Linear."""

    name = "bert"
    modality = "text"
    required_inputs = ("input_ids", "attention_mask")

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained_name: Optional[str] = "bert-base-uncased",
                 bert_config: Optional[dict[str, Any]] = None, freeze_encoder: bool = False,
                 dropout: float = 0.3, head_hidden_dim: int = 128) -> None:
        super().__init__(num_classes)
        self.text_encoder = BertTextEncoder(pretrained_name, bert_config, freeze_encoder)
        self.set_config(num_classes=num_classes, pretrained_name=pretrained_name,
                        bert_config=self.text_encoder.config_dict(), freeze_encoder=freeze_encoder,
                        dropout=dropout, head_hidden_dim=head_hidden_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(self.text_encoder.out_dim, head_hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(head_hidden_dim, num_classes))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.text_encoder(input_ids, attention_mask))

    def text_embedding_layer(self) -> nn.Module:
        return self.text_encoder.embedding_layer()
