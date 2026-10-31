"""Bidirectional LSTM text encoder and LSTM text-only classifier."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from ml.architectures.base_model import BaseSentimentModel
from ml.constants import NUM_CLASSES


class LSTMTextEncoder(nn.Module):
    """Embedding -> (Bi)LSTM over packed sequences -> final hidden state ``(B, out_dim)``.

    Packing guarantees padding never influences the representation.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 128,
                 num_layers: int = 1, bidirectional: bool = True, dropout: float = 0.4,
                 padding_idx: int = 0) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.embed_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0.0)
        self.bidirectional = bidirectional
        self.out_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        emb = self.embed_dropout(self.embedding(input_ids))
        lengths = attention_mask.sum(1).clamp(min=1).cpu()
        packed = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return torch.cat([h[-2], h[-1]], dim=1) if self.bidirectional else h[-1]

    def embedding_layer(self) -> nn.Module:
        return self.embedding


class LSTMClassifier(BaseSentimentModel):
    """Text-only classifier built on :class:`LSTMTextEncoder`."""

    name = "lstm"
    modality = "text"
    required_inputs = ("input_ids", "attention_mask")

    def __init__(self, vocab_size: int, num_classes: int = NUM_CLASSES, embed_dim: int = 128,
                 hidden_dim: int = 128, num_layers: int = 1, bidirectional: bool = True,
                 dropout: float = 0.4, head_hidden_dim: int = 64) -> None:
        super().__init__(num_classes)
        self.set_config(vocab_size=vocab_size, num_classes=num_classes, embed_dim=embed_dim,
                        hidden_dim=hidden_dim, num_layers=num_layers, bidirectional=bidirectional,
                        dropout=dropout, head_hidden_dim=head_hidden_dim)
        self.text_encoder = LSTMTextEncoder(vocab_size, embed_dim, hidden_dim, num_layers,
                                            bidirectional, dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(self.text_encoder.out_dim, head_hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(head_hidden_dim, num_classes))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.text_encoder(input_ids, attention_mask))

    def text_embedding_layer(self) -> nn.Module:
        return self.text_encoder.embedding_layer()
