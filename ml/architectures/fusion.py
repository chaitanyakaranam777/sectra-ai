"""Multimodal fusion: project -> concatenate -> fusion MLP -> logits."""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from ml.architectures.base_model import BaseSentimentModel
from ml.constants import NUM_CLASSES


class ModalityProjection(nn.Sequential):
    """Linear -> LayerNorm -> GELU -> Dropout: maps one modality into the shared ``proj_dim`` space."""

    def __init__(self, in_dim: int, proj_dim: int, dropout: float) -> None:
        super().__init__(nn.Linear(in_dim, proj_dim), nn.LayerNorm(proj_dim), nn.GELU(), nn.Dropout(dropout))


class MultimodalFusion(nn.Module):
    """Fuses image and text vectors of arbitrary sizes into class logits.

    1. both vectors are projected to ``proj_dim`` (compatible dimensions);
    2. the projections are concatenated -> ``(B, 2*proj_dim)``; with
       ``use_interaction`` their element-wise product is appended as well;
    3. fusion layers (Linear -> LayerNorm -> GELU -> Dropout) per ``fusion_hidden``;
    4. a final Linear gives ``(B, num_classes)`` logits (softmax only at inference).
    LayerNorm (not BatchNorm) keeps batch size 1 inference valid.
    """

    def __init__(self, image_dim: int, text_dim: int, proj_dim: int = 256,
                 fusion_hidden: Sequence[int] = (256, 128), dropout: float = 0.4,
                 num_classes: int = NUM_CLASSES, use_interaction: bool = False) -> None:
        super().__init__()
        self.image_proj = ModalityProjection(image_dim, proj_dim, dropout)
        self.text_proj = ModalityProjection(text_dim, proj_dim, dropout)
        self.use_interaction = use_interaction
        in_dim = proj_dim * (3 if use_interaction else 2)
        layers: list[nn.Module] = []
        for h in fusion_hidden:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            in_dim = h
        self.fusion_layers = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, image_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        pi, pt = self.image_proj(image_feat), self.text_proj(text_feat)
        parts = [pi, pt] + ([pi * pt] if self.use_interaction else [])
        return self.classifier(self.fusion_layers(torch.cat(parts, dim=-1)))


class MultimodalSentimentModel(BaseSentimentModel):
    """Shared implementation for CNN+LSTM and CNN+BERT: ``image_encoder`` + ``text_encoder`` + ``fusion``."""

    modality = "multimodal"
    required_inputs = ("image", "input_ids", "attention_mask")

    def __init__(self, image_encoder: nn.Module, text_encoder: nn.Module, proj_dim: int,
                 fusion_hidden: Sequence[int], dropout: float, num_classes: int,
                 use_interaction: bool) -> None:
        super().__init__(num_classes)
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.fusion = MultimodalFusion(image_encoder.out_dim, text_encoder.out_dim, proj_dim,
                                       fusion_hidden, dropout, num_classes, use_interaction)

    def forward(self, image: torch.Tensor, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        return self.fusion(self.image_encoder(image), self.text_encoder(input_ids, attention_mask))

    def gradcam_layer(self) -> nn.Module:
        return self.image_encoder.gradcam_layer()

    def text_embedding_layer(self) -> nn.Module:
        return self.text_encoder.embedding_layer()
