"""Image encoder and CNN image-only sentiment classifier."""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from ml.architectures.base_model import BaseSentimentModel
from ml.constants import NUM_CLASSES


class ConvBlock(nn.Sequential):
    """(Conv3x3 -> BatchNorm -> ReLU) x2, optionally followed by 2x2 max-pooling."""

    def __init__(self, in_ch: int, out_ch: int, pool: bool = True) -> None:
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        super().__init__(*layers)


class ImageEncoder(nn.Module):
    """Maps ``(B, 3, H, W)`` images to ``(B, out_dim)`` feature vectors.

    ``backbone="scratch"`` trains a small VGG-style CNN; ``"resnet18"`` uses
    torchvision's ResNet-18 (ImageNet weights only if ``pretrained=True``, which
    needs a download). The final conv stage is left un-pooled so a spatial map
    is available for Grad-CAM.
    """

    def __init__(self, backbone: str = "scratch", channels: Sequence[int] = (32, 64, 128, 256),
                 pretrained: bool = False) -> None:
        super().__init__()
        self.backbone = backbone
        if backbone == "scratch":
            blocks, prev = [], 3
            for i, ch in enumerate(channels):
                blocks.append(ConvBlock(prev, ch, pool=i < len(channels) - 1))
                prev = ch
            self.features: nn.Module = nn.Sequential(*blocks)
            self.out_dim = prev
            self._target = self.features[-1]
        elif backbone == "resnet18":
            from torchvision.models import ResNet18_Weights, resnet18

            net = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
            self.features = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                                          net.layer1, net.layer2, net.layer3, net.layer4)
            self.out_dim = 512
            self._target = net.layer4
        else:
            raise ValueError(f"Unknown image backbone '{backbone}' (use 'scratch' or 'resnet18')")
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(image)).flatten(1)

    def gradcam_layer(self) -> nn.Module:
        """Last convolutional stage (spatial feature map used by Grad-CAM)."""
        return self._target


class CNNClassifier(BaseSentimentModel):
    """Image-only classifier: ImageEncoder -> Dropout -> Linear -> ReLU -> Dropout -> Linear."""

    name = "cnn"
    modality = "image"
    required_inputs = ("image",)

    def __init__(self, num_classes: int = NUM_CLASSES, backbone: str = "scratch",
                 channels: Sequence[int] = (32, 64, 128, 256), pretrained: bool = False,
                 hidden_dim: int = 128, dropout: float = 0.4,
                 freeze_backbone: bool = False) -> None:
        super().__init__(num_classes)
        self.set_config(num_classes=num_classes, backbone=backbone, channels=channels,
                        pretrained=pretrained, hidden_dim=hidden_dim, dropout=dropout,
                        freeze_backbone=freeze_backbone)
        self.image_encoder = ImageEncoder(backbone, channels, pretrained)
        self.freeze_backbone = freeze_backbone

        if self.freeze_backbone:
            for parameter in self.image_encoder.features.parameters():
                parameter.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(self.image_encoder.out_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.image_encoder.features.eval()
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.image_encoder(image))

    def gradcam_layer(self) -> nn.Module:
        return self.image_encoder.gradcam_layer()
