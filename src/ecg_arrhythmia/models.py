"""Neural network architectures for fixed-length ECG beat waveforms."""

from __future__ import annotations

from typing import Callable, Dict

import torch
from torch import nn


class CNN1D(nn.Module):
    """Compact convolutional baseline with global average pooling."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs.unsqueeze(1)).squeeze(-1)
        return self.classifier(features)


class ResidualBlock1D(nn.Module):
    """Two-convolution residual block with an optional projection shortcut."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, kernel_size=5, stride=stride,
                padding=2, bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.layers(inputs) + self.shortcut(inputs))


class ResNet1D(nn.Module):
    """Residual 1D CNN for deeper waveform feature extraction."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            ResidualBlock1D(32, 32),
            ResidualBlock1D(32, 64, stride=2),
            ResidualBlock1D(64, 128, stride=2),
            ResidualBlock1D(128, 128),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.blocks(self.stem(inputs.unsqueeze(1))).squeeze(-1)
        return self.classifier(features)


class CNNLSTM(nn.Module):
    """Convolutional feature encoder followed by a temporal LSTM."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(
            input_size=64, hidden_size=64, num_layers=1,
            batch_first=True, bidirectional=True,
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence = self.encoder(inputs.unsqueeze(1)).transpose(1, 2)
        _, (hidden, _) = self.lstm(sequence)
        features = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(features)


MODEL_FACTORIES: Dict[str, Callable[[int], nn.Module]] = {
    "cnn": CNN1D,
    "resnet1d": ResNet1D,
    "cnn_lstm": CNNLSTM,
}


def build_model(name: str, num_classes: int = 3) -> nn.Module:
    """Create one of the supported models by its CLI name."""
    try:
        factory = MODEL_FACTORIES[name]
    except KeyError as error:
        choices = ", ".join(MODEL_FACTORIES)
        raise ValueError(f"Unknown model {name!r}; choose from {choices}.") from error
    return factory(num_classes=num_classes)
