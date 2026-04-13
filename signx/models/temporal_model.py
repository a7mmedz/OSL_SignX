"""ResNet34 + TemporalConv + BiLSTM front-end for Stage 3.

The paper drives the temporal model from latent features (not raw RGB), so
the "ResNet34" here acts on the (B, T, D) latent sequence by treating it as a
1-D signal — i.e. a sequence of 1-D residual blocks. This stays faithful to
the spirit of the paper while remaining lightweight.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class _Res1DBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = self.bn2(self.conv2(x))
        return self.act(x + residual)


class ResNet34_1D(nn.Module):
    """1-D analogue of ResNet34 over a (B, C, T) sequence.

    Channel counts are kept constant; we use 16 residual blocks total
    (matches ResNet34's count of basic blocks across stages).
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[_Res1DBlock(out_channels, dropout) for _ in range(16)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.stem(x))


class TemporalConvBlock(nn.Module):
    """Stacked 1-D temporal convolutions with optional stride for downsampling."""

    def __init__(
        self,
        channels: List[int],
        kernels: List[int],
        strides: List[int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert len(channels) >= 1 and len(channels) == len(kernels) == len(strides)
        layers: list[nn.Module] = []
        prev = channels[0]
        for ch, k, s in zip(channels, kernels, strides):
            layers += [
                nn.Conv1d(prev, ch, kernel_size=k, stride=s, padding=k // 2),
                nn.BatchNorm1d(ch),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev = ch
        self.net = nn.Sequential(*layers)
        self.out_channels = channels[-1]
        self.total_stride = 1
        for s in strides:
            self.total_stride *= s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalModel(nn.Module):
    """ResNet1D -> TemporalConv -> BiLSTM front-end producing (B, T', H) features.

    Args:
        in_dim: input latent dim (2048 or pruned).
        tconv_channels: channel sizes for stacked temporal conv layers.
        tconv_kernels: kernel sizes (same length as channels).
        tconv_strides: strides (same length as channels).
        lstm_hidden: hidden size of BiLSTM (per direction).
        lstm_layers: number of BiLSTM layers.
        bidirectional: BiLSTM toggle.
        dropout: dropout in conv blocks and LSTM.
    """

    def __init__(
        self,
        in_dim: int,
        tconv_channels: List[int],
        tconv_kernels: List[int],
        tconv_strides: List[int],
        lstm_hidden: int = 512,
        lstm_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        # Stage 1: 1-D ResNet over channels=in_dim
        # To keep channel counts manageable we project to tconv_channels[0] first.
        self.input_proj = nn.Conv1d(in_dim, tconv_channels[0], kernel_size=1)
        self.resnet = ResNet34_1D(
            in_channels=tconv_channels[0],
            out_channels=tconv_channels[0],
            dropout=dropout,
        )
        self.tconv = TemporalConvBlock(tconv_channels, tconv_kernels, tconv_strides, dropout)
        self.lstm = nn.LSTM(
            input_size=tconv_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.output_dim = lstm_hidden * (2 if bidirectional else 1)
        self.total_stride = self.tconv.total_stride

    def forward(
        self,
        latent: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """latent: (B, T, in_dim). Returns ((B, T', H), new_lengths)."""
        x = latent.transpose(1, 2)            # (B, in_dim, T)
        x = self.input_proj(x)
        x = self.resnet(x)
        x = self.tconv(x)                     # (B, C, T')
        x = x.transpose(1, 2)                 # (B, T', C)
        x, _ = self.lstm(x)                   # (B, T', H)

        new_lengths = None
        if lengths is not None:
            new_lengths = torch.div(lengths, self.total_stride, rounding_mode="floor")
            new_lengths = torch.clamp(new_lengths, min=1)
        return x, new_lengths
