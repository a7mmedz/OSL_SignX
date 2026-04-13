"""Multi-head attention pose fusion (Stage 1).

Takes per-frame pose features (B, T, D_pose) and produces a 2048-dim latent
sequence (B, T, latent_dim). The fusion uses self-attention across frames
followed by a linear projection to the shared SignX latent space.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequences."""

    def __init__(self, dim: int, max_len: int = 4096) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]]


class PoseFusionEncoder(nn.Module):
    """Project + multi-head self-attention over per-frame pose features.

    Args:
        pose_input_dim: D_pose (e.g. 258 for MediaPipe-only, 1959 for full5).
        latent_dim: target latent dimension (2048 in the paper).
        num_heads: attention heads.
        num_layers: stacked self-attention layers.
        dropout: dropout prob inside the encoder.
    """

    def __init__(
        self,
        pose_input_dim: int,
        latent_dim: int = 2048,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(pose_input_dim, latent_dim)
        self.pos_enc = PositionalEncoding(latent_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(latent_dim)

    def forward(
        self,
        pose: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """pose: (B, T, D_pose). Returns (B, T, latent_dim)."""
        x = self.input_proj(pose)
        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.out_norm(x)
