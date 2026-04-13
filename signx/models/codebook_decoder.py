"""CodeBook-based gloss decoder used in Stage 1 (replaces T5 from the paper).

Idea:
  - A learned codebook of K embeddings sits between the latent space and the
    output vocabulary, encouraging discrete sign-aware features.
  - Each input latent vector is matched against the codebook (cosine sim),
    soft-assigned to entries via softmax, and the resulting code embedding is
    fed into a small Transformer decoder that emits gloss logits.
  - The vector-quantization commitment term is exposed via `last_commit_loss`
    so the trainer can include it in the total loss.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CodeBook(nn.Module):
    """Soft vector-quantization codebook with EMA-free training.

    Output shapes:
        codes:   (B, T, codebook_dim)
        attn:    (B, T, codebook_size)
    """

    def __init__(self, codebook_size: int, codebook_dim: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.temperature = temperature
        self.embedding = nn.Parameter(torch.randn(codebook_size, codebook_dim) * 0.02)
        self.last_commit_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, codebook_dim)
        x_n = F.normalize(x, dim=-1)
        cb_n = F.normalize(self.embedding, dim=-1)
        logits = x_n @ cb_n.T / self.temperature       # (B, T, K)
        attn = F.softmax(logits, dim=-1)
        codes = attn @ self.embedding                   # (B, T, codebook_dim)
        # Commitment loss: keep encoder outputs near codebook entries
        self.last_commit_loss = F.mse_loss(x, codes.detach())
        return codes, attn


class CodeBookDecoder(nn.Module):
    """Project latent -> codebook lookup -> Transformer decoder -> vocab logits.

    Args:
        latent_dim: dim of incoming pose-fused features.
        codebook_size: number of code entries.
        codebook_dim: per-entry feature dim.
        num_layers: Transformer decoder layers.
        num_heads: attention heads.
        vocab_size: output vocab size (gloss vocabulary).
        max_target_len: maximum length of target gloss sequences.
        dropout: decoder dropout.
    """

    def __init__(
        self,
        latent_dim: int,
        codebook_size: int = 4096,
        codebook_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        vocab_size: int = 802,
        max_target_len: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(latent_dim, codebook_dim)
        self.codebook = CodeBook(codebook_size, codebook_dim)
        self.target_embedding = nn.Embedding(vocab_size, codebook_dim)
        self.pos_embedding = nn.Embedding(max_target_len, codebook_dim)
        self.max_target_len = max_target_len

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=codebook_dim,
            nhead=num_heads,
            dim_feedforward=codebook_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(codebook_dim)
        self.lm_head = nn.Linear(codebook_dim, vocab_size, bias=False)

    @staticmethod
    def _causal_mask(size: int, device) -> torch.Tensor:
        return torch.triu(torch.full((size, size), float("-inf"), device=device), diagonal=1)

    def encode_memory(self, latent: torch.Tensor) -> torch.Tensor:
        """Project latent -> codebook codes (acts as the encoder memory)."""
        x = self.in_proj(latent)
        codes, _ = self.codebook(x)
        return codes

    def forward(
        self,
        latent: torch.Tensor,
        target: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None = None,
        target_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Teacher-forced decoding.

        latent: (B, T_src, latent_dim)
        target: (B, T_tgt) int64 — already shifted right by the trainer
        Returns logits of shape (B, T_tgt, vocab_size).
        """
        memory = self.encode_memory(latent)
        b, l = target.shape
        if l > self.max_target_len:
            raise ValueError(f"target length {l} exceeds max_target_len {self.max_target_len}")
        pos = torch.arange(l, device=target.device).unsqueeze(0).expand(b, l)
        tgt_emb = self.target_embedding(target) * math.sqrt(self.codebook.codebook_dim)
        tgt_emb = tgt_emb + self.pos_embedding(pos)
        tgt_mask = self._causal_mask(l, target.device)
        out = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_key_padding_mask=target_key_padding_mask,
        )
        return self.lm_head(self.out_norm(out))

    @torch.no_grad()
    def generate(
        self,
        latent: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int = 64,
    ) -> torch.Tensor:
        """Greedy decoding for quick smoke tests / qualitative outputs."""
        b = latent.shape[0]
        device = latent.device
        ys = torch.full((b, 1), bos_id, dtype=torch.long, device=device)
        for _ in range(max_len - 1):
            logits = self.forward(latent, ys)
            next_tok = logits[:, -1].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)
            if (next_tok == eos_id).all():
                break
        return ys
