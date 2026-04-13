"""Transformer encoder-decoder for the Stage 3 CSLR head.

Encoder: takes the temporal model outputs (B, T', H) and contextualizes them.
Decoder: autoregressively emits gloss IDs (used for the cross-entropy auxiliary
loss), while a CTC head on the encoder output drives the alignment-free CTC loss.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class _SinusoidalPE(nn.Module):
    def __init__(self, dim: int, max_len: int = 4096) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]]


class SignXTransformer(nn.Module):
    """Transformer encoder-decoder + CTC head for continuous SLR.

    Args:
        input_dim: dimensionality coming from the temporal model (BiLSTM out).
        d_model: internal Transformer width.
        nhead: number of attention heads.
        num_encoder_layers / num_decoder_layers: layer counts.
        dim_feedforward: FFN size.
        vocab_size: gloss vocabulary size (incl. blank).
        dropout_attn / dropout_relu / dropout_res: dropout knobs from the paper.
        max_target_len: max gloss sequence length.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        vocab_size: int = 802,
        dropout_attn: float = 0.3,
        dropout_relu: float = 0.5,
        dropout_res: float = 0.4,
        max_target_len: int = 128,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_target_len = max_target_len

        self.input_proj = nn.Linear(input_dim, d_model)
        self.enc_pe = _SinusoidalPE(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_attn,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # CTC head — projects encoder outputs to vocab_size logits
        self.ctc_head = nn.Linear(d_model, vocab_size)

        # Autoregressive decoder for the cross-entropy auxiliary loss
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)
        self.dec_pe = _SinusoidalPE(d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_attn,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Extra dropouts kept as plain modules so trainers can probe them
        self.dropout_relu = nn.Dropout(dropout_relu)
        self.dropout_res = nn.Dropout(dropout_res)

    @staticmethod
    def _causal_mask(size: int, device) -> torch.Tensor:
        return torch.triu(torch.full((size, size), float("-inf"), device=device), diagonal=1)

    def encode(
        self,
        features: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.input_proj(features)
        x = self.enc_pe(x)
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)

    def forward(
        self,
        features: torch.Tensor,
        target: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
    ) -> dict:
        """Run encoder + (optional) decoder.

        Returns dict with:
            ctc_logits: (B, T', vocab_size)  -- for CTC loss
            dec_logits: (B, L, vocab_size)   -- for CE loss (None if target is None)
            memory:     (B, T', d_model)     -- encoder output (for downstream / KD)
        """
        memory = self.encode(features, src_key_padding_mask=src_key_padding_mask)
        ctc_logits = self.ctc_head(memory)

        dec_logits = None
        if target is not None:
            b, l = target.shape
            if l > self.max_target_len:
                raise ValueError(f"target length {l} exceeds max_target_len {self.max_target_len}")
            tgt_emb = self.tgt_embedding(target) * math.sqrt(self.d_model)
            tgt_emb = self.dec_pe(tgt_emb)
            tgt_mask = self._causal_mask(l, target.device)
            dec_out = self.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )
            dec_logits = self.lm_head(dec_out)

        return {"ctc_logits": ctc_logits, "dec_logits": dec_logits, "memory": memory}
