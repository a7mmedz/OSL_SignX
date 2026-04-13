"""Pose-Aware Feature Compilation (Algorithm 1 in the SignX paper).

Pipeline:
  1. LayerNorm across feature dim.
  2. Cross-frame covariance whitening (ZCA-style).
  3. Stochastic frame dropping (training only).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PoseAwareFeatureCompiler(nn.Module):
    """LayerNorm + whitening + frame dropout for per-frame pose features.

    Args:
        feature_dim: input feature dimension D.
        frame_dropout: probability to drop each frame during training.
        whitening_eps: regularizer added to covariance diagonal before inversion.
        whitening_enabled: toggle the whitening step.
    """

    def __init__(
        self,
        feature_dim: int,
        frame_dropout: float = 0.1,
        whitening_eps: float = 1e-4,
        whitening_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.frame_dropout = frame_dropout
        self.whitening_eps = whitening_eps
        self.whitening_enabled = whitening_enabled
        self.norm = nn.LayerNorm(feature_dim)

    def _whiten(self, x: torch.Tensor) -> torch.Tensor:
        """Cross-frame covariance whitening on (T, D)."""
        t, d = x.shape
        if t < 2:
            return x
        mu = x.mean(dim=0, keepdim=True)
        xc = x - mu
        cov = (xc.T @ xc) / (t - 1)
        cov = cov + self.whitening_eps * torch.eye(d, device=x.device, dtype=x.dtype)
        # Symmetric eigendecomposition (more stable than Cholesky for low rank)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        eigvals = torch.clamp(eigvals, min=self.whitening_eps)
        inv_sqrt = eigvecs @ torch.diag(eigvals.rsqrt()) @ eigvecs.T
        return xc @ inv_sqrt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) or (T, D). Returns same shape."""
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True

        b, t, d = x.shape
        x = self.norm(x)

        if self.whitening_enabled:
            out = torch.empty_like(x)
            for i in range(b):
                out[i] = self._whiten(x[i])
            x = out

        if self.training and self.frame_dropout > 0:
            keep_mask = (torch.rand(b, t, device=x.device) > self.frame_dropout).float()
            # Avoid dropping ALL frames
            for i in range(b):
                if keep_mask[i].sum() == 0:
                    keep_mask[i, 0] = 1.0
            x = x * keep_mask.unsqueeze(-1)

        if squeeze:
            x = x.squeeze(0)
        return x
