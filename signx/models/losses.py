"""Loss functions used across all SignX training stages.

  - word_matching_loss : per-token CE that encourages the decoder to produce
                         exactly the right gloss IDs (Stage 1).
  - contrastive_loss   : InfoNCE-style alignment of pose-fused latents and
                         gloss embeddings (Stage 1).
  - distillation_loss  : KL divergence between teacher and student logits (Stage 3).
  - latent_regularizer : L2 penalty keeping latent magnitudes bounded.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def word_matching_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pad_id: int = 0,
    label_smoothing: float = 0.1,
) -> torch.Tensor:
    """Standard padded cross-entropy with label smoothing.

    logits: (B, L, V), target: (B, L) -- assumed already shifted to be the
    actual next-token target sequence.
    """
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target.reshape(-1),
        ignore_index=pad_id,
        label_smoothing=label_smoothing,
    )


def contrastive_loss(
    pose_features: torch.Tensor,
    gloss_features: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric InfoNCE between pooled pose and gloss embeddings.

    pose_features:  (B, D) — e.g. mean-pooled pose latent for each video.
    gloss_features: (B, D) — embedded ground-truth gloss for the same video.
    """
    p = F.normalize(pose_features, dim=-1)
    g = F.normalize(gloss_features, dim=-1)
    logits = p @ g.T / temperature
    targets = torch.arange(p.shape[0], device=p.device)
    loss_p2g = F.cross_entropy(logits, targets)
    loss_g2p = F.cross_entropy(logits.T, targets)
    return (loss_p2g + loss_g2p) * 0.5


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 4.0,
) -> torch.Tensor:
    """KL(softmax(teacher/T) || log_softmax(student/T)) * T^2."""
    s = F.log_softmax(student_logits / temperature, dim=-1)
    t = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (temperature ** 2)


def latent_regularizer(latent: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    """Mean squared L2 norm of latent features (keeps them bounded)."""
    return weight * latent.pow(2).mean()
