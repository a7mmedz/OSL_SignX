"""Beam search decoding for Stage 3 CSLR outputs.

Implements two decoding strategies:
  1. ctc_greedy_decode  - fast greedy CTC (argmax + collapse).
  2. BeamSearchDecoder  - prefix beam search over CTC log-probs (Algorithm 3
     from the SignX paper), with optional length penalty.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Greedy CTC
# ---------------------------------------------------------------------------

def ctc_greedy_decode(log_probs: torch.Tensor, blank_id: int = 0) -> List[int]:
    """Greedy CTC decode from (T, V) log-probabilities.

    Returns a list of non-blank gloss ids (repeats collapsed).
    """
    ids = log_probs.argmax(dim=-1).tolist()
    out: List[int] = []
    prev = None
    for i in ids:
        if i != prev:
            if i != blank_id:
                out.append(int(i))
            prev = i
    return out


# ---------------------------------------------------------------------------
# Prefix beam search
# ---------------------------------------------------------------------------

@dataclass
class _Beam:
    """One hypothesis in the beam."""
    prefix: Tuple[int, ...]
    score_nb: float = -math.inf    # log-prob of last token being non-blank
    score_b:  float = 0.0          # log-prob of last token being blank

    @property
    def total(self) -> float:
        return math.log(math.exp(self.score_nb) + math.exp(self.score_b))


def _log_add(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


class BeamSearchDecoder:
    """CTC prefix beam search with length penalty.

    Args:
        vocab_size: total vocabulary size (incl. blank).
        blank_id: CTC blank token index.
        beam_size: number of beams to keep.
        length_penalty: alpha in score / |hyp|^alpha.
    """

    def __init__(
        self,
        vocab_size: int,
        blank_id: int = 0,
        beam_size: int = 8,
        length_penalty: float = 1.0,
    ) -> None:
        self.vocab_size = vocab_size
        self.blank_id = blank_id
        self.beam_size = beam_size
        self.length_penalty = length_penalty

    def decode(self, log_probs: torch.Tensor) -> List[int]:
        """Decode a single (T, V) float tensor of log-probs.

        Returns the best hypothesis as a list of gloss ids.
        """
        lp = log_probs.float().cpu()
        T = lp.shape[0]

        # initialise beams with empty prefix
        beams: Dict[Tuple[int, ...], _Beam] = {(): _Beam(prefix=())}
        beams[()].score_b = 0.0

        for t in range(T):
            new_beams: Dict[Tuple[int, ...], _Beam] = {}

            for prefix, beam in beams.items():
                p_tot = beam.total

                # Extend with blank
                nb = _Beam(prefix=prefix)
                p_b = float(lp[t, self.blank_id])
                nb.score_b = _log_add(
                    new_beams.get(prefix, _Beam(prefix=prefix)).score_b,
                    p_tot + p_b,
                )
                _merge(new_beams, prefix, nb)

                # Extend with each non-blank token
                for c in range(self.vocab_size):
                    if c == self.blank_id:
                        continue
                    new_prefix = prefix + (c,)
                    p_c = float(lp[t, c])
                    ext = _Beam(prefix=new_prefix)
                    if prefix and prefix[-1] == c:
                        # Can only extend from blank path
                        ext.score_nb = _log_add(
                            new_beams.get(new_prefix, _Beam(prefix=new_prefix)).score_nb,
                            beam.score_b + p_c,
                        )
                    else:
                        ext.score_nb = _log_add(
                            new_beams.get(new_prefix, _Beam(prefix=new_prefix)).score_nb,
                            p_tot + p_c,
                        )
                    _merge(new_beams, new_prefix, ext)

            # Prune to top-K
            beams = dict(
                sorted(new_beams.items(), key=lambda kv: kv[1].total, reverse=True)[: self.beam_size]
            )

        # Pick best hypothesis with length penalty
        def _penalized(b: _Beam) -> float:
            ln = max(1, len(b.prefix))
            return b.total / (ln ** self.length_penalty)

        best = max(beams.values(), key=_penalized)
        return list(best.prefix)


def _merge(
    beams: Dict[Tuple[int, ...], _Beam], prefix: Tuple[int, ...], candidate: _Beam
) -> None:
    if prefix in beams:
        existing = beams[prefix]
        existing.score_nb = _log_add(existing.score_nb, candidate.score_nb)
        existing.score_b = _log_add(existing.score_b, candidate.score_b)
    else:
        beams[prefix] = candidate
