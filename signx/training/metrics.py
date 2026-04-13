"""Evaluation metrics: WER, BLEU, P-I accuracy."""
from __future__ import annotations

from typing import List, Sequence

try:
    import editdistance
except ImportError:                       # pragma: no cover
    editdistance = None


def _levenshtein(ref: Sequence, hyp: Sequence) -> int:
    """Word/token-level Levenshtein distance."""
    if editdistance is not None:
        return int(editdistance.eval(list(ref), list(hyp)))
    # Pure-python fallback (slow but correct)
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def compute_wer(references: List[Sequence], hypotheses: List[Sequence]) -> float:
    """Word/Gloss Error Rate (lower is better)."""
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have equal length")
    total_dist = 0
    total_len = 0
    for ref, hyp in zip(references, hypotheses):
        total_dist += _levenshtein(ref, hyp)
        total_len += max(1, len(ref))
    return total_dist / total_len


def compute_pi_accuracy(references: List[Sequence], hypotheses: List[Sequence]) -> float:
    """Position-Independent accuracy: |ref ∩ hyp| / |ref|, averaged over samples."""
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have equal length")
    if not references:
        return 0.0
    accs = []
    for ref, hyp in zip(references, hypotheses):
        if not ref:
            continue
        ref_multi = list(ref)
        hyp_multi = list(hyp)
        matched = 0
        for token in ref_multi:
            if token in hyp_multi:
                hyp_multi.remove(token)
                matched += 1
        accs.append(matched / len(ref_multi))
    return sum(accs) / len(accs) if accs else 0.0


def compute_bleu(references: List[Sequence], hypotheses: List[Sequence]) -> float:
    """Corpus-level BLEU. Uses sacrebleu if available, otherwise simple BLEU-1."""
    try:
        import sacrebleu
        # sacrebleu expects strings; join token ids
        ref_strs = [" ".join(map(str, r)) for r in references]
        hyp_strs = [" ".join(map(str, h)) for h in hypotheses]
        return float(sacrebleu.corpus_bleu(hyp_strs, [ref_strs]).score)
    except ImportError:                   # pragma: no cover
        # BLEU-1 fallback
        if not references:
            return 0.0
        scores = []
        for ref, hyp in zip(references, hypotheses):
            if not hyp:
                scores.append(0.0)
                continue
            ref_set = set(map(str, ref))
            matches = sum(1 for t in hyp if str(t) in ref_set)
            scores.append(matches / len(hyp))
        return 100.0 * sum(scores) / len(scores)
