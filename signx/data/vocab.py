"""Gloss vocabulary handling with Arabic UTF-8 support.

The vocabulary file is plain UTF-8 text, one entry per line:

    0000 <blank>
    0001 مستشفى
    0002 عيادة
    ...

The blank entry MUST be at index 0 (CTC convention).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence


class GlossVocab:
    """Bidirectional mapping between integer gloss IDs and Arabic gloss strings.

    Attributes:
        id2gloss: list indexed by gloss id -> Arabic string.
        gloss2id: dict from Arabic string -> gloss id.
        blank_id: id used for CTC blank (always 0 here).
    """

    BLANK_TOKEN = "<blank>"
    UNK_TOKEN = "<unk>"

    def __init__(self, id2gloss: Sequence[str], blank_id: int = 0) -> None:
        self.id2gloss: List[str] = list(id2gloss)
        self.gloss2id = {g: i for i, g in enumerate(self.id2gloss)}
        self.blank_id = blank_id
        if self.id2gloss[blank_id] != self.BLANK_TOKEN:
            raise ValueError(
                f"Expected blank token at index {blank_id}, got {self.id2gloss[blank_id]!r}"
            )

    @classmethod
    def from_file(cls, path: str | Path) -> "GlossVocab":
        """Load a vocab file. Each line is `<id> <gloss>` (whitespace separated)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {path}")
        entries: List[tuple[int, str]] = []
        with path.open("r", encoding="utf-8") as f:
            for lineno, raw in enumerate(f, 1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    raise ValueError(f"{path}:{lineno}: malformed line: {raw!r}")
                try:
                    idx = int(parts[0])
                except ValueError as e:
                    raise ValueError(f"{path}:{lineno}: bad gloss id {parts[0]!r}") from e
                entries.append((idx, parts[1]))
        entries.sort(key=lambda x: x[0])
        # densify
        id2gloss: List[str] = []
        for idx, gloss in entries:
            while len(id2gloss) < idx:
                id2gloss.append(cls.UNK_TOKEN)
            id2gloss.append(gloss)
        return cls(id2gloss, blank_id=0)

    def __len__(self) -> int:
        return len(self.id2gloss)

    @property
    def vocab_size(self) -> int:
        return len(self.id2gloss)

    def encode(self, glosses: Iterable[str]) -> List[int]:
        """Convert a sequence of Arabic gloss strings into ids."""
        out = []
        for g in glosses:
            if g not in self.gloss2id:
                raise KeyError(f"Unknown gloss: {g!r}")
            out.append(self.gloss2id[g])
        return out

    def decode(self, ids: Iterable[int], strip_blank: bool = True) -> List[str]:
        """Convert a sequence of ids back into Arabic gloss strings."""
        out = []
        for i in ids:
            if strip_blank and i == self.blank_id:
                continue
            if 0 <= i < len(self.id2gloss):
                out.append(self.id2gloss[i])
            else:
                out.append(self.UNK_TOKEN)
        return out

    def ctc_collapse(self, ids: Sequence[int]) -> List[int]:
        """Collapse repeats and remove blanks (CTC greedy decoding)."""
        out: List[int] = []
        prev = None
        for i in ids:
            if i != prev and i != self.blank_id:
                out.append(int(i))
            prev = i
        return out
