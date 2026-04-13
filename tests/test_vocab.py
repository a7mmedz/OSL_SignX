"""Unit tests for GlossVocab."""
import tempfile
from pathlib import Path

import pytest

from signx.data.vocab import GlossVocab


SAMPLE_VOCAB = """\
0000 <blank>
0001 مستشفى
0002 عيادة
0003 صيدلية
"""


def _write_vocab(content: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    )
    tmp.write(content)
    tmp.close()
    return Path(tmp.name)


def test_load_from_file():
    path = _write_vocab(SAMPLE_VOCAB)
    vocab = GlossVocab.from_file(path)
    assert vocab.vocab_size == 4
    assert vocab.blank_id == 0
    assert vocab.id2gloss[0] == "<blank>"
    assert vocab.id2gloss[1] == "مستشفى"


def test_encode_decode_roundtrip():
    path = _write_vocab(SAMPLE_VOCAB)
    vocab = GlossVocab.from_file(path)
    glosses = ["مستشفى", "عيادة", "صيدلية"]
    ids = vocab.encode(glosses)
    assert ids == [1, 2, 3]
    decoded = vocab.decode(ids, strip_blank=True)
    assert decoded == glosses


def test_ctc_collapse():
    path = _write_vocab(SAMPLE_VOCAB)
    vocab = GlossVocab.from_file(path)
    # 0=blank, 1,1,0,2,2,0,3
    raw = [0, 1, 1, 0, 2, 2, 0, 3]
    collapsed = vocab.ctc_collapse(raw)
    assert collapsed == [1, 2, 3]


def test_unknown_gloss_raises():
    path = _write_vocab(SAMPLE_VOCAB)
    vocab = GlossVocab.from_file(path)
    with pytest.raises(KeyError):
        vocab.encode(["غير_موجود"])


def test_blank_not_at_zero_raises():
    bad = "0001 <blank>\n0002 كلمة\n"
    path = _write_vocab(bad)
    with pytest.raises(ValueError):
        GlossVocab.from_file(path)
