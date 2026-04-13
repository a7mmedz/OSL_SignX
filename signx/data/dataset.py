"""PyTorch datasets for word-level and sentence-level OSL videos.

The OSL dataset is laid out as:

    final_split/{train,dev,test}/rgb/*.mp4   -- word-level (split already done)
    dataset/OSL-Sentences/rgb_format/*.mp4   -- sentence-level (not yet split)

Filename convention: {ID}_{SignerID}_{TakeID}.mp4 e.g. 0001_S02_T02.mp4

Word-level samples have a single gloss (the WordID).
Sentence-level samples are read from `data/sentence_glosses.txt` which maps
SentenceID -> space-separated gloss IDs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from .vocab import GlossVocab

# Filename: 0001_S02_T02.mp4
_NAME_RE = re.compile(r"^(?P<id>\d+)_S(?P<signer>\d+)_T(?P<take>\d+)\.(mp4|avi|mov)$")


@dataclass
class VideoSample:
    """One row in a dataset manifest."""
    video_path: Path
    item_id: int           # WordID or SentenceID
    signer_id: int
    take_id: int
    gloss_ids: List[int]   # ground-truth gloss sequence


def _parse_name(path: Path) -> Optional[Dict[str, int]]:
    m = _NAME_RE.match(path.name)
    if not m:
        return None
    return {
        "id": int(m.group("id")),
        "signer": int(m.group("signer")),
        "take": int(m.group("take")),
    }


def _load_video_decord(path: Path, max_frames: int) -> torch.Tensor:
    """Load a video as (T, H, W, 3) uint8. Falls back to torchvision if decord missing."""
    try:
        import decord  # type: ignore
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(str(path))
        n = len(vr)
        if n == 0:
            raise RuntimeError(f"Empty video: {path}")
        if n > max_frames:
            idx = torch.linspace(0, n - 1, max_frames).long().tolist()
        else:
            idx = list(range(n))
        frames = vr.get_batch(idx)  # (T, H, W, 3) uint8
        return frames
    except ImportError:
        from torchvision.io import read_video
        frames, _, _ = read_video(str(path), pts_unit="sec")
        if frames.shape[0] > max_frames:
            idx = torch.linspace(0, frames.shape[0] - 1, max_frames).long()
            frames = frames[idx]
        return frames


class _BaseOSLDataset(Dataset):
    """Common logic for both word- and sentence-level datasets."""

    def __init__(
        self,
        samples: Sequence[VideoSample],
        vocab: GlossVocab,
        transform: Optional[Callable] = None,
        max_frames: int = 256,
    ) -> None:
        self.samples = list(samples)
        self.vocab = vocab
        self.transform = transform
        self.max_frames = max_frames

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        frames = _load_video_decord(s.video_path, max_frames=self.max_frames)
        if self.transform is not None:
            frames = self.transform(frames)
        return {
            "video": frames,                              # (T, 3, H, W) float
            "gloss_ids": torch.tensor(s.gloss_ids, dtype=torch.long),
            "video_length": frames.shape[0],
            "gloss_length": len(s.gloss_ids),
            "item_id": s.item_id,
            "signer_id": s.signer_id,
            "take_id": s.take_id,
            "path": str(s.video_path),
        }


class OSLWordDataset(_BaseOSLDataset):
    """Word-level dataset. Each video has exactly one gloss label = its WordID."""

    def __init__(
        self,
        split_dir: str | Path,
        split: str,
        vocab: GlossVocab,
        transform: Optional[Callable] = None,
        max_frames: int = 256,
    ) -> None:
        rgb_dir = Path(split_dir) / split / "rgb"
        if not rgb_dir.exists():
            raise FileNotFoundError(f"Missing word split directory: {rgb_dir}")
        samples: List[VideoSample] = []
        for video_path in sorted(rgb_dir.glob("*.mp4")):
            meta = _parse_name(video_path)
            if meta is None:
                continue
            word_id = meta["id"]
            if not (0 < word_id < vocab.vocab_size):
                continue
            samples.append(
                VideoSample(
                    video_path=video_path,
                    item_id=word_id,
                    signer_id=meta["signer"],
                    take_id=meta["take"],
                    gloss_ids=[word_id],
                )
            )
        super().__init__(samples, vocab, transform, max_frames)
        self.split_dir = Path(split_dir)
        self.split = split


class OSLSentenceDataset(_BaseOSLDataset):
    """Sentence-level dataset. Reads gloss sequences from `sentence_glosses.txt`.

    File format (one sentence per line):
        {SentenceID} {gloss1} {gloss2} ... {glossN}

    where each glossK is a WordID (integer).
    """

    def __init__(
        self,
        sentence_dir: str | Path,
        sentence_glosses_file: str | Path,
        vocab: GlossVocab,
        transform: Optional[Callable] = None,
        max_frames: int = 256,
        split_filter: Optional[Callable[[VideoSample], bool]] = None,
    ) -> None:
        sentence_dir = Path(sentence_dir)
        gloss_map = self._load_sentence_glosses(sentence_glosses_file)
        samples: List[VideoSample] = []
        if sentence_dir.exists():
            for video_path in sorted(sentence_dir.glob("*.mp4")):
                meta = _parse_name(video_path)
                if meta is None:
                    continue
                sid = meta["id"]
                if sid not in gloss_map:
                    continue
                samples.append(
                    VideoSample(
                        video_path=video_path,
                        item_id=sid,
                        signer_id=meta["signer"],
                        take_id=meta["take"],
                        gloss_ids=gloss_map[sid],
                    )
                )
        if split_filter is not None:
            samples = [s for s in samples if split_filter(s)]
        super().__init__(samples, vocab, transform, max_frames)

    @staticmethod
    def _load_sentence_glosses(path: str | Path) -> Dict[int, List[int]]:
        path = Path(path)
        out: Dict[int, List[int]] = {}
        if not path.exists():
            # Allow missing file -> empty dataset (placeholder until annotated)
            return out
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                sid = int(parts[0])
                glosses = [int(x) for x in parts[1:]]
                out[sid] = glosses
        return out


def build_dataset(
    cfg,
    vocab: GlossVocab,
    split: str = "train",
    transform: Optional[Callable] = None,
):
    """Factory: pick word- or sentence-level dataset based on `cfg.data.level`."""
    level = getattr(cfg.data, "level", "word")
    max_frames = int(cfg.video.max_frames)
    if level == "word":
        return OSLWordDataset(
            split_dir=cfg.word_split_dir,
            split=split,
            vocab=vocab,
            transform=transform,
            max_frames=max_frames,
        )
    if level == "sentence":
        return OSLSentenceDataset(
            sentence_dir=cfg.sentence_dir,
            sentence_glosses_file=cfg.sentence_glosses_file,
            vocab=vocab,
            transform=transform,
            max_frames=max_frames,
        )
    raise ValueError(f"Unknown data level: {level}")
