"""Validate the OSL dataset layout and emit per-split manifests.

This script does NOT copy or move any video data. It only:
  1. Verifies the configured dataset paths exist.
  2. Counts videos in each split and reports per-signer/per-class stats.
  3. Optionally writes JSON manifests to outputs/manifests/.

Usage:
    python data/prepare_data.py --config configs/stage1_pose2gloss.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Make repo importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signx.data.dataset import OSLWordDataset, OSLSentenceDataset, _parse_name
from signx.data.vocab import GlossVocab
from signx.utils.config import load_config


def _validate_word_split(split_dir: Path, split: str) -> dict:
    rgb_dir = split_dir / split / "rgb"
    if not rgb_dir.exists():
        return {"split": split, "exists": False, "path": str(rgb_dir)}
    videos = sorted(rgb_dir.glob("*.mp4"))
    classes = Counter()
    signers = Counter()
    for v in videos:
        meta = _parse_name(v)
        if meta:
            classes[meta["id"]] += 1
            signers[meta["signer"]] += 1
    return {
        "split": split,
        "exists": True,
        "path": str(rgb_dir),
        "num_videos": len(videos),
        "num_unique_classes": len(classes),
        "num_signers": len(signers),
        "signers": dict(sorted(signers.items())),
    }


def _validate_sentences(sentence_dir: Path) -> dict:
    if not sentence_dir.exists():
        return {"exists": False, "path": str(sentence_dir)}
    videos = sorted(sentence_dir.glob("*.mp4"))
    return {
        "exists": True,
        "path": str(sentence_dir),
        "num_videos": len(videos),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage1_pose2gloss.yaml")
    ap.add_argument("--write-manifests", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Vocab check
    vocab_path = Path(cfg.vocab_file)
    if not vocab_path.exists():
        print(f"[ERROR] Missing vocab file: {vocab_path}")
    else:
        vocab = GlossVocab.from_file(vocab_path)
        print(f"[OK]    Loaded vocab: {vocab.vocab_size} entries (incl. blank)")

    # Word splits
    word_root = Path(cfg.word_split_dir)
    print(f"\n=== Word-level splits @ {word_root} ===")
    for split in ("train", "dev", "test"):
        info = _validate_word_split(word_root, split)
        if not info["exists"]:
            print(f"[MISS]  {split}: {info['path']}")
            continue
        print(
            f"[OK]    {split}: {info['num_videos']} videos, "
            f"{info['num_unique_classes']} classes, {info['num_signers']} signers"
        )

    # Sentence dir
    sent_root = Path(cfg.sentence_dir)
    print(f"\n=== Sentence-level @ {sent_root} ===")
    sent_info = _validate_sentences(sent_root)
    if not sent_info["exists"]:
        print(f"[MISS]  {sent_info['path']}")
    else:
        print(f"[OK]    {sent_info['num_videos']} sentence videos")

    sg_path = Path(cfg.sentence_glosses_file)
    if not sg_path.exists():
        print(
            f"[WARN]  No sentence_glosses.txt at {sg_path} — sentence training will be empty.\n"
            f"        Create it with format: '<SentenceID> <wordId1> <wordId2> ...' per line."
        )
    else:
        n = sum(1 for line in sg_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith("#"))
        print(f"[OK]    sentence_glosses.txt: {n} annotated sentences")

    # Optional manifest writing
    if args.write_manifests:
        out_dir = Path(cfg.output_dir) / "manifests"
        out_dir.mkdir(parents=True, exist_ok=True)
        for split in ("train", "dev", "test"):
            info = _validate_word_split(word_root, split)
            (out_dir / f"word_{split}.json").write_text(
                json.dumps(info, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        print(f"\n[OK] Manifests written to {out_dir}")


if __name__ == "__main__":
    main()
