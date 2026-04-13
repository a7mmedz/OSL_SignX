#!/usr/bin/env bash
# Extract MediaPipe pose features for all word-level videos and cache them.
# This is OPTIONAL — training can extract on-the-fly but caching is faster.
#
# Usage: bash scripts/extract_poses.sh [--split train|dev|test]
set -euo pipefail
cd "$(dirname "$0")/.."

SPLIT=${1:-train}
DATASET_ROOT=$(grep "dataset_root:" configs/paths.yaml | awk '{print $2}')
WORD_SPLIT_DIR="${DATASET_ROOT}/final_split"
CACHE_DIR="outputs/cache/pose_${SPLIT}"

echo "=== Extracting MediaPipe poses for split: ${SPLIT} ==="
echo "Source : ${WORD_SPLIT_DIR}/${SPLIT}/rgb/"
echo "Cache  : ${CACHE_DIR}"
echo ""

mkdir -p "${CACHE_DIR}"

.venv/bin/python - <<'EOF'
import sys, os, torch
from pathlib import Path
from tqdm import tqdm

split = os.environ.get("SPLIT", "train")
src   = Path(os.environ["WORD_SPLIT_DIR"]) / split / "rgb"
cache = Path(os.environ["CACHE_DIR"])

sys.path.insert(0, ".")
from signx.pose.mediapipe_extractor import MediaPipePoseExtractor
extractor = MediaPipePoseExtractor(output_dim=258)

videos = sorted(src.glob("*.mp4"))
print(f"Found {len(videos)} videos")
for v in tqdm(videos, desc="Extracting"):
    out_path = cache / f"{v.stem}.pt"
    if out_path.exists():
        continue
    try:
        feats = extractor.extract(v)
        torch.save(feats, out_path)
    except Exception as e:
        print(f"  [WARN] {v.name}: {e}", file=sys.stderr)
print("Done.")
EOF
