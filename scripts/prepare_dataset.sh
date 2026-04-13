#!/usr/bin/env bash
# Validate dataset structure and optionally write manifests.
# Usage: bash scripts/prepare_dataset.sh [--write-manifests]
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== OSL-SignX Dataset Preparation ==="
echo "Config: configs/stage1_pose2gloss.yaml"
echo ""

.venv/bin/python data/prepare_data.py --config configs/stage1_pose2gloss.yaml "$@"
