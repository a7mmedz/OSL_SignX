#!/usr/bin/env bash
# Run the three-stage SignX training pipeline.
#
# Usage:
#   bash scripts/train.sh                  # run all three stages
#   bash scripts/train.sh --stage 1        # only stage 1
#   bash scripts/train.sh --stage 2        # only stage 2
#   bash scripts/train.sh --stage 3        # only stage 3
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE=${STAGE:-all}

# Parse optional --stage and --dataset-root flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage) STAGE="$2"; shift 2 ;;
        --dataset-root) export DATASET_ROOT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Resolve the Python interpreter: prefer activated venv, then local .venv, then PATH
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
elif [[ -f ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="$(command -v python3 || command -v python)"
fi

run_stage() {
    local n=$1; shift
    echo ""
    echo "========================================"
    echo "  Training Stage ${n}"
    echo "========================================"
    "$PYTHON" "$@"
}

if [[ "$STAGE" == "all" || "$STAGE" == "1" ]]; then
    run_stage 1 -m signx.training.train_stage1 --config configs/stage1_pose2gloss.yaml
fi

if [[ "$STAGE" == "all" || "$STAGE" == "2" ]]; then
    run_stage 2 -m signx.training.train_stage2 --config configs/stage2_video2pose.yaml
fi

if [[ "$STAGE" == "all" || "$STAGE" == "3" ]]; then
    run_stage 3 -m signx.training.train_stage3 --config configs/stage3_cslr.yaml
fi

echo ""
echo "Training complete."
