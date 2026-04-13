#!/usr/bin/env bash
# Evaluate Stage 3 on the test set.
#
# Usage:
#   bash scripts/evaluate.sh \
#       --stage2-checkpoint outputs/checkpoints/stage2/best.pt \
#       --stage3-checkpoint outputs/checkpoints/stage3/best.pt
set -euo pipefail
cd "$(dirname "$0")/.."

S2_CKPT=${S2_CKPT:-outputs/checkpoints/stage2/best.pt}
S3_CKPT=${S3_CKPT:-outputs/checkpoints/stage3/best.pt}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage2-checkpoint) S2_CKPT="$2"; shift 2 ;;
        --stage3-checkpoint) S3_CKPT="$2"; shift 2 ;;
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

echo "=== Evaluation ==="
echo "Stage 2 checkpoint : ${S2_CKPT}"
echo "Stage 3 checkpoint : ${S3_CKPT}"
echo ""

"$PYTHON" -m signx.inference.evaluate \
    --config configs/stage3_cslr.yaml \
    --stage2-checkpoint "${S2_CKPT}" \
    --stage3-checkpoint "${S3_CKPT}" \
    --split test \
    --beam-size 8 \
    --output outputs/eval_results.json

echo ""
echo "Results saved to outputs/eval_results.json"
