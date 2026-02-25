#!/bin/bash
#SBATCH --job-name=taxembed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/taxembed_%j.out
#SBATCH --error=logs/taxembed_%j.err

# --- LRZ GPU Training Script for TaxPointCare ---
# Usage:
#   sbatch scripts/train_lrz.sh Arthropoda arthropoda_v6f 100
#   sbatch scripts/train_lrz.sh Mollusca mollusca_v6e 50
#   sbatch scripts/train_lrz.sh 1 ncbi_full_v8 150  # TaxID 1 = root of life

set -euo pipefail

IDENTIFIER="${1:?Usage: train_lrz.sh <identifier> <tag> [dim]}"
TAG="${2:?Usage: train_lrz.sh <identifier> <tag> [dim]}"
DIM="${3:-100}"

echo "=== TaxEmbed LRZ Training ==="
echo "  Identifier: $IDENTIFIER"
echo "  Tag:        $TAG"
echo "  Dimension:  $DIM"
echo "  GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  SLURM Job:  ${SLURM_JOB_ID:-local}"
echo ""

mkdir -p logs

# Activate environment (adjust path for your LRZ setup)
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -n "${CONDA_PREFIX:-}" ]; then
    echo "Using conda env: $CONDA_PREFIX"
fi

# Run training with GPU, AMP, gradient accumulation, and all improvements
python -m taxembed.cli.main train "$IDENTIFIER" \
    -as "$TAG" \
    --dim "$DIM" \
    --epochs 200 \
    --batch-size 128 \
    --n-negatives 100 \
    --lr 0.005 \
    --gpu 0 \
    --amp \
    --grad-accum-steps 4 \
    --curriculum \
    --curriculum-phases auto \
    --early-stopping 25 \
    --radial-nudge 0.05 \
    --radial-schedule log \
    --depth-scale-margin \
    --margin-min 0.05 \
    --margin-max 1.0 \
    --epoch-fraction 0.3

echo ""
echo "=== Training complete ==="
echo "  Artifacts: artifacts/tags/$TAG/"
