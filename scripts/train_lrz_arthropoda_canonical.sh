#!/bin/bash
#SBATCH --job-name=taxembed_arthropoda_canonical
#SBATCH --partition=lrz-v100x2
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/logs/arthropoda_canonical_%j.out
#SBATCH --error=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/logs/arthropoda_canonical_%j.err
#SBATCH --container-image=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/pytorch-2.4.0-cuda12.1.sqsh
#SBATCH --container-workdir=/app
#SBATCH --container-mounts=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/src:/app:rw,/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/data:/data:ro,/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/artifacts:/app/artifacts:rw

set -euo pipefail

# MKL/libgomp threading-layer clash fix (load-bearing — see PROJECT_STATE.md).
export MKL_THREADING_LAYER=GNU

# =============================================================================
# GENERALIZATION run — the canonical metazoa recipe applied to ARTHROPODA
# (324,983 nodes; ~metazoa-scale, so LRZ-only; mollusca at 32k is the local proof).
# IDENTICAL recipe to the locked metazoa Exp1, only --file/--mapping/-as differ.
#
# ⚠ PREREQUISITE — the arthropoda_6656_clean dataset must be present at /data on LRZ:
#     /data/taxonomy_edges_arthropoda_6656_clean_transitive.npz
#     /data/taxonomy_edges_arthropoda_6656_clean.mapping.tsv
#   It IS local at data/taxopy/arthropoda_6656_clean/ but may not be uploaded yet.
#   If missing, scp it up first (see the next-session handoff), OR build it on the
#   login node with:  taxembed build Arthropoda --clean
# =============================================================================

echo "=== TaxEmbed ARTHROPODA generalization (canonical recipe) ==="
echo "Job: ${SLURM_JOB_ID}, Host: $(hostname)"
nvidia-smi -L
python --version
echo

cd /app
pip install --no-cache-dir -e . 2>&1 | tail -3
echo

if [ ! -f /data/taxonomy_edges_arthropoda_6656_clean_transitive.npz ]; then
    echo "ERROR: arthropoda dataset not found at /data — upload or build it first (see header)." >&2
    exit 1
fi

echo "=== train ==="
python -m taxembed.cli.main train \
    --file /data/taxonomy_edges_arthropoda_6656_clean_transitive.npz \
    --mapping /data/taxonomy_edges_arthropoda_6656_clean.mapping.tsv \
    -as arthropoda_canonical \
    --dim 100 \
    --epochs 200 \
    --batch-size 256 \
    --n-negatives 300 \
    --lr 0.001 \
    --gpu 0 \
    --amp \
    --grad-accum-steps 8 \
    --curriculum \
    --curriculum-phases auto \
    --lr-schedule cosine_warmrestart \
    --warm-restart-on-phase \
    --lr-min-multiplier 0.01 \
    --early-stopping 999 \
    --radial-nudge 0.05 \
    --radial-schedule log \
    --depth-scale-margin \
    --margin-min 0.05 \
    --margin-max 1.0 \
    --epoch-fraction 0.3 \
    --euclidean-param \
    --loss softmax \
    --save-every 10

echo
echo "=== training complete ==="
ls -la /app/artifacts/tags/arthropoda_canonical/
