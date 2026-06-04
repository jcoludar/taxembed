#!/bin/bash
#SBATCH --job-name=taxembed_eukaryota_canonical
#SBATCH --partition=lrz-v100x2
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/logs/eukaryota_canonical_%j.out
#SBATCH --error=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/logs/eukaryota_canonical_%j.err
#SBATCH --container-image=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/pytorch-2.4.0-cuda12.1.sqsh
#SBATCH --container-workdir=/app
#SBATCH --container-mounts=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/src:/app:rw,/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/data:/data:ro,/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/artifacts:/app/artifacts:rw

set -euo pipefail

# MKL/libgomp threading-layer clash fix (load-bearing — see PROJECT_STATE.md).
export MKL_THREADING_LAYER=GNU

# =============================================================================
# EUKARYOTA (taxid 2759) — the all-Life-scale run. ~930k-1M clean nodes (~2x metazoa),
# max depth ~39. Recipe = the LOCKED metazoa canonical recipe; Eukaryota is BIGGER than
# metazoa so the big-batch regime is appropriate (more data -> more grad-steps/epoch; the
# mollusca "shrink eff-batch" fix is only for SMALL clades). Build the dataset LOCALLY with
# `taxembed build 2759 --clean`, VERIFY it (no sp./environmental junk; kingdom balance), then
# scp the .npz + mapping to /data before submitting.
#
# ⚠ WATCH (the one real risk): within-clade hard-negative starvation worsens with scale+depth
# (within-grandparent neg fraction: echino 0.13 -> metazoa 0.011 -> eukaryota lower still). It was
# NOT binding at metazoa, but if per-rank separation STALLS around ep80-120 (read the milestones),
# that's the signal to enable the hard-negative sampler (design + diagnostic already on the shelf),
# OR bump --n-negatives 300 -> 500. Don't let a stall masquerade as "the architecture can't scale".
# =============================================================================

echo "=== TaxEmbed EUKARYOTA (canonical recipe, all-Life scale) ==="
echo "Job: ${SLURM_JOB_ID}, Host: $(hostname)"
nvidia-smi -L
python --version
echo

cd /app
pip install --no-cache-dir -e . 2>&1 | tail -3
echo

if [ ! -f /data/taxonomy_edges_eukaryota_2759_clean_transitive.npz ]; then
    echo "ERROR: eukaryota dataset not found at /data — build locally + scp it up first (see header)." >&2
    exit 1
fi

echo "=== train ==="
python -m taxembed.cli.main train \
    --file /data/taxonomy_edges_eukaryota_2759_clean_transitive.npz \
    --mapping /data/taxonomy_edges_eukaryota_2759_clean.mapping.tsv \
    -as eukaryota_canonical \
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
ls -la /app/artifacts/tags/eukaryota_canonical/
