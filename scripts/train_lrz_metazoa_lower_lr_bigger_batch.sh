#!/bin/bash
#SBATCH --job-name=taxembed_metazoa_lower_lr_bigger_batch
#SBATCH --partition=lrz-v100x2
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/logs/metazoa_lower_lr_bigger_batch_%j.out
#SBATCH --error=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/logs/metazoa_lower_lr_bigger_batch_%j.err
#SBATCH --container-image=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/pytorch-2.4.0-cuda12.1.sqsh
#SBATCH --container-workdir=/app
#SBATCH --container-mounts=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/src:/app:rw,/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/data:/data:ro,/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/artifacts:/app/artifacts:rw

set -euo pipefail

# MKL/libgomp threading-layer clash fix (load-bearing — see PROJECT_STATE.md).
export MKL_THREADING_LAYER=GNU

echo "=== TaxEmbed Experiment 1 — dd<=18 survival ==="
echo "Job: ${SLURM_JOB_ID}, Host: $(hostname)"
nvidia-smi -L
python --version
echo

cd /app
pip install --no-cache-dir -e . 2>&1 | tail -3
echo

# Experiment 1: targets the dd<=18 transition (2026-06-02 diagnostic verdict).
# Job C (metazoa_softmax_milestones) sustained +0.875 / 1.14-1.20x for 20 epochs
# inside dd<=9 (ep 50-70), then crashed to +0.796 the moment dd<=18 loaded at ep 80.
#
# This run keeps Job C's recipe (softmax + euclidean-param + curriculum auto +
# radial-nudge 0.05 + log radial schedule + depth-scale margin + epoch_fraction 0.3
# + save-every 10) and changes:
#   batch-size       128   -> 256
#   grad-accum-steps 4     -> 8
#   n-negatives      100   -> 300
#   lr               0.005 -> 0.001
#   + lr-schedule cosine_warmrestart --warm-restart-on-phase
#   (cosine decays each phase to 1% of base, then restarts at ep 40/80/120 phase
#    boundaries — base_lr returns each time the curriculum loads a harder dd window)
echo "=== train ==="
python -m taxembed.cli.main train \
    --file /data/taxonomy_edges_metazoa_33208_clean_transitive.npz \
    --mapping /data/taxonomy_edges_metazoa_33208_clean.mapping.tsv \
    -as metazoa_lower_lr_bigger_batch \
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
ls -la /app/artifacts/tags/metazoa_lower_lr_bigger_batch/
