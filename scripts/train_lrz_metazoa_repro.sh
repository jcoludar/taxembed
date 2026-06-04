#!/bin/bash
#SBATCH --job-name=taxembed_metazoa_repro
#SBATCH --partition=lrz-v100x2
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/logs/metazoa_repro_%j.out
#SBATCH --error=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/logs/metazoa_repro_%j.err
#SBATCH --container-image=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/pytorch-2.4.0-cuda12.1.sqsh
#SBATCH --container-workdir=/app
#SBATCH --container-mounts=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/src:/app:rw,/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/data:/data:ro,/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/artifacts:/app/artifacts:rw

set -euo pipefail

# MKL/libgomp threading-layer clash fix (load-bearing — see PROJECT_STATE.md).
export MKL_THREADING_LAYER=GNU

# =============================================================================
# REPRODUCIBILITY RE-RUN of Experiment 1 (job 5664609, tag
# metazoa_lower_lr_bigger_batch — the run that SOLVED metazoa scale).
#
# Purpose: confirm the EXCELLENT result is NOT a one-off. IDENTICAL config to
# Exp1; ONLY the tag changes → writes to artifacts/tags/metazoa_lower_lr_bigger_batch_repro/
# so the locked original is never overwritten. `--seed` is not wired into training
# yet, so this is a fresh-init re-run: success = the seeded analyzer lands the same
# ballpark (phylum ~2.6 / class ~3.8 / order ~6.7 / family ~10× ± a bit), NOT bit-identical.
# =============================================================================

echo "=== TaxEmbed metazoa REPRODUCIBILITY re-run (Exp1 config, new tag) ==="
echo "Job: ${SLURM_JOB_ID}, Host: $(hostname)"
nvidia-smi -L
python --version
echo

cd /app
pip install --no-cache-dir -e . 2>&1 | tail -3
echo

echo "=== train ==="
python -m taxembed.cli.main train \
    --file /data/taxonomy_edges_metazoa_33208_clean_transitive.npz \
    --mapping /data/taxonomy_edges_metazoa_33208_clean.mapping.tsv \
    -as metazoa_lower_lr_bigger_batch_repro \
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
ls -la /app/artifacts/tags/metazoa_lower_lr_bigger_batch_repro/
