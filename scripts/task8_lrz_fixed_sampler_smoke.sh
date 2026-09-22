#!/bin/bash
#SBATCH --job-name=taxembed_task8_smoke
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/logs/task8_smoke_%j.out
#SBATCH --error=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/logs/task8_smoke_%j.err
#SBATCH --container-image=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/pytorch-2.4.0-cuda12.1.sqsh
#SBATCH --container-workdir=/app
#SBATCH --container-mounts=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/src_task8:/app:rw,/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/data:/data:ro,/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/artifacts:/app/artifacts:rw

set -euo pipefail
export MKL_THREADING_LAYER=GNU

# =============================================================================
# Rule 16 CANARY for task8_lrz_fixed_sampler.sh, run on CPU (the GPU queue is ~1 day;
# the CPU queue is minutes). The new code is the negative sampler, which is numpy on
# the CPU in BOTH cases, so a CPU run exercises exactly what changed. The CUDA/AMP
# path is unchanged and was proven by the Task 9 smoke 5743270.
# Proves on the real metazoa closure, in the src_task8 tree:
#   - py_compile clean; every flag the full job passes is a real CLI option
#   - the ancestry index builds at 498k nodes; root pairs are excluded
#   - 2 epochs run through two curriculum phases without IndexError
#   - the per-epoch sampler line prints (fix active) and the run reports its seed
#   - run.json records both new flags
# =============================================================================

echo "=== TaxEmbed Task 8 SMOKE (CPU) ==="
echo "Job: ${SLURM_JOB_ID}, Host: $(hostname)"
cd /app
pip install --no-cache-dir -e . 2>&1 | tail -3

DATA=/data/taxonomy_edges_metazoa_33208_clean_transitive.npz
MAP=/data/taxonomy_edges_metazoa_33208_clean.mapping.tsv
python -m py_compile /app/train_small.py /app/train_hierarchical.py /app/src/taxembed/cli/main.py /app/src/taxembed/eval/subtree.py
echo "py_compile clean"

python -m taxembed.cli.main train --help > /tmp/train_help.txt
for flag in --exclude-descendant-negatives --drop-root-anchored --seed --grad-accum-steps \
            --lr-schedule --warm-restart-on-phase --lr-min-multiplier --epoch-fraction \
            --depth-scale-margin --radial-schedule --euclidean-param --save-every --amp; do
    if ! grep -q -- "${flag}" /tmp/train_help.txt; then
        echo "SMOKE FAILED: ${flag} is not a real option on the CLI" >&2
        exit 1
    fi
done
echo "all flags real"

python -m taxembed.cli.main train \
    --file "${DATA}" --mapping "${MAP}" \
    --dim 100 --gpu -1 \
    --curriculum --curriculum-phases auto --early-stopping 999 \
    --radial-nudge 0.05 --radial-schedule log \
    --depth-scale-margin --margin-min 0.05 --margin-max 1.0 \
    --epoch-fraction 0.1 --euclidean-param --loss softmax \
    --epochs 2 --seed 0 \
    --batch-size 256 --n-negatives 300 --lr 0.001 --grad-accum-steps 8 \
    --lr-schedule cosine_warmrestart --warm-restart-on-phase --lr-min-multiplier 0.01 \
    --exclude-descendant-negatives --drop-root-anchored \
    -as smoke_task8_fixed 2>&1 | tee /tmp/smoke_task8.log

for pat in "seeded: 0" "excluding" "sampler: observed FN rate"; do
    if ! grep -q "${pat}" /tmp/smoke_task8.log; then
        echo "SMOKE FAILED: log lacks '${pat}'" >&2
        exit 1
    fi
done
for key in '"exclude_descendant_negatives": true' '"drop_root_anchored": true'; do
    if ! grep -q "${key}" /app/artifacts/tags/smoke_task8_fixed/run.json; then
        echo "SMOKE FAILED: run.json lacks ${key}" >&2
        exit 1
    fi
done
grep "sampler:" /tmp/smoke_task8.log
echo "=== SMOKE PASSED -- safe to submit task8_lrz_fixed_sampler.sh ==="
