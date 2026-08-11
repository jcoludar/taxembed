#!/bin/bash
#SBATCH --job-name=taxembed_task9_smoke
#SBATCH --partition=lrz-v100x2
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/logs/task9_smoke_%j.out
#SBATCH --error=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/logs/task9_smoke_%j.err
#SBATCH --container-image=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/pytorch-2.4.0-cuda12.1.sqsh
#SBATCH --container-workdir=/app
#SBATCH --container-mounts=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/src:/app:rw,/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/data:/data:ro,/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/artifacts:/app/artifacts:rw

set -euo pipefail
export MKL_THREADING_LAYER=GNU

# =============================================================================
# Rule 16 CANARY for task9_lrz_recipe_contrast.sh. Short walltime, 2 epochs, one
# seed, both arms. Proves, before the real job consumes a queue slot:
#   - the container starts and the editable install succeeds
#   - the metazoa dataset is actually present at /data
#   - EVERY flag both arms pass is a real argparse option (the failure mode that
#     would otherwise surface after hours of queue wait)
#   - --seed is honoured end to end: the run must PRINT its seeded line, which is
#     the difference between the flag being accepted and the flag being USED
#   - run.json records the seed
#
# Run this, confirm SMOKE PASSED, and only then submit the full job.
# =============================================================================

echo "=== TaxEmbed Task 9 SMOKE ==="
echo "Job: ${SLURM_JOB_ID}, Host: $(hostname)"
nvidia-smi -L || true
python --version
echo

cd /app
pip install --no-cache-dir -e . 2>&1 | tail -3
echo

DATA=/data/taxonomy_edges_metazoa_33208_clean_transitive.npz
MAP=/data/taxonomy_edges_metazoa_33208_clean.mapping.tsv

for f in "${DATA}" "${MAP}"; do
    if [ ! -f "${f}" ]; then
        echo "SMOKE FAILED: missing ${f}" >&2
        exit 1
    fi
done
echo "inputs present"

# Static check inside the container, against the interpreter that will run it.
python -m py_compile /app/train_small.py /app/train_hierarchical.py /app/src/taxembed/cli/main.py
echo "py_compile clean"

# Every flag the real job passes must exist. --seed is the new one and the one
# whose absence would waste the whole queue slot.
python -m taxembed.cli.main train --help > /tmp/train_help.txt
for flag in --seed --grad-accum-steps --lr-schedule --warm-restart-on-phase \
            --lr-min-multiplier --epoch-fraction --depth-scale-margin \
            --radial-schedule --euclidean-param --save-every --amp; do
    if ! grep -q -- "${flag}" /tmp/train_help.txt; then
        echo "SMOKE FAILED: ${flag} is not a real option on the CLI" >&2
        exit 1
    fi
done
echo "all flags real"
echo

SHARED=(
    --file "${DATA}" --mapping "${MAP}"
    --dim 100 --gpu 0 --amp
    --curriculum --curriculum-phases auto --early-stopping 999
    --radial-nudge 0.05 --radial-schedule log
    --depth-scale-margin --margin-min 0.05 --margin-max 1.0
    --epoch-fraction 0.3 --euclidean-param --loss softmax
    --epochs 2 --seed 0
)

echo "=== canonical (2 epochs) ==="
python -m taxembed.cli.main train "${SHARED[@]}" \
    -as smoke_task9_canonical \
    --batch-size 256 --n-negatives 300 --lr 0.001 --grad-accum-steps 8 \
    --lr-schedule cosine_warmrestart --warm-restart-on-phase --lr-min-multiplier 0.01 \
    2>&1 | tee /tmp/smoke_canonical.log

echo "=== prior (2 epochs) ==="
python -m taxembed.cli.main train "${SHARED[@]}" \
    -as smoke_task9_prior \
    --batch-size 128 --n-negatives 100 --lr 0.005 --grad-accum-steps 4 \
    2>&1 | tee /tmp/smoke_prior.log

# A seed that is accepted but ignored looks identical in --help. Demand the proof.
for log in /tmp/smoke_canonical.log /tmp/smoke_prior.log; do
    if ! grep -q "seeded: 0" "${log}"; then
        echo "SMOKE FAILED: ${log} never reported being seeded -- flag accepted but not used" >&2
        exit 1
    fi
done
echo "both arms reported being seeded"

for tag in smoke_task9_canonical smoke_task9_prior; do
    if ! grep -q '"seed"' "/app/artifacts/tags/${tag}/run.json"; then
        echo "SMOKE FAILED: ${tag}/run.json does not record the seed" >&2
        exit 1
    fi
done
echo "run.json records the seed in both arms"

echo
echo "=== SMOKE PASSED -- safe to submit task9_lrz_recipe_contrast.sh ==="
