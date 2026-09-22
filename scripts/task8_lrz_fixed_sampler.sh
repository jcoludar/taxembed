#!/bin/bash
#SBATCH --job-name=taxembed_task8
#SBATCH --partition=lrz-v100x2
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2
#SBATCH --output=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/logs/task8_%A_%a.out
#SBATCH --error=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/logs/task8_%A_%a.err
#SBATCH --container-image=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/pytorch-2.4.0-cuda12.1.sqsh
#SBATCH --container-workdir=/app
#SBATCH --container-mounts=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/src_task8:/app:rw,/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/data:/data:ro,/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/artifacts:/app/artifacts:rw

set -euo pipefail
export MKL_THREADING_LAYER=GNU

# =============================================================================
# Plan v2 Task 8 -- the FIXED-sampler arm: canonical recipe + ancestry-aware negatives.
# Pre-registration: results/objective_integrity_delta_preregistration.json
#
# ARRAY 0-2 = seeds 0/1/2. The UNFIXED arm is NOT run here: it is Task 9's canonical
# seeds (array 5802007 tasks 0-2), with identical flags minus the two below.
#
# ⚠ Mounts src_task8, NOT src: array 5802007 (Task 9, shipped sampler) mounts src and
#   must keep running the code it was smoke-tested with. src_task8 = copy of src +
#   the Task 6 files (submodule commit 3e871e3).
# --save-every 20 (not 10): ~6.8 GB per run instead of ~10.8 GB on a shared container.
# =============================================================================

DATA=/data/taxonomy_edges_metazoa_33208_clean_transitive.npz
MAP=/data/taxonomy_edges_metazoa_33208_clean.mapping.tsv
EPOCHS="${TASK8_EPOCHS:-200}"
SEED="${SLURM_ARRAY_TASK_ID}"

echo "=== TaxEmbed Task 8 -- fixed sampler, canonical, seed ${SEED}, metazoa ==="
echo "Job: ${SLURM_ARRAY_JOB_ID}, array task ${SEED}, Host: $(hostname)"
nvidia-smi -L || true
python --version
cd /app
pip install --no-cache-dir -e . 2>&1 | tail -3

for f in "${DATA}" "${MAP}"; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: missing ${f}" >&2
        exit 1
    fi
done

python -m taxembed.cli.main train \
    --file "${DATA}" --mapping "${MAP}" \
    --dim 100 --gpu 0 --amp \
    --curriculum --curriculum-phases auto --early-stopping 999 \
    --radial-nudge 0.05 --radial-schedule log \
    --depth-scale-margin --margin-min 0.05 --margin-max 1.0 \
    --epoch-fraction 0.3 --euclidean-param --loss softmax \
    --save-every 20 --epochs "${EPOCHS}" --seed "${SEED}" \
    --batch-size 256 --n-negatives 300 --lr 0.001 --grad-accum-steps 8 \
    --lr-schedule cosine_warmrestart --warm-restart-on-phase --lr-min-multiplier 0.01 \
    --exclude-descendant-negatives --drop-root-anchored \
    -as "task8_fixed_canonical_s${SEED}"

echo "=== fixed canonical seed ${SEED} complete ==="
ls -la "/app/artifacts/tags/task8_fixed_canonical_s${SEED}/"
