#!/bin/bash
#SBATCH --job-name=taxembed_task8_score_seeds
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/logs/task8_score_seeds_%j.out
#SBATCH --error=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/logs/task8_score_seeds_%j.err
#SBATCH --container-image=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/pytorch-2.4.0-cuda12.1.sqsh
#SBATCH --container-workdir=/app
#SBATCH --container-mounts=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/src_task8:/app:rw,/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/data:/data:ro,/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/artifacts:/app/artifacts:rw

set -euo pipefail
export MKL_THREADING_LAYER=GNU

# =============================================================================
# Plan v2 Task 8 -- does the ancestry-aware sampler change what the recipe learns?
# Pre-registration: results/objective_integrity_delta_preregistration.json
#   (MATERIAL / ROBUST / MIXED; validity gate shared with Task 9's v2.)
#
#   fixed    tags task8_fixed_canonical_s{0,1,2}  (array 5802297, --save-every 20)
#   unfixed  tags task9_canonical_s{0,1,2}        (array 5802007, --save-every 10)
#
# THE UNFIXED ARM IS SCORED AGAIN HERE, not read across from the Task 9 job.
# Two reasons, and the second is the important one:
#   1. --pair computes its paired cluster bootstrap from per-query arrays held in
#      ONE process, so both arms must be scored in the same invocation.
#   2. It buys a check that COULD have failed. The query set and clade level are
#      deterministic in (tree, seed=0), so task9_canonical_s* must score
#      bit-identically here and in task9_lrz_score_seeds.sh. If the two jobs
#      disagree, something is wrong with a tree, a mount, or a checkpoint -- and
#      that is worth knowing BEFORE either verdict is written.
#
# Arms have different milestone cadences (20 vs 10 epochs). That is declared in
# the pre-registration; the fixed arm's milestones are a subset of the unfixed
# arm's, and the per-run VALUE is the rolling 196-200 mean in both arms, which is
# identical in cadence. Do not compare milestone counts.
#
# TASK8_SCORE_SMOKE=1 -> Rule 16 canary: one seed per arm, one checkpoint each.
# =============================================================================

NPZ=/data/taxonomy_edges_metazoa_33208_clean_transitive.npz
TAGS=/app/artifacts/tags
OUTDIR=/app/artifacts/task8_scoring
STAMP="$(date +%Y%m%d_%H%M%S)"
SMOKE="${TASK8_SCORE_SMOKE:-}"

if [ -n "${SMOKE}" ]; then
    MODE=smoke
    OUT="${OUTDIR}/task8_seeds_SMOKE_${STAMP}.json"
    SEEDS=(0)
    EXTRA=(--max-checkpoints 1 --pair fixed_s0_roll:unfixed_s0_roll)
else
    MODE=full
    OUT="${OUTDIR}/task8_seeds_${STAMP}.json"
    SEEDS=(0 1 2)
    EXTRA=(--pair fixed_s0_roll:unfixed_s0_roll
           --pair fixed_s1_roll:unfixed_s1_roll
           --pair fixed_s2_roll:unfixed_s2_roll)
fi

echo "=== TaxEmbed Task 8 scoring -- fixed vs unfixed sampler ==="
echo "Job: ${SLURM_JOB_ID:-?}, Host: $(hostname), mode: ${MODE}"
python --version
cd /app

python -m py_compile /app/scripts/score_recipe_checkpoints.py /app/src/taxembed/eval/angular.py
python -c 'import sys; sys.path.insert(0, "/app/src"); from taxembed.eval.angular import level_auc; print("level_auc present:", level_auc.__module__)'

pip install --no-cache-dir -e . 2>&1 | tail -3

RUNFLAGS=()
for arm in fixed unfixed; do
    for s in "${SEEDS[@]}"; do
        if [ "${arm}" = "fixed" ]; then
            tag="task8_fixed_canonical_s${s}"
        else
            tag="task9_canonical_s${s}"
        fi
        dir="${TAGS}/${tag}"
        for f in "${dir}/run.json" "${dir}/${tag}_milestone_epoch200.pth" "${dir}/${tag}_epoch200.pth"; do
            if [ ! -f "${f}" ]; then
                echo "ERROR: missing ${f}" >&2
                exit 1
            fi
        done
        RUNFLAGS+=(--run "${arm}_s${s}_ms=${dir}/${tag}_milestone_epoch*.pth")
        RUNFLAGS+=(--run "${arm}_s${s}_roll=${dir}/${tag}_epoch*.pth")
    done
done
if [ ! -f "${NPZ}" ]; then
    echo "ERROR: missing ${NPZ}" >&2
    exit 1
fi
echo "inputs present, py_compile clean, ${#RUNFLAGS[@]} run flags"

python /app/scripts/score_recipe_checkpoints.py \
    --npz "${NPZ}" \
    --out "${OUT}" \
    "${RUNFLAGS[@]}" \
    --n-queries 10000 --k 10 --seed 0 \
    "${EXTRA[@]}"

echo "=== scoring complete: ${OUT} ==="
ls -la "${OUT}" "${OUT%.json}.npz"
md5sum "${OUT}" "${OUT%.json}.npz"
