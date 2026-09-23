#!/bin/bash
#SBATCH --job-name=taxembed_task9_score_seeds
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/logs/task9_score_seeds_%j.out
#SBATCH --error=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/logs/task9_score_seeds_%j.err
#SBATCH --container-image=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/pytorch-2.4.0-cuda12.1.sqsh
#SBATCH --container-workdir=/app
#SBATCH --container-mounts=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/src_task8:/app:rw,/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/data:/data:ro,/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/artifacts:/app/artifacts:rw

set -euo pipefail
export MKL_THREADING_LAYER=GNU

# =============================================================================
# Plan v2 Task 9 -- score the SEEDED array (job 5802007) on the radius-free metric.
#
# THIS is the job whose output the pre-registration is read on. The Figure 4 pair
# (fig4_runs_*.json, n=1 per arm, unseeded) is REPORTED, never read:
#   preregistration_v2_20260922.who_is_read = "ONLY the seeded array, job 5802007".
#
#   6 runs = 2 arms x seeds 0/1/2, metazoa (498,246 nodes), 200 epochs.
#   artifacts/tags/task9_{canonical,prior}_s{0,1,2}/
#
# Per run, TWO arms are registered with the scorer, because the pre-registration
# uses them for different things:
#   *_ms   20 milestone checkpoints (ep 10..200) -> validity gate (a), and the
#          trajectory that re-plots Figure 4 on a learned metric.
#   *_roll rolling checkpoints ep 196-200        -> THE per-run value
#          ("mean of S_angle over the rolling epoch 196-200 checkpoints,
#            within-run jitter reported as their SD").
#
# MOUNTS src_task8, NOT src. Only src_task8's scorer has angular.level_auc, which
# amendment 1 makes a required condition. src stays frozen while 5802007 runs.
#
# Cost is dominated by 150 checkpoints x (S_angle + S_poincare + level AUC) at
# 498k nodes. Walltime is deliberately 20 h on a 14-day-max CPU partition: a
# scoring job that dies at its walltime boundary wastes the array it was scoring.
#
# TASK9_SCORE_SMOKE=1 -> Rule 16 canary: the two runs that were COMPLETED first,
# one checkpoint each, plus the pairing code path. It also MEASURES seconds per
# checkpoint, which is the number that sets this job's walltime.
# =============================================================================

R=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz
NPZ=/data/taxonomy_edges_metazoa_33208_clean_transitive.npz
TAGS=/app/artifacts/tags
OUTDIR=/app/artifacts/task9_scoring
STAMP="$(date +%Y%m%d_%H%M%S)"
SMOKE="${TASK9_SCORE_SMOKE:-}"

if [ -n "${SMOKE}" ]; then
    MODE=smoke
    OUT="${OUTDIR}/seeds_SMOKE_${STAMP}.json"
    ARMS=(canonical)
    SEEDS=(0 1)
    EXTRA=(--max-checkpoints 1 --pair canonical_s0_roll:canonical_s1_roll)
else
    MODE=full
    OUT="${OUTDIR}/seeds_${STAMP}.json"
    ARMS=(canonical prior)
    SEEDS=(0 1 2)
    # Seed-matched pairs: the paired cluster bootstrap is supporting evidence for
    # the all-above-all reading rule, not a substitute for it.
    EXTRA=(--pair canonical_s0_roll:prior_s0_roll
           --pair canonical_s1_roll:prior_s1_roll
           --pair canonical_s2_roll:prior_s2_roll)
fi

echo "=== TaxEmbed Task 9 scoring -- SEEDED array 5802007 ==="
echo "Job: ${SLURM_JOB_ID:-?}, Host: $(hostname), mode: ${MODE}"
python --version
cd /app

# Rule 16: prove the tree we are mounted on is the one with level_auc, BEFORE the
# container spends an hour discovering it is not.
python -m py_compile /app/scripts/score_recipe_checkpoints.py /app/src/taxembed/eval/angular.py
python -c 'import sys; sys.path.insert(0, "/app/src"); from taxembed.eval.angular import level_auc; print("level_auc present:", level_auc.__module__)'

pip install --no-cache-dir -e . 2>&1 | tail -3

RUNFLAGS=()
for arm in "${ARMS[@]}"; do
    for s in "${SEEDS[@]}"; do
        tag="task9_${arm}_s${s}"
        dir="${TAGS}/${tag}"
        # Fail fast and by name. A missing ep200 means that array element did not
        # finish, and scoring a truncated run is worse than not scoring it.
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
