#!/bin/bash
#SBATCH --job-name=taxembed_p2_train
#SBATCH --partition=lrz-v100x2
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-8
#SBATCH --output=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/logs/p2_train_%A_%a.out
#SBATCH --error=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/logs/p2_train_%A_%a.err
#SBATCH --container-image=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/pytorch-2.4.0-cuda12.1.sqsh
#SBATCH --container-workdir=/app
#SBATCH --container-mounts=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/src:/app:rw,/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/data:/data:ro,/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/artifacts:/app/artifacts:rw

set -euo pipefail
export MKL_THREADING_LAYER=GNU

# =============================================================================
# Plan v1 Task 8 -- train the P2 held-out-link-prediction array: 9 tasks = 3 arms x seeds 0/1/2.
# One run per array task, so each gets its own walltime and they queue independently instead of
# chaining into one multi-day job that dies at the walltime boundary (same shape as Task 9's
# task9_lrz_recipe_contrast.sh).
#
# ARMS (USER-approved 2026-09-24, the third arm added after the brief was first written):
#     0-2  vis00_s{0,1,2}       Task 2 visibility=0%  split, REAL metazoa_33208_clean closure
#     3-5  vis50_s{0,1,2}       Task 2 visibility=50% split, REAL metazoa_33208_clean closure
#     6-8  randomdag_s{0,1,2}   scripts/build_p2_randomdag_split.py split, RANDOMISED closure
#          (src/taxembed/eval/randomdag.py::randomize_parents+closure_from_parent) -- depth
#          preserved exactly, fan-out NOT preserved (p2_amendment_1_20260924: mean chance-floor
#          ratio randomised/real measured 5.403 on mollusca_6447_clean). Holds out parent edges
#          from the RANDOMISED tree and will be SCORED against the RANDOMISED parents.
#
# ALL THREE ARMS train the SAME "canonical" recipe that Task 9 showed learning throughout
# (S_angle 0.892->0.933 while loss rose 3.90->4.02): batch 256 x grad-accum 8 (effective 2048),
# 300 negatives, lr 0.001 with cosine_warmrestart + warm-restart-on-phase, softmax loss,
# euclidean parametrization. P2 is not a recipe contrast -- one recipe, three data conditions.
#
# INPUTS. Task 2's split .npz files (vis00/vis50) and scripts/build_p2_randomdag_split.py's
# output (randomdag) must be uploaded to /data/p2_splits on LRZ BEFORE this array is submitted,
# alongside the metazoa mapping TSV; the P2 eval modules (src/taxembed/eval/{p2_split,randomdag,
# linkpred,baselines,preregistration}.py) and scripts/{build_p2_split,build_p2_randomdag_split,
# score_p2_linkpred}.py must be synced into the mounted src/ tree. Submit p2_lrz_train_smoke.sh
# first and READ its log (Rule 16) -- do not submit this array on smoke exit 0 alone.
#
# Pre-registration: results/p2_heldout_preregistration.json, including its
# p2_amendment_1_20260924 block. READING RULE: cross-tree comparisons (an arm vs RandomDAG) use
# normalized_rank -- chance is 0.5 for ANY pool size, unlike raw MRR which RandomDAG's collapsed
# fan-out makes an easier task on. Within-tree comparisons (vis00 vs vis50; an arm vs its own
# baselines.sibling_chance_mean) use MRR. Do not read scripts/score_p2_linkpred.py output by hand
# off this array's checkpoints -- go through src/taxembed/eval/preregistration.py::p2_verdict.
# =============================================================================

MAP=/data/taxonomy_edges_metazoa_33208_clean.mapping.tsv
SPLITDIR=/data/p2_splits
EPOCHS="${P2_EPOCHS:-200}"

# Array index -> (arm, seed), per the Task 8 brief's amendment (9 elements, not 6).
IDX="${SLURM_ARRAY_TASK_ID}"
if [ "${IDX}" -lt 3 ]; then
    ARM=vis00
    SEED="${IDX}"
elif [ "${IDX}" -lt 6 ]; then
    ARM=vis50
    SEED="$((IDX-3))"
else
    ARM=randomdag
    SEED="$((IDX-6))"
fi
TAG="p2_${ARM}_s${SEED}"

case "${ARM}" in
    vis00)
        TRAIN="${SPLITDIR}/p2_metazoa_33208_clean_vis00_seed${SEED}_train.npz"
        ;;
    vis50)
        TRAIN="${SPLITDIR}/p2_metazoa_33208_clean_vis50_seed${SEED}_train.npz"
        ;;
    randomdag)
        TRAIN="${SPLITDIR}/p2_metazoa_33208_clean_randomdag_vis00_seed${SEED}_train.npz"
        ;;
esac

echo "=== TaxEmbed P2 Task 8 -- ${ARM}, seed ${SEED}, metazoa ==="
echo "Job: ${SLURM_ARRAY_JOB_ID}, array task ${IDX}, Host: $(hostname)"
nvidia-smi -L || true          # never kill the job on this (Rule 16 SIGPIPE incident)
python --version
echo

cd /app
pip install --no-cache-dir -e . 2>&1 | tail -3
echo

for f in "${TRAIN}" "${MAP}"; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: missing ${f}" >&2
        exit 1
    fi
done

python -m taxembed.cli.main train \
    --file "${TRAIN}" \
    --mapping "${MAP}" \
    --dim 100 \
    --gpu 0 \
    --amp \
    --curriculum \
    --curriculum-phases auto \
    --early-stopping 999 \
    --radial-nudge 0.05 \
    --radial-schedule log \
    --depth-scale-margin \
    --margin-min 0.05 \
    --margin-max 1.0 \
    --epoch-fraction 0.3 \
    --euclidean-param \
    --loss softmax \
    --save-every 10 \
    --epochs "${EPOCHS}" \
    --seed "${SEED}" \
    --batch-size 256 \
    --n-negatives 300 \
    --lr 0.001 \
    --grad-accum-steps 8 \
    --lr-schedule cosine_warmrestart \
    --warm-restart-on-phase \
    --lr-min-multiplier 0.01 \
    -as "${TAG}"

echo
echo "=== ${ARM} seed ${SEED} complete ==="
ls -la "/app/artifacts/tags/${TAG}/"
