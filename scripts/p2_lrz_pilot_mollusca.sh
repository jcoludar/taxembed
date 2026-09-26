#!/bin/bash
#SBATCH --job-name=taxembed_p2_pilot_moll
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/logs/p2_pilot_moll_%j.out
#SBATCH --error=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/logs/p2_pilot_moll_%j.err
#SBATCH --container-image=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/pytorch-2.4.0-cuda12.1.sqsh
#SBATCH --container-workdir=/app
#SBATCH --container-mounts=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/src:/app:rw,/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/data:/data:ro,/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/artifacts:/app/artifacts:rw

set -euo pipefail
export MKL_THREADING_LAYER=GNU

# =============================================================================
# THE MOLLUSCA PLUMBING PILOT -- the whole P2 chain, end to end, on CPU, at 1/15th the scale.
#
# 🛑 READ ITS VERDICT FOR PLUMBING ONLY, NEVER FOR SCIENCE. mollusca_6447_clean has 32,017 nodes
# and ~740 held-out nodes per seed, and it populates ONLY the 11-15 depth stratum (measured:
# the 16-21 and 22-28 strata come back n=0). Its verdict is therefore a statement about whether
# the PIPES CONNECT, not about whether the geometry generalises. Nothing here may be quoted.
#
# WHY IT EXISTS. Until 2026-09-26 no scorer JSON existed anywhere in this project and the
# merge -> p2_verdict path had NEVER run on real scorer output -- every test of it used
# hand-authored fixtures written by the same person who wrote the engine. The class of defect
# that costs a GPU array is exactly the one fixtures cannot see: a key the scorer writes and the
# engine does not look for (C4), a control silently resolving another tree's baselines (C-B), a
# roll glob picking up more checkpoints than the window (C-C). All three were found by reading,
# and all three would ALSO have been caught here, after minutes of CPU instead of ~41 GPU-hours.
#
# It needs its own script because scripts/p2_lrz_train.sh hardcodes the metazoa mapping and
# split filenames (only EPOCHS is env-overridable).
#
# WHAT IT ASSERTS, in order:
#   1. 12 arms x 3 seeds train and write BOTH a rolling and a milestone checkpoint set
#   2. the scorer runs on all 12, writing 9 JSONs with SEED-TAGGED baselines keys
#   3. merge_p2_scorer_outputs accepts them -- exercising assert_baselines_agree_across_seeds
#      AND assert_control_baselines_exist on real output
#   4. p2_verdict returns under the FULL production flag set, amendment_6 included
#   5. the reading reports per-stratum prior basis, the raw head-to-head, and a roll window of
#      exactly P2_ROLL_WINDOW
# =============================================================================

CLADE=mollusca_6447_clean
MOLL_DIR="/data/taxopy/${CLADE}"
MAP="${MOLL_DIR}/taxonomy_edges_${CLADE}.mapping.tsv"
CLOSURE="${MOLL_DIR}/taxonomy_edges_${CLADE}_transitive.npz"
SPLITDIR=/data/p2_splits
TAGS=/app/artifacts/tags
OUTDIR=/app/artifacts/p2_pilot_mollusca
STAMP="$(date +%Y%m%d_%H%M%S)"
SEEDS=(0 1 2)

# The pilot trains SHORT. P2_FINAL_EPOCH is 200 in the engine and gate (c) requires a milestone
# there, so a short pilot CANNOT satisfy gate (c) and its verdict will read UNINFORMATIVE. That
# is expected and is not a failure of the pilot: UNINFORMATIVE still exercises the merge, the
# baselines asserts, the control resolution and the gate machinery -- everything except the
# direction-reading branch. PILOT_FULL=1 trains the full 200 epochs instead, which makes the
# direction branch reachable at mollusca scale; budget for that before setting it.
EPOCHS="${P2_PILOT_EPOCHS:-30}"
SAVE_EVERY="${P2_PILOT_SAVE_EVERY:-10}"

echo "=== P2 MOLLUSCA PLUMBING PILOT (CPU) -- ${CLADE}, ${EPOCHS} epochs/arm ==="
echo "Job: ${SLURM_JOB_ID:-?}, Host: $(hostname)"
python --version
cd /app

# Rule 16: compile before spending even CPU wall-clock.
python -m py_compile /app/scripts/score_p2_linkpred.py /app/scripts/apply_preregistration.py \
    /app/src/taxembed/eval/preregistration.py /app/src/taxembed/eval/linkpred.py \
    /app/src/taxembed/eval/baselines.py /app/src/taxembed/eval/p2_split.py \
    /app/src/taxembed/eval/subtree.py
echo "py_compile clean"

pip install --no-cache-dir -e . 2>&1 | tail -3
mkdir -p "${OUTDIR}"

for f in "${MAP}" "${CLOSURE}"; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: missing ${f}" >&2
        exit 1
    fi
done

# ---- 1. train all 12 arms ------------------------------------------------------------------
for s in "${SEEDS[@]}"; do
    for arm in vis00 vis50 degmatch_vis00 degmatch_vis50; do
        TRAIN="${SPLITDIR}/p2_${CLADE}_${arm}_seed${s}_train.npz"
        if [ ! -f "${TRAIN}" ]; then
            echo "ERROR: missing ${TRAIN}" >&2
            exit 1
        fi
        TAG="pilot_${arm}_s${s}"
        TAGDIR="${TAGS}/${TAG}"
        # Same rm-class posture as p2_lrz_train.sh: clear only an INCOMPLETE previous attempt,
        # scoped to this tag, and never the shared tags/ root.
        if [ -d "${TAGDIR}" ]; then
            find "${TAGDIR}" -maxdepth 1 -name "${TAG}_epoch*.pth" -delete
            find "${TAGDIR}" -maxdepth 1 -name "${TAG}_milestone_epoch*.pth" -delete
        fi
        echo "--- training ${TAG} ---"
        python -m taxembed.cli.main train \
            --file "${TRAIN}" --mapping "${MAP}" \
            --dim 100 --gpu -1 \
            --curriculum --curriculum-phases auto --early-stopping 999 \
            --radial-nudge 0.05 --radial-schedule log \
            --depth-scale-margin --margin-min 0.05 --margin-max 1.0 \
            --epoch-fraction 0.3 --euclidean-param --loss softmax \
            --save-every "${SAVE_EVERY}" \
            --epochs "${EPOCHS}" --seed "${s}" \
            --batch-size 256 --n-negatives 300 --lr 0.001 --grad-accum-steps 8 \
            --lr-schedule cosine_warmrestart --warm-restart-on-phase --lr-min-multiplier 0.01 \
            -as "${TAG}" 2>&1 | tail -3
    done
done

# ---- 2. score: 9 JSONs, seed-tagged baselines keys, exactly as the metazoa job writes -------
OUT_FILES=()
for s in "${SEEDS[@]}"; do
    REAL_MANIFEST="${SPLITDIR}/p2_${CLADE}_vis00_seed${s}_manifest.json"
    REAL_HELDOUT="${SPLITDIR}/p2_${CLADE}_seed${s}_heldout.npz"

    REAL_FLAGS=()
    for arm in vis00 vis50; do
        tag="pilot_${arm}_s${s}"
        dir="${TAGS}/${tag}"
        REAL_FLAGS+=(--checkpoints "${arm}_s${s}_ms=${dir}/${tag}_milestone_epoch*.pth")
        REAL_FLAGS+=(--checkpoints "${arm}_s${s}_roll=${dir}/${tag}_epoch*.pth")
    done
    OUT_REAL="${OUTDIR}/pilot_real_s${s}_${STAMP}.json"
    python /app/scripts/score_p2_linkpred.py \
        --manifest "${REAL_MANIFEST}" --heldout "${REAL_HELDOUT}" --closure "${CLOSURE}" \
        "${REAL_FLAGS[@]}" --metric cosine --seed 0 \
        --baselines-key "baselines_s${s}" --out "${OUT_REAL}"
    OUT_FILES+=("${OUT_REAL}")

    for arm in degmatch_vis00 degmatch_vis50; do
        DM_MANIFEST="${SPLITDIR}/p2_${CLADE}_${arm}_seed${s}_manifest.json"
        DM_HELDOUT="${SPLITDIR}/p2_${CLADE}_degmatch_seed${s}_heldout.npz"
        DM_CLOSURE="${SPLITDIR}/taxonomy_edges_${CLADE}_degmatch_seed${s}_transitive.npz"
        tag="pilot_${arm}_s${s}"
        dir="${TAGS}/${tag}"
        OUT_DM="${OUTDIR}/pilot_${arm}_s${s}_${STAMP}.json"
        python /app/scripts/score_p2_linkpred.py \
            --manifest "${DM_MANIFEST}" --heldout "${DM_HELDOUT}" --closure "${DM_CLOSURE}" \
            --checkpoints "${arm}_s${s}_ms=${dir}/${tag}_milestone_epoch*.pth" \
            --checkpoints "${arm}_s${s}_roll=${dir}/${tag}_epoch*.pth" \
            --metric cosine --seed 0 \
            --baselines-key "baselines_${arm}_s${s}" --out "${OUT_DM}"
        OUT_FILES+=("${OUT_DM}")
    done
done

echo "=== scored ${#OUT_FILES[@]} JSON(s) ==="

# ---- 3-5. the parser bridge, then the FULL production reading -------------------------------
python /app/helpers/check_p2_prereg_parses_real_scorer_output.py "${OUT_FILES[@]}"

python /app/scripts/apply_preregistration.py --task p2 \
    --json "${OUT_FILES[@]}" \
    --amendment-1 --amendment-2 --amendment-4 --amendment-6 \
    --out "${OUTDIR}/pilot_verdict_${STAMP}.json"

echo
echo "=== PILOT COMPLETE ==="
echo "🛑 PLUMBING ONLY. mollusca populates only the 11-15 depth stratum, and at ${EPOCHS} epochs"
echo "   gate (c) (a milestone at epoch 200) cannot pass, so UNINFORMATIVE is the EXPECTED"
echo "   verdict. What this run establishes is that 12 arms train, 9 JSONs score, the merge"
echo "   accepts them with its baselines asserts armed, and p2_verdict returns under the full"
echo "   production flag set. Do not quote its direction."
ls -la "${OUTDIR}"
