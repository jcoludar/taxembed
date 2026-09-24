#!/bin/bash
#SBATCH --job-name=taxembed_p2_score
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/logs/p2_score_%j.out
#SBATCH --error=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/logs/p2_score_%j.err
#SBATCH --container-image=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/pytorch-2.4.0-cuda12.1.sqsh
#SBATCH --container-workdir=/app
#SBATCH --container-mounts=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/src:/app:rw,/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/data:/data:ro,/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/artifacts:/app/artifacts:rw

set -euo pipefail
export MKL_THREADING_LAYER=GNU

# =============================================================================
# Plan v1 Task 8 -- score the P2 12-run array (scripts/p2_lrz_train.sh) with
# scripts/score_p2_linkpred.py: for every held-out LEAF node whose parent edge was withheld, rank
# its true parent among its grandparent's other children using the checkpoint's embedding.
#
#   12 runs = 4 arms x seeds 0/1/2, metazoa scale, 200 epochs (p2_amendment_2_20260924, USER
#   DESIGN decision 2026-09-24: RandomDAG now runs at BOTH visibilities, matched per real arm).
#   artifacts/tags/p2_{vis00,vis50,randomdag_vis00,randomdag_vis50}_s{0,1,2}/
#
# Per run, TWO checkpoint groups are registered, mirroring Task 9's *_ms/*_roll convention
# (results/p2_heldout_preregistration.json's primary-metric definition):
#   *_ms   milestone checkpoints (ep 10..200, --save-every 10)     -> completion check (gate c)
#          and the trajectory used for the learning gate (gate a).
#   *_roll rolling checkpoints nearest epoch 200                   -> THE per-run value (mean MRR
#          over the trailing rolling window nearest epoch 200; within-run jitter = that window's SD).
#
# THREE SEPARATE score_p2_linkpred.py CALLS PER SEED, not one for all four arms: vis00 and vis50
# share one --manifest/--heldout pair (the held-out NODE SET and the closure's parent/depth are
# properties of the REAL tree, unaffected by the visibility knob -- Task 2's build_p2_split.py
# writes one heldout.npz per (clade, seed), not per visibility). randomdag_vis00 and
# randomdag_vis50 do NOT share that real-tree pair: each holds out parent edges from the SAME
# randomised closure (scripts/build_p2_randomdag_split.py, same seed) but is scored against its
# OWN visibility-tagged manifest -- using the real-tree pair, or the OTHER randomdag arm's
# manifest, for either randomdag checkpoint set would rank candidates from the WRONG tree. The two
# randomdag arms DO share one heldout.npz (the held-out node set does not depend on visibility,
# exactly as for vis00/vis50), so there are three manifest/heldout pairs per seed in total: the
# real pair, and the two randomdag manifests against the one shared randomdag heldout.
#
# Pre-registration: results/p2_heldout_preregistration.json, including its
# p2_amendment_1_20260924 and p2_amendment_2_20260924 blocks. READING RULE: do not read a verdict
# off these JSON files by hand. Cross-tree comparisons (an arm vs its MATCHED RandomDAG control --
# vis00 vs randomdag_vis00, vis50 vs randomdag_vis50, never crossed) use normalized_rank -- chance
# is 0.5 for ANY pool size, which is what makes it comparable across vis00/vis50's real pools and
# RandomDAG's collapsed-fan-out pools; within-tree comparisons (vis00 vs vis50; an arm vs its own
# baselines.sibling_chance_mean) use MRR. Go through
# src/taxembed/eval/preregistration.py::p2_verdict(result, seeds, amendment_1=True, amendment_2=True).
# =============================================================================

SPLITDIR=/data/p2_splits
TAGS=/app/artifacts/tags
OUTDIR=/app/artifacts/p2_scoring
STAMP="$(date +%Y%m%d_%H%M%S)"
SEEDS=(0 1 2)

echo "=== TaxEmbed P2 Task 8 scoring -- 12-run array ==="
echo "Job: ${SLURM_JOB_ID:-?}, Host: $(hostname)"
python --version
cd /app

# Rule 16: prove the scorer and the modules it imports compile before spending CPU wall-clock.
python -m py_compile /app/scripts/score_p2_linkpred.py /app/src/taxembed/eval/linkpred.py \
    /app/src/taxembed/eval/baselines.py /app/src/taxembed/eval/p2_split.py \
    /app/src/taxembed/eval/subtree.py

pip install --no-cache-dir -e . 2>&1 | tail -3

mkdir -p "${OUTDIR}"
OUT_FILES=()

for s in "${SEEDS[@]}"; do
    # -------- real-tree pair: vis00 + vis50, same seed, same manifest/heldout --------
    REAL_MANIFEST="${SPLITDIR}/p2_metazoa_33208_clean_vis00_seed${s}_manifest.json"
    REAL_HELDOUT="${SPLITDIR}/p2_metazoa_33208_clean_seed${s}_heldout.npz"
    for f in "${REAL_MANIFEST}" "${REAL_HELDOUT}"; do
        if [ ! -f "${f}" ]; then
            echo "ERROR: missing ${f}" >&2
            exit 1
        fi
    done

    REAL_FLAGS=()
    for arm in vis00 vis50; do
        tag="p2_${arm}_s${s}"
        dir="${TAGS}/${tag}"
        # Fail fast and by name. A missing ep200 means that array element did not finish, and
        # scoring a truncated run is worse than not scoring it.
        for f in "${dir}/run.json" "${dir}/${tag}_milestone_epoch200.pth" "${dir}/${tag}_epoch200.pth"; do
            if [ ! -f "${f}" ]; then
                echo "ERROR: missing ${f}" >&2
                exit 1
            fi
        done
        REAL_FLAGS+=(--checkpoints "${arm}_s${s}_ms=${dir}/${tag}_milestone_epoch*.pth")
        REAL_FLAGS+=(--checkpoints "${arm}_s${s}_roll=${dir}/${tag}_epoch*.pth")
    done

    OUT_REAL="${OUTDIR}/p2_real_s${s}_${STAMP}.json"
    python /app/scripts/score_p2_linkpred.py \
        --manifest "${REAL_MANIFEST}" \
        --heldout "${REAL_HELDOUT}" \
        "${REAL_FLAGS[@]}" \
        --metric cosine \
        --seed 0 \
        --out "${OUT_REAL}"
    OUT_FILES+=("${OUT_REAL}")

    # -------- RandomDAG: TWO matched controls, each its own visibility-tagged manifest, never
    # the real-tree pair and never each other's manifest (p2_amendment_2_20260924) --------
    RAND_HELDOUT="${SPLITDIR}/p2_metazoa_33208_clean_randomdag_seed${s}_heldout.npz"
    if [ ! -f "${RAND_HELDOUT}" ]; then
        echo "ERROR: missing ${RAND_HELDOUT}" >&2
        exit 1
    fi

    for rand_arm in randomdag_vis00 randomdag_vis50; do
        RAND_MANIFEST="${SPLITDIR}/p2_metazoa_33208_clean_${rand_arm}_seed${s}_manifest.json"
        if [ ! -f "${RAND_MANIFEST}" ]; then
            echo "ERROR: missing ${RAND_MANIFEST}" >&2
            exit 1
        fi

        tag="p2_${rand_arm}_s${s}"
        dir="${TAGS}/${tag}"
        for f in "${dir}/run.json" "${dir}/${tag}_milestone_epoch200.pth" "${dir}/${tag}_epoch200.pth"; do
            if [ ! -f "${f}" ]; then
                echo "ERROR: missing ${f}" >&2
                exit 1
            fi
        done

        OUT_RAND="${OUTDIR}/p2_${rand_arm}_s${s}_${STAMP}.json"
        python /app/scripts/score_p2_linkpred.py \
            --manifest "${RAND_MANIFEST}" \
            --heldout "${RAND_HELDOUT}" \
            --checkpoints "${rand_arm}_s${s}_ms=${dir}/${tag}_milestone_epoch*.pth" \
            --checkpoints "${rand_arm}_s${s}_roll=${dir}/${tag}_epoch*.pth" \
            --metric cosine \
            --seed 0 \
            --out "${OUT_RAND}"
        OUT_FILES+=("${OUT_RAND}")
    done
done

echo "=== scoring complete: ${#OUT_FILES[@]} result file(s) ==="
ls -la "${OUT_FILES[@]}"
md5sum "${OUT_FILES[@]}"
