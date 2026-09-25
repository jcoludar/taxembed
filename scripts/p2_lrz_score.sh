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
#   12 runs = 4 arms x seeds 0/1/2, metazoa scale, 200 epochs (p2_amendment_2_20260924's matched-
#   control shape; p2_amendment_4_20260924, USER DESIGN decision 2026-09-24, BEFORE any array was
#   submitted and BEFORE any P2 outcome data existed: the RandomDAG control is RETIRED and
#   replaced by a DEGREE-MATCHED shuffle that preserves the fan-out multiset exactly instead of
#   collapsing it -- see scripts/p2_lrz_train.sh's header for the measured numbers).
#   artifacts/tags/p2_{vis00,vis50,degmatch_vis00,degmatch_vis50}_s{0,1,2}/
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
# writes one heldout.npz per (clade, seed), not per visibility). degmatch_vis00 and
# degmatch_vis50 do NOT share that real-tree pair: each holds out parent edges from the SAME
# degree-matched closure (scripts/build_p2_degmatch_split.py, same seed) but is scored against its
# OWN visibility-tagged manifest -- using the real-tree pair, or the OTHER degmatch arm's
# manifest, for either degmatch checkpoint set would rank candidates from the WRONG tree. The two
# degmatch arms DO share one heldout.npz (the held-out node set does not depend on visibility,
# exactly as for vis00/vis50), so there are three manifest/heldout pairs per seed in total: the
# real pair, and the two degmatch manifests against the one shared degmatch heldout.
#
# Pre-registration: results/p2_heldout_preregistration.json, including its
# p2_amendment_1_20260924, p2_amendment_2_20260924 and p2_amendment_4_20260924 blocks. READING
# RULE: do not read a verdict off these JSON files by hand. Cross-tree comparisons (an arm vs its
# MATCHED degree-matched control -- vis00 vs degmatch_vis00, vis50 vs degmatch_vis50, never
# crossed) use normalized_rank -- chance is 0.5 for ANY pool size, which is what makes it
# comparable across vis00/vis50's real pools and the degmatch control's pools even though (per
# p2_amendment_4_20260924) those pools are not perfectly chance-matched either; within-tree
# comparisons (vis00 vs vis50; an arm vs its own baselines.chance_mrr_mean, and, per
# p2_amendment_3_20260924, its own baselines.degree_prior) use MRR. PART 2 fix (2026-09-24, no
# amendment flag -- a correctness fix, not a design choice with two legitimate readings, same
# posture as amendment_3's C1-C5): each seed draws its OWN held-out split and therefore has its
# OWN sibling_chance_mean/chance_mrr_mean/degree_prior -- every --baselines-key below is now
# SEED-TAGGED (baselines_s${s}, baselines_degmatch_vis00_s${s}, ...) so the merge step never
# silently applies seed 0's floor to seeds 1/2's runs; see
# taxembed.eval.preregistration._p2_baselines_for_seed and
# assert_baselines_agree_across_seeds. This job writes 9 SEPARATE JSON files (never one combined
# file) -- score with scripts/apply_preregistration.py --task p2 --json <all 9 OUT_FILES>
# --amendment-1 --amendment-2 --amendment-4, which merges them
# (taxembed.eval.preregistration.merge_p2_scorer_outputs) before scoring. Go through
# p2_verdict(result, seeds, amendment_1=True, amendment_2=True, amendment_4=True); never read a
# verdict off these JSON files by hand.
# =============================================================================

SPLITDIR=/data/p2_splits
TAGS=/app/artifacts/tags
OUTDIR=/app/artifacts/p2_scoring
STAMP="$(date +%Y%m%d_%H%M%S)"
SEEDS=(0 1 2)
# C5 (2026-09-24, p2_amendment_3_20260924): build_p2_split.py's manifest records
# manifest["source_npz"] as the absolute macOS build-time path -- it does not exist inside this
# container. --closure overrides it explicitly with the container-mounted path instead.
REAL_CLOSURE=/data/taxopy/metazoa_33208_clean/taxonomy_edges_metazoa_33208_clean_transitive.npz

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
    # C5: pre-flight the closure too, not just the manifest and heldout -- a missing closure
    # would otherwise only surface as FileNotFoundError deep inside score_p2_linkpred.py, after
    # everything else already passed.
    for f in "${REAL_MANIFEST}" "${REAL_HELDOUT}" "${REAL_CLOSURE}"; do
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

    # PART 2 (2026-09-24, correctness fix, no amendment flag): --baselines-key is SEED-TAGGED
    # (baselines_s${s}, never the bare "baselines") -- this seed's own sibling_chance_mean/
    # chance_mrr_mean/degree_prior, drawn from THIS seed's own manifest/heldout, must survive the
    # merge distinctly from the other two seeds' own measurements, never collapsed onto whichever
    # seed's file happens to merge first.
    OUT_REAL="${OUTDIR}/p2_real_s${s}_${STAMP}.json"
    python /app/scripts/score_p2_linkpred.py \
        --manifest "${REAL_MANIFEST}" \
        --heldout "${REAL_HELDOUT}" \
        --closure "${REAL_CLOSURE}" \
        "${REAL_FLAGS[@]}" \
        --metric cosine \
        --seed 0 \
        --baselines-key "baselines_s${s}" \
        --out "${OUT_REAL}"
    OUT_FILES+=("${OUT_REAL}")

    # -------- degree-matched control (p2_amendment_4_20260924): TWO matched controls, each its
    # own visibility-tagged manifest, never the real-tree pair and never each other's manifest
    # (p2_amendment_2_20260924's matched-control shape, now built from the degree-matched
    # shuffle instead of RandomDAG's uniform rewire) --------
    DEGMATCH_HELDOUT="${SPLITDIR}/p2_metazoa_33208_clean_degmatch_seed${s}_heldout.npz"
    # C5: the degree-matched closure build_p2_degmatch_split.py writes -- SHARED by
    # degmatch_vis00 and degmatch_vis50 at this seed (only the TRAIN split's visibility thinning
    # differs between them; the closure itself depends only on --seed, not --visibility).
    DEGMATCH_CLOSURE="${SPLITDIR}/taxonomy_edges_metazoa_33208_clean_degmatch_seed${s}_transitive.npz"
    for f in "${DEGMATCH_HELDOUT}" "${DEGMATCH_CLOSURE}"; do
        if [ ! -f "${f}" ]; then
            echo "ERROR: missing ${f}" >&2
            exit 1
        fi
    done

    for degmatch_arm in degmatch_vis00 degmatch_vis50; do
        DEGMATCH_MANIFEST="${SPLITDIR}/p2_metazoa_33208_clean_${degmatch_arm}_seed${s}_manifest.json"
        if [ ! -f "${DEGMATCH_MANIFEST}" ]; then
            echo "ERROR: missing ${DEGMATCH_MANIFEST}" >&2
            exit 1
        fi

        tag="p2_${degmatch_arm}_s${s}"
        dir="${TAGS}/${tag}"
        for f in "${dir}/run.json" "${dir}/${tag}_milestone_epoch200.pth" "${dir}/${tag}_epoch200.pth"; do
            if [ ! -f "${f}" ]; then
                echo "ERROR: missing ${f}" >&2
                exit 1
            fi
        done

        # C4 #3 + PART 2: --baselines-key writes this control's OWN floor, for THIS SEED, under
        # the seed-tagged key p2_verdict(..., amendment_2=True, amendment_4=True) reads
        # (baselines_degmatch_vis00_s${s} / baselines_degmatch_vis50_s${s}) instead of every
        # invocation writing the same plain "baselines" key (the pre-Part-2 C4 #3 fix) or the
        # same seed-less "baselines_${degmatch_arm}" key across all 3 seeds (the seed-0-wins
        # defect Part 2 fixes) -- each seed's own control measurement now survives the merge
        # distinctly.
        OUT_DEGMATCH="${OUTDIR}/p2_${degmatch_arm}_s${s}_${STAMP}.json"
        python /app/scripts/score_p2_linkpred.py \
            --manifest "${DEGMATCH_MANIFEST}" \
            --heldout "${DEGMATCH_HELDOUT}" \
            --closure "${DEGMATCH_CLOSURE}" \
            --checkpoints "${degmatch_arm}_s${s}_ms=${dir}/${tag}_milestone_epoch*.pth" \
            --checkpoints "${degmatch_arm}_s${s}_roll=${dir}/${tag}_epoch*.pth" \
            --metric cosine \
            --seed 0 \
            --baselines-key "baselines_${degmatch_arm}_s${s}" \
            --out "${OUT_DEGMATCH}"
        OUT_FILES+=("${OUT_DEGMATCH}")
    done
done

echo "=== scoring complete: ${#OUT_FILES[@]} result file(s) ==="
ls -la "${OUT_FILES[@]}"
md5sum "${OUT_FILES[@]}"
