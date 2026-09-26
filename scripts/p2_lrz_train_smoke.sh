#!/bin/bash
#SBATCH --job-name=taxembed_p2_train_smoke
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/logs/p2_train_smoke_%j.out
#SBATCH --error=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/logs/p2_train_smoke_%j.err
#SBATCH --container-image=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/pytorch-2.4.0-cuda12.1.sqsh
#SBATCH --container-workdir=/app
#SBATCH --container-mounts=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/src:/app:rw,/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/data:/data:ro,/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/artifacts:/app/artifacts:rw

set -euo pipefail
export MKL_THREADING_LAYER=GNU

# =============================================================================
# Rule 16 CANARY for scripts/p2_lrz_train.sh AND, on its CLI surface only, scripts/p2_lrz_score.sh
# -- run on CPU (the GPU queue is ~11-19h per run; the CPU queue is minutes). Both jobs share one
# failure mode this smoke exists to catch: a flag that looks right in prose but is not a real
# option on the CLI it is handed to, discovered only after a scarce resource is finally granted
# (the ESMFold incident: job 5710076 died 37s after finally getting an H100, on a one-line error
# a 2-second static check would have caught).
#
# Proves, on the small mollusca_6447_clean P2 split (already built locally, Task 2's pattern),
# in minutes rather than the GPU array's ~11-19h per run:
#   - py_compile clean across every module the train + score jobs import
#   - every flag scripts/p2_lrz_train.sh passes is a real `taxembed.cli.main train` option
#   - every flag scripts/p2_lrz_score.sh passes is a real `scripts/score_p2_linkpred.py` option
#   - a P2 split .npz (scripts/build_p2_split.py's own output schema) trains end to end for a
#     couple of epochs without error, and the run reports the seed it was given
#
# Pre-registration: results/p2_heldout_preregistration.json, including its
# p2_amendment_1_20260924 and p2_amendment_2_20260924 blocks. READING RULE (this smoke does not
# exercise it, and must not be read as if it did): cross-tree comparisons (an arm vs its MATCHED
# RandomDAG control -- vis00 vs randomdag_vis00, vis50 vs randomdag_vis50, never crossed) use
# normalized_rank -- chance is 0.5 for ANY pool size; within-tree comparisons (vis00 vs vis50, an
# arm vs its own baselines.sibling_chance_mean) use MRR. This run's own tiny checkpoint is not
# scored and contributes nothing to p2_verdict; it exists only to prove the pipeline runs.
# =============================================================================

echo "=== TaxEmbed P2 Task 8 train SMOKE (CPU) ==="
echo "Job: ${SLURM_JOB_ID}, Host: $(hostname)"
cd /app
pip install --no-cache-dir -e . 2>&1 | tail -3

MOLL_DIR=/data/taxopy/mollusca_6447_clean
MAP="${MOLL_DIR}/taxonomy_edges_mollusca_6447_clean.mapping.tsv"
SPLITDIR=/data/p2_splits
TRAIN="${SPLITDIR}/p2_mollusca_6447_clean_vis00_seed0_train.npz"

for f in "${MAP}" "${TRAIN}"; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: missing ${f}" >&2
        exit 1
    fi
done

python -m py_compile \
    /app/src/taxembed/cli/main.py \
    /app/src/taxembed/eval/p2_split.py \
    /app/src/taxembed/eval/randomdag.py \
    /app/src/taxembed/eval/linkpred.py \
    /app/src/taxembed/eval/baselines.py \
    /app/src/taxembed/eval/preregistration.py \
    /app/scripts/build_p2_split.py \
    /app/scripts/build_p2_degmatch_split.py \
    /app/scripts/score_p2_linkpred.py
echo "py_compile clean"

python -m taxembed.cli.main train --help > /tmp/p2_train_help.txt
for flag in --file --mapping --dim --gpu --amp --curriculum --curriculum-phases \
            --early-stopping --radial-nudge --radial-schedule --depth-scale-margin \
            --margin-min --margin-max --epoch-fraction --euclidean-param --loss \
            --save-every --epochs --seed --batch-size --n-negatives --lr \
            --grad-accum-steps --lr-schedule --warm-restart-on-phase --lr-min-multiplier -as; do
    if ! grep -q -- "${flag}" /tmp/p2_train_help.txt; then
        echo "SMOKE FAILED: ${flag} is not a real option on taxembed.cli.main train" >&2
        exit 1
    fi
done
echo "all p2_lrz_train.sh flags real"

python /app/scripts/score_p2_linkpred.py --help > /tmp/p2_score_help.txt
for flag in --manifest --heldout --checkpoints --out --metric --max-checkpoints --seed \
            --closure --baselines-key; do
    if ! grep -q -- "${flag}" /tmp/p2_score_help.txt; then
        echo "SMOKE FAILED: ${flag} is not a real option on scripts/score_p2_linkpred.py" >&2
        exit 1
    fi
done
echo "all p2_lrz_score.sh flags real"

# B4 (2026-09-26): --save-every AND enough epochs to actually WRITE a milestone.
# The smoke used to train 2 epochs with no --save-every, so the MILESTONE-WRITING PATH was
# unexercised -- while scripts/p2_lrz_score.sh's pre-flight REQUIRES ${tag}_milestone_epoch200.pth
# and refuses without it. A canary that does not touch the path the consumer requires is covered
# by precedent, not by the canary. 10 epochs with --save-every 10 writes exactly one milestone,
# which is all that is needed to prove the path exists and is named as the score job expects.
EPOCHS="${P2_SMOKE_EPOCHS:-10}"
SAVE_EVERY="${P2_SMOKE_SAVE_EVERY:-10}"
# NOT exercised here, deliberately and with the reason recorded rather than left implicit:
# --amp. The real job runs `--gpu 0 --amp`; this canary runs on the CPU queue (minutes, versus
# hours for a GPU slot), and mixed precision on CPU would exercise a different code path than
# the one the array uses, so "it passed here" would not transfer. --amp's reality as a FLAG is
# checked above against the CLI's own --help, which is what this canary can honestly establish.
python -m taxembed.cli.main train \
    --file "${TRAIN}" --mapping "${MAP}" \
    --dim 100 --gpu -1 \
    --curriculum --curriculum-phases auto --early-stopping 999 \
    --radial-nudge 0.05 --radial-schedule log \
    --depth-scale-margin --margin-min 0.05 --margin-max 1.0 \
    --epoch-fraction 0.3 --euclidean-param --loss softmax \
    --save-every "${SAVE_EVERY}" \
    --epochs "${EPOCHS}" --seed 0 \
    --batch-size 256 --n-negatives 300 --lr 0.001 --grad-accum-steps 8 \
    --lr-schedule cosine_warmrestart --warm-restart-on-phase --lr-min-multiplier 0.01 \
    -as smoke_p2_train 2>&1 | tee /tmp/p2_train_smoke.log

if ! grep -q "Training complete" /tmp/p2_train_smoke.log; then
    echo "SMOKE FAILED: log lacks 'Training complete'" >&2
    exit 1
fi
if ! grep -q '"seed": 0' /app/artifacts/tags/smoke_p2_train/run.json; then
    echo "SMOKE FAILED: run.json lacks the seed it was given" >&2
    exit 1
fi

SMOKE_TAGDIR=/app/artifacts/tags/smoke_p2_train
n_milestone="$(find "${SMOKE_TAGDIR}" -maxdepth 1 -name "smoke_p2_train_milestone_epoch*.pth" | wc -l | tr -d ' ')"
if [ "${n_milestone}" -lt 1 ]; then
    echo "SMOKE FAILED: --save-every ${SAVE_EVERY} wrote NO milestone checkpoint, but" >&2
    echo "              p2_lrz_score.sh's pre-flight requires \${tag}_milestone_epoch200.pth" >&2
    ls -la "${SMOKE_TAGDIR}" >&2
    exit 1
fi
echo "milestone-writing path exercised: ${n_milestone} milestone checkpoint(s)"

n_roll="$(find "${SMOKE_TAGDIR}" -maxdepth 1 -name "smoke_p2_train_epoch*.pth" | wc -l | tr -d ' ')"
echo "rolling checkpoints written: ${n_roll} (the trainer's queue keeps the last 5)"

# B4/V6: run the SCORER end to end. Until 2026-09-26 score_p2_linkpred.py had never been executed
# on data anywhere in this project -- only --help. It is ~1 second on the mollusca split and it
# is the single cheapest thing that de-risks the whole post-GPU read, so it belongs in the canary
# rather than in a reviewer's scratch directory.
SMOKE_SCORES=/tmp/p2_smoke_scores.json
python /app/scripts/score_p2_linkpred.py \
    --manifest "${SPLITDIR}/p2_mollusca_6447_clean_vis00_seed0_manifest.json" \
    --heldout "${SPLITDIR}/p2_mollusca_6447_clean_seed0_heldout.npz" \
    --closure "${MOLL_DIR}/taxonomy_edges_mollusca_6447_clean_transitive.npz" \
    --checkpoints "vis00_s0_ms=${SMOKE_TAGDIR}/smoke_p2_train_milestone_epoch*.pth" \
    --checkpoints "vis00_s0_roll=${SMOKE_TAGDIR}/smoke_p2_train_epoch*.pth" \
    --baselines-key baselines_s0 \
    --out "${SMOKE_SCORES}" 2>&1 | tail -5

python - "${SMOKE_SCORES}" <<'PYCHECK'
import json, sys
res = json.load(open(sys.argv[1]))
b = res["baselines_s0"]
missing = [k for k in ("sibling_chance_mean", "chance_mrr_mean", "degree_prior") if k not in b]
if missing:
    sys.exit(f"SMOKE FAILED: scorer output lacks {missing}")
# C2 (2026-09-26): the per-stratum degree prior the engine's amendment_6 sign test divides by.
if "by_depth" not in b["degree_prior"]:
    sys.exit("SMOKE FAILED: degree_prior.by_depth absent -- amendment_6's per-stratum divisor "
             "would silently fall back to the aggregate, which over-allows the sign test")
arms = sorted(res["arms"])
if not any(a.endswith("_roll") for a in arms) or not any(a.endswith("_ms") for a in arms):
    sys.exit(f"SMOKE FAILED: scorer registered {arms}, expected both a _ms and a _roll group")
print(f"scorer output OK: arms {arms}, degree_prior.by_depth strata "
      f"{sorted(b['degree_prior']['by_depth'])}")
PYCHECK

echo "=== SMOKE PASSED -- safe to submit p2_lrz_train.sh (after confirming Task 2's vis00/vis50 splits, scripts/build_p2_degmatch_split.py's MATCHED degmatch_vis00 AND degmatch_vis50 splits (p2_amendment_4_20260924) -- run the builder once per --visibility, same --seed -- and the metazoa mapping TSV are all present under /data/p2_splits) ==="
ls -la /app/artifacts/tags/smoke_p2_train/
