#!/bin/bash
#SBATCH --job-name=taxembed_task9
#SBATCH --partition=lrz-v100x2
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-5
#SBATCH --output=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/logs/task9_%A_%a.out
#SBATCH --error=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/logs/task9_%A_%a.err
#SBATCH --container-image=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/pytorch-2.4.0-cuda12.1.sqsh
#SBATCH --container-workdir=/app
#SBATCH --container-mounts=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/src:/app:rw,/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/data:/data:ro,/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/artifacts:/app/artifacts:rw

set -euo pipefail
export MKL_THREADING_LAYER=GNU

# =============================================================================
# Plan v2 Task 9 -- does Figure 4's recipe claim survive on the UNPLANTED metric?
#
# ARRAY JOB: 6 tasks = 2 arms x seeds 0/1/2. One run per array task, so each gets
# its own walltime and they queue independently instead of chaining into one
# multi-day job that dies at the walltime boundary.
#
# ON METAZOA (498,246 nodes) -- the exact scale at which Figure 4's contrast was
# measured, so these numbers are directly comparable to the published ones.
#
# WHY NOT A SMALLER CLADE. The canonical recipe is scale-aware by construction:
# batch 256 x grad-accum 8 takes 4x FEWER optimizer steps per epoch than the prior
# arm (128 x 4). Below its design scale that difference decides the outcome and the
# contrast measures nothing about the recipe. Steps/epoch at epoch_fraction 0.3:
#     echinodermata   3,965 nodes ->     5.1   run 1: canonical NEVER TRAINED, INVALID
#     mollusca       32,017       ->    40.8   run 2: trained, did NOT converge
#     arthropoda    324,983       ->   946.6   (not uploaded to /data)
#     metazoa       498,246       -> 1,734.5   <- THIS JOB, and Figure 4's own scale
#
# ARM DEFINITION is read off the two recorded run.json files, NOT the manuscript's
# one-line description. The arms differ in FOUR substantive factors:
#     effective batch  512 (128x4)  ->  2048 (256x8)
#     n_negatives      100          ->  300
#     lr               0.005        ->  0.001
#     lr_schedule      const        ->  cosine_warmrestart + warm-restart-on-phase
# Everything else identical. Both arms carry the curriculum, so NEITHER is
# literally Nickel-Kiela.
#
# Both arms use the SHIPPED (unfixed) sampler. Task 6's ancestry-aware sampler has
# not landed; the 47.375240% false-negative defect is present equally in both arms,
# so it does not confound the contrast, and this audits the figure AS SHIPPED.
#
# Pre-registration and validity gate: results/recipe_angular_comparison.json.
# Do NOT read the contrast until BOTH arms show a loss decrease clearly above their
# own last-10-epoch noise span AND a trending learned metric. Otherwise the result
# is UNINFORMATIVE in both directions -- never a pass, never a fail.
#
# Submit the smoke first. See task9_lrz_recipe_contrast_smoke.sh (Rule 16).
# =============================================================================

DATA=/data/taxonomy_edges_metazoa_33208_clean_transitive.npz
MAP=/data/taxonomy_edges_metazoa_33208_clean.mapping.tsv
EPOCHS="${TASK9_EPOCHS:-200}"

# Array index -> (arm, seed). 0,1,2 = canonical s0/s1/s2; 3,4,5 = prior s0/s1/s2.
IDX="${SLURM_ARRAY_TASK_ID}"
if [ "${IDX}" -lt 3 ]; then
    ARM=canonical
    SEED="${IDX}"
else
    ARM=prior
    SEED=$(( IDX - 3 ))
fi

echo "=== TaxEmbed Task 9 -- ${ARM}, seed ${SEED}, metazoa ==="
echo "Job: ${SLURM_ARRAY_JOB_ID}, array task ${IDX}, Host: $(hostname)"
nvidia-smi -L || true          # never kill the job on this (Rule 16 SIGPIPE incident)
python --version
echo

cd /app
pip install --no-cache-dir -e . 2>&1 | tail -3
echo

for f in "${DATA}" "${MAP}"; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: missing ${f}" >&2
        exit 1
    fi
done

SHARED=(
    --file "${DATA}"
    --mapping "${MAP}"
    --dim 100
    --gpu 0
    --amp
    --curriculum
    --curriculum-phases auto
    --early-stopping 999
    --radial-nudge 0.05
    --radial-schedule log
    --depth-scale-margin
    --margin-min 0.05
    --margin-max 1.0
    --epoch-fraction 0.3
    --euclidean-param
    --loss softmax
    --save-every 10
    --epochs "${EPOCHS}"
    --seed "${SEED}"
)

if [ "${ARM}" = "canonical" ]; then
    ARM_FLAGS=(
        --batch-size 256
        --n-negatives 300
        --lr 0.001
        --grad-accum-steps 8
        --lr-schedule cosine_warmrestart
        --warm-restart-on-phase
        --lr-min-multiplier 0.01
    )
else
    # no --lr-schedule: const, as recorded in metazoa_softmax/run.json
    ARM_FLAGS=(
        --batch-size 128
        --n-negatives 100
        --lr 0.005
        --grad-accum-steps 4
    )
fi

python -m taxembed.cli.main train "${SHARED[@]}" "${ARM_FLAGS[@]}" \
    -as "task9_${ARM}_s${SEED}"

echo
echo "=== ${ARM} seed ${SEED} complete ==="
ls -la "/app/artifacts/tags/task9_${ARM}_s${SEED}/"
