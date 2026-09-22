#!/bin/bash
#SBATCH --job-name=taxembed_task9_score
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/logs/task9_score_%j.out
#SBATCH --error=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/logs/task9_score_%j.err
#SBATCH --container-image=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/pytorch-2.4.0-cuda12.1.sqsh
#SBATCH --container-workdir=/app
#SBATCH --container-mounts=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/src:/app:rw,/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/data:/data:ro,/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/artifacts:/app/artifacts:rw

set -euo pipefail
export MKL_THREADING_LAYER=GNU

# =============================================================================
# Plan v2 Task 9 -- score the two runs Figure 4 was DRAWN FROM on the radius-free metric.
# Design: docs/specs/2026-09-22-task9-radius-free-scorer-design.md
#
#   prior      artifacts/tags/metazoa_softmax_milestones       (job 5661777, 200 ep)
#   canonical  artifacts/tags/metazoa_lower_lr_bigger_batch    (job 5664609, 200 ep)
#   20 milestone checkpoints each (epochs 10..200), plus a reconstructed init null row.
#
# CPU only: numpy + torch.load. Read-only on the checkpoints; writes one JSON + one .npz
# under artifacts/task9_scoring/. n=1 per arm and unseeded -> PROVISIONAL reading; the
# seeded array (job 5802007) gives run-to-run spread.
#
# TASK9_MAX_CKPT=1 turns this into the Rule 16 smoke (1 checkpoint per arm).
# =============================================================================

MAX="${TASK9_MAX_CKPT:-}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR=/app/artifacts/task9_scoring
if [ -n "${MAX}" ]; then
    OUT="${OUTDIR}/fig4_runs_SMOKE_${STAMP}.json"
    MAXFLAG=(--max-checkpoints "${MAX}")
else
    OUT="${OUTDIR}/fig4_runs_${STAMP}.json"
    MAXFLAG=()
fi

echo "=== TaxEmbed Task 9 scoring (Figure 4 runs) ==="
echo "Job: ${SLURM_JOB_ID:-?}, Host: $(hostname), max-ckpt: ${MAX:-all}"
python --version
cd /app
pip install --no-cache-dir -e . 2>&1 | tail -3

NPZ=/data/taxonomy_edges_metazoa_33208_clean_transitive.npz
PRIOR_DIR=/app/artifacts/tags/metazoa_softmax_milestones
CANON_DIR=/app/artifacts/tags/metazoa_lower_lr_bigger_batch
for f in "${NPZ}" "${PRIOR_DIR}/metazoa_softmax_milestones_milestone_epoch200.pth" \
         "${CANON_DIR}/metazoa_lower_lr_bigger_batch_milestone_epoch200.pth"; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: missing ${f}" >&2
        exit 1
    fi
done
python -m py_compile /app/scripts/score_recipe_checkpoints.py /app/src/taxembed/eval/angular.py
echo "inputs present, py_compile clean"

python /app/scripts/score_recipe_checkpoints.py \
    --npz "${NPZ}" \
    --out "${OUT}" \
    --run "prior=${PRIOR_DIR}/metazoa_softmax_milestones_milestone_epoch*.pth" \
    --run "canonical=${CANON_DIR}/metazoa_lower_lr_bigger_batch_milestone_epoch*.pth" \
    --run "prior_roll=${PRIOR_DIR}/metazoa_softmax_milestones_epoch*.pth" \
    --run "canonical_roll=${CANON_DIR}/metazoa_lower_lr_bigger_batch_epoch*.pth" \
    --pair canonical_roll:prior_roll \
    --n-queries 10000 --k 10 --seed 0 \
    ${MAXFLAG[@]+"${MAXFLAG[@]}"}

echo "=== scoring complete: ${OUT} ==="
