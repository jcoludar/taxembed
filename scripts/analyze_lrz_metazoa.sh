#!/bin/bash
#SBATCH --job-name=taxembed_metazoa_regen
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/logs/metazoa_regen_%j.out
#SBATCH --error=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/logs/metazoa_regen_%j.err
#SBATCH --container-image=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/pytorch-2.4.0-cuda12.1.sqsh
#SBATCH --container-workdir=/app
#SBATCH --container-mounts=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/src:/app:rw,/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/data:/data:ro,/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/artifacts:/app/artifacts:rw

set -euo pipefail
export MKL_THREADING_LAYER=GNU

# Regenerate the 498k canonical ablation row (metazoa_lower_lr_bigger_batch) as a 5-SEED MEAN,
# so it matches the estimator used for the 877k/1.1M rows (table currently mixes a single
# final-ep200 run here with seeded means below). Writes analysis_results.json (raw artifact).

STAMP=20260721
TAG=metazoa_lower_lr_bigger_batch
CKPT="/app/artifacts/tags/${TAG}/${TAG}.pth"
MAPPING="/data/taxonomy_edges_metazoa_33208_clean.mapping.tsv"
OUT="/app/artifacts/tags/${TAG}/regen_analysis_${STAMP}"

echo "=== metazoa 498k canonical regen (5-seed mean) ==="
echo "Job: ${SLURM_JOB_ID:-?}, Host: $(hostname)"
python --version
cd /app
pip install --no-cache-dir -e . 2>&1 | tail -3

WORK_TAXDUMP="${TMPDIR:-/tmp}/taxdump_work"
mkdir -p "$WORK_TAXDUMP"
cp /data/taxdump_current/nodes.dmp /data/taxdump_current/names.dmp /data/taxdump_current/merged.dmp "$WORK_TAXDUMP"/

for f in "$CKPT" "$MAPPING"; do
    if [ ! -f "$f" ]; then echo "ERROR: missing input: $f" >&2; exit 1; fi
done
mkdir -p "$OUT"

python scripts/analyze_hierarchy_hyperbolic.py \
    --checkpoint "$CKPT" --mapping "$MAPPING" --data-dir "$WORK_TAXDUMP" \
    --ranks phylum class order family --seed 0 --repeats 5 \
    -o "$OUT"

echo "--- metazoa raw metrics ---"
cat "$OUT/analysis_results.json"
