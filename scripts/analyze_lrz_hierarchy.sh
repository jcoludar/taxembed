#!/bin/bash
#SBATCH --job-name=taxembed_hierarchy
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/logs/hierarchy_%j.out
#SBATCH --error=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/logs/hierarchy_%j.err
#SBATCH --container-image=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/pytorch-2.4.0-cuda12.1.sqsh
#SBATCH --container-workdir=/app
#SBATCH --container-mounts=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/src:/app:rw,/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/data:/data:ro,/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/artifacts:/app/artifacts:rw

set -euo pipefail
export MKL_THREADING_LAYER=GNU   # MKL/libgomp threading-layer clash fix (see PROJECT_STATE.md)

# =============================================================================
# Headline-number regeneration to RAW artifacts (angular separation + depth-norm).
# analyze_hierarchy_hyperbolic.py is CPU-only (numpy/scipy/taxopy; torch.load on CPU).
# It now emits analysis_results.json alongside the PNGs — the manuscript's headline
# values (depth-norm +0.954, 1.16x domain -> 7.69x family, etc.) previously had no
# on-disk numeric backing file, only PROJECT_STATE.md prose (methods-audit 2026-07-15).
#
# Runs BOTH headline-scale checkpoints (final ep200, unsuffixed .pth), --seed 0
# --repeats 5 (mean+/-std over 5 seeds, matching the PROJECT_STATE seeded sweep).
# Writes to a fresh regen_analysis_20260721/ per tag (never overwrites the June
# analysis_final_seeded/ PNGs).
# =============================================================================

STAMP=20260721
BASE=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz

echo "=== TaxEmbed headline-number regen (CPU) ==="
echo "Job: ${SLURM_JOB_ID:-?}, Host: $(hostname)"
python --version
echo

cd /app
pip install --no-cache-dir -e . 2>&1 | tail -3
echo

# TaxoPy needs a WRITABLE taxdb_dir (it touches the .dmp files); /data is mounted ro -> copy to scratch.
WORK_TAXDUMP="${TMPDIR:-/tmp}/taxdump_work"
mkdir -p "$WORK_TAXDUMP"
cp /data/taxdump_current/nodes.dmp /data/taxdump_current/names.dmp /data/taxdump_current/merged.dmp "$WORK_TAXDUMP"/
echo "  staged writable taxdump at $WORK_TAXDUMP"
echo

run_one () {
    local TAG="$1"; local MAPPING_BN="$2"; shift 2
    local CKPT="/app/artifacts/tags/${TAG}/${TAG}.pth"
    local MAPPING="/data/${MAPPING_BN}"
    local OUT="/app/artifacts/tags/${TAG}/regen_analysis_${STAMP}"
    echo "=================================================================="
    echo "=== ${TAG}: ranks $* ==="
    echo "=================================================================="
    for f in "$CKPT" "$MAPPING"; do
        if [ ! -f "$f" ]; then echo "ERROR: missing input: $f" >&2; exit 1; fi
    done
    mkdir -p "$OUT"
    python scripts/analyze_hierarchy_hyperbolic.py \
        --checkpoint "$CKPT" --mapping "$MAPPING" --data-dir "$WORK_TAXDUMP" \
        --ranks "$@" --seed 0 --repeats 5 \
        -o "$OUT"
    echo "--- ${TAG} raw metrics ---"
    cat "$OUT/analysis_results.json"
    echo
}

# eukaryota 877k (phylum ~ kingdom here); corroborates the headline scale.
run_one eukaryota_canonical taxonomy_edges_eukaryota_2759_clean.mapping.tsv phylum class order family

# cellular 1.1M (all three domains) — THE headline. Pass both domain-rank labels
# (NCBI switched superkingdom -> domain); the script skips whichever is absent.
run_one cellular_canonical taxonomy_edges_cellular_organisms_131567_clean.mapping.tsv domain superkingdom phylum class order family

echo "=== done — JSON + PNGs under artifacts/tags/{eukaryota_canonical,cellular_canonical}/regen_analysis_${STAMP}/ ==="
