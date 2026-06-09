#!/bin/bash
#SBATCH --job-name=taxembed_anomaly
#SBATCH --partition=lrz-v100x2
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/logs/anomaly_%j.out
#SBATCH --error=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/logs/anomaly_%j.err
#SBATCH --container-image=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/pytorch-2.4.0-cuda12.1.sqsh
#SBATCH --container-workdir=/app
#SBATCH --container-mounts=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/src:/app:rw,/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/data:/data:ro,/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/artifacts:/app/artifacts:rw

set -euo pipefail
export MKL_THREADING_LAYER=GNU   # MKL/libgomp threading-layer clash fix (see PROJECT_STATE.md)

# =============================================================================
# Application #2 — taxonomy QC / anomaly detection — GPU run on LRZ.
# Runs the device-aware (CUDA) per-node scorer + leg A (synthetic ROC by
# displacement) + leg C (incertae-sedis enrichment) on the GPU. Leg B (the
# NCBI release-diff headline) is a SEPARATE manual step — it needs an OLD
# archived taxdump + the training/old/new dates (see the tail of this file).
#
# Usage (positional, defaults to eukaryota):
#   sbatch analyze_lrz_anomaly.sh <TAG> <MAPPING_BASENAME>
#   eukaryota: sbatch analyze_lrz_anomaly.sh eukaryota_canonical taxonomy_edges_eukaryota_2759_clean.mapping.tsv
#   cellular:  sbatch analyze_lrz_anomaly.sh cellular_canonical  taxonomy_edges_cellular_organisms_131567_clean.mapping.tsv
#
# PREREQ on LRZ (one-time): the taxdump must be staged at /data/taxdump_current/
# (names.dmp + nodes.dmp). Stage it with scripts/_extract_taxdump_dmps.py from a
# new_taxdump.tar.gz scp'd to /data, e.g.:
#   python scripts/_extract_taxdump_dmps.py --tarball /data/new_taxdump.tar.gz --out-dir /data/taxdump_current
#
# Re-derive EVERY rank-dependent baseline on the dataset at hand — do NOT carry a
# eukaryota constant into the cellular figure (spec §9C cellular rank-label trap).
# Use the `final` (ep200) checkpoint (the unsuffixed .pth), never `best`.
# =============================================================================

TAG="${1:-eukaryota_canonical}"
MAPPING_BASENAME="${2:-taxonomy_edges_eukaryota_2759_clean.mapping.tsv}"

CKPT="/app/artifacts/tags/${TAG}/${TAG}.pth"
MAPPING="/data/${MAPPING_BASENAME}"
NAMES_DMP="/data/taxdump_current/names.dmp"
OUT="/app/artifacts/tags/${TAG}/anomaly"

echo "=== TaxEmbed Application #2 (anomaly QC) — GPU run ==="
echo "Job: ${SLURM_JOB_ID:-?}, Host: $(hostname), TAG=${TAG}"
nvidia-smi -L || true
python --version
echo

cd /app
pip install --no-cache-dir -e . 2>&1 | tail -3
echo

# --- preflight (cheap; fail loud BEFORE GPU time) ---
for f in "$CKPT" "$MAPPING" "$NAMES_DMP"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: required input missing: $f" >&2
        echo "  (TAG=$TAG; cellular only after job 5673097 lands; taxdump must be staged — see header)" >&2
        exit 1
    fi
done

echo "=== Step 1: per-node score + BH-FDR (rank=family) ==="
python scripts/taxonomy_anomaly.py \
    --checkpoint "$CKPT" --mapping "$MAPPING" --data-dir /data \
    --rank family --k 10 --n-null 200 --n-bins 5 --seed 0 \
    --device cuda --knn-batch 2048 \
    -o "$OUT"
echo

echo "=== Step 2: leg A — synthetic ROC stratified by displacement ==="
python scripts/_anomaly_validation.py roc \
    --checkpoint "$CKPT" --mapping "$MAPPING" --data-dir /data \
    --rank family --k 10 --n-relocate 3000 --n-null 200 --seed 0 \
    --device cuda --knn-batch 2048 \
    -o "$OUT"
echo

echo "=== Step 3: leg C — incertae-sedis / environmental enrichment ==="
python scripts/_anomaly_validation.py enrichment \
    --pool-npz "$OUT/anomaly_pool.npz" --mapping "$MAPPING" \
    --names-dmp "$NAMES_DMP" --top-frac 0.1 \
    -o "$OUT"
echo

echo "=== done — outputs in $OUT ==="
ls -la "$OUT"

# =============================================================================
# Leg B (HEADLINE) — NCBI release-diff — run MANUALLY after staging an OLD archive.
# Needs: an archived taxdump ~3 yr older than training (e.g. taxdmp_2022-01-01.zip)
# staged at /data/taxdump_archive_<old>/, plus the three dates with the no-leakage
# ordering training_date <= old_date < new_date enforced by the CLI:
#
#   python scripts/_anomaly_validation.py releasediff \
#     --pool-npz $OUT/anomaly_pool.npz --mapping $MAPPING \
#     --old-nodes /data/taxdump_archive_2022/nodes.dmp \
#     --old-merged /data/taxdump_archive_2022/merged.dmp \
#     --old-delnodes /data/taxdump_archive_2022/delnodes.dmp \
#     --new-nodes /data/taxdump_current/nodes.dmp \
#     --new-merged /data/taxdump_current/merged.dmp \
#     --new-delnodes /data/taxdump_current/delnodes.dmp \
#     --training-date <TRAINING-DUMP-DATE> --old-date <OLD> --new-date <CURRENT-DUMP-DATE> \
#     --top-frac 0.1 --n-bins 5 -o $OUT
# =============================================================================
