#!/bin/bash
#SBATCH --job-name=taxembed_cellular_canonical
#SBATCH --partition=lrz-v100x2
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/logs/cellular_canonical_%j.out
#SBATCH --error=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/logs/cellular_canonical_%j.err
#SBATCH --container-image=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/pytorch-2.4.0-cuda12.1.sqsh
#SBATCH --container-workdir=/app
#SBATCH --container-mounts=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/src:/app:rw,/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/data:/data:ro,/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/artifacts:/app/artifacts:rw

set -euo pipefail

# MKL/libgomp threading-layer clash fix (load-bearing — see PROJECT_STATE.md).
export MKL_THREADING_LAYER=GNU

# =============================================================================
# CELLULAR ORGANISMS (taxid 131567) — the TRUE all-Life run: Bacteria + Archaea +
# Eukaryota, NO viruses (polyphyletic, no shared root w/ cellular life). Built locally
# with `taxembed build 131567 --clean`: 2.625M raw -> 1,102,163 clean nodes (58% noise
# stripped), 21,399,053 transitive pairs, max depth 40. Domain balance verified
# (Bacteria 216k / Archaea 7k / Eukaryota 878k; Euk count matches the standalone
# eukaryota_2759_clean build = consistency check). Residual name-noise 0.023% (internal
# "environmental samples"/informal-bacterium containers w/ real children — topology only).
#
# Recipe = the LOCKED canonical recipe, UNCHANGED. 21.4M pairs is only ~1.13x eukaryota's
# 18.9M (which ran 20h30m), so epoch_fraction 0.3 + eff-batch 2048 stays put and the run
# fits the 48h walltime with margin (est. ~23-27h). Do NOT shrink eff-batch (that fix is
# for SMALL clades like mollusca); this is the LARGEST clade we have.
#
# ⚠ WATCH (same as eukaryota): if per-rank separation STALLS around ep80-120 (read the
# milestones), enable the hard-negative sampler OR bump --n-negatives 300 -> 500. At
# eukaryota the ep100-150 window was a soft plateau, NOT a collapse, then climbed hard to
# family 8.23x by ep200 — expect the same shape. Use `final` (ep200), NOT `best`.
# =============================================================================

echo "=== TaxEmbed CELLULAR ORGANISMS (canonical recipe, true all-Life scale) ==="
echo "Job: ${SLURM_JOB_ID}, Host: $(hostname)"
nvidia-smi -L
python --version
echo

cd /app
pip install --no-cache-dir -e . 2>&1 | tail -3
echo

if [ ! -f /data/taxonomy_edges_cellular_organisms_131567_clean_transitive.npz ]; then
    echo "ERROR: cellular dataset not found at /data — build locally + scp it up first (see header)." >&2
    exit 1
fi

echo "=== train ==="
python -m taxembed.cli.main train \
    --file /data/taxonomy_edges_cellular_organisms_131567_clean_transitive.npz \
    --mapping /data/taxonomy_edges_cellular_organisms_131567_clean.mapping.tsv \
    -as cellular_canonical \
    --dim 100 \
    --epochs 200 \
    --batch-size 256 \
    --n-negatives 300 \
    --lr 0.001 \
    --gpu 0 \
    --amp \
    --grad-accum-steps 8 \
    --curriculum \
    --curriculum-phases auto \
    --lr-schedule cosine_warmrestart \
    --warm-restart-on-phase \
    --lr-min-multiplier 0.01 \
    --early-stopping 999 \
    --radial-nudge 0.05 \
    --radial-schedule log \
    --depth-scale-margin \
    --margin-min 0.05 \
    --margin-max 1.0 \
    --epoch-fraction 0.3 \
    --euclidean-param \
    --loss softmax \
    --save-every 10

echo
echo "=== training complete ==="
ls -la /app/artifacts/tags/cellular_canonical/
