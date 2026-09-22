#!/bin/bash
#SBATCH --job-name=taxembed_task9_transplant
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/logs/task9_transplant_%j.out
#SBATCH --error=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/logs/task9_transplant_%j.err
#SBATCH --container-image=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/pytorch-2.4.0-cuda12.1.sqsh
#SBATCH --container-workdir=/app
#SBATCH --container-mounts=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/src_task8:/app:rw,/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/data:/data:ro,/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/artifacts:/app/artifacts:rw

set -euo pipefail
export MKL_THREADING_LAYER=GNU

# Radius-transplant 2x2 on the two Figure 4 runs at epoch 200 (CPU, read-only on checkpoints).
# Mounts src_task8 (has transplant.py); never the src tree array 5802007 runs from.

cd /app
pip install --no-cache-dir -e . 2>&1 | tail -3
python -m py_compile /app/scripts/radius_transplant_2x2.py /app/src/taxembed/eval/transplant.py

STAMP="$(date +%Y%m%d_%H%M%S)"
python /app/scripts/radius_transplant_2x2.py \
    --npz /data/taxonomy_edges_metazoa_33208_clean_transitive.npz \
    --prior-ckpt /app/artifacts/tags/metazoa_softmax_milestones/metazoa_softmax_milestones_milestone_epoch200.pth \
    --canonical-ckpt /app/artifacts/tags/metazoa_lower_lr_bigger_batch/metazoa_lower_lr_bigger_batch_milestone_epoch200.pth \
    --out "/app/artifacts/task9_scoring/transplant_2x2_${STAMP}.json" \
    --seeds 0 1 2 --n-sample 2000
