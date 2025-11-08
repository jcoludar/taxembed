#!/bin/bash
# Quick-start script for hierarchical training

echo "Starting hierarchical Poincaré training..."
echo "This will train with:"
echo "  - 975K ancestor-descendant pairs (not just 100K parent-child)"
echo "  - Depth-aware initialization"
echo "  - Radial regularization"
echo "  - Hard negative sampling"
echo "  - Depth weighting"
echo "  - Early stopping (patience=3 epochs)"
echo ""

python train_hierarchical.py \
    --data data/taxonomy_edges_small_transitive.pkl \
    --checkpoint taxonomy_model_hierarchical_small.pth \
    --dim 10 \
    --epochs 10000 \
    --early-stopping 3 \
    --batch-size 64 \
    --n-negatives 50 \
    --lr 0.005 \
    --margin 0.2 \
    --lambda-reg 0.1 \
    --gpu -1

echo ""
echo "Training complete! Now run analysis:"
echo "  python analyze_hierarchy_hyperbolic.py"
