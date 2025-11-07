#!/bin/bash
# Train Poincaré embeddings on NCBI taxonomy data
# Uses Apple Silicon GPU (MPS) if available

python3 embed.py \
  -dim 10 \
  -lr 0.3 \
  -epochs 50 \
  -negs 50 \
  -burnin 10 \
  -ndproc 4 \
  -model distance \
  -manifold poincare \
  -dset data/taxonomy_edges.csv \
  -checkpoint taxonomy_model.pth \
  -batchsize 32 \
  -eval_each 5 \
  -fresh \
  -sparse \
  -train_threads 1
