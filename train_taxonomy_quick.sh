#!/bin/bash
# Quick test training on a subset of taxonomy data
# Uses only first 100k edges for fast iteration

head -100001 data/taxonomy_edges.csv > data/taxonomy_edges_small.csv

python3 embed.py \
  -dim 10 \
  -lr 0.3 \
  -epochs 10 \
  -negs 50 \
  -burnin 5 \
  -ndproc 2 \
  -model distance \
  -manifold poincare \
  -dset data/taxonomy_edges_small.csv \
  -checkpoint taxonomy_model_small.pth \
  -batchsize 32 \
  -eval_each 2 \
  -fresh \
  -sparse \
  -train_threads 1
