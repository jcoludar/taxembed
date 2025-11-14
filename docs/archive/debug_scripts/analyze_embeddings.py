#!/usr/bin/env python3
"""Quick analysis of embedding quality and distribution."""

import torch
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict

print("=" * 80)
print("EMBEDDING ANALYSIS")
print("=" * 80)

# 1. Check embeddings
print("\n1. EMBEDDING STATISTICS")
ckpt = torch.load('taxonomy_model_small_best.pth', map_location='cpu')
embs = ckpt['embeddings']
norms = embs.norm(dim=1).detach().numpy()

print(f"Shape: {embs.shape}")
print(f"Norms: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}")
print(f"Outside ball (>1.0): {(norms > 1.0).sum()} / {len(norms)} = {100*(norms > 1.0).mean():.2f}%")
print(f"Near boundary (>0.9): {(norms > 0.9).sum()} / {len(norms)} = {100*(norms > 0.9).mean():.2f}%")

percentiles = [0, 10, 25, 50, 75, 90, 99, 100]
print("\nNorm distribution (percentiles):")
for p in percentiles:
    print(f"  {p:3d}th: {np.percentile(norms, p):.4f}")

# 2. Check training data
print("\n" + "=" * 80)
print("2. TRAINING DATA ANALYSIS")
with open('data/taxonomy_edges_small_transitive.pkl', 'rb') as f:
    training_data = pickle.load(f)

depths = [item['depth_diff'] for item in training_data]
print(f"Total pairs: {len(training_data):,}")
print(f"Depth range: {min(depths)} to {max(depths)}")
print(f"Mean depth: {np.mean(depths):.2f}, Median: {np.median(depths):.1f}")

from collections import Counter
depth_counts = Counter(depths)
print("\nTop 10 depth differences:")
for depth, count in depth_counts.most_common(10):
    pct = 100 * count / len(training_data)
    print(f"  Depth {depth:2d}: {count:7,} pairs ({pct:5.1f}%)")

parent_child = sum(1 for d in depths if d == 1)
print(f"\nParent-child (depth=1): {parent_child:,} ({100*parent_child/len(training_data):.1f}%)")
deep_pairs = sum(1 for d in depths if d > 5)
print(f"Deep pairs (depth>5): {deep_pairs:,} ({100*deep_pairs/len(training_data):.1f}%)")

# 3. Check mapping
print("\n" + "=" * 80)
print("3. MAPPING ANALYSIS")
df = pd.read_csv("data/taxonomy_edges_small.mapping.tsv", sep="\t", header=None, names=["taxid", "idx"])
# Convert to numeric
df['idx'] = pd.to_numeric(df['idx'], errors='coerce')
df['taxid'] = pd.to_numeric(df['taxid'], errors='coerce')
df = df.dropna()
df['idx'] = df['idx'].astype(int)
df['taxid'] = df['taxid'].astype(int)

print(f"Mapping entries: {len(df):,}")
print(f"Unique indices: {df['idx'].nunique():,}")
print(f"Unique TaxIDs: {df['taxid'].nunique():,}")
print(f"Index range: {df['idx'].min()} to {df['idx'].max()}")

# Check if indices are continuous
expected = set(range(df['idx'].max() + 1))
actual = set(df['idx'])
missing = expected - actual
if missing:
    print(f"WARNING: {len(missing)} missing indices!")
else:
    print("✓ Indices are continuous")

# 4. Check for issues
print("\n" + "=" * 80)
print("4. POTENTIAL ISSUES")

issues = []

# Issue 1: Too many embeddings vs nodes
n_embs = embs.shape[0]
n_nodes = df['idx'].max() + 1
if n_embs != n_nodes:
    issues.append(f"Mismatch: {n_embs:,} embeddings but {n_nodes:,} nodes in mapping")
    print(f"⚠️  {issues[-1]}")

# Issue 2: Extreme norms
if norms.max() > 0.999:
    issues.append(f"Embeddings too close to boundary: max norm = {norms.max():.6f}")
    print(f"⚠️  {issues[-1]}")

# Issue 3: Data imbalance
if parent_child < len(training_data) * 0.1:
    issues.append(f"Very few parent-child pairs: only {100*parent_child/len(training_data):.1f}%")
    print(f"⚠️  {issues[-1]}")

# Issue 4: Norms too concentrated
norm_std = norms.std()
if norm_std < 0.1:
    issues.append(f"Norms too uniform: std = {norm_std:.4f}")
    print(f"⚠️  {issues[-1]}")

if not issues:
    print("✓ No major issues detected")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Embeddings: {embs.shape[0]:,} × {embs.shape[1]}")
print(f"Training pairs: {len(training_data):,}")
print(f"Norm range: [{norms.min():.3f}, {norms.max():.3f}]")
print(f"Issues found: {len(issues)}")
