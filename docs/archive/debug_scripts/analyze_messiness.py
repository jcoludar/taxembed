#!/usr/bin/env python3
"""Analyze why new embeddings might look messier in UMAP."""

import torch
import numpy as np
import pandas as pd
import pickle

print("=" * 80)
print("ANALYZING EMBEDDING STRUCTURE: OLD vs NEW")
print("=" * 80)

# Load checkpoints
old_ckpt = torch.load("taxonomy_model_small_epoch57.pth", map_location='cpu')
new_ckpt = torch.load("taxonomy_model_small_epoch36.pth", map_location='cpu')

embs_old = old_ckpt['embeddings'].detach().numpy()
embs_new = new_ckpt['embeddings'].detach().numpy()

# Load training data to identify node roles
with open('data/taxonomy_edges_small_transitive.pkl', 'rb') as f:
    training_data = pickle.load(f)

# Identify which nodes appear in training
ancestors = set(item['ancestor_idx'] for item in training_data)
descendants = set(item['descendant_idx'] for item in training_data)
in_training = ancestors | descendants

print(f"\n📊 DATASET COMPOSITION")
print(f"OLD model: {embs_old.shape[0]:,} nodes")
print(f"NEW model: {embs_new.shape[0]:,} nodes")
print(f"Nodes in training: {len(in_training):,}")
print(f"Missing from OLD: {embs_new.shape[0] - embs_old.shape[0]:,}")

# Analyze the nodes that are in NEW but not in OLD
print(f"\n📊 THE {embs_new.shape[0] - embs_old.shape[0]:,} RECOVERED NODES:")

# For NEW model, separate nodes by training presence
norms_new = np.linalg.norm(embs_new, axis=1)
norms_in_training = norms_new[list(in_training)]
not_in_training = set(range(embs_new.shape[0])) - in_training
norms_not_in_training = norms_new[list(not_in_training)]

print(f"\nNodes IN training ({len(in_training):,}):")
print(f"  Mean norm: {norms_in_training.mean():.4f}")
print(f"  Std dev:   {norms_in_training.std():.4f}")
print(f"  Range:     [{norms_in_training.min():.4f}, {norms_in_training.max():.4f}]")

print(f"\nNodes NOT in training ({len(not_in_training):,}):")
print(f"  Mean norm: {norms_not_in_training.mean():.4f}")
print(f"  Std dev:   {norms_not_in_training.std():.4f}")
print(f"  Range:     [{norms_not_in_training.min():.4f}, {norms_not_in_training.max():.4f}]")

# Check if missing nodes are all at max_depth (leaf nodes)
print(f"\n📊 DEPTH ANALYSIS OF MISSING NODES:")

# Build TaxID -> depth mapping
taxid_to_depth = {}
idx_to_taxid = {}
for item in training_data:
    taxid_to_depth[item['ancestor_taxid']] = item['ancestor_depth']
    taxid_to_depth[item['descendant_taxid']] = item['descendant_depth']

# Load mapping to get TaxIDs for missing nodes
mapping_df = pd.read_csv("data/taxonomy_edges_small.mapping.tsv",
                         sep="\t", header=None, names=["taxid", "idx"])
mapping_df['idx'] = pd.to_numeric(mapping_df['idx'], errors='coerce')
mapping_df['taxid'] = pd.to_numeric(mapping_df['taxid'], errors='coerce')
mapping_df = mapping_df.dropna()

# Check depths of nodes not in training
missing_depths = []
for idx in not_in_training:
    taxid = mapping_df[mapping_df['idx'] == idx]['taxid'].values[0]
    depth = taxid_to_depth.get(taxid, 37)  # default to max if not found
    missing_depths.append(depth)

missing_depths = np.array(missing_depths)
print(f"  Mean depth: {missing_depths.mean():.1f}")
print(f"  Median depth: {np.median(missing_depths):.1f}")
print(f"  At max depth (37): {(missing_depths == 37).sum():,} ({100*(missing_depths == 37).sum()/len(missing_depths):.1f}%)")

print(f"\n📊 EMBEDDING VARIANCE (Structure vs Noise):")
print(f"\nOLD model ({embs_old.shape[0]:,} nodes):")
print(f"  Variance per dimension: {embs_old.var(axis=0).mean():.6f}")
print(f"  Total variance: {embs_old.var():.6f}")

print(f"\nNEW model ({embs_new.shape[0]:,} nodes):")
# Separate trained vs untrained nodes
embs_trained = embs_new[list(in_training)]
embs_untrained = embs_new[list(not_in_training)]

print(f"  All nodes variance: {embs_new.var():.6f}")
print(f"  Trained nodes ({len(in_training):,}): {embs_trained.var():.6f}")
print(f"  Untrained nodes ({len(not_in_training):,}): {embs_untrained.var():.6f}")

print(f"\n📊 POTENTIAL CAUSES OF MESSINESS:")

issues = []

# Check 1: High variance in untrained nodes
if embs_untrained.var() > embs_trained.var() * 0.5:
    issues.append("⚠️  Untrained nodes have high variance (random initialization)")
    print(f"\n1. ⚠️  Untrained nodes variance ({embs_untrained.var():.6f}) is significant")
    print(f"   These {len(not_in_training):,} nodes were initialized at depth=37 but never trained")

# Check 2: All missing nodes at boundary
boundary_pct = (norms_not_in_training > 0.9).sum() / len(norms_not_in_training)
if boundary_pct > 0.8:
    issues.append(f"⚠️  {100*boundary_pct:.0f}% of untrained nodes clustered at boundary")
    print(f"\n2. ⚠️  {100*boundary_pct:.0f}% of untrained nodes at norm > 0.9")
    print(f"   They form a dense cluster at boundary, creating visual clutter")

# Check 3: Training epochs
old_epoch = old_ckpt['epoch']
new_epoch = new_ckpt['epoch']
if new_epoch < old_epoch * 0.7:
    issues.append(f"⚠️  NEW trained fewer epochs ({new_epoch} vs {old_epoch})")
    print(f"\n3. ⚠️  NEW trained for {new_epoch} epochs vs OLD {old_epoch} epochs")
    print(f"   Weaker regularization (λ=0.01) needs more time to organize structure")

# Check 4: Regularization strength
print(f"\n4. ℹ️  Regularization reduced: λ=0.1 → 0.01 (10x weaker)")
print(f"   Embeddings have more freedom but less enforced structure")

print(f"\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

if issues:
    print(f"\n🔴 Found {len(issues)} issues causing messiness:\n")
    for issue in issues:
        print(f"   {issue}")
else:
    print("\n✅ No obvious structural issues found")

print(f"\n💡 RECOMMENDATIONS:")

print(f"\n1. **Ignore untrained nodes in visualization**")
print(f"   Only visualize the {len(in_training):,} nodes that were actually trained")
print(f"   The {len(not_in_training):,} missing nodes are random/poorly initialized")

print(f"\n2. **Train longer with weaker regularization**")
print(f"   OLD: epoch {old_epoch} with λ=0.1 (strong guidance)")
print(f"   NEW: epoch {new_epoch} with λ=0.01 (weak guidance)")
print(f"   → NEW needs ~3x more epochs to converge with weaker λ")

print(f"\n3. **Increase regularization temporarily**")
print(f"   Try λ=0.05 (middle ground between 0.01 and 0.1)")
print(f"   Or train with λ=0.1 for first 50 epochs, then reduce")

print(f"\n4. **Better initialization for missing nodes**")
print(f"   Instead of depth=37 for all missing nodes,")
print(f"   Use actual taxonomy depth or set to depth=0 (center)")

print("\n" + "=" * 80)
