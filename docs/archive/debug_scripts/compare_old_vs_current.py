#!/usr/bin/env python3
"""Deep comparison: Old model (28 epoch, before fixes) vs Current model (35 epoch, after fixes)."""

import torch
import numpy as np
import pickle

print("=" * 80)
print("DEEP COMPARISON: OLD (28 epoch) vs CURRENT (35 epoch)")
print("=" * 80)

# Load old model (before fixes)
old_path = "small_model_28epoch/taxonomy_model_small_best.pth"
print(f"\n📦 OLD MODEL (before fixes)")
print(f"   Path: {old_path}")

old_model = torch.load(old_path, map_location='cpu')
embs_old = old_model['embeddings'].detach().numpy()
old_epoch = old_model['epoch']
old_loss = old_model['loss']

print(f"   Epoch: {old_epoch}")
print(f"   Loss: {old_loss:.6f}")
print(f"   Shape: {embs_old.shape}")

# Load current model (after fixes)
current_path = "taxonomy_model_small_best.pth"
print(f"\n📦 CURRENT MODEL (after fixes)")
print(f"   Path: {current_path}")

current_model = torch.load(current_path, map_location='cpu')
embs_current = current_model['embeddings'].detach().numpy()
current_epoch = current_model['epoch']
current_loss = current_model['loss']

print(f"   Epoch: {current_epoch}")
print(f"   Loss: {current_loss:.6f}")
print(f"   Shape: {embs_current.shape}")

# Load training data to identify trained nodes
print("\n📊 Loading training info...")
with open('data/taxonomy_edges_small_transitive.pkl', 'rb') as f:
    training_data = pickle.load(f)

ancestors = set(item['ancestor_idx'] for item in training_data)
descendants = set(item['descendant_idx'] for item in training_data)
trained_indices = ancestors | descendants

print(f"   Nodes in training data: {len(trained_indices):,}")

# Calculate norms
norms_old = np.linalg.norm(embs_old, axis=1)
norms_current = np.linalg.norm(embs_current, axis=1)

# For OLD: Only look at nodes that exist
old_n_nodes = embs_old.shape[0]
old_trained = [i for i in trained_indices if i < old_n_nodes]
norms_old_trained = norms_old[old_trained]

# For CURRENT: Separate trained vs untrained
current_trained = [i for i in trained_indices if i < embs_current.shape[0]]
current_untrained = [i for i in range(embs_current.shape[0]) if i not in trained_indices]
norms_current_trained = norms_current[current_trained]
norms_current_untrained = norms_current[current_untrained] if current_untrained else np.array([])

print("\n" + "=" * 80)
print("COMPARISON: TRAINED NODES ONLY")
print("=" * 80)

print(f"\n📊 OLD MODEL (trained nodes only, {len(old_trained):,} nodes):")
print(f"   Min norm:   {norms_old_trained.min():.4f}")
print(f"   Mean norm:  {norms_old_trained.mean():.4f}")
print(f"   Max norm:   {norms_old_trained.max():.4f}")
print(f"   Std dev:    {norms_old_trained.std():.4f}")
print(f"   >0.90: {(norms_old_trained > 0.90).sum():,} ({100*(norms_old_trained > 0.90).sum()/len(norms_old_trained):.1f}%)")
print(f"   >0.95: {(norms_old_trained > 0.95).sum():,} ({100*(norms_old_trained > 0.95).sum()/len(norms_old_trained):.1f}%)")

print(f"\n📊 CURRENT MODEL (trained nodes only, {len(current_trained):,} nodes):")
print(f"   Min norm:   {norms_current_trained.min():.4f}")
print(f"   Mean norm:  {norms_current_trained.mean():.4f}")
print(f"   Max norm:   {norms_current_trained.max():.4f}")
print(f"   Std dev:    {norms_current_trained.std():.4f}")
print(f"   >0.90: {(norms_current_trained > 0.90).sum():,} ({100*(norms_current_trained > 0.90).sum()/len(norms_current_trained):.1f}%)")
print(f"   >0.95: {(norms_current_trained > 0.95).sum():,} ({100*(norms_current_trained > 0.95).sum()/len(norms_current_trained):.1f}%)")

# Variance analysis (spread)
print("\n📊 EMBEDDING SPREAD (Variance):")
print(f"   OLD:     {embs_old.var():.6f}")
print(f"   CURRENT (all): {embs_current.var():.6f}")
print(f"   CURRENT (trained only): {embs_current[current_trained].var():.6f}")

# Percentiles
print("\n📊 NORM PERCENTILES (Trained nodes only):")
percentiles = [25, 50, 75, 90, 95, 99]
print(f"\n   {'Percentile':<12} {'OLD':<10} {'CURRENT':<10} {'Δ':<10}")
print(f"   {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
for p in percentiles:
    old_val = np.percentile(norms_old_trained, p)
    current_val = np.percentile(norms_current_trained, p)
    delta = current_val - old_val
    print(f"   {p}%{'':<10} {old_val:.4f}     {current_val:.4f}     {delta:+.4f}")

# Loss comparison
print("\n📊 TRAINING LOSS:")
print(f"   OLD (epoch {old_epoch}):     {old_loss:.6f}")
print(f"   CURRENT (epoch {current_epoch}): {current_loss:.6f}")
print(f"   Δ: {current_loss - old_loss:+.6f} ({100*(current_loss - old_loss)/old_loss:+.1f}%)")

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

# Check if embeddings are more compressed
mean_diff = norms_current_trained.mean() - norms_old_trained.mean()
p90_diff = np.percentile(norms_current_trained, 90) - np.percentile(norms_old_trained, 90)
boundary_old = (norms_old_trained > 0.90).sum() / len(norms_old_trained)
boundary_current = (norms_current_trained > 0.90).sum() / len(norms_current_trained)

issues = []

if mean_diff > 0.05:
    issues.append(f"⚠️  Mean norm increased by {mean_diff:.3f} - embeddings pushed to boundary")

if p90_diff > 0.05:
    issues.append(f"⚠️  90th percentile increased by {p90_diff:.3f} - more compression")

if boundary_current > boundary_old + 0.1:
    issues.append(f"⚠️  Boundary clustering increased by {100*(boundary_current - boundary_old):.1f}%")

if norms_current_trained.std() < norms_old_trained.std() * 0.8:
    issues.append(f"⚠️  Std dev decreased - less spread in radial direction")

if embs_current[current_trained].var() < embs_old.var() * 0.8:
    issues.append(f"⚠️  Total variance decreased - embeddings more clustered")

if issues:
    print("\n🔴 PROBLEMS FOUND:\n")
    for issue in issues:
        print(f"   {issue}")
else:
    print("\n✅ CURRENT model looks comparable or better")

print("\n💡 LIKELY CAUSES:")

# Check hyperparameters
print("\n1. **Regularization strength**")
print("   - If λ increased, embeddings get pushed to target radii")
print("   - Check: Was λ=0.1 → 0.01 → back to higher?")

print("\n2. **Initialization**")
print("   - If init range changed, nodes start closer to boundary")
print("   - OLD init: likely [0.1, 0.95]")
print("   - CURRENT init: [0.05, 0.85]")

print("\n3. **Number of nodes**")
print("   - OLD: {old_n_nodes:,} nodes")
print(f"   - CURRENT: {embs_current.shape[0]:,} nodes")
print("   - More nodes = more crowding")

print("\n4. **Training data**")
print(f"   - OLD: {len(training_data) - 26590:,} pairs (approx)")
print(f"   - CURRENT: {len(training_data):,} pairs")
print("   - More pairs might cause different dynamics")

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

print("\n1. **Revert to OLD hyperparameters temporarily**")
print("   - Use same λ, init range, max_norm as old model")
print("   - This isolates whether the data fix is the problem")

print("\n2. **Check what changed between models**")
print("   - Compare train_small.py and train_hierarchical.py")
print("   - Look for λ, init range, max_norm, projection frequency")

print("\n3. **Train longer with weaker regularization**")
print("   - Current λ might be too strong")
print("   - Try λ=0.01 for 100+ epochs")

print("\n4. **Adjust initialization**")
print("   - Current range [0.05, 0.85] might be too conservative")
print("   - Try [0.1, 0.90] to match old model")

print("\n" + "=" * 80)
