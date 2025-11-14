#!/usr/bin/env python3
"""Comprehensive diagnosis of training issues."""

import pickle
import pandas as pd
import numpy as np

print("=" * 80)
print("COMPREHENSIVE DIAGNOSIS")
print("=" * 80)

# Load training data
with open('data/taxonomy_edges_small_transitive.pkl', 'rb') as f:
    training_data = pickle.load(f)

# Load mapping
df = pd.read_csv("data/taxonomy_edges_small.mapping.tsv", sep="\t", header=None, names=["taxid", "idx"])
df['idx'] = pd.to_numeric(df['idx'], errors='coerce')
df = df.dropna()
df['idx'] = df['idx'].astype(int)

all_indices_in_mapping = set(df['idx'])
print(f"\nAll indices in mapping file: {len(all_indices_in_mapping):,}")
print(f"  Range: {min(all_indices_in_mapping)} to {max(all_indices_in_mapping)}")

# Get nodes from training data (what train_small.py does)
ancestor_indices = set(item['ancestor_idx'] for item in training_data)
descendant_indices = set(item['descendant_idx'] for item in training_data)
all_indices_in_training = ancestor_indices | descendant_indices

n_nodes_from_training = max(all_indices_in_training) + 1  # What train_small.py uses
n_nodes_from_mapping = max(all_indices_in_mapping) + 1    # What it SHOULD use

print(f"\nIndices appearing in training data: {len(all_indices_in_training):,}")
print(f"  Range: {min(all_indices_in_training)} to {max(all_indices_in_training)}")
print(f"  n_nodes calculated (max+1): {n_nodes_from_training:,}")

print(f"\nExpected n_nodes from mapping: {n_nodes_from_mapping:,}")
print(f"MISMATCH: {n_nodes_from_mapping - n_nodes_from_training:,} nodes missing from model!")

# Find missing indices
missing_indices = all_indices_in_mapping - all_indices_in_training
print(f"\nMissing {len(missing_indices):,} indices:")
print(f"  They exist in mapping but never appear in training data")

# Check what these missing indices are
only_ancestors = ancestor_indices - descendant_indices
only_descendants = descendant_indices - ancestor_indices
both = ancestor_indices & descendant_indices

print(f"\nNode roles in training data:")
print(f"  Only ancestors (never descendants): {len(only_ancestors):,}")
print(f"  Only descendants (never ancestors): {len(only_descendants):,}")
print(f"  Both: {len(both):,}")
print(f"  Total in training: {len(all_indices_in_training):,}")
print(f"  Never appear: {len(missing_indices):,}")

# Analyze missing nodes
if len(missing_indices) < 100:
    print(f"\nMissing indices: {sorted(missing_indices)[:20]}")
else:
    print(f"\nFirst 20 missing indices: {sorted(missing_indices)[:20]}")
    print(f"Last 20 missing indices: {sorted(missing_indices)[-20:]}")

# Check gaps in training data indices
max_idx_training = max(all_indices_in_training)
expected_range = set(range(max_idx_training + 1))
gaps_in_training = expected_range - all_indices_in_training

print(f"\nGaps in training indices (0 to {max_idx_training}):")
print(f"  {len(gaps_in_training):,} indices missing from continuous range")
if len(gaps_in_training) < 50:
    print(f"  Gaps: {sorted(gaps_in_training)}")

print("\n" + "=" * 80)
print("ROOT CAUSE ANALYSIS")
print("=" * 80)

print("\n1. BUG IN train_small.py (line 346-347):")
print("   Current: n_nodes = max(training_data indices) + 1")
print("   Problem: Excludes nodes not in training pairs")
print("   Fix: n_nodes = max(mapping indices) + 1")

print("\n2. BOUNDARY COMPRESSION:")
print("   90% of embeddings at norm > 0.9997")
print("   Problem: No space for hierarchical structure")
print("   Causes:")
print("   - Radial regularizer pushing to specific radii")
print("   - Ball projection clamping at 1.0")
print("   - Deep pairs (74%) all pushed near boundary")

print("\n3. DATA IMBALANCE:")
print("   Parent-child: 6%, Deep pairs (>5): 74%")
print("   Problem: Model overfits to deep relationships")
print("   Effect: Local structure (siblings, parents) lost")

print("\n" + "=" * 80)
print("RECOMMENDED FIXES")
print("=" * 80)

print("\n1. CRITICAL - Fix n_nodes calculation:")
print("   Replace lines 346-347 in train_small.py with:")
print("   ```")
print("   # Load mapping to get true n_nodes")
print("   mapping_df = pd.read_csv('data/taxonomy_edges_small.mapping.tsv',")
print("                            sep='\\t', header=None, names=['taxid', 'idx'])")
print("   mapping_df['idx'] = pd.to_numeric(mapping_df['idx'], errors='coerce')")
print("   mapping_df = mapping_df.dropna()")
print("   n_nodes = int(mapping_df['idx'].max()) + 1")
print("   ```")

print("\n2. URGENT - Relax boundary constraint:")
print("   Change project_to_ball max radius from 0.999.. to 0.98")
print("   Leave room for hierarchical spread")

print("\n3. IMPORTANT - Balance training data:")
print("   Option A: Oversample parent-child pairs (weight them 10x)")
print("   Option B: Cap depth to 5-10 levels")
print("   Option C: Curriculum learning (start with depth 1-2, gradually add deeper)")

print("\n4. OPTIONAL - Reduce radial regularization:")
print("   Try lambda_reg = 0.01 instead of 0.1")
print("   Or disable completely for first epochs")
