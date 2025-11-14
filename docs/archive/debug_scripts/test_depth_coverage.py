#!/usr/bin/env python3
"""Quick test to verify depth coverage fix."""

import pandas as pd
import pickle

print("Testing depth coverage fix...\n")

# Load training data
with open('data/taxonomy_edges_small_transitive.pkl', 'rb') as f:
    training_data = pickle.load(f)

# Load mapping
mapping_df = pd.read_csv("data/taxonomy_edges_small.mapping.tsv",
                         sep="\t", header=None, names=["taxid", "idx"])
mapping_df['idx'] = pd.to_numeric(mapping_df['idx'], errors='coerce')
mapping_df['taxid'] = pd.to_numeric(mapping_df['taxid'], errors='coerce')
mapping_df = mapping_df.dropna()
mapping_df['idx'] = mapping_df['idx'].astype(int)
mapping_df['taxid'] = mapping_df['taxid'].astype(int)
n_nodes = int(mapping_df['idx'].max()) + 1

max_depth = max(item['descendant_depth'] for item in training_data)

print(f"Total nodes in dataset: {n_nodes:,}")
print(f"Max depth: {max_depth}\n")

# Build depth mapping from training data (old way)
idx_to_depth_old = {}
for item in training_data:
    idx_to_depth_old[item['descendant_idx']] = item['descendant_depth']
    if item['ancestor_idx'] not in idx_to_depth_old:
        idx_to_depth_old[item['ancestor_idx']] = item['ancestor_depth']

print(f"OLD approach: {len(idx_to_depth_old):,} nodes with depths")

# Build TaxID -> depth mapping
taxid_to_depth = {}
for item in training_data:
    taxid_to_depth[item['ancestor_taxid']] = item['ancestor_depth']
    taxid_to_depth[item['descendant_taxid']] = item['descendant_depth']

print(f"TaxID -> depth mapping: {len(taxid_to_depth):,} TaxIDs")

# NEW approach: fill in missing nodes
idx_to_depth_new = dict(idx_to_depth_old)
missing_count = 0
assigned_from_taxid = 0
assigned_default = 0

for idx in range(n_nodes):
    if idx not in idx_to_depth_new:
        missing_count += 1
        # Find TaxID for this index
        taxid = mapping_df[mapping_df['idx'] == idx]['taxid'].values[0]
        if taxid in taxid_to_depth:
            idx_to_depth_new[idx] = taxid_to_depth[taxid]
            assigned_from_taxid += 1
        else:
            # Node not in taxonomy at all - likely a leaf, assign max depth
            idx_to_depth_new[idx] = max_depth
            assigned_default += 1

print(f"\nNEW approach: {len(idx_to_depth_new):,} nodes with depths")
print(f"  Missing from training: {missing_count:,}")
print(f"  Assigned via TaxID lookup: {assigned_from_taxid:,}")
print(f"  Assigned default (max_depth): {assigned_default:,}")

if len(idx_to_depth_new) == n_nodes:
    print(f"\n✅ SUCCESS: All {n_nodes:,} nodes have depth information!")
else:
    print(f"\n❌ PROBLEM: Only {len(idx_to_depth_new):,} / {n_nodes:,} nodes have depths")

print(f"\nExpected regularization message:")
print(f"  OLD: 'Regularizing {len(idx_to_depth_old):,} nodes'")
print(f"  NEW: 'Regularizing {len(idx_to_depth_new):,} nodes'")
