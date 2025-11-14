#!/usr/bin/env python3
"""Visualize only the nodes that were actually trained (appeared in training pairs)."""

import torch
import numpy as np
import pandas as pd
import pickle
import umap
import matplotlib.pyplot as plt
from collections import defaultdict
import sys

print("=" * 80)
print("VISUALIZATION: TRAINED NODES ONLY")
print("=" * 80)

# Load checkpoint
checkpoint_file = sys.argv[1] if len(sys.argv) > 1 else "taxonomy_model_small_best.pth"
print(f"\nLoading embeddings from {checkpoint_file}...")

model = torch.load(checkpoint_file, map_location='cpu')
if isinstance(model, dict) and 'embeddings' in model:
    embeddings = model['embeddings'].detach().numpy()
else:
    print("Error: Unexpected checkpoint format")
    sys.exit(1)

print(f"  ✓ Total embeddings: {embeddings.shape}")

# Load training data to identify trained nodes
print("\nLoading training data to identify trained nodes...")
with open('data/taxonomy_edges_small_transitive.pkl', 'rb') as f:
    training_data = pickle.load(f)

ancestors = set(item['ancestor_idx'] for item in training_data)
descendants = set(item['descendant_idx'] for item in training_data)
trained_indices = sorted(ancestors | descendants)

print(f"  ✓ Nodes in training: {len(trained_indices):,}")
print(f"  ✗ Nodes NOT in training: {embeddings.shape[0] - len(trained_indices):,}")

# Filter to trained nodes only
embeddings_trained = embeddings[trained_indices]
norms = np.linalg.norm(embeddings_trained, axis=1)

print(f"\n📊 Trained nodes statistics:")
print(f"  Shape: {embeddings_trained.shape}")
print(f"  Norm range: [{norms.min():.3f}, {norms.max():.3f}]")
print(f"  Norm mean: {norms.mean():.3f}")
print(f"  At boundary (>0.95): {(norms > 0.95).sum()} ({100*(norms > 0.95).sum()/len(norms):.1f}%)")

# Load mapping
print("\nLoading taxonomy mapping...")
mapping_df = pd.read_csv("data/taxonomy_edges_small.mapping.tsv",
                         sep="\t", header=None, names=["taxid", "idx"])
mapping_df['idx'] = pd.to_numeric(mapping_df['idx'], errors='coerce')
mapping_df['taxid'] = pd.to_numeric(mapping_df['taxid'], errors='coerce')
mapping_df = mapping_df.dropna()
mapping_df['idx'] = mapping_df['idx'].astype(int)
mapping_df['taxid'] = mapping_df['taxid'].astype(int)

# Load taxonomy tree
print("Loading taxonomy tree...")
idx_to_taxid = dict(zip(mapping_df['idx'], mapping_df['taxid']))
valid_taxids = set(idx_to_taxid.values())

# Load nodes (parent relationships)
nodes = {}
with open("data/nodes.dmp", "r") as f:
    for line in f:
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 2:
            taxid = int(parts[0])
            parent = int(parts[1])
            if taxid in valid_taxids:
                nodes[taxid] = parent

print(f"  ✓ Loaded {len(nodes):,} taxonomy nodes")

# Identify groups
print("\nFinding taxonomic groups...")

def find_group_descendants(root_taxid, nodes, idx_to_taxid, trained_indices):
    """Find all descendants of a taxonomic group in trained indices."""
    group_indices = []
    
    def find_descendants(taxid):
        # Check if this taxid is in our trained nodes
        for idx in trained_indices:
            if idx_to_taxid.get(idx) == taxid:
                group_indices.append(idx)
        
        # Find children
        for child, parent in nodes.items():
            if parent == taxid:
                find_descendants(child)
    
    find_descendants(root_taxid)
    return set(group_indices)

groups = {
    'Mammals': 40674,
    'Birds': 8782,
    'Insects': 50557,
    'Bacteria': 2,
    'Fungi': 4751,
    'Plants': 33090
}

group_indices = {}
for name, root_taxid in groups.items():
    indices = find_group_descendants(root_taxid, nodes, idx_to_taxid, trained_indices)
    group_indices[name] = indices
    print(f"  ✓ {name}: {len(indices):,} organisms")

# Sample if needed
sample_size = 25000
if len(trained_indices) > sample_size:
    print(f"\nSampling {sample_size:,} from {len(trained_indices):,} trained nodes...")
    sample_idx = np.random.choice(len(trained_indices), sample_size, replace=False)
    embeddings_plot = embeddings_trained[sample_idx]
    sampled_indices = [trained_indices[i] for i in sample_idx]
else:
    embeddings_plot = embeddings_trained
    sampled_indices = trained_indices
    sample_idx = np.arange(len(trained_indices))

print(f"  ✓ Using {len(embeddings_plot):,} points")

# Assign colors
colors = []
color_map = {
    'Mammals': '#e74c3c',
    'Birds': '#f1c40f',
    'Insects': '#2ecc71',
    'Bacteria': '#9b59b6',
    'Fungi': '#e67e22',
    'Plants': '#1abc9c'
}

for idx in sampled_indices:
    assigned = False
    for group_name, group_set in group_indices.items():
        if idx in group_set:
            colors.append(color_map[group_name])
            assigned = True
            break
    if not assigned:
        colors.append('#bdc3c7')

# UMAP
print(f"\nRunning UMAP on {len(embeddings_plot):,} points...")
reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    metric='euclidean',
    random_state=42
)
embedding_2d = reducer.fit_transform(embeddings_plot)

# Plot
print("Creating visualization...")
plt.figure(figsize=(16, 12))
plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
           c=colors, s=1, alpha=0.6, rasterized=True)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=color_map[name], label=f"{name} ({len(group_indices[name]):,})")
                   for name in groups.keys() if len(group_indices[name]) > 0]
legend_elements.append(Patch(facecolor='#bdc3c7', label='Other'))
plt.legend(handles=legend_elements, loc='upper right', fontsize=10)

plt.title(f'Poincaré Embeddings - TRAINED NODES ONLY ({len(trained_indices):,} nodes)\n'
          f'Excluding {embeddings.shape[0] - len(trained_indices):,} untrained nodes',
          fontsize=16, pad=20)
plt.xlabel('UMAP 1', fontsize=12)
plt.ylabel('UMAP 2', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

output_file = 'taxonomy_embeddings_trained_only.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✅ Saved: {output_file}")
print(f"\n   Showing: {len(trained_indices):,} trained nodes")
print(f"   Hidden:  {embeddings.shape[0] - len(trained_indices):,} untrained nodes")
print("\n" + "=" * 80)
