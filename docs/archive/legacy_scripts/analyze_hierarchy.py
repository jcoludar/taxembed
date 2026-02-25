#!/usr/bin/env python3
"""
Analyze hierarchical structure in embeddings.
Check if organisms within the same high-level taxon (phylum/class/order)
are closer to each other than to organisms from different taxa.
"""

import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt


def load_embeddings(ckpt_path):
    """Load embeddings from checkpoint."""
    print(f"Loading embeddings from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        emb = sd["lt.weight"].detach().cpu().numpy()
    elif "embeddings" in ckpt:
        emb = ckpt["embeddings"].cpu().numpy()
    else:
        raise ValueError("Cannot find embeddings in checkpoint")
    
    print(f"  ✓ Shape: {emb.shape}")
    return emb


def load_mapping(mapping_file):
    """Load index to TaxID mapping."""
    print(f"Loading mapping from {mapping_file}...")
    df = pd.read_csv(mapping_file, sep="\t", header=None, names=["idx", "taxid"])
    numeric_df = df[df["taxid"].str.isnumeric()]
    idx2tax = dict(zip(numeric_df["idx"], numeric_df["taxid"]))
    print(f"  ✓ Loaded {len(idx2tax):,} mappings")
    return idx2tax


def load_taxonomy_ranks(valid_taxids):
    """Load taxonomy with ranks for valid TaxIDs."""
    print("Loading taxonomy tree with ranks...")
    
    # Load names
    names = {}
    with open("data/names.dmp", "r") as f:
        for line in f:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 4 and parts[3] == "scientific name":
                taxid = int(parts[0])
                if taxid in valid_taxids:
                    names[taxid] = parts[1]
    
    # Load nodes with ranks
    taxonomy = {}
    with open("data/nodes.dmp", "r") as f:
        for line in f:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 5:
                taxid = int(parts[0])
                if taxid in valid_taxids:
                    parent = int(parts[1])
                    rank = parts[2]
                    taxonomy[taxid] = {
                        "parent": parent,
                        "rank": rank,
                        "name": names.get(taxid, f"TaxID_{taxid}")
                    }
    
    print(f"  ✓ Loaded {len(taxonomy):,} taxonomy nodes")
    return taxonomy


def get_ancestor_at_rank(taxid, taxonomy, target_rank):
    """Find the ancestor of a taxid at a specific rank."""
    visited = set()
    current = taxid
    
    while current in taxonomy and current not in visited:
        visited.add(current)
        node = taxonomy[current]
        
        if node["rank"] == target_rank:
            return current
        
        parent = node["parent"]
        if parent == current:  # Root
            break
        current = parent
    
    return None


def analyze_hierarchical_clustering(emb, idx2tax, taxonomy, rank="phylum"):
    """Analyze if organisms cluster by taxonomic rank."""
    print(f"\n{'='*80}")
    print(f"HIERARCHICAL CLUSTERING ANALYSIS - {rank.upper()}")
    print(f"{'='*80}\n")
    
    # Get valid taxids
    valid_taxids = set(int(t) for t in idx2tax.values())
    
    # Map each organism to its ancestor at target rank
    organism_to_group = {}
    group_names = {}
    
    for idx, taxid_str in idx2tax.items():
        taxid = int(taxid_str)
        ancestor = get_ancestor_at_rank(taxid, taxonomy, rank)
        if ancestor:
            organism_to_group[idx] = ancestor
            if ancestor not in group_names and ancestor in taxonomy:
                group_names[ancestor] = taxonomy[ancestor]["name"]
    
    print(f"Found {len(set(organism_to_group.values()))} distinct {rank}s")
    print(f"Mapped {len(organism_to_group)}/{len(idx2tax)} organisms to {rank} level")
    
    # Group organisms by their ancestor
    # Only include indices that are valid for the embedding matrix
    max_idx = emb.shape[0] - 1
    groups = defaultdict(list)
    for idx, group_id in organism_to_group.items():
        idx_int = int(idx) if isinstance(idx, str) else idx
        if idx_int <= max_idx:
            groups[group_id].append(idx_int)
        else:
            print(f"Warning: idx {idx} out of bounds (max={max_idx})")
    
    # Filter to groups with at least 10 members for meaningful analysis
    min_size = 10
    large_groups = {gid: indices for gid, indices in groups.items() if len(indices) >= min_size}
    
    print(f"\n{rank.capitalize()}s with ≥{min_size} organisms: {len(large_groups)}")
    
    # Show top groups by size
    group_sizes = [(gid, len(indices), group_names.get(gid, f"TaxID_{gid}")) 
                   for gid, indices in large_groups.items()]
    group_sizes.sort(key=lambda x: -x[1])
    
    print(f"\nTop 20 {rank}s by organism count:")
    for i, (gid, size, name) in enumerate(group_sizes[:20], 1):
        print(f"  {i:2d}. {name:40s}: {size:6,} organisms")
    
    # Compute distances
    print(f"\nComputing pairwise distances...")
    
    # Sample for efficiency if needed
    max_per_group = 500
    sampled_groups = {}
    for gid, indices in large_groups.items():
        indices_list = list(indices) if not isinstance(indices, list) else indices
        if len(indices_list) > max_per_group:
            sampled_groups[gid] = list(np.random.choice(indices_list, max_per_group, replace=False))
        else:
            sampled_groups[gid] = indices_list
    
    # Calculate intra-group vs inter-group distances
    intra_distances = []
    inter_distances = []
    
    group_list = list(sampled_groups.items())
    
    for i, (gid1, indices1) in enumerate(group_list):
        # Intra-group distances
        if len(indices1) >= 2:
            # Convert to numpy array of embeddings
            group_emb = np.array([emb[int(idx)] for idx in indices1])
            dists = pdist(group_emb)
            intra_distances.extend(dists)
        
        # Inter-group distances (sample to avoid O(n²) explosion)
        for j, (gid2, indices2) in enumerate(group_list[i+1:], i+1):
            # Sample pairs for efficiency
            idx1_sample = list(np.random.choice(indices1, min(100, len(indices1)), replace=False))
            idx2_sample = list(np.random.choice(indices2, min(100, len(indices2)), replace=False))
            
            for idx1 in idx1_sample[:10]:  # Limit per group pair
                for idx2 in idx2_sample[:10]:
                    dist = np.linalg.norm(emb[int(idx1)] - emb[int(idx2)])
                    inter_distances.append(dist)
    
    intra_distances = np.array(intra_distances)
    inter_distances = np.array(inter_distances)
    
    print(f"\n{'='*80}")
    print(f"DISTANCE STATISTICS")
    print(f"{'='*80}")
    print(f"\nIntra-{rank} distances (within same {rank}):")
    print(f"  Count: {len(intra_distances):,}")
    print(f"  Mean:  {np.mean(intra_distances):.6f}")
    print(f"  Std:   {np.std(intra_distances):.6f}")
    print(f"  Min:   {np.min(intra_distances):.6f}")
    print(f"  Max:   {np.max(intra_distances):.6f}")
    print(f"  Median: {np.median(intra_distances):.6f}")
    
    print(f"\nInter-{rank} distances (between different {rank}s):")
    print(f"  Count: {len(inter_distances):,}")
    print(f"  Mean:  {np.mean(inter_distances):.6f}")
    print(f"  Std:   {np.std(inter_distances):.6f}")
    print(f"  Min:   {np.min(inter_distances):.6f}")
    print(f"  Max:   {np.max(inter_distances):.6f}")
    print(f"  Median: {np.median(inter_distances):.6f}")
    
    # Separation score
    separation = np.mean(inter_distances) / np.mean(intra_distances)
    print(f"\n{'='*80}")
    print(f"HIERARCHICAL QUALITY METRICS")
    print(f"{'='*80}")
    print(f"\nSeparation Ratio (higher is better):")
    print(f"  Inter-{rank} / Intra-{rank} = {separation:.3f}x")
    
    if separation > 2.0:
        quality = "✅ EXCELLENT"
    elif separation > 1.5:
        quality = "✅ GOOD"
    elif separation > 1.2:
        quality = "⚠️ MODERATE"
    else:
        quality = "❌ POOR"
    
    print(f"  Quality: {quality}")
    
    # Overlap analysis
    overlap = np.sum(intra_distances > np.median(inter_distances)) / len(intra_distances) * 100
    print(f"\nDistance overlap:")
    print(f"  {overlap:.1f}% of intra-{rank} distances exceed median inter-{rank} distance")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Histogram
    ax = axes[0]
    bins = np.linspace(0, max(np.max(intra_distances), np.max(inter_distances)), 50)
    ax.hist(intra_distances, bins=bins, alpha=0.6, label=f'Intra-{rank}', color='blue', density=True)
    ax.hist(inter_distances, bins=bins, alpha=0.6, label=f'Inter-{rank}', color='red', density=True)
    ax.axvline(np.mean(intra_distances), color='blue', linestyle='--', linewidth=2, label=f'Intra mean: {np.mean(intra_distances):.3f}')
    ax.axvline(np.mean(inter_distances), color='red', linestyle='--', linewidth=2, label=f'Inter mean: {np.mean(inter_distances):.3f}')
    ax.set_xlabel('Euclidean Distance', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Distance Distribution by {rank.capitalize()}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Box plot
    ax = axes[1]
    data_to_plot = [intra_distances, inter_distances]
    bp = ax.boxplot(data_to_plot, labels=[f'Intra-{rank}', f'Inter-{rank}'], 
                    patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax.set_ylabel('Euclidean Distance', fontsize=12)
    ax.set_title(f'{rank.capitalize()}-level Clustering Quality', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add separation ratio text
    ax.text(0.5, 0.95, f'Separation Ratio: {separation:.2f}x\n{quality}',
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            horizontalalignment='center')
    
    plt.tight_layout()
    output_file = f'hierarchy_analysis_{rank}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot: {output_file}")
    plt.close()
    
    return {
        'intra_mean': np.mean(intra_distances),
        'inter_mean': np.mean(inter_distances),
        'separation': separation,
        'quality': quality,
        'num_groups': len(large_groups),
        'num_organisms': sum(len(indices) for indices in large_groups.values())
    }


def main():
    # Load data
    checkpoint = "taxonomy_model_small_early_stop_epoch2341.pth"
    mapping_file = "data/taxonomy_edges_small.mapping.tsv"
    
    emb = load_embeddings(checkpoint)
    idx2tax = load_mapping(mapping_file)
    
    # Load taxonomy
    valid_taxids = set(int(t) for t in idx2tax.values())
    taxonomy = load_taxonomy_ranks(valid_taxids)
    
    # Analyze at different taxonomic levels
    results = {}
    for rank in ["phylum", "class", "order", "family"]:
        try:
            results[rank] = analyze_hierarchical_clustering(emb, idx2tax, taxonomy, rank=rank)
        except Exception as e:
            print(f"\nError analyzing {rank}: {e}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY ACROSS RANKS")
    print(f"{'='*80}\n")
    print(f"{'Rank':<12} {'Groups':>8} {'Organisms':>10} {'Separation':>12} {'Quality':>15}")
    print("-" * 80)
    for rank, res in results.items():
        print(f"{rank.capitalize():<12} {res['num_groups']:>8} {res['num_organisms']:>10,} "
              f"{res['separation']:>11.2f}x {res['quality']:>15}")


if __name__ == "__main__":
    main()
