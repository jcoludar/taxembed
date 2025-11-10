#!/usr/bin/env python3
"""
Visualize multiple taxonomic groups on the same plot with different colors.

Usage:
    python visualize_multi_groups.py taxonomy_model_small_best.pth
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from umap import UMAP


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


def load_mapping(map_path):
    """Load TaxID to index mapping."""
    if not Path(map_path).exists():
        print(f"⚠️  Mapping file not found: {map_path}")
        return None, None
    
    print(f"Loading mapping from {map_path}...")
    df = pd.read_csv(map_path, sep="\t", dtype={"taxid": str, "idx": int})
    
    # Filter out non-numeric taxids
    numeric_df = df[df["taxid"].str.isnumeric()]
    tax2idx = dict(zip(numeric_df["taxid"], numeric_df["idx"]))
    idx2tax = dict(zip(numeric_df["idx"], numeric_df["taxid"]))
    
    print(f"  ✓ Loaded {len(tax2idx):,} mappings")
    return tax2idx, idx2tax


def load_taxonomy_tree(valid_taxids=None):
    """Load NCBI taxonomy tree structure."""
    try:
        # Load names
        names = {}
        with open("data/names.dmp", "r") as f:
            for line in f:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 4 and parts[3] == "scientific name":
                    taxid = int(parts[0])
                    if valid_taxids is None or taxid in valid_taxids:
                        names[taxid] = parts[1]
        
        # Load nodes (parent relationships)
        nodes = {}
        with open("data/nodes.dmp", "r") as f:
            for line in f:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 5:
                    taxid = int(parts[0])
                    parent = int(parts[1])
                    if valid_taxids is None or taxid in valid_taxids:
                        nodes[taxid] = parent
        
        print(f"Loading taxonomy tree...")
        print(f"  ✓ Loaded {len(nodes):,} taxonomy nodes (filtered to dataset)")
        
        return names, nodes
    
    except Exception as e:
        print(f"⚠️  Could not load taxonomy tree: {e}")
        return None, None


# Taxonomic groups with their root TaxIDs
TAXONOMIC_GROUPS = {
    "Mammals": 40674,
    "Birds": 8782,
    "Insects": 50557,
    "Bacteria": 2,
    "Fungi": 4751,
    "Plants": 33090,
}

# Color palette for groups
COLORS = {
    "Mammals": "#e74c3c",        # Red
    "Birds": "#f1c40f",          # Yellow
    "Insects": "#2ecc71",        # Green  
    "Bacteria": "#9b59b6",       # Purple
    "Fungi": "#e67e22",          # Orange
    "Plants": "#1abc9c",         # Teal
    "Other": "#bdc3c7",          # Light Gray
}


def find_group_descendants(root_taxid, nodes, tax2idx):
    """Find all descendants of a taxonomic group."""
    group_indices = set()
    
    def find_descendants(taxid):
        # Convert taxid to string for lookup
        taxid_str = str(taxid)
        if taxid_str in tax2idx:
            group_indices.add(tax2idx[taxid_str])
        
        # Find children
        for child, parent in nodes.items():
            if parent == taxid:
                find_descendants(child)
    
    find_descendants(root_taxid)
    return group_indices


def visualize_multi_groups(emb, tax2idx, idx2tax, nodes, names, sample_size=25000, output_file=None):
    """Create UMAP visualization with multiple highlighted groups."""
    
    # Find members of each group
    print("\nFinding taxonomic groups...")
    group_members = {}
    for group_name, root_taxid in TAXONOMIC_GROUPS.items():
        if root_taxid in nodes or root_taxid == 1:  # 1 is root of all life
            indices = find_group_descendants(root_taxid, nodes, tax2idx)
            group_members[group_name] = indices
            print(f"  ✓ {group_name}: {len(indices):,} organisms")
        else:
            print(f"  ⚠️  {group_name}: root TaxID {root_taxid} not found in taxonomy tree")
            group_members[group_name] = set()
    
    # Assign colors to all indices
    n_total = emb.shape[0]
    idx_to_group = {}
    
    for idx in range(n_total):
        assigned = False
        for group_name, members in group_members.items():
            if idx in members:
                idx_to_group[idx] = group_name
                assigned = True
                break
        if not assigned:
            idx_to_group[idx] = "Other"
    
    # Sample points
    print(f"\nSampling {sample_size:,} points from {n_total:,} total...")
    all_indices = list(range(n_total))
    
    # Stratified sampling: ensure each group is represented
    sampled_indices = []
    samples_per_group = {}
    
    for group_name in list(TAXONOMIC_GROUPS.keys()) + ["Other"]:
        group_indices = [i for i in all_indices if idx_to_group[i] == group_name]
        if group_indices:
            # Sample proportionally
            n_sample = min(len(group_indices), max(1, int(len(group_indices) / n_total * sample_size)))
            sampled = np.random.choice(group_indices, n_sample, replace=False)
            sampled_indices.extend(sampled)
            samples_per_group[group_name] = len(sampled)
    
    # If we haven't reached sample_size, add more from "Other"
    if len(sampled_indices) < sample_size:
        other_indices = [i for i in all_indices if i not in sampled_indices]
        additional = np.random.choice(other_indices, 
                                     min(len(other_indices), sample_size - len(sampled_indices)), 
                                     replace=False)
        sampled_indices.extend(additional)
    
    sampled_indices = np.array(sampled_indices)
    
    print(f"  ✓ Sampled {len(sampled_indices):,} points")
    for group_name in list(TAXONOMIC_GROUPS.keys()) + ["Other"]:
        if group_name in samples_per_group:
            print(f"     - {group_name}: {samples_per_group[group_name]:,}")
    
    # Extract embeddings
    sample_emb = emb[sampled_indices]
    
    # Run UMAP
    print(f"\nRunning UMAP on {len(sampled_indices):,} points...")
    umap_model = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    projection = umap_model.fit_transform(sample_emb)
    
    # Plot
    print("Creating visualization...")
    fig, ax = plt.subplots(figsize=(18, 14))
    
    # Plot each group
    for group_name in ["Other"] + list(TAXONOMIC_GROUPS.keys()):
        mask = np.array([idx_to_group[idx] == group_name for idx in sampled_indices])
        n_points = mask.sum()
        
        if n_points > 0:
            color = COLORS.get(group_name, "#cccccc")
            
            # Other group: smaller, more transparent
            if group_name == "Other":
                ax.scatter(
                    projection[mask, 0],
                    projection[mask, 1],
                    c=color,
                    s=15,
                    alpha=0.2,
                    label=f"{group_name} (n={n_points:,})",
                    edgecolors="none",
                    zorder=1
                )
            else:
                # Highlighted groups: larger, more visible
                ax.scatter(
                    projection[mask, 0],
                    projection[mask, 1],
                    c=color,
                    s=50,
                    alpha=0.8,
                    label=f"{group_name} (n={n_points:,})",
                    edgecolors="black",
                    linewidth=0.5,
                    zorder=2
                )
    
    ax.set_xlabel("UMAP 1", fontsize=16)
    ax.set_ylabel("UMAP 2", fontsize=16)
    ax.set_title("Taxonomy Embeddings - Multiple Groups Highlighted", fontsize=20, fontweight="bold")
    ax.legend(loc="best", fontsize=13, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file is None:
        output_file = "taxonomy_embeddings_multi_groups.png"
    
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    print(f"\n✅ Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Visualize multiple taxonomic groups")
    parser.add_argument("checkpoint", help="Path to checkpoint file")
    parser.add_argument("--mapping", help="Path to mapping file (auto-detected if not specified)")
    parser.add_argument("--sample", type=int, default=25000, help="Number of points to sample")
    parser.add_argument("--output", help="Output filename")
    
    args = parser.parse_args()
    
    # Load embeddings
    emb = load_embeddings(args.checkpoint)
    
    # Auto-detect mapping file
    if args.mapping is None:
        # Try to find mapping file
        for candidate in ["data/taxonomy_edges_small.mapping.tsv", 
                         "data/taxonomy_edges.mapping.tsv"]:
            if Path(candidate).exists():
                args.mapping = candidate
                break
    
    if args.mapping is None:
        print("❌ Could not find mapping file. Please specify with --mapping")
        sys.exit(1)
    
    # Load mapping
    tax2idx, idx2tax = load_mapping(args.mapping)
    if tax2idx is None:
        sys.exit(1)
    
    # Convert to set of numeric taxids
    valid_taxids = set(int(t) for t in tax2idx.keys())
    print(f"Dataset contains {len(valid_taxids):,} unique organisms")
    
    # Load taxonomy tree
    names, nodes = load_taxonomy_tree(valid_taxids)
    if names is None or nodes is None:
        print("❌ Could not load taxonomy tree")
        sys.exit(1)
    
    # Visualize
    visualize_multi_groups(emb, tax2idx, idx2tax, nodes, names, 
                          sample_size=args.sample,
                          output_file=args.output)


if __name__ == "__main__":
    main()
