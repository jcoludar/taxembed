#!/usr/bin/env python3
"""
Universal visualization tool for Poincaré embeddings.

Features:
- Works with any checkpoint
- Can highlight specific taxonomic groups (primates, mammals, bacteria, etc.)
- Supports different sampling strategies
- Generates UMAP projections
- Nearest neighbor analysis

Usage:
    python scripts/visualize_embeddings.py <checkpoint.pth> [options]
    
Examples:
    # Basic visualization
    python scripts/visualize_embeddings.py model.pth
    
    # Highlight primates
    python scripts/visualize_embeddings.py model.pth --highlight primates
    
    # Highlight mammals with custom sample size
    python scripts/visualize_embeddings.py model.pth --highlight mammals --sample 50000
    
    # Only visualize specific group
    python scripts/visualize_embeddings.py model.pth --only primates
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
    """Load NCBI taxonomy tree structure, optionally filtered to valid TaxIDs.
    
    Args:
        valid_taxids: Set of TaxIDs to include. If None, loads all.
    """
    try:
        # Load names
        names = {}
        with open("data/names.dmp", "r") as f:
            for line in f:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 4 and parts[3] == "scientific name":
                    taxid = int(parts[0])
                    # Only load if in valid set or loading all
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
                    rank = parts[2]
                    # Only load if in valid set or loading all
                    if valid_taxids is None or taxid in valid_taxids:
                        nodes[taxid] = {"parent": parent, "rank": rank, "name": names.get(taxid, "")}
        
        if valid_taxids:
            print(f"  ✓ Loaded {len(nodes):,} taxonomy nodes (filtered to dataset)")
        else:
            print(f"  ✓ Loaded {len(nodes):,} taxonomy nodes")
        return nodes
    
    except FileNotFoundError:
        print("  ⚠️  Taxonomy files not found (data/nodes.dmp, data/names.dmp)")
        return {}


def find_taxonomic_group(nodes, group_name):
    """Find all TaxIDs in a taxonomic group."""
    # Common groups and their NCBI TaxIDs
    group_roots = {
        "primates": 9443,
        "mammals": 40674,
        "mammalia": 40674,
        "vertebrates": 7742,
        "bacteria": 2,
        "archaea": 2157,
        "fungi": 4751,
        "plants": 33090,
        "insects": 50557,
        "rodents": 9989,
    }
    
    group_name = group_name.lower()
    if group_name not in group_roots:
        print(f"  ⚠️  Unknown group: {group_name}")
        print(f"  Available groups: {', '.join(group_roots.keys())}")
        return set()
    
    root_taxid = group_roots[group_name]
    
    # Find all descendants
    print(f"Finding {group_name} (root TaxID: {root_taxid})...")
    group_taxids = set()
    
    def find_descendants(taxid):
        group_taxids.add(taxid)
        for child_id, child_info in nodes.items():
            if child_info["parent"] == taxid:
                find_descendants(child_id)
    
    find_descendants(root_taxid)
    print(f"  ✓ Found {len(group_taxids):,} {group_name}")
    return group_taxids


def nearest_neighbors(emb, idx, k=10):
    """Find k nearest neighbors."""
    x = emb[idx]
    d = np.linalg.norm(emb - x, axis=1)
    nbrs = np.argsort(d)[:k+1]
    nbrs = [j for j in nbrs if j != idx][:k]
    return nbrs, d[nbrs]


def visualize_embeddings(emb, indices, colors, labels, title, output_file, sample_size=None):
    """Create UMAP visualization."""
    # Sample if needed
    if sample_size and len(indices) > sample_size:
        print(f"Sampling {sample_size:,} points from {len(indices):,} total...")
        sample_idx = np.random.choice(len(indices), sample_size, replace=False)
        indices = [indices[i] for i in sample_idx]
        colors = [colors[i] for i in sample_idx]
    
    # Extract embeddings
    sample_emb = emb[indices]
    
    # Run UMAP
    print(f"Running UMAP on {len(indices):,} points...")
    umap_model = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    projection = umap_model.fit_transform(sample_emb)
    
    # Plot
    print("Creating visualization...")
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Plot points
    for label, color in labels.items():
        mask = np.array([c == color for c in colors])
        if mask.sum() > 0:
            ax.scatter(
                projection[mask, 0],
                projection[mask, 1],
                c=color,
                s=30 if label == "Highlighted" else 20,
                alpha=0.7 if label == "Highlighted" else 0.3,
                label=f"{label} (n={mask.sum():,})",
                edgecolors="darkred" if label == "Highlighted" else "none",
                linewidth=0.5 if label == "Highlighted" else 0,
            )
    
    ax.set_xlabel("UMAP 1", fontsize=14)
    ax.set_ylabel("UMAP 2", fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.legend(loc="best", fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    print(f"✓ Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Poincaré embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("checkpoint", help="Path to checkpoint file")
    parser.add_argument(
        "--mapping",
        help="Path to mapping file (auto-detected if not specified)",
    )
    parser.add_argument(
        "--highlight",
        help="Taxonomic group to highlight (primates, mammals, bacteria, etc.)",
    )
    parser.add_argument(
        "--only",
        help="Only show specific taxonomic group",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=25000,
        help="Number of points to sample (default: 25000)",
    )
    parser.add_argument(
        "--output",
        help="Output filename (default: auto-generated)",
    )
    parser.add_argument(
        "--nearest",
        type=int,
        default=5,
        help="Show N nearest neighbors for key organisms (default: 5)",
    )
    
    args = parser.parse_args()
    
    # Load embeddings
    emb = load_embeddings(args.checkpoint)
    n_nodes = emb.shape[0]
    
    # Auto-detect mapping file
    if not args.mapping:
        checkpoint_path = Path(args.checkpoint)
        
        # Try to infer from checkpoint name
        if "small" in checkpoint_path.name:
            args.mapping = "data/taxonomy_edges_small.mapping.tsv"
        else:
            args.mapping = "data/taxonomy_edges.mapping.tsv"
    
    # Load mapping
    tax2idx, idx2tax = load_mapping(args.mapping)
    
    # Get valid TaxIDs from the mapping (only organisms in training data)
    valid_taxids = set(int(taxid) for taxid in idx2tax.values())
    print(f"Dataset contains {len(valid_taxids):,} unique organisms")
    
    # Load taxonomy if highlighting (filtered to dataset organisms only)
    nodes = {}
    if args.highlight or args.only:
        print("Loading taxonomy tree...")
        nodes = load_taxonomy_tree(valid_taxids=valid_taxids)
    
    # Determine which indices to visualize
    if args.only:
        # Only show specific group
        group_taxids = find_taxonomic_group(nodes, args.only)
        indices = [idx for idx, tax in idx2tax.items() if int(tax) in group_taxids]
        colors = ["red"] * len(indices)
        labels = {"Highlighted": "red"}
        title = f"{args.only.capitalize()} - UMAP Projection\n({len(indices):,} organisms)"
    elif args.highlight:
        # Show all, highlight specific group
        group_taxids = find_taxonomic_group(nodes, args.highlight)
        indices = list(range(n_nodes))
        colors = ["red" if idx2tax.get(idx) and int(idx2tax[idx]) in group_taxids else "lightgray" for idx in indices]
        n_highlighted = colors.count("red")
        labels = {"Highlighted": "red", "Other": "lightgray"}
        title = f"Embedding Visualization - {args.highlight.capitalize()} Highlighted\n({n_highlighted:,} highlighted, {n_nodes-n_highlighted:,} other)"
    else:
        # Show all
        indices = list(range(min(n_nodes, args.sample)))
        colors = ["steelblue"] * len(indices)
        labels = {"All": "steelblue"}
        title = f"Embedding Visualization\n({len(indices):,} organisms)"
    
    # Generate output filename
    if not args.output:
        checkpoint_name = Path(args.checkpoint).stem
        if args.only:
            args.output = f"umap_{checkpoint_name}_{args.only}_only.png"
        elif args.highlight:
            args.output = f"umap_{checkpoint_name}_{args.highlight}_highlighted.png"
        else:
            args.output = f"umap_{checkpoint_name}.png"
    
    # Visualize
    visualize_embeddings(
        emb, indices, colors, labels, title, args.output, args.sample if not args.only else None
    )
    
    # Nearest neighbors analysis
    if tax2idx and args.nearest > 0:
        print(f"\n{'='*70}")
        print("NEAREST NEIGHBORS ANALYSIS")
        print(f"{'='*70}\n")
        
        key_organisms = {
            "9606": "Homo sapiens (Human)",
            "10090": "Mus musculus (Mouse)",
            "6239": "Caenorhabditis elegans",
            "7227": "Drosophila melanogaster",
            "562": "Escherichia coli",
        }
        
        for taxid, name in key_organisms.items():
            if taxid in tax2idx:
                idx = tax2idx[taxid]
                nbrs, dists = nearest_neighbors(emb, idx, args.nearest)
                print(f"{name}:")
                for i, (nbr, dist) in enumerate(zip(nbrs, dists), 1):
                    nbr_tax = idx2tax.get(nbr, "Unknown")
                    print(f"  {i}. TaxID {nbr_tax} (distance: {dist:.6f})")
                print()
    
    print(f"\n{'='*70}")
    print("✅ VISUALIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
