#!/usr/bin/env python3
"""
Evaluate trained Poincaré embeddings and create UMAP visualization.
"""
import torch
import numpy as np
import pandas as pd
from umap import UMAP
import matplotlib.pyplot as plt
import sys

def load_embeddings(ckpt_path):
    """Load embeddings from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"]
    emb = sd["lt.weight"].detach().cpu().numpy()
    return emb

def load_mapping(map_path):
    """Load TaxID to index mapping."""
    m = pd.read_csv(map_path, sep="\t")
    tax2idx = dict(zip(m["taxid"].astype(str), m["idx"]))
    idx2tax = dict(zip(m["idx"], m["taxid"].astype(str)))
    return tax2idx, idx2tax

def nearest_neighbors(emb, idx, k=10):
    """Find k nearest neighbors using Euclidean distance."""
    x = emb[idx]
    d = np.linalg.norm(emb - x, axis=1)
    nbrs = np.argsort(d)[:k+1]
    nbrs = [j for j in nbrs if j != idx][:k]
    return nbrs, d[nbrs]

def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "taxonomy_model_full.pth"
    map_path = sys.argv[2] if len(sys.argv) > 2 else "data/taxonomy_edges.mapping.tsv"
    
    print(f"Loading embeddings from {ckpt_path}...")
    emb = load_embeddings(ckpt_path)
    print(f"Embeddings shape: {emb.shape}")
    
    print(f"Loading mapping from {map_path}...")
    tax2idx, idx2tax = load_mapping(map_path)
    
    # Test with a few known organisms
    test_taxids = ["9606", "10090", "6239", "7227", "562"]  # Human, Mouse, C.elegans, Drosophila, E.coli
    test_names = {
        "9606": "Homo sapiens (Human)",
        "10090": "Mus musculus (Mouse)",
        "6239": "Caenorhabditis elegans (C. elegans)",
        "7227": "Drosophila melanogaster (Fruit fly)",
        "562": "Escherichia coli (E. coli)"
    }
    
    print("\n" + "="*80)
    print("NEAREST NEIGHBORS ANALYSIS")
    print("="*80)
    
    for taxid in test_taxids:
        if taxid not in tax2idx:
            print(f"\n⚠️  TaxID {taxid} not in dataset")
            continue
        
        idx = tax2idx[taxid]
        nbrs, dists = nearest_neighbors(emb, idx, k=5)
        
        print(f"\n{test_names.get(taxid, f'TaxID {taxid}')}:")
        for rank, (j, d) in enumerate(zip(nbrs, dists), 1):
            neighbor_taxid = idx2tax[j]
            print(f"  {rank}. TaxID {neighbor_taxid} (distance: {d:.6f})")
    
    # UMAP projection
    print("\n" + "="*80)
    print("COMPUTING UMAP PROJECTION (this may take a few minutes)...")
    print("="*80)
    
    # Sample for visualization if dataset is large
    if emb.shape[0] > 10000:
        print(f"Sampling 10,000 points for visualization...")
        sample_idx = np.random.choice(emb.shape[0], 10000, replace=False)
        emb_sample = emb[sample_idx]
        idx_map = {i: sample_idx[i] for i in range(len(sample_idx))}
    else:
        emb_sample = emb
        idx_map = {i: i for i in range(len(emb))}
    
    print("Running UMAP...")
    umap_proj = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(emb_sample)
    
    print("Creating visualization...")
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Color by some property (e.g., distance from human)
    if "9606" in tax2idx:
        human_idx = tax2idx["9606"]
        if human_idx in idx_map.values():
            # Find position in sample
            human_pos = [k for k, v in idx_map.items() if v == human_idx][0]
            colors = np.linalg.norm(umap_proj - umap_proj[human_pos], axis=1)
            scatter = ax.scatter(umap_proj[:, 0], umap_proj[:, 1], c=colors, cmap='viridis', s=10, alpha=0.6)
            plt.colorbar(scatter, ax=ax, label='Distance from Human')
        else:
            ax.scatter(umap_proj[:, 0], umap_proj[:, 1], s=10, alpha=0.6)
    else:
        ax.scatter(umap_proj[:, 0], umap_proj[:, 1], s=10, alpha=0.6)
    
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('Poincaré Embeddings - NCBI Taxonomy (UMAP Projection)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('umap_projection.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: umap_projection.png")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Total nodes: {emb.shape[0]}")
    print(f"Embedding dimension: {emb.shape[1]}")
    print(f"UMAP projection saved to: umap_projection.png")

if __name__ == "__main__":
    main()
