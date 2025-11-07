#!/usr/bin/env python3
"""
UMAP visualization with ALL Primates colored distinctly.
Uses NCBI taxonomy to identify all primate TaxIDs.
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
    m = pd.read_csv(map_path, sep="\t", dtype={"taxid": str, "idx": int})
    tax2idx = dict(zip(m["taxid"], m["idx"]))
    idx2tax = dict(zip(m["idx"], m["taxid"]))
    return tax2idx, idx2tax

def get_all_descendants(taxid, nodes_path="data/nodes.dmp"):
    """Get all descendants of a taxid (all children, grandchildren, etc.)."""
    descendants = set([taxid])
    to_process = [taxid]
    
    try:
        # Build child map
        children_map = {}
        with open(nodes_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t|\t')
                if len(parts) >= 2:
                    child_id = parts[0].strip()
                    parent_id = parts[1].strip()
                    if parent_id not in children_map:
                        children_map[parent_id] = []
                    children_map[parent_id].append(child_id)
        
        # BFS to get all descendants
        while to_process:
            current = to_process.pop(0)
            if current in children_map:
                for child in children_map[current]:
                    if child not in descendants:
                        descendants.add(child)
                        to_process.append(child)
    except Exception as e:
        print(f"Warning: Could not build full primate tree: {e}")
    
    return descendants

def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "taxonomy_model_full.pth"
    map_path = sys.argv[2] if len(sys.argv) > 2 else "data/taxonomy_edges.mapping.tsv"
    
    print(f"Loading embeddings from {ckpt_path}...")
    emb = load_embeddings(ckpt_path)
    print(f"Embeddings shape: {emb.shape}")
    
    print(f"Loading mapping from {map_path}...")
    tax2idx, idx2tax = load_mapping(map_path)
    
    # Get all primates (TaxID 9443 is the root of Primates order)
    print("\nIdentifying all Primates (this takes ~30 seconds)...")
    primate_taxids = get_all_descendants("9443")
    primate_indices = set()
    for taxid in primate_taxids:
        if taxid in tax2idx:
            primate_indices.add(tax2idx[taxid])
    
    print(f"Found {len(primate_indices)} primates in the dataset")
    
    # Sample for visualization
    print("\nSampling 25,000 points for visualization...")
    np.random.seed(42)
    sample_indices = np.random.choice(emb.shape[0], min(25000, emb.shape[0]), replace=False)
    emb_sample = emb[sample_indices]
    
    # Assign colors: Primates vs Others
    print("Assigning colors...")
    colors = []
    labels = []
    for idx in sample_indices:
        if idx in primate_indices:
            colors.append('#FF1493')  # Deep pink for primates
            labels.append('Primates')
        else:
            colors.append('#CCCCCC')  # Light gray for others
            labels.append('Other')
    
    colors = np.array(colors)
    labels = np.array(labels)
    
    # Compute UMAP
    print("\nRunning UMAP (this takes ~2-3 minutes)...")
    umap_proj = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, 
                     random_state=42, n_jobs=4).fit_transform(emb_sample)
    
    # Create visualization
    print("Creating visualization...")
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Plot Others first (background)
    mask_other = labels == 'Other'
    ax.scatter(umap_proj[mask_other, 0], umap_proj[mask_other, 1], 
              c='#CCCCCC', s=15, alpha=0.3, label=f'Other ({mask_other.sum()} organisms)')
    
    # Plot Primates on top
    mask_primate = labels == 'Primates'
    ax.scatter(umap_proj[mask_primate, 0], umap_proj[mask_primate, 1], 
              c='#FF1493', s=30, alpha=0.8, label=f'Primates ({mask_primate.sum()} organisms)',
              edgecolors='darkred', linewidth=0.5)
    
    # Highlight specific primates with stars
    highlight_taxids = {
        '9606': ('Homo sapiens\n(Human)', 'darkred', 500),
        '10090': ('Mus musculus\n(Mouse)', 'blue', 400),  # Not primate, for reference
        '6239': ('C. elegans', 'green', 400),  # Not primate, for reference
    }
    
    for taxid, (name, color, size) in highlight_taxids.items():
        if taxid in tax2idx:
            idx = tax2idx[taxid]
            pos = np.where(sample_indices == idx)[0]
            if len(pos) > 0:
                pos = pos[0]
                ax.scatter(umap_proj[pos, 0], umap_proj[pos, 1], 
                          c=color, s=size, marker='*', edgecolors='black', 
                          linewidth=2, zorder=10)
                # Add label
                ax.annotate(name, (umap_proj[pos, 0], umap_proj[pos, 1]),
                           xytext=(15, 15), textcoords='offset points',
                           fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.8),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', lw=2))
    
    ax.set_xlabel('UMAP 1', fontsize=13, fontweight='bold')
    ax.set_ylabel('UMAP 2', fontsize=13, fontweight='bold')
    ax.set_title('Poincaré Embeddings - NCBI Taxonomy\nAll Primates Highlighted (Deep Pink)', 
                fontsize=15, fontweight='bold')
    ax.legend(loc='best', fontsize=12, framealpha=0.95, title='Groups', title_fontsize=13)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('umap_primates_highlighted.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: umap_primates_highlighted.png")
    
    # Print summary
    print("\n" + "="*80)
    print("VISUALIZATION SUMMARY")
    print("="*80)
    print(f"Total sampled points: {len(sample_indices):,}")
    print(f"Primates in sample: {mask_primate.sum():,} ({100*mask_primate.sum()/len(sample_indices):.1f}%)")
    print(f"Other organisms: {mask_other.sum():,} ({100*mask_other.sum()/len(sample_indices):.1f}%)")
    print(f"\nTotal primates in full model: {len(primate_indices):,}")
    print(f"Total organisms in model: {emb.shape[0]:,}")
    print(f"\nKey organisms highlighted:")
    for taxid, (name, _, _) in highlight_taxids.items():
        status = "✓" if taxid in tax2idx else "✗"
        is_primate = "PRIMATE" if taxid in primate_taxids else "other"
        print(f"  {status} {name:30s} (TaxID {taxid:6s}) - {is_primate}")

if __name__ == "__main__":
    main()
