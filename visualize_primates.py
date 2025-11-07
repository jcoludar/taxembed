#!/usr/bin/env python3
"""
Quick UMAP visualization highlighting Primates and other key groups.
Uses pre-defined TaxID lists for fast coloring.
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

def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "taxonomy_model_full.pth"
    map_path = sys.argv[2] if len(sys.argv) > 2 else "data/taxonomy_edges.mapping.tsv"
    
    print(f"Loading embeddings from {ckpt_path}...")
    emb = load_embeddings(ckpt_path)
    print(f"Embeddings shape: {emb.shape}")
    
    print(f"Loading mapping from {map_path}...")
    tax2idx, idx2tax = load_mapping(map_path)
    
    # Define key taxonomic groups by TaxID ranges and specific IDs
    # These are approximate ranges - adjust as needed
    groups = {
        'Primates': ['9606', '10090', '9913', '6239', '7227'],  # Human, Mouse, Cow, C.elegans, Drosophila
        'Mammals': list(range(9200, 9300)) + list(range(10000, 10100)),
        'Other Vertebrates': list(range(7700, 7800)),
        'Invertebrates': list(range(6200, 6300)),
        'Plants': list(range(3600, 3700)),
        'Fungi': list(range(4700, 4800)),
        'Bacteria': list(range(500, 600)),
        'Other': []
    }
    
    # Convert to string and filter to existing taxids
    group_colors = {
        'Primates': '#FF1493',      # Deep pink
        'Mammals': '#FF69B4',       # Hot pink
        'Other Vertebrates': '#4169E1',  # Royal blue
        'Invertebrates': '#32CD32', # Lime green
        'Plants': '#228B22',        # Forest green
        'Fungi': '#8B4513',         # Saddle brown
        'Bacteria': '#FF6347',      # Tomato
        'Other': '#CCCCCC'          # Light gray
    }
    
    # Sample for visualization
    print("\nSampling 20,000 points for visualization...")
    np.random.seed(42)
    sample_indices = np.random.choice(emb.shape[0], min(20000, emb.shape[0]), replace=False)
    emb_sample = emb[sample_indices]
    
    # Assign colors based on group membership
    print("Assigning colors to sampled points...")
    colors = np.array([group_colors['Other']] * len(sample_indices), dtype=object)
    
    for group_name, taxid_list in groups.items():
        if group_name == 'Other':
            continue
        for taxid in taxid_list:
            taxid_str = str(taxid)
            if taxid_str in tax2idx:
                idx = tax2idx[taxid_str]
                pos = np.where(sample_indices == idx)[0]
                if len(pos) > 0:
                    colors[pos] = group_colors[group_name]
    
    # Compute UMAP
    print("\nRunning UMAP (this takes ~2-3 minutes)...")
    umap_proj = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, 
                     random_state=42, n_jobs=4).fit_transform(emb_sample)
    
    # Create visualization
    print("Creating visualization...")
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Plot background (Other)
    mask_other = colors == group_colors['Other']
    ax.scatter(umap_proj[mask_other, 0], umap_proj[mask_other, 1], 
              c=group_colors['Other'], s=10, alpha=0.3, label='Other')
    
    # Plot each group
    for group_name in ['Bacteria', 'Fungi', 'Plants', 'Invertebrates', 
                       'Other Vertebrates', 'Mammals', 'Primates']:
        mask = colors == group_colors[group_name]
        if mask.sum() > 0:
            ax.scatter(umap_proj[mask, 0], umap_proj[mask, 1], 
                      c=group_colors[group_name], label=group_name, 
                      s=20, alpha=0.7, edgecolors='none')
    
    # Highlight specific organisms with stars
    highlight_taxids = {
        '9606': ('Homo sapiens\n(Human)', 'darkred', 400),
        '10090': ('Mus musculus\n(Mouse)', 'darkred', 300),
        '6239': ('C. elegans', 'darkgreen', 300),
        '7227': ('D. melanogaster\n(Fruit fly)', 'darkgreen', 300),
        '562': ('E. coli', 'darkred', 300),
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
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    ax.set_xlabel('UMAP 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('UMAP 2', fontsize=12, fontweight='bold')
    ax.set_title('Poincaré Embeddings - NCBI Taxonomy\nColored by Major Groups (Stars = Key Organisms)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.95, title='Taxonomic Groups')
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('umap_by_groups.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: umap_by_groups.png")
    
    # Print summary
    print("\n" + "="*80)
    print("VISUALIZATION SUMMARY")
    print("="*80)
    print(f"Total sampled points: {len(sample_indices)}")
    print(f"Embedding dimension: {emb.shape[1]}")
    print(f"Total organisms in model: {emb.shape[0]:,}")
    print("\nKey organisms highlighted:")
    for taxid, (name, _, _) in highlight_taxids.items():
        status = "✓" if taxid in tax2idx else "✗"
        print(f"  {status} {name} (TaxID {taxid})")

if __name__ == "__main__":
    main()
