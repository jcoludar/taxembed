#!/usr/bin/env python3
"""
Create UMAP visualization colored by taxonomic rank (Kingdom/Phylum).
Highlights Primates and other interesting groups.
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

def get_taxonomy_rank(taxid, names_path="data/names.dmp"):
    """Get taxonomic rank for a TaxID from NCBI names.dmp."""
    try:
        with open(names_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t|\t')
                if len(parts) >= 4 and parts[0] == taxid:
                    rank = parts[3]
                    if rank in ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']:
                        return rank
    except:
        pass
    return 'unknown'

def get_kingdom(taxid, nodes_path="data/nodes.dmp"):
    """Get kingdom for a TaxID by traversing up the tree."""
    visited = set()
    current = taxid
    
    try:
        # Build parent map
        parent_map = {}
        with open(nodes_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t|\t')
                if len(parts) >= 2:
                    child_id = parts[0].strip()
                    parent_id = parts[1].strip()
                    parent_map[child_id] = parent_id
        
        # Traverse up to root
        while current not in visited and current in parent_map:
            visited.add(current)
            current = parent_map[current]
        
        # Map to kingdom name
        kingdom_map = {
            '1': 'Archaea',
            '2': 'Bacteria',
            '2157': 'Archaea',
            '2759': 'Eukaryota',
            '10239': 'Viruses'
        }
        
        if current in kingdom_map:
            return kingdom_map[current]
        return 'Other'
    except:
        return 'Unknown'

def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "taxonomy_model_full.pth"
    map_path = sys.argv[2] if len(sys.argv) > 2 else "data/taxonomy_edges.mapping.tsv"
    
    print(f"Loading embeddings from {ckpt_path}...")
    emb = load_embeddings(ckpt_path)
    print(f"Embeddings shape: {emb.shape}")
    
    print(f"Loading mapping from {map_path}...")
    tax2idx, idx2tax = load_mapping(map_path)
    
    # Sample for visualization
    print("\nSampling 15,000 points for visualization...")
    np.random.seed(42)
    sample_indices = np.random.choice(emb.shape[0], 15000, replace=False)
    emb_sample = emb[sample_indices]
    
    # Get kingdom info for sampled points
    print("Retrieving kingdom information (this may take a minute)...")
    kingdoms = []
    kingdom_colors = {
        'Bacteria': '#FF6B6B',
        'Archaea': '#4ECDC4',
        'Eukaryota': '#45B7D1',
        'Viruses': '#FFA07A',
        'Unknown': '#CCCCCC'
    }
    
    for i, idx in enumerate(sample_indices):
        if i % 1000 == 0:
            print(f"  Processed {i}/15000...")
        taxid = idx2tax[idx]
        kingdom = get_kingdom(taxid)
        kingdoms.append(kingdom)
    
    kingdoms = np.array(kingdoms)
    
    # Compute UMAP
    print("\nRunning UMAP...")
    umap_proj = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(emb_sample)
    
    # Create visualization
    print("Creating visualization...")
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Plot each kingdom with different color
    for kingdom, color in kingdom_colors.items():
        mask = kingdoms == kingdom
        if mask.sum() > 0:
            ax.scatter(umap_proj[mask, 0], umap_proj[mask, 1], 
                      c=color, label=kingdom, s=15, alpha=0.6, edgecolors='none')
    
    # Highlight specific organisms
    highlight_taxids = {
        '9606': ('Homo sapiens', 'red', 200),
        '10090': ('Mus musculus', 'darkred', 200),
        '6239': ('C. elegans', 'blue', 200),
        '7227': ('D. melanogaster', 'darkblue', 200),
        '562': ('E. coli', 'green', 200),
    }
    
    for taxid, (name, color, size) in highlight_taxids.items():
        if taxid in tax2idx:
            idx = tax2idx[taxid]
            # Find position in sample
            pos = np.where(sample_indices == idx)[0]
            if len(pos) > 0:
                pos = pos[0]
                ax.scatter(umap_proj[pos, 0], umap_proj[pos, 1], 
                          c=color, s=size, marker='*', edgecolors='black', linewidth=2,
                          label=name, zorder=10)
    
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title('Poincaré Embeddings - NCBI Taxonomy\nColored by Kingdom (Stars = Notable Organisms)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('umap_by_kingdom.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: umap_by_kingdom.png")
    
    # Print statistics
    print("\n" + "="*80)
    print("KINGDOM DISTRIBUTION")
    print("="*80)
    for kingdom in sorted(kingdom_colors.keys()):
        count = (kingdoms == kingdom).sum()
        pct = 100 * count / len(kingdoms)
        print(f"{kingdom:15s}: {count:6d} ({pct:5.1f}%)")

if __name__ == "__main__":
    main()
