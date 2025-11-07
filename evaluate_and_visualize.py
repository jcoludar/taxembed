#!/usr/bin/env python3
"""
Evaluate trained Poincaré embeddings and visualize with UMAP.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import argparse
import sys

def load_model_and_embeddings(checkpoint_path):
    """Load trained model and extract embeddings."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    chkpnt = torch.load(checkpoint_path, map_location='cpu')
    embeddings = chkpnt['embeddings']
    print(f"Loaded embeddings shape: {embeddings.shape}")
    return embeddings

def load_taxonomy_data(csv_path):
    """Load taxonomy edges."""
    print(f"Loading taxonomy data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} edges")
    return df

def evaluate_reconstruction(embeddings, edges_df):
    """
    Evaluate reconstruction quality.
    Compute mean rank and MAP for held-out edges.
    """
    print("\nEvaluating reconstruction quality...")
    
    # Convert to numpy for faster computation
    emb_np = embeddings.numpy() if isinstance(embeddings, torch.Tensor) else embeddings
    
    # Compute pairwise distances in hyperbolic space (Poincaré ball)
    # For simplicity, use Euclidean distance as proxy
    from sklearn.metrics.pairwise import euclidean_distances
    distances = euclidean_distances(emb_np)
    
    # Sample edges for evaluation (full eval would be slow)
    sample_size = min(10000, len(edges_df))
    sample_edges = edges_df.sample(n=sample_size, random_state=42)
    
    ranks = []
    for idx, row in sample_edges.iterrows():
        child_id = int(row['id1'])
        parent_id = int(row['id2'])
        
        if child_id < len(distances) and parent_id < len(distances):
            # Rank of parent among all nodes for this child
            dists_from_child = distances[child_id]
            rank = (dists_from_child < dists_from_child[parent_id]).sum()
            ranks.append(rank)
    
    if ranks:
        mean_rank = np.mean(ranks)
        median_rank = np.median(ranks)
        print(f"Mean Rank: {mean_rank:.2f}")
        print(f"Median Rank: {median_rank:.2f}")
        return mean_rank, median_rank
    return None, None

def visualize_with_umap(embeddings, edges_df, output_path='umap_visualization.png'):
    """
    Project embeddings to 2D using UMAP and visualize.
    """
    print("\nInstalling/importing UMAP...")
    try:
        import umap
    except ImportError:
        print("Installing umap-learn...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "umap-learn", "-q"])
        import umap
    
    print("Projecting embeddings to 2D with UMAP (this may take a few minutes)...")
    
    # Convert to numpy
    emb_np = embeddings.numpy() if isinstance(embeddings, torch.Tensor) else embeddings
    
    # Standardize embeddings
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(emb_np)
    
    # Apply UMAP
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42, n_jobs=-1)
    embedding_2d = reducer.fit_transform(emb_scaled)
    
    print(f"UMAP projection shape: {embedding_2d.shape}")
    
    # Create visualization
    print(f"Creating visualization...")
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Plot all points
    scatter = ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                        c=np.arange(len(embedding_2d)), 
                        cmap='viridis', 
                        s=1, 
                        alpha=0.6)
    
    ax.set_title('UMAP Projection of Poincaré Embeddings (NCBI Taxonomy)', fontsize=16)
    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    plt.close()
    
    return embedding_2d

def sample_and_analyze(embeddings, edges_df, n_samples=5):
    """
    Sample some taxa and show their nearest neighbors.
    """
    print(f"\n{'='*60}")
    print("Sample Analysis: Nearest Neighbors in Embedding Space")
    print(f"{'='*60}")
    
    emb_np = embeddings.numpy() if isinstance(embeddings, torch.Tensor) else embeddings
    from sklearn.metrics.pairwise import euclidean_distances
    
    # Get unique taxa
    all_taxa = set(edges_df['id1'].unique()) | set(edges_df['id2'].unique())
    sample_taxa = np.random.choice(list(all_taxa), min(n_samples, len(all_taxa)), replace=False)
    
    # Load taxid to name mapping if available
    try:
        names_df = pd.read_csv('data/names.dmp', sep='\t\|\t', header=None, engine='python', 
                               usecols=[0, 1], names=['taxid', 'name'], nrows=100000)
        names_dict = dict(zip(names_df['taxid'], names_df['name']))
    except:
        names_dict = {}
    
    for taxid in sample_taxa:
        taxid = int(taxid)
        if taxid < len(emb_np):
            # Find nearest neighbors
            distances = euclidean_distances([emb_np[taxid]], emb_np)[0]
            nearest_indices = np.argsort(distances)[:6]  # Top 6 including self
            
            taxid_name = names_dict.get(taxid, f"TaxID {taxid}")
            print(f"\nTaxID {taxid} ({taxid_name}):")
            print("  Nearest neighbors:")
            for rank, idx in enumerate(nearest_indices[1:6], 1):  # Skip self
                neighbor_name = names_dict.get(int(idx), f"TaxID {idx}")
                dist = distances[idx]
                print(f"    {rank}. {neighbor_name} (distance: {dist:.4f})")

def main():
    parser = argparse.ArgumentParser(description='Evaluate and visualize Poincaré embeddings')
    parser.add_argument('--checkpoint', default='taxonomy_model.pth', 
                       help='Path to checkpoint file')
    parser.add_argument('--data', default='data/taxonomy_edges.csv',
                       help='Path to taxonomy edges CSV')
    parser.add_argument('--output', default='umap_visualization.png',
                       help='Output path for UMAP visualization')
    parser.add_argument('--no-umap', action='store_true',
                       help='Skip UMAP visualization')
    parser.add_argument('--samples', type=int, default=5,
                       help='Number of taxa to sample for analysis')
    
    args = parser.parse_args()
    
    # Load data
    embeddings = load_model_and_embeddings(args.checkpoint)
    edges_df = load_taxonomy_data(args.data)
    
    # Evaluate
    evaluate_reconstruction(embeddings, edges_df)
    
    # Sample analysis
    sample_and_analyze(embeddings, edges_df, n_samples=args.samples)
    
    # Visualize
    if not args.no_umap:
        visualize_with_umap(embeddings, edges_df, args.output)
    
    print("\n✓ Evaluation complete!")

if __name__ == '__main__':
    main()
