#!/usr/bin/env python3
"""Example of visualizing embeddings with custom group highlighting."""

import argparse

from taxembed.visualization import (
    load_embeddings,
    load_mapping,
)


def main():
    """Create custom visualizations of trained embeddings."""
    parser = argparse.ArgumentParser(description="Visualize embeddings")
    parser.add_argument("checkpoint", help="Path to checkpoint file")
    parser.add_argument(
        "--mapping",
        default="data/taxonomy_edges_small.mapping.tsv",
        help="Path to mapping file",
    )
    parser.add_argument("--output", default="embedding_viz.png", help="Output image path")
    parser.add_argument("--sample", type=int, default=10000, help="Number of points to visualize")

    args = parser.parse_args()

    print("=" * 60)
    print("EMBEDDING VISUALIZATION")
    print("=" * 60)

    # Load embeddings
    print(f"\nLoading embeddings from {args.checkpoint}...")
    embeddings = load_embeddings(args.checkpoint)
    print(f"  ✓ Loaded {embeddings.shape[0]:,} embeddings of dimension {embeddings.shape[1]}")

    # Load mapping
    print(f"\nLoading mapping from {args.mapping}...")
    tax2idx, idx2tax = load_mapping(args.mapping)
    if tax2idx:
        print(f"  ✓ Loaded {len(tax2idx):,} TaxID mappings")

    # Create visualization
    print(f"\nCreating UMAP visualization (sampling {args.sample:,} points)...")
    print("  This may take a few minutes...")

    # Note: The full create_umap_visualization function needs to be properly
    # extracted from visualize_multi_groups.py. For now, this is a placeholder.
    print("\n  ⚠️  Full visualization function needs to be refactored.")
    print(f"  For now, use: taxembed-visualize {args.checkpoint}")

    print("\n" + "=" * 60)
    print("TIP: Use the taxembed CLI for full visualization features:")
    print(f"     taxembed visualize <tag> --sample {args.sample}")
    print("=" * 60)


if __name__ == "__main__":
    main()
