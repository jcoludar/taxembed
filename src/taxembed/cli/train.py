#!/usr/bin/env python3
"""Train hierarchical Poincaré embeddings."""

import argparse
import pickle

import pandas as pd
import torch
import torch.optim as optim

from taxembed.models import HierarchicalPoincareEmbedding
from taxembed.training import HierarchicalDataLoader, train_model


def main():
    """Train hierarchical model on taxonomy data."""
    parser = argparse.ArgumentParser(description="Train hierarchical Poincaré embeddings")
    parser.add_argument(
        "--data",
        default="data/taxonomy_edges_small_transitive.pkl",
        help="Training data (default: small dataset)",
    )
    parser.add_argument(
        "--checkpoint",
        default="taxonomy_model_small.pth",
        help="Output checkpoint path",
    )
    parser.add_argument(
        "--mapping",
        default="data/taxonomy_edges_small.mapping.tsv",
        help="Mapping file path aligned with training data",
    )
    parser.add_argument("--dim", type=int, default=10, help="Embedding dimension")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--n-negatives", type=int, default=50, help="Number of negative samples")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--margin", type=float, default=0.2, help="Ranking loss margin")
    parser.add_argument("--lambda-reg", type=float, default=0.1, help="Regularization strength")
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=5,
        help="Early stopping patience (0 to disable)",
    )
    parser.add_argument("--gpu", type=int, default=-1, help="GPU device index (-1 for CPU)")

    args = parser.parse_args()

    # Device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    print("Loading training data...")
    with open(args.data, "rb") as f:
        training_data = pickle.load(f)

    print(f"  ✓ Loaded {len(training_data):,} training pairs")

    # Load mapping to get node count
    print("Loading mapping...")
    mapping_df = pd.read_csv(args.mapping, sep="\t", header=None, names=["idx", "taxid"])
    n_nodes = len(mapping_df)
    print(f"  ✓ {n_nodes:,} unique nodes")

    # Build depth map
    print("Building depth map...")
    idx_to_depth = {}
    max_depth = 0
    for item in training_data:
        ancestor_idx = item["ancestor_idx"]
        descendant_idx = item["descendant_idx"]
        ancestor_depth = item["ancestor_depth"]
        descendant_depth = item["descendant_depth"]

        idx_to_depth[ancestor_idx] = ancestor_depth
        idx_to_depth[descendant_idx] = descendant_depth
        max_depth = max(max_depth, ancestor_depth, descendant_depth)

    print(f"  ✓ Depth range: [0, {max_depth}]")

    # Create model
    print(f"\nInitializing {args.dim}D Poincaré embeddings...")
    model = HierarchicalPoincareEmbedding(
        n_nodes=n_nodes, dim=args.dim, max_depth=max_depth, init_depth_data=idx_to_depth
    )

    # Create data loader
    print("Creating data loader...")
    dataloader = HierarchicalDataLoader(
        training_data=training_data,
        n_nodes=n_nodes,
        batch_size=args.batch_size,
        n_negatives=args.n_negatives,
        depth_stratify=True,
    )

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train
    train_model(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        n_epochs=args.epochs,
        idx_to_depth=idx_to_depth,
        max_depth=max_depth,
        device=device,
        margin=args.margin,
        lambda_reg=args.lambda_reg,
        early_stopping_patience=args.early_stopping,
        checkpoint_base=args.checkpoint,
    )

    print(
        f"\n✓ Training complete! Best model saved to {args.checkpoint.replace('.pth', '_best.pth')}"
    )


if __name__ == "__main__":
    main()
