#!/usr/bin/env python3
"""Basic example of training Poincaré embeddings on taxonomy data."""

import pickle
from pathlib import Path

import pandas as pd
import torch
import torch.optim as optim

from taxembed.models import HierarchicalPoincareEmbedding
from taxembed.training import HierarchicalDataLoader, train_model


def main():
    """Train a simple model on the small taxonomy dataset."""

    # Paths
    data_dir = Path("data")
    training_data_file = data_dir / "taxonomy_edges_small_transitive.pkl"
    mapping_file = data_dir / "taxonomy_edges_small.mapping.tsv"

    # Check if data exists
    if not training_data_file.exists():
        print(f"❌ Training data not found: {training_data_file}")
        print("   Run: taxembed-download && taxembed-prepare")
        return

    print("=" * 60)
    print("BASIC TRAINING EXAMPLE")
    print("=" * 60)

    # Load training data
    print("\n1. Loading training data...")
    with open(training_data_file, "rb") as f:
        training_data = pickle.load(f)
    print(f"   ✓ Loaded {len(training_data):,} training pairs")

    # Load mapping
    print("\n2. Loading node mapping...")
    mapping_df = pd.read_csv(mapping_file, sep="\t", header=None, names=["idx", "taxid"])
    n_nodes = len(mapping_df)
    print(f"   ✓ {n_nodes:,} unique nodes")

    # Build depth map
    print("\n3. Building depth information...")
    idx_to_depth = {}
    max_depth = 0
    for item in training_data:
        idx_to_depth[item["ancestor_idx"]] = item["ancestor_depth"]
        idx_to_depth[item["descendant_idx"]] = item["descendant_depth"]
        max_depth = max(max_depth, item["ancestor_depth"], item["descendant_depth"])
    print(f"   ✓ Depth range: [0, {max_depth}]")

    # Create model
    print("\n4. Creating Poincaré embedding model...")
    model = HierarchicalPoincareEmbedding(
        n_nodes=n_nodes,
        dim=10,  # 10-dimensional embeddings
        max_depth=max_depth,
        init_depth_data=idx_to_depth,
    )
    print(f"   ✓ Model initialized with {n_nodes:,} nodes in 10D")

    # Create data loader
    print("\n5. Creating data loader...")
    dataloader = HierarchicalDataLoader(
        training_data=training_data,
        n_nodes=n_nodes,
        batch_size=64,
        n_negatives=50,
        depth_stratify=True,
    )

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # Train for a few epochs (just for demo)
    print("\n6. Training model...")
    print("   (Training for 5 epochs as a demo - increase for better results)\n")

    device = torch.device("cpu")  # Use CPU for demo

    train_model(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        n_epochs=5,  # Short for demo
        idx_to_depth=idx_to_depth,
        max_depth=max_depth,
        device=device,
        margin=0.2,
        lambda_reg=0.1,
        early_stopping_patience=0,  # Disabled for demo
        checkpoint_base="examples/demo_model.pth",
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("  - Visualize: taxembed-visualize examples/demo_model_best.pth")
    print("  - Train longer: Increase n_epochs for better quality")
    print("  - Analyze: python examples/nn_demo.py")


if __name__ == "__main__":
    main()
