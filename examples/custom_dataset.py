#!/usr/bin/env python3
"""Example of building and training on a custom taxonomic clade."""

import argparse
from pathlib import Path

from taxembed.builders import build_clade_dataset


def main():
    """Build a custom dataset and show how to use it."""
    parser = argparse.ArgumentParser(description="Build custom taxonomy dataset")
    parser.add_argument(
        "--root-taxid",
        type=int,
        required=True,
        help="Root TaxID for the clade (e.g., 33208 for Metazoa)",
    )
    parser.add_argument("--name", required=True, help="Dataset name (e.g., animals, plants)")
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum depth to include (optional)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("CUSTOM DATASET BUILDER")
    print("=" * 60)

    # Build the dataset
    print(f"\nBuilding dataset for TaxID {args.root_taxid} ({args.name})...")

    data_dir = Path("data")
    output_dir = data_dir / "taxopy" / args.name

    result = build_clade_dataset(
        root_taxid=args.root_taxid,
        dataset_name=args.name,
        output_dir=output_dir,
        taxdump_dir=data_dir,
        max_depth=args.max_depth,
    )

    print("\n" + "=" * 60)
    print("DATASET READY!")
    print("=" * 60)
    print("\nStatistics:")
    print(f"  Root TaxID: {result.root_taxid}")
    print(f"  Nodes: {result.node_count:,}")
    print(f"  Training pairs: {result.pairs_count:,}")
    print(f"  Max depth: {result.max_depth}")
    print("\nFiles created:")
    for name, path in result.files.items():
        print(f"  {name}: {path}")

    print("\nNext steps:")
    print(f"  Train: taxembed train {args.root_taxid} -as {args.name} --epochs 100")
    print("  Or: python -c 'from taxembed.cli.train import main; main()' \\")
    print(f"      --data {result.files['transitive_pickle']} \\")
    print(f"      --mapping {result.files['mapping']} \\")
    print(f"      --checkpoint {args.name}_model.pth")


if __name__ == "__main__":
    main()
