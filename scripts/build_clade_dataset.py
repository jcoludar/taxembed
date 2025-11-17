#!/usr/bin/env python3
"""CLI wrapper around the TaxoPy-backed clade dataset builder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from taxembed.builders import build_clade_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a taxonomy dataset for a specific clade.")
    parser.add_argument("--root-taxid", type=int, required=True, help="Root TaxID for the clade.")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Optional dataset name override (defaults to sanitized root name).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "taxopy",
        help="Directory where the dataset artifacts will be written.",
    )
    parser.add_argument(
        "--taxdump-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Directory containing nodes.dmp/names.dmp (downloaded automatically if missing).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Optional depth limit (relative to the root taxon).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_clade_dataset(
        args.root_taxid,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        taxdump_dir=args.taxdump_dir,
        max_depth=args.max_depth,
    )

    print("\n✅ Finished building clade dataset")
    print(f"  Dataset: {result.dataset_name}")
    print(f"  Root TaxID: {result.root_taxid}")
    print(f"  Nodes: {result.node_count:,}")
    print(f"  Edges: {result.edge_count:,}")
    print(f"  Max depth observed: {result.max_depth}")
    print(f"  Transitive pairs: {result.pairs_count:,}")
    print(f"  Output directory: {result.output_dir}")
    print("\n  Files:")
    for key, path in result.files.items():
        print(f"    - {key}: {path}")


if __name__ == "__main__":
    main()

