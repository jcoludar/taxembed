#!/usr/bin/env python3
"""
Build a clean taxonomy dataset by filtering a taxid allowlist.

Takes an existing clade dataset and a clean taxid list (from audit_taxonomy_noise.py),
then rebuilds the edgelists, mapping, and transitive closure with only clean nodes.

Usage (from repo root):
    uv run python scripts/build_clean_dataset.py \
        --source mollusca_6447 \
        --clean-taxids artifacts/audit_mollusca_v3.clean_taxids.txt \
        --name mollusca_clean
"""

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Re-use existing writer infrastructure
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TAXOPY_DIR = DATA_DIR / "taxopy"


def load_clean_taxids(path: Path) -> set:
    with open(path) as f:
        return {int(line.strip()) for line in f if line.strip()}


def build_filtered_tree(source_dir: Path, source_name: str, keep: set):
    """Load original edges, filter to keep set, re-parent through removed internals."""
    edge_path = source_dir / f"taxonomy_edges_{source_name}.edgelist"
    parent_of = {}
    with open(edge_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                parent, child = int(parts[0]), int(parts[1])
                if child != parent:
                    parent_of[child] = parent

    # For each kept node, find its nearest kept ancestor
    # This "collapses" removed internal nodes
    new_parent = {}
    for tid in keep:
        p = parent_of.get(tid)
        while p is not None and p not in keep:
            p = parent_of.get(p)
        if p is not None and p != tid:
            new_parent[tid] = p

    # Build new edges and compute depths via BFS from root
    # Find root (kept node with no kept parent)
    roots = [t for t in keep if t not in new_parent]
    if len(roots) != 1:
        print(f"  Warning: {len(roots)} roots found, using first: {roots[:3]}")

    children = defaultdict(list)
    for child, parent in new_parent.items():
        children[parent].append(child)

    # BFS to get depths and edges
    depths = {}
    edges = []
    queue = deque()
    for r in roots:
        queue.append((r, 0))
        depths[r] = 0

    while queue:
        node, depth = queue.popleft()
        for child in children.get(node, []):
            if child not in depths:
                depths[child] = depth + 1
                edges.append((node, child))
                queue.append((child, depth + 1))

    return depths, edges, new_parent


def build_transitive_pairs(depths, parent_of, taxid_to_idx):
    """Build transitive closure: for each node, walk up to root."""
    pairs = []
    for taxid, depth in tqdm(depths.items(), desc="  Building transitive pairs"):
        desc_idx = taxid_to_idx[taxid]
        ancestor = parent_of.get(taxid)
        while ancestor is not None and ancestor in depths:
            anc_depth = depths[ancestor]
            pairs.append({
                "ancestor_idx": taxid_to_idx[ancestor],
                "descendant_idx": desc_idx,
                "depth_diff": depth - anc_depth,
                "ancestor_depth": anc_depth,
                "descendant_depth": depth,
                "ancestor_taxid": ancestor,
                "descendant_taxid": taxid,
            })
            ancestor = parent_of.get(ancestor)
    return pairs


def write_dataset(dataset_dir: Path, name: str, depths, edges, pairs, root_taxid):
    """Write all dataset files."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    prefix = dataset_dir / f"taxonomy_edges_{name}"

    # Mapping
    sorted_taxids = sorted(depths.keys())
    mapping_df = pd.DataFrame({
        "taxid": sorted_taxids,
        "idx": list(range(len(sorted_taxids))),
    })
    mapping_path = prefix.with_name(f"{prefix.name}.mapping.tsv")
    mapping_df.to_csv(mapping_path, sep="\t", index=False)

    # Raw edges
    edge_path = prefix.with_name(f"{prefix.name}.edgelist")
    with open(edge_path, "w") as f:
        for parent, child in edges:
            f.write(f"{parent} {child}\n")

    # Mapped edges
    taxid_to_idx = dict(zip(mapping_df["taxid"], mapping_df["idx"]))
    mapped_path = prefix.with_name(f"{prefix.name}.mapped.edgelist")
    with open(mapped_path, "w") as f:
        for parent, child in edges:
            if parent in taxid_to_idx and child in taxid_to_idx:
                f.write(f"{taxid_to_idx[parent]} {taxid_to_idx[child]}\n")

    # Transitive closure as NPZ (columnar)
    npz_path = prefix.with_name(f"{prefix.name}_transitive.npz")
    if pairs:
        np.savez_compressed(
            npz_path,
            ancestor_idx=np.array([p["ancestor_idx"] for p in pairs], dtype=np.int32),
            descendant_idx=np.array([p["descendant_idx"] for p in pairs], dtype=np.int32),
            depth_diff=np.array([p["depth_diff"] for p in pairs], dtype=np.int16),
            ancestor_depth=np.array([p["ancestor_depth"] for p in pairs], dtype=np.int16),
            descendant_depth=np.array([p["descendant_depth"] for p in pairs], dtype=np.int16),
            ancestor_taxid=np.array([p["ancestor_taxid"] for p in pairs], dtype=np.int32),
            descendant_taxid=np.array([p["descendant_taxid"] for p in pairs], dtype=np.int32),
        )

    # Transitive as TSV
    tsv_path = prefix.with_name(f"{prefix.name}_transitive.tsv")
    pd.DataFrame(pairs).to_csv(tsv_path, sep="\t", index=False)

    # Manifest
    manifest = {
        "dataset_name": name,
        "root_taxid": root_taxid,
        "root_name": "Mollusca",
        "nodes": len(mapping_df),
        "edges": len(edges),
        "max_depth_observed": max(depths.values()) if depths else 0,
        "max_depth_requested": None,
        "transitive_pairs": len(pairs),
    }
    manifest_path = prefix.with_name(f"{prefix.name}_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return {
        "mapping": mapping_path,
        "edgelist": edge_path,
        "mapped_edgelist": mapped_path,
        "transitive_npz": npz_path,
        "transitive_tsv": tsv_path,
        "manifest": manifest_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Build clean taxonomy dataset")
    parser.add_argument("--source", required=True, help="Source dataset name")
    parser.add_argument("--clean-taxids", required=True, type=Path, help="Clean taxid list file")
    parser.add_argument("--name", required=True, help="Output dataset name")
    parser.add_argument("--root-taxid", type=int, default=6447, help="Root taxid (default: 6447 = Mollusca)")
    args = parser.parse_args()

    source_dir = TAXOPY_DIR / args.source
    print(f"Source dataset: {source_dir}")

    # Load clean taxids
    keep = load_clean_taxids(args.clean_taxids)
    print(f"Clean taxids: {len(keep):,}")

    # Ensure root is in the keep set
    if args.root_taxid not in keep:
        keep.add(args.root_taxid)
        print(f"  Added root {args.root_taxid} to keep set")

    # Build filtered tree
    print("\nBuilding filtered tree...")
    depths, edges, parent_of = build_filtered_tree(source_dir, args.source, keep)
    print(f"  Nodes: {len(depths):,}")
    print(f"  Edges: {len(edges):,}")
    print(f"  Max depth: {max(depths.values()) if depths else 0}")

    # Build mapping
    sorted_taxids = sorted(depths.keys())
    taxid_to_idx = {tid: i for i, tid in enumerate(sorted_taxids)}

    # Build transitive pairs
    print("\nBuilding transitive closure...")
    pairs = build_transitive_pairs(depths, parent_of, taxid_to_idx)
    print(f"  Transitive pairs: {len(pairs):,}")

    # Write dataset
    out_dir = TAXOPY_DIR / args.name
    print(f"\nWriting dataset to {out_dir}...")
    files = write_dataset(out_dir, args.name, depths, edges, pairs, args.root_taxid)
    for k, v in files.items():
        print(f"  {k}: {v}")

    print(f"\n  Done! Dataset '{args.name}' ready for training.")
    print(f"  Use: VIRTUAL_ENV= uv run taxembed train Mollusca -as mollusca_clean_v1 --file {files['transitive_npz']}")


if __name__ == "__main__":
    main()
