#!/usr/bin/env python3
"""
Visualize multiple taxonomic groups on the same plot with different colors.

Usage:
    python visualize_multi_groups.py taxonomy_model_small_best.pth
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import taxopy
import torch
from umap import UMAP

from taxembed.data import ensure_taxdump


def load_embeddings(ckpt_path):
    """Load embeddings from checkpoint."""
    print(f"Loading embeddings from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        emb = sd["lt.weight"].detach().cpu().numpy()
    elif "embeddings" in ckpt:
        emb = ckpt["embeddings"].cpu().numpy()
    else:
        raise ValueError("Cannot find embeddings in checkpoint")

    print(f"  ✓ Shape: {emb.shape}")
    return emb


def load_mapping(map_path):
    """Load TaxID to index mapping."""
    if not Path(map_path).exists():
        print(f"⚠️  Mapping file not found: {map_path}")
        return None, None

    print(f"Loading mapping from {map_path}...")
    df = pd.read_csv(map_path, sep="\t", dtype={"taxid": str, "idx": int})

    # Filter out non-numeric taxids
    numeric_df = df[df["taxid"].str.isnumeric()]
    tax2idx = dict(zip(numeric_df["taxid"], numeric_df["idx"], strict=False))
    idx2tax = dict(zip(numeric_df["idx"], numeric_df["taxid"], strict=False))

    print(f"  ✓ Loaded {len(tax2idx):,} mappings")
    return tax2idx, idx2tax


def load_taxonomy_tree(valid_taxids=None, base_dir: Path = Path("data")):
    """Load NCBI taxonomy tree structure from dump files or TaxoPy fallback."""
    names = {}
    nodes = {}

    # Ensure dump files are extracted (will download if needed)
    try:
        nodes_file, names_file, _ = ensure_taxdump(base_dir)
        names_path = names_file
        nodes_path = nodes_file
    except Exception as e:
        print(f"⚠️  Could not ensure taxdump files: {e}")
        names_path = base_dir / "names.dmp"
        nodes_path = base_dir / "nodes.dmp"

    if names_path.exists() and nodes_path.exists():
        try:
            # Load names
            with names_path.open("r") as f:
                for line in f:
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) >= 4 and parts[3] == "scientific name":
                        taxid = int(parts[0])
                        if valid_taxids is None or taxid in valid_taxids:
                            names[taxid] = parts[1]

            # Load nodes (parent relationships)
            with nodes_path.open("r") as f:
                for line in f:
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) >= 5:
                        taxid = int(parts[0])
                        parent = int(parts[1])
                        if valid_taxids is None or taxid in valid_taxids:
                            nodes[taxid] = parent

            print("Loading taxonomy tree from dump files...")
            print(f"  ✓ Loaded {len(nodes):,} taxonomy nodes (filtered to dataset)")
            return names, nodes
        except Exception as e:
            print(f"⚠️  Error reading dump files: {e}, falling back to TaxoPy...")

    # Fallback to TaxoPy
    try:
        print("Loading taxonomy tree via TaxoPy (dump files not found)...")
        taxdb = taxopy.TaxDb(taxdb_dir=str(base_dir))

        # Build names dict
        for taxid_str, name in taxdb.taxid2name.items():
            taxid = int(taxid_str)
            if valid_taxids is None or taxid in valid_taxids:
                names[taxid] = name

        # Build nodes dict (parent relationships)
        for taxid_str, parent_str in taxdb.taxid2parent.items():
            taxid = int(taxid_str)
            parent = int(parent_str)
            if valid_taxids is None or taxid in valid_taxids:
                nodes[taxid] = parent

        print(f"  ✓ Loaded {len(nodes):,} taxonomy nodes via TaxoPy (filtered to dataset)")
        return names, nodes
    except Exception as e:
        print(f"⚠️  Could not load taxonomy tree: {e}")
        return None, None


# Taxonomic groups with their root TaxIDs
def build_parent_children(nodes):
    parent_children = defaultdict(list)
    for child, parent in nodes.items():
        parent_children[parent].append(child)
    return parent_children


def collect_descendants(root_taxid, parent_children, tax2idx):
    """Find all descendants of root_taxid within the dataset."""
    stack = [root_taxid]
    indices = set()
    while stack:
        node = stack.pop()
        idx = tax2idx.get(str(node))
        if idx is not None:
            indices.add(idx)
        stack.extend(parent_children.get(node, []))
    return indices


def get_nodes_at_depth(root_taxid, parent_children, depth):
    """Get all nodes at a specific depth from root (0=children, 1=grandchildren, etc.)."""
    if depth == 0:
        return parent_children.get(root_taxid, [])

    current_level = [root_taxid]
    for _ in range(depth):
        next_level = []
        for node in current_level:
            next_level.extend(parent_children.get(node, []))
        current_level = next_level
        if not current_level:
            break
    return current_level


def visualize_multi_groups(
    emb,
    tax2idx,
    idx2tax,
    nodes,
    names,
    sample_size=25000,
    output_file=None,
    child_coloring=None,
    coloring_depth=0,
    clade_name=None,
    epoch=None,
    loss=None,
):
    """Create UMAP visualization with multiple highlighted groups.

    Args:
        child_coloring: Root TaxID to color by children
        coloring_depth: Depth level for coloring (0=children, 1=grandchildren, 2=great-grandchildren, etc.)
        clade_name: Name of the clade for title
        epoch: Training epoch for title
        loss: Training loss for title
    """

    # Find members of each group
    print("\nFinding taxonomic groups...")
    group_members = {}
    if child_coloring is not None:
        parent_children = build_parent_children(nodes)
        root_taxid = child_coloring

        # Get nodes at the specified depth
        depth_nodes = get_nodes_at_depth(root_taxid, parent_children, coloring_depth)

        if not depth_nodes:
            depth_label = ["children", "grandchildren", "great-grandchildren"][
                min(coloring_depth, 2)
            ]
            if coloring_depth > 2:
                depth_label = f"{coloring_depth}-level descendants"
            print(
                f"  ⚠️  No {depth_label} found at depth {coloring_depth} under root TaxID {root_taxid}"
            )
        else:
            depth_label = ["children", "grandchildren", "great-grandchildren"][
                min(coloring_depth, 2)
            ]
            if coloring_depth > 2:
                depth_label = f"{coloring_depth}-level descendants"
            print(f"  Coloring by {depth_label} (depth {coloring_depth})...")

        for node in depth_nodes:
            # Collect all descendants of this node for coloring
            indices = collect_descendants(node, parent_children, tax2idx)
            label = names.get(node, f"TaxID {node}")
            group_members[label] = indices
            print(f"  ✓ {label}: {len(indices):,} organisms")
    else:
        default_groups = {
            "Mammals": 40674,
            "Birds": 8782,
            "Insects": 50557,
            "Bacteria": 2,
            "Fungi": 4751,
            "Plants": 33090,
        }
        parent_children = build_parent_children(nodes)
        for group_name, root_taxid in default_groups.items():
            if root_taxid in nodes or root_taxid == 1:
                indices = collect_descendants(root_taxid, parent_children, tax2idx)
                group_members[group_name] = indices
                print(f"  ✓ {group_name}: {len(indices):,} organisms")
            else:
                print(f"  ⚠️  {group_name}: root TaxID {root_taxid} not found in taxonomy tree")
                group_members[group_name] = set()

    # Assign colors to all indices
    n_total = emb.shape[0]
    idx_to_group = {}

    for idx in range(n_total):
        assigned = False
        for group_name, members in group_members.items():
            if idx in members:
                idx_to_group[idx] = group_name
                assigned = True
                break
        if not assigned:
            idx_to_group[idx] = "Other"

    # Sample points
    print(f"\nSampling {sample_size:,} points from {n_total:,} total...")
    all_indices = list(range(n_total))

    # Stratified sampling: ensure each group is represented
    sampled_indices = []
    samples_per_group = {}

    label_order = list(group_members.keys())
    if not label_order:
        label_order = ["Other"]
    palette = plt.get_cmap("tab20", max(1, len(label_order)))
    color_lookup = {
        group_name: palette(idx) if len(label_order) > 0 else "#1f77b4"
        for idx, group_name in enumerate(label_order)
    }
    color_lookup["Other"] = "#bdc3c7"

    for group_name in label_order + ["Other"]:
        group_indices = [i for i in all_indices if idx_to_group[i] == group_name]
        if group_indices:
            # Sample proportionally
            n_sample = min(
                len(group_indices), max(1, int(len(group_indices) / n_total * sample_size))
            )
            sampled = np.random.choice(group_indices, n_sample, replace=False)
            sampled_indices.extend(sampled)
            samples_per_group[group_name] = len(sampled)

    # If we haven't reached sample_size, add more from "Other"
    if len(sampled_indices) < sample_size:
        other_indices = [i for i in all_indices if i not in sampled_indices]
        additional = np.random.choice(
            other_indices,
            min(len(other_indices), sample_size - len(sampled_indices)),
            replace=False,
        )
        sampled_indices.extend(additional)

    sampled_indices = np.array(sampled_indices)

    print(f"  ✓ Sampled {len(sampled_indices):,} points")
    for group_name in label_order + ["Other"]:
        if group_name in samples_per_group:
            print(f"     - {group_name}: {samples_per_group[group_name]:,}")

    # Extract embeddings
    sample_emb = emb[sampled_indices]

    # Run UMAP
    print(f"\nRunning UMAP on {len(sampled_indices):,} points...")
    umap_model = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    projection = umap_model.fit_transform(sample_emb)

    # Plot
    print("Creating visualization...")
    fig, ax = plt.subplots(figsize=(18, 14))

    # Plot each group
    for group_name in ["Other"] + label_order:
        mask = np.array([idx_to_group[idx] == group_name for idx in sampled_indices])
        n_points = mask.sum()

        if n_points > 0:
            color = color_lookup.get(group_name, "#cccccc")

            # Other group: smaller, more transparent
            if group_name == "Other":
                ax.scatter(
                    projection[mask, 0],
                    projection[mask, 1],
                    color=color,
                    s=15,
                    alpha=0.2,
                    label=f"{group_name} (n={n_points:,})",
                    edgecolors="none",
                    zorder=1,
                )
            else:
                # Highlighted groups: larger, more visible
                ax.scatter(
                    projection[mask, 0],
                    projection[mask, 1],
                    color=color,
                    s=50,
                    alpha=0.8,
                    label=f"{group_name} (n={n_points:,})",
                    edgecolors="black",
                    linewidth=0.5,
                    zorder=2,
                )

    ax.set_xlabel("UMAP 1", fontsize=16)
    ax.set_ylabel("UMAP 2", fontsize=16)

    # Build title
    title_parts = []
    if clade_name:
        title_parts.append(f"TaxEmbed: {clade_name}")
    else:
        title_parts.append("TaxEmbed")

    title_parts.append(f"Children Level {coloring_depth}")

    if epoch is not None:
        title_parts.append(f"epochs {epoch}")

    if loss is not None:
        title_parts.append(f"Loss {loss:.6f}")

    title = ", ".join(title_parts)
    ax.set_title(title, fontsize=20, fontweight="bold")
    ax.legend(loc="best", fontsize=13, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file is None:
        output_file = "taxonomy_embeddings_multi_groups.png"

    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    print(f"\n✅ Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Visualize multiple taxonomic groups")
    parser.add_argument("checkpoint", help="Path to checkpoint file")
    parser.add_argument("--mapping", help="Path to mapping file (auto-detected if not specified)")
    parser.add_argument("--sample", type=int, default=25000, help="Number of points to sample")
    parser.add_argument("--output", help="Output filename")
    parser.add_argument("--names", help="Path to names.dmp (default: data/names.dmp)")
    parser.add_argument("--nodes", help="Path to nodes.dmp (default: data/nodes.dmp)")
    parser.add_argument("--root-taxid", type=int, help="Root TaxID for child-level coloring")
    parser.add_argument(
        "--children",
        type=int,
        default=0,
        help="Depth level for coloring (0=children, 1=grandchildren, 2=great-grandchildren, etc.)",
    )
    parser.add_argument("--clade-name", help="Name of the clade for title")
    parser.add_argument("--epoch", type=int, help="Training epoch for title")
    parser.add_argument("--loss", type=float, help="Training loss for title")

    args = parser.parse_args()

    # Load embeddings
    emb = load_embeddings(args.checkpoint)

    # Auto-detect mapping file
    if args.mapping is None:
        # Try to find mapping file
        for candidate in [
            "data/taxonomy_edges_small.mapping.tsv",
            "data/taxonomy_edges.mapping.tsv",
        ]:
            if Path(candidate).exists():
                args.mapping = candidate
                break

    if args.mapping is None:
        print("❌ Could not find mapping file. Please specify with --mapping")
        sys.exit(1)

    # Load mapping
    tax2idx, idx2tax = load_mapping(args.mapping)
    if tax2idx is None:
        sys.exit(1)

    # Convert to set of numeric taxids
    valid_taxids = {int(t) for t in tax2idx.keys()}
    print(f"Dataset contains {len(valid_taxids):,} unique organisms")

    # Load taxonomy tree - find data directory at project root
    # Script is at: src/taxembed/visualization/umap_viz.py
    # Project root is 3 levels up
    project_root = Path(__file__).resolve().parents[3]
    default_data_dir = project_root / "data"

    if args.names or args.nodes:
        names_path = Path(args.names).resolve() if args.names else default_data_dir / "names.dmp"
        Path(args.nodes).resolve() if args.nodes else names_path.parent / "nodes.dmp"
        base_dir = names_path.parent
        names, nodes = load_taxonomy_tree(valid_taxids, base_dir=base_dir)
    else:
        # Use data directory relative to script location
        names, nodes = load_taxonomy_tree(valid_taxids, base_dir=default_data_dir)
    if names is None or nodes is None:
        print("❌ Could not load taxonomy tree")
        sys.exit(1)

    # Prepare child coloring if root taxid provided
    child_coloring = None
    if args.root_taxid is not None:
        child_coloring = args.root_taxid

    # Visualize
    visualize_multi_groups(
        emb,
        tax2idx,
        idx2tax,
        nodes,
        names,
        sample_size=args.sample,
        output_file=args.output,
        child_coloring=child_coloring,
        coloring_depth=args.children,
        clade_name=args.clade_name,
        epoch=args.epoch,
        loss=args.loss,
    )


if __name__ == "__main__":
    main()
