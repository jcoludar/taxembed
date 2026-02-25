#!/usr/bin/env python3
"""
Analyze hierarchical structure using HYPERBOLIC (Poincaré) distance.
This is the correct metric for Poincaré embeddings, not Euclidean!

Usage:
    # By tag (auto-discovers paths from artifacts/tags/<tag>/run.json):
    python analyze_hierarchy_hyperbolic.py --tag echinoderms_v3

    # By explicit paths:
    python analyze_hierarchy_hyperbolic.py \
        --checkpoint model_best.pth \
        --mapping data/mapping.tsv

    # Override output directory:
    python analyze_hierarchy_hyperbolic.py --tag echinoderms_v3 -o results/
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import taxopy
import torch
from scipy.stats import pearsonr, spearmanr


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # poincare-embeddings/
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "tags"


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_embeddings(ckpt_path):
    """Load embeddings from checkpoint."""
    print(f"Loading embeddings from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        emb = sd["lt.weight"].detach().cpu().numpy()
    elif "embeddings" in ckpt:
        emb = ckpt["embeddings"].detach().cpu().numpy()
    else:
        raise ValueError("Cannot find embeddings in checkpoint")

    print(f"  Shape: {emb.shape}")

    norms = np.linalg.norm(emb, axis=1)
    print(f"  Norms: min={norms.min():.4f}, mean={norms.mean():.4f}, max={norms.max():.4f}")
    if norms.max() >= 1.0:
        print(f"  WARNING: Some embeddings are outside the Poincare ball (norm >= 1.0)!")

    return emb


def load_mapping(mapping_file):
    """Load index to TaxID mapping.

    Handles both header (taxid, idx) and headerless (two-column) TSV files.
    """
    print(f"Loading mapping from {mapping_file}...")
    df = pd.read_csv(mapping_file, sep="\t", dtype=str)

    # Detect header vs headerless
    if "taxid" in df.columns and "idx" in df.columns:
        pass  # already has headers
    else:
        df = pd.read_csv(mapping_file, sep="\t", dtype=str,
                         header=None, names=["taxid", "idx"])

    numeric_mask = df["taxid"].str.isnumeric() & df["idx"].str.isnumeric()
    df = df[numeric_mask]
    idx2tax = dict(zip(df["idx"].astype(int), df["taxid"].astype(int)))
    print(f"  Loaded {len(idx2tax):,} mappings")
    return idx2tax


def load_taxonomy_with_depth(valid_taxids, data_dir=None):
    """Load taxonomy using TaxoPy (handles both old and new NCBI dump formats).

    Returns a dict: taxid -> {"parent", "rank", "name", "depth"}.
    """
    data_dir = Path(data_dir or DATA_DIR)
    print(f"Loading taxonomy via TaxoPy from {data_dir}...")

    try:
        taxdb = taxopy.TaxDb(taxdb_dir=str(data_dir))
    except Exception as e:
        print(f"  ERROR: Could not load taxonomy: {e}")
        sys.exit(1)

    taxonomy = {}
    for taxid in valid_taxids:
        parent = taxdb.taxid2parent.get(taxid)
        rank = taxdb.taxid2rank.get(taxid, "no rank")
        name = taxdb.taxid2name.get(taxid, f"TaxID_{taxid}")
        if parent is not None:
            taxonomy[taxid] = {
                "parent": int(parent),
                "rank": rank,
                "name": name,
                "depth": None,
            }

    # Compute depth via parent traversal
    def get_depth(taxid, visited=None):
        if visited is None:
            visited = set()
        if taxid not in taxonomy:
            return 0
        if taxonomy[taxid]["depth"] is not None:
            return taxonomy[taxid]["depth"]
        if taxid in visited:
            return 0
        visited.add(taxid)
        parent = taxonomy[taxid]["parent"]
        if parent == taxid:
            taxonomy[taxid]["depth"] = 0
        else:
            taxonomy[taxid]["depth"] = get_depth(parent, visited) + 1
        return taxonomy[taxid]["depth"]

    for taxid in taxonomy:
        if taxonomy[taxid]["depth"] is None:
            get_depth(taxid)

    print(f"  Loaded {len(taxonomy):,} taxonomy nodes with depths")
    return taxonomy


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def poincare_distance(u, v, eps=1e-5):
    """Compute Poincare distance between points u and v."""
    u_norm_sq = min(np.sum(u ** 2), 1 - eps)
    v_norm_sq = min(np.sum(v ** 2), 1 - eps)
    diff_norm_sq = np.sum((u - v) ** 2)

    x = 2 * diff_norm_sq / ((1 - u_norm_sq) * (1 - v_norm_sq) + eps)
    return np.arccosh(1 + x + eps)


def get_ancestor_at_rank(taxid, taxonomy, target_rank):
    """Find the ancestor of a taxid at a specific rank."""
    visited = set()
    current = taxid
    while current in taxonomy and current not in visited:
        visited.add(current)
        node = taxonomy[current]
        if node["rank"] == target_rank:
            return current
        parent = node["parent"]
        if parent == current:
            break
        current = parent
    return None


# ---------------------------------------------------------------------------
# Analysis routines
# ---------------------------------------------------------------------------

def analyze_depth_vs_norm(emb, idx2tax, taxonomy, output_dir):
    """Check if depth correlates with radial position."""
    print(f"\n{'=' * 80}")
    print("RADIAL MONOTONICITY CHECK")
    print(f"{'=' * 80}\n")

    max_idx = emb.shape[0] - 1
    depths, norms = [], []

    for idx, taxid in idx2tax.items():
        if idx > max_idx or taxid not in taxonomy:
            continue
        depth = taxonomy[taxid]["depth"]
        if depth is None:
            continue
        depths.append(depth)
        norms.append(np.linalg.norm(emb[idx]))

    depths = np.array(depths)
    norms = np.array(norms)

    pearson_r, pearson_p = pearsonr(depths, norms)
    spearman_r, spearman_p = spearmanr(depths, norms)

    print(f"Depth vs. Norm correlation:")
    print(f"  Pearson:  r = {pearson_r:+.4f} (p = {pearson_p:.2e})")
    print(f"  Spearman: rho = {spearman_r:+.4f} (p = {spearman_p:.2e})")

    if pearson_r > 0.5:
        assessment = "GOOD - Deeper nodes are near boundary"
    elif pearson_r > 0.3:
        assessment = "WEAK - Some depth structure exists"
    elif pearson_r > 0:
        assessment = "POOR - Very weak depth structure"
    else:
        assessment = "BROKEN - Negative correlation! Hierarchy is inverted"
    print(f"\nAssessment: {assessment}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.hexbin(depths, norms, gridsize=50, cmap="YlOrRd", mincnt=1)
    ax1.set_xlabel("Taxonomic Depth", fontsize=12)
    ax1.set_ylabel("Embedding Norm (Distance from Origin)", fontsize=12)
    ax1.set_title(
        f"Depth vs. Radial Position\nPearson r = {pearson_r:.3f}",
        fontsize=14, fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)
    z = np.polyfit(depths, norms, 1)
    p = np.poly1d(z)
    ax1.plot(depths, p(depths), "r--", linewidth=2, alpha=0.8,
             label=f"Trend: y = {z[0]:.4f}x + {z[1]:.4f}")
    ax1.legend()

    depth_bins = np.percentile(depths, [0, 25, 50, 75, 100])
    depth_labels = [f"{int(depth_bins[i])}-{int(depth_bins[i + 1])}"
                    for i in range(len(depth_bins) - 1)]
    depth_binned = np.digitize(depths, depth_bins[1:-1])
    data_by_bin = [norms[depth_binned == i] for i in range(len(depth_bins) - 1)]
    bp = ax2.boxplot(data_by_bin, tick_labels=depth_labels, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
    ax2.set_xlabel("Depth Quartile", fontsize=12)
    ax2.set_ylabel("Embedding Norm", fontsize=12)
    ax2.set_title("Norm Distribution by Depth", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = output_dir / "depth_vs_norm_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot: {out}")
    plt.close()

    return pearson_r, spearman_r


def analyze_hierarchical_clustering_hyperbolic(emb, idx2tax, taxonomy, rank, output_dir):
    """Analyze hierarchical clustering using HYPERBOLIC distance."""
    print(f"\n{'=' * 80}")
    print(f"HIERARCHICAL CLUSTERING - {rank.upper()} (HYPERBOLIC DISTANCE)")
    print(f"{'=' * 80}\n")

    max_idx = emb.shape[0] - 1

    organism_to_group = {}
    group_names = {}

    for idx, taxid in idx2tax.items():
        if idx > max_idx:
            continue
        ancestor = get_ancestor_at_rank(taxid, taxonomy, rank)
        if ancestor:
            organism_to_group[idx] = ancestor
            if ancestor not in group_names and ancestor in taxonomy:
                group_names[ancestor] = taxonomy[ancestor]["name"]

    groups = defaultdict(list)
    for idx, group_id in organism_to_group.items():
        groups[group_id].append(idx)

    min_size = 10
    large_groups = {gid: indices for gid, indices in groups.items()
                    if len(indices) >= min_size}

    print(f"Found {len(set(organism_to_group.values()))} distinct {rank}s")
    print(f"{rank.capitalize()}s with >={min_size} organisms: {len(large_groups)}")

    group_sizes = [(gid, len(indices), group_names.get(gid, f"TaxID_{gid}"))
                   for gid, indices in large_groups.items()]
    group_sizes.sort(key=lambda x: -x[1])

    print(f"\nTop 10 {rank}s by organism count:")
    for i, (gid, size, name) in enumerate(group_sizes[:10], 1):
        print(f"  {i:2d}. {name:40s}: {size:6,} organisms")

    print(f"\nComputing pairwise HYPERBOLIC distances...")

    max_per_group = 100
    sampled_groups = {}
    for gid, indices in large_groups.items():
        if len(indices) > max_per_group:
            sampled_groups[gid] = list(np.random.choice(indices, max_per_group, replace=False))
        else:
            sampled_groups[gid] = list(indices)

    intra_distances = []
    inter_distances = []

    group_list = list(sampled_groups.items())

    for i, (gid1, indices1) in enumerate(group_list):
        if len(indices1) >= 2:
            for ii in range(len(indices1)):
                for jj in range(ii + 1, min(ii + 20, len(indices1))):
                    dist = poincare_distance(emb[indices1[ii]], emb[indices1[jj]])
                    intra_distances.append(dist)

        for j in range(i + 1, min(i + 10, len(group_list))):
            gid2, indices2 = group_list[j]
            for _ in range(min(100, len(indices1) * len(indices2))):
                idx1 = np.random.choice(indices1)
                idx2 = np.random.choice(indices2)
                dist = poincare_distance(emb[idx1], emb[idx2])
                inter_distances.append(dist)

    intra_distances = np.array(intra_distances)
    inter_distances = np.array(inter_distances)

    if len(intra_distances) == 0 or len(inter_distances) == 0:
        which = "intra" if len(intra_distances) == 0 else "inter"
        print(f"\n  Skipping {rank}: no {which}-group distances "
              f"(need >=2 groups with >={min_size} members)")
        return float("nan"), "N/A (single group)"

    print(f"\n{'=' * 80}")
    print(f"HYPERBOLIC DISTANCE STATISTICS")
    print(f"{'=' * 80}")
    print(f"\nIntra-{rank} distances:")
    print(f"  Count:  {len(intra_distances):,}")
    print(f"  Mean:   {np.mean(intra_distances):.6f}")
    print(f"  Median: {np.median(intra_distances):.6f}")
    print(f"  Std:    {np.std(intra_distances):.6f}")

    print(f"\nInter-{rank} distances:")
    print(f"  Count:  {len(inter_distances):,}")
    print(f"  Mean:   {np.mean(inter_distances):.6f}")
    print(f"  Median: {np.median(inter_distances):.6f}")
    print(f"  Std:    {np.std(inter_distances):.6f}")

    separation = np.mean(inter_distances) / np.mean(intra_distances)
    print(f"\nSeparation Ratio: {separation:.3f}x")

    if separation > 2.0:
        quality = "EXCELLENT"
    elif separation > 1.5:
        quality = "GOOD"
    elif separation > 1.2:
        quality = "MODERATE"
    else:
        quality = "POOR"
    print(f"Quality: {quality}")

    fig, ax = plt.subplots(figsize=(12, 6))
    bins = np.linspace(0, max(np.max(intra_distances), np.max(inter_distances)), 50)
    ax.hist(intra_distances, bins=bins, alpha=0.6, label=f"Intra-{rank}", color="blue", density=True)
    ax.hist(inter_distances, bins=bins, alpha=0.6, label=f"Inter-{rank}", color="red", density=True)
    ax.axvline(np.mean(intra_distances), color="blue", linestyle="--", linewidth=2,
               label=f"Intra mean: {np.mean(intra_distances):.3f}")
    ax.axvline(np.mean(inter_distances), color="red", linestyle="--", linewidth=2,
               label=f"Inter mean: {np.mean(inter_distances):.3f}")
    ax.set_xlabel("Poincare Distance (Hyperbolic)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"{rank.capitalize()}-level Clustering (HYPERBOLIC)\n"
                 f"Separation: {separation:.2f}x - {quality}",
                 fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = output_dir / f"hierarchy_hyperbolic_{rank}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot: {out}")
    plt.close()

    return separation, quality


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def resolve_paths_from_tag(tag):
    """Auto-discover checkpoint, mapping, and output dir from an artifacts tag."""
    import re
    slug = re.sub(r"[^a-z0-9_-]+", "_", tag.lower()).strip("_") or "run"
    tag_dir = ARTIFACTS_DIR / slug
    meta_path = tag_dir / "run.json"

    if not meta_path.exists():
        raise SystemExit(f"No run metadata for tag '{tag}' ({meta_path} missing)")

    metadata = json.loads(meta_path.read_text())
    paths = metadata.get("training", {}).get("paths", {})

    checkpoint = Path(paths.get("best_checkpoint", paths.get("checkpoint_base", "")))
    mapping = Path(paths.get("mapping", ""))

    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    if not mapping.exists():
        raise SystemExit(f"Mapping not found: {mapping}")

    return checkpoint, mapping, tag_dir


def main():
    parser = argparse.ArgumentParser(
        description="Analyze hierarchical structure using Poincare (hyperbolic) distance",
    )
    parser.add_argument("--tag", help="Artifact tag (auto-discovers paths from run.json)")
    parser.add_argument("--checkpoint", help="Path to checkpoint file")
    parser.add_argument("--mapping", help="Path to mapping TSV file")
    parser.add_argument("-o", "--output-dir", help="Output directory for plots")
    parser.add_argument("--data-dir", default=str(DATA_DIR),
                        help=f"Taxonomy dump directory (default: {DATA_DIR})")
    parser.add_argument("--ranks", nargs="+", default=["phylum", "class", "order"],
                        help="Taxonomic ranks to analyze (default: phylum class order)")

    args = parser.parse_args()

    # Resolve paths
    if args.tag:
        checkpoint, mapping, tag_dir = resolve_paths_from_tag(args.tag)
        output_dir = Path(args.output_dir) if args.output_dir else tag_dir
    elif args.checkpoint and args.mapping:
        checkpoint = Path(args.checkpoint)
        mapping = Path(args.mapping)
        output_dir = Path(args.output_dir) if args.output_dir else Path(".")
    else:
        parser.error("Provide --tag or both --checkpoint and --mapping")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    emb = load_embeddings(checkpoint)
    idx2tax = load_mapping(mapping)

    valid_taxids = set(int(t) for t in idx2tax.values())
    taxonomy = load_taxonomy_with_depth(valid_taxids, data_dir=args.data_dir)

    # 1. Check radial monotonicity
    pearson_r, spearman_r = analyze_depth_vs_norm(emb, idx2tax, taxonomy, output_dir)

    # 2. Analyze hierarchy with hyperbolic distance
    results = {}
    for rank in args.ranks:
        try:
            sep, qual = analyze_hierarchical_clustering_hyperbolic(
                emb, idx2tax, taxonomy, rank, output_dir,
            )
            results[rank] = {"separation": sep, "quality": qual}
        except Exception as e:
            print(f"\nError analyzing {rank}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY - HYPERBOLIC ANALYSIS")
    print(f"{'=' * 80}\n")
    print(f"Depth-Norm Correlation: {pearson_r:+.3f} (Pearson), {spearman_r:+.3f} (Spearman)")
    print(f"\n{'Rank':<12} {'Separation':>12} {'Quality':>15}")
    print("-" * 40)
    for rank, res in results.items():
        print(f"{rank.capitalize():<12} {res['separation']:>11.2f}x {res['quality']:>15}")

    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()
