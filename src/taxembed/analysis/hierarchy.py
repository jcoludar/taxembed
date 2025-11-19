#!/usr/bin/env python3
"""
Analyze hierarchical structure using HYPERBOLIC (Poincaré) distance.
This is the correct metric for Poincaré embeddings, not Euclidean!
"""

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def poincare_distance(u, v, eps=1e-5):
    """
    Compute Poincaré distance between points u and v.

    d(u,v) = arcosh(1 + 2 * ||u-v||^2 / ((1-||u||^2)(1-||v||^2)))

    This is the TRUE hyperbolic distance in the Poincaré ball model.
    """
    # Compute squared norms
    u_norm_sq = np.sum(u**2)
    v_norm_sq = np.sum(v**2)

    # Clamp to stay inside the ball (norm < 1)
    u_norm_sq = min(u_norm_sq, 1 - eps)
    v_norm_sq = min(v_norm_sq, 1 - eps)

    # Compute squared distance
    diff_norm_sq = np.sum((u - v) ** 2)

    # Poincaré distance formula
    numerator = 2 * diff_norm_sq
    denominator = (1 - u_norm_sq) * (1 - v_norm_sq)

    # arcosh(1 + x) with numerical stability
    x = numerator / (denominator + eps)
    dist = np.arccosh(1 + x + eps)

    return dist


def poincare_distance_matrix(embeddings, indices):
    """Compute pairwise Poincaré distances for a subset of embeddings."""
    n = len(indices)
    distances = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            dist = poincare_distance(embeddings[indices[i]], embeddings[indices[j]])
            distances[i, j] = dist
            distances[j, i] = dist

    return distances


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

    # Check norms
    norms = np.linalg.norm(emb, axis=1)
    print(f"  Norms: min={norms.min():.4f}, mean={norms.mean():.4f}, max={norms.max():.4f}")
    if norms.max() >= 1.0:
        print("  ⚠️  WARNING: Some embeddings are outside the Poincaré ball (norm >= 1.0)!")

    return emb


def load_mapping(mapping_file):
    """Load index to TaxID mapping."""
    print(f"Loading mapping from {mapping_file}...")
    df = pd.read_csv(mapping_file, sep="\t", header=None, names=["idx", "taxid"])
    numeric_df = df[df["taxid"].str.isnumeric()]
    idx2tax = dict(zip(numeric_df["idx"], numeric_df["taxid"], strict=False))
    print(f"  ✓ Loaded {len(idx2tax):,} mappings")
    return idx2tax


def load_taxonomy_with_depth(valid_taxids):
    """Load taxonomy and compute depth for each node."""
    print("Loading taxonomy tree with depths...")

    # Load names
    names = {}
    with open("data/names.dmp") as f:
        for line in f:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 4 and parts[3] == "scientific name":
                taxid = int(parts[0])
                if taxid in valid_taxids:
                    names[taxid] = parts[1]

    # Load nodes with ranks
    taxonomy = {}
    with open("data/nodes.dmp") as f:
        for line in f:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 5:
                taxid = int(parts[0])
                if taxid in valid_taxids:
                    parent = int(parts[1])
                    rank = parts[2]
                    taxonomy[taxid] = {
                        "parent": parent,
                        "rank": rank,
                        "name": names.get(taxid, f"TaxID_{taxid}"),
                        "depth": None,  # Will compute
                    }

    # Compute depth for each node
    def get_depth(taxid, visited=None):
        if visited is None:
            visited = set()

        if taxid not in taxonomy:
            return 0

        if taxonomy[taxid]["depth"] is not None:
            return taxonomy[taxid]["depth"]

        if taxid in visited:  # Cycle detection
            return 0

        visited.add(taxid)
        parent = taxonomy[taxid]["parent"]

        if parent == taxid:  # Root
            taxonomy[taxid]["depth"] = 0
        else:
            taxonomy[taxid]["depth"] = get_depth(parent, visited) + 1

        return taxonomy[taxid]["depth"]

    for taxid in taxonomy:
        if taxonomy[taxid]["depth"] is None:
            get_depth(taxid)

    print(f"  ✓ Loaded {len(taxonomy):,} taxonomy nodes with depths")
    return taxonomy


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
        if parent == current:  # Root
            break
        current = parent

    return None


def analyze_depth_vs_norm(emb, idx2tax, taxonomy):
    """Check if depth correlates with radial position (CRITICAL check)."""
    print(f"\n{'=' * 80}")
    print("RADIAL MONOTONICITY CHECK")
    print(f"{'=' * 80}\n")

    max_idx = emb.shape[0] - 1

    depths = []
    norms = []

    for idx_str, taxid_str in idx2tax.items():
        idx = int(idx_str)
        if idx > max_idx:
            continue

        taxid = int(taxid_str)
        if taxid not in taxonomy:
            continue

        depth = taxonomy[taxid]["depth"]
        if depth is None:
            continue

        norm = np.linalg.norm(emb[idx])

        depths.append(depth)
        norms.append(norm)

    depths = np.array(depths)
    norms = np.array(norms)

    # Compute correlation
    from scipy.stats import pearsonr, spearmanr

    pearson_r, pearson_p = pearsonr(depths, norms)
    spearman_r, spearman_p = spearmanr(depths, norms)

    print("Depth vs. Norm correlation:")
    print(f"  Pearson:  r = {pearson_r:+.4f} (p = {pearson_p:.2e})")
    print(f"  Spearman: ρ = {spearman_r:+.4f} (p = {spearman_p:.2e})")

    if pearson_r > 0.5:
        assessment = "✅ GOOD - Deeper nodes are near boundary"
    elif pearson_r > 0.3:
        assessment = "⚠️ WEAK - Some depth structure exists"
    elif pearson_r > 0:
        assessment = "❌ POOR - Very weak depth structure"
    else:
        assessment = "❌ BROKEN - Negative correlation! Hierarchy is inverted"

    print(f"\nAssessment: {assessment}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Scatter plot with density
    ax1.hexbin(depths, norms, gridsize=50, cmap="YlOrRd", mincnt=1)
    ax1.set_xlabel("Taxonomic Depth", fontsize=12)
    ax1.set_ylabel("Embedding Norm (Distance from Origin)", fontsize=12)
    ax1.set_title(
        f"Depth vs. Radial Position\nPearson r = {pearson_r:.3f}", fontsize=14, fontweight="bold"
    )
    ax1.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(depths, norms, 1)
    p = np.poly1d(z)
    ax1.plot(
        depths,
        p(depths),
        "r--",
        linewidth=2,
        alpha=0.8,
        label=f"Trend: y = {z[0]:.4f}x + {z[1]:.4f}",
    )
    ax1.legend()

    # Box plot by depth bins
    depth_bins = np.percentile(depths, [0, 25, 50, 75, 100])
    depth_labels = [
        f"{int(depth_bins[i])}-{int(depth_bins[i + 1])}" for i in range(len(depth_bins) - 1)
    ]
    depth_binned = np.digitize(depths, depth_bins[1:-1])

    data_by_bin = [norms[depth_binned == i] for i in range(len(depth_bins) - 1)]
    bp = ax2.boxplot(data_by_bin, labels=depth_labels, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
    ax2.set_xlabel("Depth Quartile", fontsize=12)
    ax2.set_ylabel("Embedding Norm", fontsize=12)
    ax2.set_title("Norm Distribution by Depth", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("depth_vs_norm_analysis.png", dpi=150, bbox_inches="tight")
    print("\nSaved plot: depth_vs_norm_analysis.png")
    plt.close()

    return pearson_r, spearman_r


def analyze_hierarchical_clustering_hyperbolic(emb, idx2tax, taxonomy, rank="phylum"):
    """Analyze hierarchical clustering using HYPERBOLIC distance."""
    print(f"\n{'=' * 80}")
    print(f"HIERARCHICAL CLUSTERING - {rank.upper()} (HYPERBOLIC DISTANCE)")
    print(f"{'=' * 80}\n")

    max_idx = emb.shape[0] - 1

    # Map organisms to groups
    organism_to_group = {}
    group_names = {}

    for idx_str, taxid_str in idx2tax.items():
        idx = int(idx_str)
        if idx > max_idx:
            continue

        taxid = int(taxid_str)
        ancestor = get_ancestor_at_rank(taxid, taxonomy, rank)
        if ancestor:
            organism_to_group[idx] = ancestor
            if ancestor not in group_names and ancestor in taxonomy:
                group_names[ancestor] = taxonomy[ancestor]["name"]

    # Group by ancestor
    groups = defaultdict(list)
    for idx, group_id in organism_to_group.items():
        groups[group_id].append(idx)

    # Filter large groups
    min_size = 10
    large_groups = {gid: indices for gid, indices in groups.items() if len(indices) >= min_size}

    print(f"Found {len(set(organism_to_group.values()))} distinct {rank}s")
    print(f"{rank.capitalize()}s with ≥{min_size} organisms: {len(large_groups)}")

    # Show top groups
    group_sizes = [
        (gid, len(indices), group_names.get(gid, f"TaxID_{gid}"))
        for gid, indices in large_groups.items()
    ]
    group_sizes.sort(key=lambda x: -x[1])

    print(f"\nTop 10 {rank}s by organism count:")
    for i, (_gid, size, name) in enumerate(group_sizes[:10], 1):
        print(f"  {i:2d}. {name:40s}: {size:6,} organisms")

    # Compute hyperbolic distances (sample for efficiency)
    print("\nComputing pairwise HYPERBOLIC distances...")

    max_per_group = 100  # Limit for efficiency
    sampled_groups = {}
    for gid, indices in large_groups.items():
        if len(indices) > max_per_group:
            sampled_groups[gid] = list(np.random.choice(indices, max_per_group, replace=False))
        else:
            sampled_groups[gid] = list(indices)

    intra_distances = []
    inter_distances = []

    group_list = list(sampled_groups.items())

    for i, (_gid1, indices1) in enumerate(group_list):
        # Intra-group distances
        if len(indices1) >= 2:
            for ii in range(len(indices1)):
                for jj in range(ii + 1, min(ii + 20, len(indices1))):  # Limit pairs
                    dist = poincare_distance(emb[indices1[ii]], emb[indices1[jj]])
                    intra_distances.append(dist)

        # Inter-group distances (sample)
        for j in range(i + 1, min(i + 10, len(group_list))):  # Limit group pairs
            gid2, indices2 = group_list[j]

            # Sample pairs
            for _ in range(min(100, len(indices1) * len(indices2))):
                idx1 = np.random.choice(indices1)
                idx2 = np.random.choice(indices2)
                dist = poincare_distance(emb[idx1], emb[idx2])
                inter_distances.append(dist)

    intra_distances = np.array(intra_distances)
    inter_distances = np.array(inter_distances)

    print(f"\n{'=' * 80}")
    print("HYPERBOLIC DISTANCE STATISTICS")
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

    # Separation
    separation = np.mean(inter_distances) / np.mean(intra_distances)
    print(f"\nSeparation Ratio: {separation:.3f}x")

    if separation > 2.0:
        quality = "✅ EXCELLENT"
    elif separation > 1.5:
        quality = "✅ GOOD"
    elif separation > 1.2:
        quality = "⚠️ MODERATE"
    else:
        quality = "❌ POOR"

    print(f"Quality: {quality}")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bins = np.linspace(0, max(np.max(intra_distances), np.max(inter_distances)), 50)
    ax.hist(
        intra_distances, bins=bins, alpha=0.6, label=f"Intra-{rank}", color="blue", density=True
    )
    ax.hist(inter_distances, bins=bins, alpha=0.6, label=f"Inter-{rank}", color="red", density=True)
    ax.axvline(
        np.mean(intra_distances),
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Intra mean: {np.mean(intra_distances):.3f}",
    )
    ax.axvline(
        np.mean(inter_distances),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Inter mean: {np.mean(inter_distances):.3f}",
    )
    ax.set_xlabel("Poincaré Distance (Hyperbolic)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"{rank.capitalize()}-level Clustering (HYPERBOLIC)\nSeparation: {separation:.2f}x - {quality}",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = f"hierarchy_hyperbolic_{rank}.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot: {output_file}")
    plt.close()

    return separation, quality


def main():
    checkpoint = "taxonomy_model_hierarchical_small_v3_best.pth"
    mapping_file = "data/taxonomy_edges_small.mapping.tsv"

    # Load data
    emb = load_embeddings(checkpoint)
    idx2tax = load_mapping(mapping_file)

    valid_taxids = {int(t) for t in idx2tax.values()}
    taxonomy = load_taxonomy_with_depth(valid_taxids)

    # 1. Check radial monotonicity
    pearson_r, spearman_r = analyze_depth_vs_norm(emb, idx2tax, taxonomy)

    # 2. Analyze hierarchy with hyperbolic distance
    results = {}
    for rank in ["phylum", "class", "order"]:
        try:
            sep, qual = analyze_hierarchical_clustering_hyperbolic(
                emb, idx2tax, taxonomy, rank=rank
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
    print(f"Depth-Norm Correlation: {pearson_r:+.3f} (Pearson)")
    print(f"\n{'Rank':<12} {'Separation':>12} {'Quality':>15}")
    print("-" * 40)
    for rank, res in results.items():
        print(f"{rank.capitalize():<12} {res['separation']:>11.2f}x {res['quality']:>15}")


if __name__ == "__main__":
    main()
