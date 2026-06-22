"""Slice metazoa_v2 embedding by phylum, run hyperbolic separation per slice.

Tests the fork: does v2's POOR 1.05x metazoa-wide separation come from
(a) the global density of 498k nodes crowding lateral structure (SCALE), or
(b) the ranking loss capping lateral structure regardless of subset (LOSS)?

For each target phylum, we slice the embedding to just that phylum's indices
and compute intra/inter-group separation at class/order/family ranks. If
echino-slice matches standalone echino_euclparam (~1.21x), the recipe transfers
and scale is the bottleneck. If echino-slice also stays ~1.05x, ranking loss
is the hard ceiling and softmax is the only remaining lever.
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

# Reuse loaders + distance + ancestor walk from the canonical analyzer.
from analyze_hierarchy_hyperbolic import (
    load_embeddings,
    load_mapping,
    load_taxonomy_with_depth,
    poincare_distance,
    get_ancestor_at_rank,
)


CHECKPOINT = REPO_ROOT / "artifacts/tags/metazoa_v2_euclparam/metazoa_v2_euclparam_best.pth"
MAPPING = REPO_ROOT / "data/taxopy/metazoa_33208_clean/taxonomy_edges_metazoa_33208_clean.mapping.tsv"
DATA_DIR = REPO_ROOT / "data"

# Phyla to slice. The four largest, plus the canonical echino reference.
TARGET_PHYLA = [
    "Echinodermata",   # 3,965 — direct comparison to standalone echino_euclparam (1.21x)
    "Mollusca",        # 32,016
    "Chordata",        # 92,811
    "Arthropoda",      # 324,983 — close to whole-metazoa density
]

RANKS = ["class", "order", "family"]
MIN_GROUP_SIZE = 10
MAX_PER_GROUP = 100
RNG = np.random.default_rng(seed=42)


def quality_label(sep):
    if sep > 2.0:
        return "EXCELLENT"
    if sep > 1.5:
        return "GOOD"
    if sep > 1.2:
        return "MODERATE"
    return "POOR"


def compute_separation(emb, indices_in_slice, idx2tax, taxonomy, rank):
    """Compute intra/inter mean hyperbolic-distance ratio for one rank within a slice.

    Mirrors analyze_hierarchical_clustering_hyperbolic's logic with the same
    sampling caps (100 per group, 20 nearest pairs per anchor, 10 group-pairs
    looked at per anchor group). Returns (n_groups, separation, intra_mean,
    inter_mean) or None if fewer than 2 groups meet the size floor.
    """
    organism_to_group = {}
    for idx in indices_in_slice:
        taxid = idx2tax.get(idx)
        if taxid is None:
            continue
        ancestor = get_ancestor_at_rank(taxid, taxonomy, rank)
        if ancestor is None:
            continue
        organism_to_group[idx] = ancestor

    groups = defaultdict(list)
    for idx, gid in organism_to_group.items():
        groups[gid].append(idx)
    large = {gid: idxs for gid, idxs in groups.items() if len(idxs) >= MIN_GROUP_SIZE}
    if len(large) < 2:
        return None

    sampled = {}
    for gid, idxs in large.items():
        if len(idxs) > MAX_PER_GROUP:
            sampled[gid] = list(RNG.choice(idxs, MAX_PER_GROUP, replace=False))
        else:
            sampled[gid] = list(idxs)

    intra, inter = [], []
    group_list = list(sampled.items())
    for i, (gid1, idxs1) in enumerate(group_list):
        if len(idxs1) >= 2:
            for ii in range(len(idxs1)):
                for jj in range(ii + 1, min(ii + 20, len(idxs1))):
                    intra.append(poincare_distance(emb[idxs1[ii]], emb[idxs1[jj]]))
        for j in range(i + 1, min(i + 10, len(group_list))):
            _, idxs2 = group_list[j]
            for _ in range(min(100, len(idxs1) * len(idxs2))):
                a = RNG.choice(idxs1)
                b = RNG.choice(idxs2)
                inter.append(poincare_distance(emb[a], emb[b]))

    if not intra or not inter:
        return None
    intra_mean = float(np.mean(intra))
    inter_mean = float(np.mean(inter))
    return len(large), inter_mean / intra_mean, intra_mean, inter_mean


def find_indices_in_phylum(emb, idx2tax, taxonomy, target_phylum_name):
    """Return indices whose lineage passes through a phylum-rank ancestor with the given name."""
    target_taxids = {
        tid for tid, node in taxonomy.items()
        if node["rank"] == "phylum" and node["name"] == target_phylum_name
    }
    if not target_taxids:
        return [], None
    target_taxid = next(iter(target_taxids))

    indices = []
    max_idx = emb.shape[0] - 1
    for idx, taxid in idx2tax.items():
        if idx > max_idx:
            continue
        if get_ancestor_at_rank(taxid, taxonomy, "phylum") == target_taxid:
            indices.append(idx)
    return indices, target_taxid


if __name__ == "__main__":
    print(f"Loading embeddings ({CHECKPOINT.name})...")
    emb = load_embeddings(CHECKPOINT)
    idx2tax = load_mapping(MAPPING)
    taxonomy = load_taxonomy_with_depth(set(idx2tax.values()), data_dir=DATA_DIR)

    print(f"\n{'=' * 84}")
    print("PER-PHYLUM HYPERBOLIC SEPARATION (sliced from metazoa_v2_euclparam_best)")
    print(f"{'=' * 84}\n")
    print(f"{'Phylum':<18}{'N (slice)':>11}{'Rank':<10}{'Groups':>9}{'Sep':>9}{'Quality':>13}")
    print("-" * 84)

    rows = []
    for phylum in TARGET_PHYLA:
        indices, target_tax = find_indices_in_phylum(emb, idx2tax, taxonomy, phylum)
        if not indices:
            print(f"{phylum:<18}  (phylum taxid not found in taxonomy; skipping)")
            continue
        for rank in RANKS:
            result = compute_separation(emb, indices, idx2tax, taxonomy, rank)
            if result is None:
                print(f"{phylum:<18}{len(indices):>11,}{rank:<10}{'-':>9}{'N/A':>9}{'(<2 grps)':>13}")
                continue
            n_groups, sep, intra_m, inter_m = result
            qual = quality_label(sep)
            print(f"{phylum:<18}{len(indices):>11,}{rank:<10}{n_groups:>9}{sep:>8.3f}x{qual:>13}")
            rows.append({
                "phylum": phylum,
                "n_slice": len(indices),
                "rank": rank,
                "n_groups": n_groups,
                "separation": sep,
                "intra_mean": intra_m,
                "inter_mean": inter_m,
                "quality": qual,
            })

    print()
    print("Reference (whole-Metazoa from earlier analyze run):")
    print("  phylum 1.048x | class 1.079x | order 1.085x | family 1.054x   (all POOR)")
    print()
    print("Reference (standalone echino_euclparam, 4k subset, same ranking recipe):")
    print("  separation 1.21x  (MODERATE) per PROJECT_STATE.md")
    print()
    print("Reference (standalone echino_softmax, 4k subset, winning recipe):")
    print("  class 1.46x / order 3.01x / family 2.63x  (EXCELLENT)")
    print()
    import csv
    out = REPO_ROOT / "artifacts/tags/metazoa_v2_euclparam/analysis/per_phylum_separation.tsv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"Wrote: {out}")
