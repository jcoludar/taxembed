#!/usr/bin/env python3
"""
Audit NCBI taxonomy noise in a clade dataset.

Resolves every taxid in a .mapping.tsv to its scientific name and rank via the
NCBI dump files, then categorises nodes as clean or noisy. Uses a leaf-pruning
strategy: only remove noisy LEAF nodes (and "unclassified X" container nodes
whose children are all removed). Legitimate internal clades are never removed.

Usage (from repo root):
    uv run python scripts/audit_taxonomy_noise.py \
        --dataset mollusca_6447 \
        --out artifacts/audit_mollusca.png
"""

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TAXOPY_DIR = DATA_DIR / "taxopy"

# Noise categories for name-based detection.
# These are applied to scientific names; only LEAF nodes matching are removed.
NOISE_PATTERNS = [
    ("sp.", r"\bsp\."),                               # Genus sp. (unnamed species)
    ("nr./cf./aff.", r"\b(?:nr|cf|aff)\.\s"),       # Near/confer/affinity
    ("environmental", r"\benvironmental\b"),          # Environmental samples
    ("unidentified", r"\bunidentified\b"),            # Unidentified
    ("uncultured", r"\buncultured\b"),                # Uncultured organisms
    ("hybrid", r"\b[Hh]ybrid\b"),                     # Hybrid taxa
]

# "Container" noise: internal nodes that group junk. Removed only if ALL
# their children are also removed (bottom-up pruning).
CONTAINER_PATTERNS = [
    ("unclassified", r"^unclassified\s"),             # "unclassified Gastropoda"
    ("incertae sedis", r"\bincertae\s+sedis\b"),      # Uncertain placement
    ("environmental samples", r"^environmental\s+samples$"),
]

# Ranks we consider "stale NCBI artefacts" at the leaf level only
STALE_LEAF_RANKS = {"no rank"}  # Only flag if also matching a noise pattern


def parse_names_dmp(names_path: Path, taxids: set) -> dict:
    """Parse names.dmp, return {taxid: scientific_name} for taxids in set."""
    names = {}
    with open(names_path) as f:
        for line in f:
            parts = line.strip().rstrip("|").split("\t|\t")
            if len(parts) < 4:
                continue
            tid = int(parts[0].strip())
            if tid in taxids and parts[3].strip() == "scientific name":
                names[tid] = parts[1].strip()
    return names


def parse_nodes_dmp(nodes_path: Path, taxids: set) -> dict:
    """Parse nodes.dmp, return {taxid: rank} for taxids in set."""
    ranks = {}
    with open(nodes_path) as f:
        for line in f:
            parts = line.strip().rstrip("|").split("\t|\t")
            tid = int(parts[0].strip())
            if tid in taxids:
                ranks[tid] = parts[2].strip()
    return ranks


def classify_name_noise(name: str) -> list[str]:
    """Return noise category labels matching this name (leaf patterns)."""
    labels = []
    for cat, pattern in NOISE_PATTERNS:
        if re.search(pattern, name, re.IGNORECASE):
            labels.append(cat)
    return labels


def is_container(name: str) -> str | None:
    """Return container category if name matches, else None."""
    for cat, pattern in CONTAINER_PATTERNS:
        if re.search(pattern, name, re.IGNORECASE):
            return cat
    return None


def build_tree(dataset_dir: Path, dataset_name: str, all_taxids: set):
    """Load edgelist and build parent_of and children_of dicts."""
    edge_path = dataset_dir / f"taxonomy_edges_{dataset_name}.edgelist"
    parent_of = {}
    children_of = defaultdict(list)
    with open(edge_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                parent, child = int(parts[0]), int(parts[1])
                if child != parent:
                    parent_of[child] = parent
                    children_of[parent].append(child)
    return parent_of, children_of


def compute_subtree_sizes(children_of: dict, all_taxids: set) -> dict:
    """Compute subtree size for each node."""
    sizes = {}
    def _dfs(node):
        s = 1
        for c in children_of.get(node, []):
            s += _dfs(c)
        sizes[node] = s
        return s

    roots = [t for t in all_taxids
             if all(t not in children_of.get(p, []) for p in [])
             or t not in {c for kids in children_of.values() for c in kids}]
    # Simpler: any node that is NOT a child of anyone
    all_children = {c for kids in children_of.values() for c in kids}
    roots = [t for t in all_taxids if t not in all_children]
    for r in roots:
        _dfs(r)
    for t in all_taxids:
        if t not in sizes:
            sizes[t] = 1
    return sizes


def bottom_up_prune(children_of: dict, names: dict, ranks: dict, all_taxids: set):
    """
    Bottom-up leaf pruning strategy:
    1. Mark leaves that match noise patterns → remove
    2. After removing noisy leaves, if a container node ("unclassified X")
       has ALL children removed → remove it too
    3. Repeat until stable

    Returns: (removed_set, removal_reason dict)
    """
    removed = set()
    reason = {}

    # Build mutable children sets
    live_children = {t: set(children_of.get(t, [])) for t in all_taxids}

    changed = True
    while changed:
        changed = False
        for tid in list(all_taxids - removed):
            # Skip if has live children (not a leaf)
            alive_kids = live_children.get(tid, set()) - removed
            if alive_kids:
                continue

            # This is a leaf. Check if noisy.
            name = names.get(tid, "")
            cats = classify_name_noise(name)
            if cats:
                removed.add(tid)
                reason[tid] = "|".join(cats)
                changed = True
                continue

            # Not a noisy leaf — but is it a taxid not in the NCBI dump?
            # (4,585 "unknown" rank nodes = stale taxids)
            rank = ranks.get(tid, "unknown")
            if rank == "unknown":
                removed.add(tid)
                reason[tid] = "stale_taxid"
                changed = True
                continue

        # Container collapse: "unclassified X" or "incertae sedis" nodes
        # where ALL children have been removed → remove the container too
        for tid in list(all_taxids - removed):
            name = names.get(tid, "")
            container_cat = is_container(name)
            if container_cat is None:
                continue
            alive_kids = live_children.get(tid, set()) - removed
            if not alive_kids and tid not in removed:
                removed.add(tid)
                reason[tid] = f"empty_container:{container_cat}"
                changed = True

    return removed, reason


def main():
    parser = argparse.ArgumentParser(description="Audit NCBI taxonomy noise in a clade")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. mollusca_6447)")
    parser.add_argument("--out", type=Path, default=None, help="Output plot path (.png)")
    parser.add_argument("--names-dmp", type=Path, default=DATA_DIR / "names.dmp")
    parser.add_argument("--nodes-dmp", type=Path, default=DATA_DIR / "nodes.dmp")
    parser.add_argument("--csv", type=Path, default=None, help="Write full audit to CSV")
    args = parser.parse_args()

    dataset_dir = TAXOPY_DIR / args.dataset
    mapping_path = dataset_dir / f"taxonomy_edges_{args.dataset}.mapping.tsv"

    # ── Load taxids ───────────────────────────────────────────────────────
    print(f"Loading mapping from {mapping_path}...")
    mapping_df = pd.read_csv(mapping_path, sep="\t")
    taxids = set(mapping_df["taxid"].astype(int))
    print(f"  {len(taxids):,} unique taxids")

    # ── Resolve names and ranks ───────────────────────────────────────────
    print(f"Parsing {args.names_dmp.name}...")
    names = parse_names_dmp(args.names_dmp, taxids)
    print(f"  Resolved {len(names):,} / {len(taxids):,} names")

    print(f"Parsing {args.nodes_dmp.name}...")
    ranks = parse_nodes_dmp(args.nodes_dmp, taxids)
    print(f"  Resolved {len(ranks):,} / {len(taxids):,} ranks")

    unresolved = taxids - set(names.keys())
    if unresolved:
        print(f"  ⚠ {len(unresolved):,} taxids not in dump (stale/merged)")

    # ── Build tree ────────────────────────────────────────────────────────
    parent_of, children_of = build_tree(dataset_dir, args.dataset, taxids)
    subtree_sizes = compute_subtree_sizes(children_of, taxids)

    # ── Classify each taxid (for reporting) ───────────────────────────────
    print("\nClassifying nodes...")
    noise_counts = Counter()
    noise_by_category = defaultdict(set)
    all_noisy_names = {}  # tid → [categories]

    for tid in sorted(taxids):
        name = names.get(tid, f"taxid:{tid}")
        cats = classify_name_noise(name)
        container = is_container(name)
        if container:
            cats.append(f"container:{container}")
        if ranks.get(tid, "unknown") == "unknown":
            cats.append("stale_taxid")
        if cats:
            all_noisy_names[tid] = cats
            for c in cats:
                noise_counts[c] += 1
                noise_by_category[c].add(tid)

    # ── Bottom-up pruning ─────────────────────────────────────────────────
    print("\nRunning bottom-up leaf pruning...")
    removed, removal_reasons = bottom_up_prune(children_of, names, ranks, taxids)
    kept = taxids - removed

    # ── Build records for DataFrame ───────────────────────────────────────
    records = []
    for tid in sorted(taxids):
        name = names.get(tid, f"taxid:{tid}")
        rank = ranks.get(tid, "unknown")
        cats = all_noisy_names.get(tid, [])
        records.append({
            "taxid": tid,
            "name": name,
            "rank": rank,
            "subtree_size": subtree_sizes.get(tid, 1),
            "noise_categories": "|".join(cats) if cats else "",
            "is_flagged": len(cats) > 0,
            "is_removed": tid in removed,
            "removal_reason": removal_reasons.get(tid, ""),
        })

    df = pd.DataFrame(records)
    n_total = len(df)
    n_flagged = df["is_flagged"].sum()
    n_removed = df["is_removed"].sum()
    n_kept = n_total - n_removed

    # Count removal reasons
    reason_counts = Counter()
    for r in removal_reasons.values():
        for part in r.split("|"):
            reason_counts[part] += 1

    # ── Print summary ─────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  TAXONOMY AUDIT: {args.dataset}")
    print(f"{'='*65}")
    print(f"  Total nodes          : {n_total:>8,}")
    print(f"  Flagged (any noise)  : {n_flagged:>8,}  ({100*n_flagged/n_total:.1f}%)")
    print(f"  Removed (leaf-prune) : {n_removed:>8,}  ({100*n_removed/n_total:.1f}%)")
    print(f"  Kept (clean dataset) : {n_kept:>8,}  ({100*n_kept/n_total:.1f}%)")
    print()

    print("  Noise flags (a node can appear in multiple categories):")
    for cat, count in noise_counts.most_common():
        print(f"    {cat:>25s} : {count:>6,}  ({100*count/n_total:.1f}%)")

    print(f"\n  Removal reasons:")
    for reason, count in reason_counts.most_common():
        print(f"    {reason:>25s} : {count:>6,}")

    # ── Rank distribution ─────────────────────────────────────────────────
    print("\n  Rank distribution (top 15):")
    rank_counts = df["rank"].value_counts().head(15)
    for rank_name, count in rank_counts.items():
        removed_in_rank = df[(df["rank"] == rank_name) & df["is_removed"]].shape[0]
        pct = 100 * removed_in_rank / count if count > 0 else 0
        print(f"    {rank_name:>20s} : {count:>6,}  ({removed_in_rank:,} removed = {pct:.0f}%)")

    # ── Example removed names ─────────────────────────────────────────────
    print("\n  Example removed names (by reason):")
    reason_examples = defaultdict(list)
    for tid, r in removal_reasons.items():
        primary = r.split("|")[0]
        if len(reason_examples[primary]) < 5:
            reason_examples[primary].append(names.get(tid, str(tid)))
    for reason, examples in sorted(reason_examples.items()):
        print(f"    {reason}: {', '.join(examples)}")

    # ── Save CSV ──────────────────────────────────────────────────────────
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\n  Full audit saved to {args.csv}")

    # ── Plots ─────────────────────────────────────────────────────────────
    out_path = args.out or Path(f"artifacts/audit_{args.dataset}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Taxonomy Noise Audit: {args.dataset}\n"
        f"{n_total:,} total → {n_removed:,} removed → {n_kept:,} clean "
        f"({100*n_kept/n_total:.1f}%)\n"
        f"Strategy: bottom-up leaf pruning (no cascade through legitimate clades)",
        fontsize=13, fontweight="bold"
    )

    # ── Plot 1: Noise category bar chart ──────────────────────────────────
    ax = axes[0, 0]
    cats_sorted = noise_counts.most_common()
    cat_labels = [c[0] for c in cats_sorted]
    cat_vals = [c[1] for c in cats_sorted]
    bars = ax.barh(cat_labels[::-1], cat_vals[::-1], color="#e74c3c", alpha=0.8)
    ax.set_xlabel("Number of flagged nodes")
    ax.set_title("Noise flags by category")
    for bar, val in zip(bars, cat_vals[::-1]):
        ax.text(bar.get_width() + max(cat_vals) * 0.02,
                bar.get_y() + bar.get_height()/2,
                f"{val:,}", va="center", fontsize=9)

    # ── Plot 2: Composition pie ───────────────────────────────────────────
    ax = axes[0, 1]
    # Break removed into: noisy leaves, stale taxids, empty containers
    n_noisy_leaf = sum(1 for r in removal_reasons.values()
                       if not r.startswith("stale") and not r.startswith("empty"))
    n_stale = sum(1 for r in removal_reasons.values() if "stale" in r)
    n_empty_cont = sum(1 for r in removal_reasons.values() if "empty_container" in r)
    sizes = [n_kept, n_noisy_leaf, n_stale, n_empty_cont]
    labels = [
        f"Clean\n{n_kept:,}",
        f"Noisy leaves\n{n_noisy_leaf:,}",
        f"Stale taxids\n{n_stale:,}",
        f"Empty containers\n{n_empty_cont:,}",
    ]
    colors = ["#2ecc71", "#e74c3c", "#95a5a6", "#e67e22"]
    # Remove zero-size slices
    nonzero = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
    if nonzero:
        sizes, labels, colors = zip(*nonzero)
        ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
               startangle=90, textprops={"fontsize": 9})
    ax.set_title("Node composition after pruning")

    # ── Plot 3: Rank distribution (kept vs removed stacked) ───────────────
    ax = axes[1, 0]
    top_ranks = df["rank"].value_counts().head(12).index.tolist()
    rank_kept = []
    rank_removed = []
    for rk in top_ranks:
        subset = df[df["rank"] == rk]
        rank_kept.append((~subset["is_removed"]).sum())
        rank_removed.append(subset["is_removed"].sum())

    x = np.arange(len(top_ranks))
    w = 0.6
    ax.bar(x, rank_kept, w, label="Kept", color="#2ecc71", alpha=0.8)
    ax.bar(x, rank_removed, w, bottom=rank_kept, label="Removed", color="#e74c3c", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(top_ranks, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Number of nodes")
    ax.set_title("Rank distribution (kept vs removed)")
    ax.legend(fontsize=9)

    # ── Plot 4: Before/after tree depth distribution ──────────────────────
    ax = axes[1, 1]
    # Compute depth for each node via parent chain
    depths = {}
    def get_depth(tid):
        if tid in depths:
            return depths[tid]
        p = parent_of.get(tid)
        if p is None or p == tid:
            depths[tid] = 0
            return 0
        d = get_depth(p) + 1
        depths[tid] = d
        return d

    for tid in taxids:
        get_depth(tid)

    before_depths = [depths.get(t, 0) for t in taxids]
    after_depths = [depths.get(t, 0) for t in kept]

    max_d = max(before_depths) + 1
    bins = np.arange(0, max_d + 1) - 0.5
    ax.hist(before_depths, bins=bins, alpha=0.5, label=f"Before ({n_total:,})",
            color="#3498db", edgecolor="white")
    ax.hist(after_depths, bins=bins, alpha=0.7, label=f"After ({n_kept:,})",
            color="#2ecc71", edgecolor="white")
    ax.set_xlabel("Depth in tree")
    ax.set_ylabel("Number of nodes")
    ax.set_title("Depth distribution before/after cleanup")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved to {out_path}")

    # ── Write clean taxid list for rebuilding ─────────────────────────────
    clean_list_path = out_path.with_suffix(".clean_taxids.txt")
    with open(clean_list_path, "w") as f:
        for tid in sorted(kept):
            f.write(f"{tid}\n")
    print(f"  Clean taxid list ({len(kept):,} nodes) saved to {clean_list_path}")


if __name__ == "__main__":
    main()
