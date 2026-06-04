#!/usr/bin/env python3
"""UMAP of a trained taxonomy embedding, coloured by taxonomic RANK — upgraded viz.

Builds on `visualize_multi_groups.py` (reuses its numba Poincaré metric) and on
`analyze_hierarchy_hyperbolic.py` (reuses its rank-ancestor machinery, so the colour groups are
*exactly* the groups the separation metric scores). Three upgrades over the original viz:

  1. **Rank-based grouping** — colour each node by its ancestor at a taxonomic RANK (phylum, class, …),
     NOT by tree-depth-from-root. (Metazoa's tree children are "Eumetazoa"/"Porifera"; the phyla
     Arthropoda/Cnidaria/… sit many unranked levels deeper, so depth-based colouring is wrong.)
  2. **Balanced per-group sampling** — a per-group cap so a giant clade (Arthropoda, 325k) doesn't
     swamp small ones (Cnidaria 8k, Echinodermata 4k). Every coloured group stays visible.
  3. **Readable palette + zoom** — top-K groups by size get distinct colours, the long tail is pooled
     into a grey background; `--restrict-to-taxid` zooms into one clade (e.g. Arthropoda) and recolours
     by a finer rank (e.g. class) for a "different levels" panel.

Default = metazoa coloured by phylum (Arthropoda / Chordata / Mollusca / Cnidaria / …).

Usage:
    .venv/bin/python scripts/figure_umap_taxa.py \
        --checkpoint artifacts/tags/metazoa_lower_lr_bigger_batch/metazoa_lower_lr_bigger_batch.pth \
        --mapping data/taxopy/metazoa_33208_clean/taxonomy_edges_metazoa_33208_clean.mapping.tsv \
        --rank phylum --metric poincare --cap-per-group 1500 --max-groups 14 \
        --title "Metazoa embedding — coloured by phylum" \
        --out paper/figures/metazoa_umap_phylum

    # Zoom: Arthropoda (6656) recoloured by class
    ... --restrict-to-taxid 6656 --rank class --title "Arthropoda — coloured by class" \
        --out paper/figures/arthropoda_umap_class
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from visualize_multi_groups import _poincare_dist_numba  # noqa: E402  (numba UMAP metric)
from analyze_hierarchy_hyperbolic import (  # noqa: E402
    load_embeddings,
    load_mapping,
    load_taxonomy_with_depth,
    get_ancestor_at_rank,
)

PALETTE = [
    "#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6",
    "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9A6324", "#800000", "#808000",
    "#000075", "#f58231", "#a9a9a9", "#aaffc3", "#ffd8b1", "#000000",
]
GREY = "#d0d0d0"


def is_descendant_of(taxid, ancestor, taxonomy):
    """True if `ancestor` is on `taxid`'s parent chain (within the loaded taxonomy)."""
    seen = set()
    cur = taxid
    while cur in taxonomy and cur not in seen:
        if cur == ancestor:
            return True
        seen.add(cur)
        p = taxonomy[cur]["parent"]
        if p == cur:
            break
        cur = p
    return ancestor == taxid


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--rank", default="phylum", help="Colour each node by its ancestor at this rank")
    ap.add_argument("--node-rank", default=None,
                    help="Plot ONLY nodes whose own rank is this (e.g. 'order'/'family') — one point per coarse "
                         "taxon, which surfaces clean phylum REGIONS instead of species micro-clusters")
    ap.add_argument("--restrict-to-taxid", type=int, default=None,
                    help="Only plot descendants of this taxid (zoom into a clade), then colour by --rank")
    ap.add_argument("--exclude-taxids", type=int, nargs="*", default=[],
                    help="Drop descendants of these taxids (e.g. exclude the giant phyla Arthropoda 6656 + "
                         "Chordata 7711 so the smaller phyla get UMAP room to separate)")
    ap.add_argument("--metric", choices=["poincare", "euclidean"], default="poincare")
    ap.add_argument("--cap-per-group", type=int, default=1500)
    ap.add_argument("--other-cap", type=int, default=4000)
    ap.add_argument("--max-groups", type=int, default=14)
    ap.add_argument("--min-group", type=int, default=30)
    ap.add_argument("--n-neighbors", type=int, default=15,
                    help="UMAP n_neighbors — raise (50-100) to surface coarse (phylum) structure over local clusters")
    ap.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--title", default="Taxonomy embedding — UMAP")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    emb = load_embeddings(args.checkpoint)
    idx2tax = load_mapping(args.mapping)
    valid = set(int(t) for t in idx2tax.values())
    taxonomy = load_taxonomy_with_depth(valid, data_dir=args.data_dir)

    n_total = emb.shape[0]
    labels = np.array(["other"] * n_total, dtype=object)
    keep = np.zeros(n_total, dtype=bool)          # which idx are eligible to plot
    group_sizes = {}

    for idx, taxid in idx2tax.items():
        if idx >= n_total or taxid not in taxonomy:
            continue
        if args.restrict_to_taxid is not None and not is_descendant_of(taxid, args.restrict_to_taxid, taxonomy):
            continue
        if args.exclude_taxids and any(is_descendant_of(taxid, x, taxonomy) for x in args.exclude_taxids):
            continue
        if args.node_rank is not None and taxonomy[taxid].get("rank") != args.node_rank:
            continue
        keep[idx] = True
        anc = get_ancestor_at_rank(taxid, taxonomy, args.rank)
        if anc is None:
            continue
        name = taxonomy.get(anc, {}).get("name", f"TaxID {anc}")
        labels[idx] = name
        group_sizes[name] = group_sizes.get(name, 0) + 1

    group_sizes = {g: n for g, n in group_sizes.items() if n >= args.min_group}
    if not group_sizes:
        raise SystemExit(f"No groups at rank '{args.rank}' (restrict={args.restrict_to_taxid}).")

    ranked = sorted(group_sizes.items(), key=lambda kv: -kv[1])
    named = [g for g, _ in ranked[:args.max_groups]]
    named_set = set(named)
    print(f"\nRank '{args.rank}': colouring {len(named)} of {len(group_sizes)} groups "
          f"(rest → background):")
    for g in named:
        print(f"  {g:28s} {group_sizes[g]:>8,}")

    # Balanced sampling within the kept set.
    sampled = []
    for g in named:
        pool = np.where((labels == g) & keep)[0]
        sampled.append(rng.choice(pool, min(len(pool), args.cap_per_group), replace=False))
    bg_pool = np.where(keep & np.array([(l == "other") or (l not in named_set) for l in labels]))[0]
    if len(bg_pool):
        sampled.append(rng.choice(bg_pool, min(len(bg_pool), args.other_cap), replace=False))
    sample_idx = np.concatenate(sampled)
    rng.shuffle(sample_idx)
    print(f"\nUMAP on {len(sample_idx):,} sampled points (metric={args.metric})...")

    from umap import UMAP
    kw = dict(n_components=2, random_state=args.seed, n_neighbors=args.n_neighbors, min_dist=args.min_dist)
    if args.metric == "poincare":
        kw["metric"] = _poincare_dist_numba
    proj = UMAP(**kw).fit_transform(emb[sample_idx].astype(np.float64))

    samp_labels = labels[sample_idx]
    color_of = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(named)}

    fig, ax = plt.subplots(figsize=(15, 12))
    bg = np.array([(l == "other") or (l not in named_set) for l in samp_labels])
    if bg.any():
        ax.scatter(proj[bg, 0], proj[bg, 1], s=5, c=GREY, alpha=0.35, linewidths=0,
                   label=f"other {args.rank} (n={int(bg.sum()):,})", zorder=1)
    for g in named:
        m = samp_labels == g
        if m.sum():
            ax.scatter(proj[m, 0], proj[m, 1], s=22, c=color_of[g], alpha=0.85,
                       linewidths=0.2, edgecolors="black",
                       label=f"{g} (n={int(m.sum()):,})", zorder=3)

    ax.set_xlabel("UMAP 1", fontsize=13)
    ax.set_ylabel("UMAP 2", fontsize=13)
    ax.set_title(args.title, fontsize=17, fontweight="bold")
    ax.legend(loc="best", fontsize=9, framealpha=0.92, markerscale=1.6)
    ax.grid(True, alpha=0.2)
    ax.set_aspect("equal", adjustable="datalim")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out.with_suffix('.png')}\n      {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
