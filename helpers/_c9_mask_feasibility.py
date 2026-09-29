#!/usr/bin/env python3
"""C9 masking run — FEASIBILITY probe. Are the 91,317 placeholder taxa prunable leaves?

The C9 decision (do not retrain; measure by masking at evaluation time) rests on being able to
re-score S_angle with the placeholder taxa removed. eval/angular.py's TreeIndex builds its
same-depth pools from `depth == d` over the WHOLE node array and takes no subset argument, so the
masked condition needs a PRUNED tree. Pruning is only trivial if the placeholders are leaves --
an internal placeholder would orphan a subtree and require reparenting, which is a different and
much more arguable operation.

So: count how many of the 91,317 are leaves, and how many children the non-leaf ones carry.
Counts only; decides nothing.

Written 2026-09-29.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings")
sys.path.insert(0, str(ROOT / "helpers"))
sys.path.insert(0, str(ROOT / "src"))

from _p3_placement_score import MAPPING, OLD_DATE, build_tree, load_mapping  # noqa: E402
from taxembed.eval.release_diff import parse_parents  # noqa: E402

NAMES = ROOT / "data" / f"taxdump_archive_{OLD_DATE}" / "names.dmp"
NODES = ROOT / "data" / f"taxdump_archive_{OLD_DATE}" / "nodes.dmp"
PLACEHOLDER = re.compile(r"\b(?:bacterium|archaeon)\b", re.IGNORECASE)


def main() -> None:
    print("=" * 88)
    print("C9 masking — feasibility: are the placeholder taxa prunable LEAVES?")
    print("=" * 88)

    taxid2row = load_mapping(MAPPING)
    names: dict[int, str] = {}
    with NAMES.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "scientific name" not in line:
                continue
            p = line.split("\t|\t")
            if len(p) < 2:
                continue
            try:
                names[int(p[0].strip())] = p[1].strip()
            except ValueError:
                continue

    parent_map = parse_parents(NODES)
    taxids, idx, parent, depth, tin, tout = build_tree(parent_map)
    n = len(taxids)

    n_children = np.bincount(parent, minlength=n)
    n_children[parent == np.arange(n)] -= 1          # root parents itself

    embedded = np.array([t for t in taxid2row if t in idx], dtype=np.int64)
    rows = np.array([idx[int(t)] for t in embedded], dtype=np.int64)

    ph_mask = np.array([bool(PLACEHOLDER.search(names.get(int(t), ""))) for t in embedded])
    ph_rows = rows[ph_mask]
    print(f"\n  embedded taxa in tree : {len(rows):,}")
    print(f"  placeholder taxa      : {len(ph_rows):,}")

    # leaf = no children AT ALL in the full NCBI tree (not just among embedded nodes)
    kids = n_children[ph_rows]
    n_leaf = int((kids == 0).sum())
    print(f"  ...of which LEAVES    : {n_leaf:,} ({100*n_leaf/max(len(ph_rows),1):.2f} %)")
    print(f"  ...with children      : {len(ph_rows)-n_leaf:,}")
    if len(ph_rows) - n_leaf:
        nz = kids[kids > 0]
        print(f"       children: min {nz.min()}, median {int(np.median(nz))}, max {nz.max()}, "
              f"total descendants-1 hop {int(nz.sum()):,}")

    # depth profile -- S_angle pools are per-depth, so this says which pools are affected
    dph = depth[ph_rows]
    dall = depth[rows]
    print(f"\n  placeholder depth: min {dph.min()}, median {int(np.median(dph))}, max {dph.max()}")
    print(f"  {'depth':>6}{'embedded':>12}{'placeholder':>14}{'share':>9}")
    for d in range(int(dall.min()), int(dall.max()) + 1):
        tot = int((dall == d).sum())
        if tot < 2000:
            continue
        p = int((dph == d).sum())
        if p == 0:
            continue
        print(f"  {d:>6}{tot:>12,}{p:>14,}{100*p/tot:>8.1f}%")

    outp = ROOT / "results" / "c9_mask_feasibility_20260929.json"
    json.dump({
        "purpose": "can the C9 placeholder taxa be pruned as leaves for a masked S_angle re-score?",
        "embedded_in_tree": int(len(rows)),
        "placeholder": int(len(ph_rows)),
        "placeholder_leaves": n_leaf,
        "placeholder_with_children": int(len(ph_rows) - n_leaf),
        "placeholder_leaf_pct": round(100 * n_leaf / max(len(ph_rows), 1), 2),
    }, outp.open("w"), indent=2)
    print(f"\nwritten: {outp}")


if __name__ == "__main__":
    main()
