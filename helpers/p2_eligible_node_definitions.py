"""Which node-eligibility rule yields spec v3 §P2.3's 593,576 eligible nodes?

WHY THIS EXISTS (2026-09-24, P2 design).

Spec v3 §P2.3 asserts "593,576 eligible nodes; a 10% holdout is 59,357 test links" for the
cellular closure. 59,357 is 10% of 593,576 -- i.e. ONE test link per held-out node, which is the
signature of a PARENT-edge holdout, not the non-basic-edge holdout that §P2.1 inherits from Ganea.
The spec does not define "eligible" anywhere, so the number is an unreproducible prose figure
until a rule reproduces it.

This enumerates candidate definitions and counts each, so the split builder can adopt the one that
matches instead of inventing a fourth. Prints a falsifiable number per rule BEFORE any is chosen.

Read-only over data/. Cellular scale by default; --clade NAME to check another.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path("/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings")
DATA = REPO / "data" / "taxopy"
TARGET = 593_576  # the figure spec v3 §P2.3 asserts


def main() -> int:
    clade = "cellular_organisms_131567_clean"
    if "--clade" in sys.argv:
        clade = sys.argv[sys.argv.index("--clade") + 1]
    path = DATA / clade / f"taxonomy_edges_{clade}_transitive.npz"
    z = np.load(path, allow_pickle=True)

    a = np.asarray(z["ancestor_idx"])
    d = np.asarray(z["descendant_idx"])
    dd = np.asarray(z["depth_diff"])
    d_depth = np.asarray(z["descendant_depth"])

    basic = dd == 1
    par = a[basic]          # parent of each node, in basic-edge order
    chi = d[basic]          # the node itself
    depth_of_child = d_depth[basic]

    n_nodes = len(np.union1d(np.unique(a), np.unique(d)))
    print(f"clade = {clade}")
    print(f"n_nodes                          = {n_nodes:,}")
    print(f"n_basic (nodes with a parent)    = {len(chi):,}")
    print(f"spec §P2.3 target                = {TARGET:,}\n")

    # who is a leaf?  a node that is never a parent
    parents = set(par.tolist())
    is_leaf = np.array([c not in parents for c in chi.tolist()])
    print(f"leaves (never a parent)          = {int(is_leaf.sum()):,}")
    print(f"internal nodes (excl. root)      = {int((~is_leaf).sum()):,}")

    # sibling counts: how many children does each node's parent have?
    fanout = Counter(par.tolist())
    n_sibs = np.array([fanout[p] for p in par.tolist()])
    print(f"nodes whose parent has >=2 kids  = {int((n_sibs >= 2).sum()):,}")
    print(f"nodes that are an only child     = {int((n_sibs == 1).sum()):,}\n")

    # depth quartiles the spec quotes: Q1 = 11, Q3 = 28 -- check against descendant_depth
    q1, med, q3 = np.percentile(d_depth, [25, 50, 75])
    print(f"depth_diff-free check -- descendant_depth quartiles: "
          f"Q1={q1:.1f} median={med:.1f} Q3={q3:.1f}  (spec says Q1=11, Q3=28)")
    dq1, dmed, dq3 = np.percentile(depth_of_child, [25, 50, 75])
    print(f"over basic edges only:                               "
          f"Q1={dq1:.1f} median={dmed:.1f} Q3={dq3:.1f}\n")

    rules = {
        "all nodes with a parent (non-root)": np.ones(len(chi), dtype=bool),
        "non-root AND non-leaf": ~is_leaf,
        "non-root AND parent has >=2 children": n_sibs >= 2,
        "non-root, non-leaf, parent has >=2 children": (~is_leaf) & (n_sibs >= 2),
        "non-root AND leaf": is_leaf,
        "leaf AND parent has >=2 children": is_leaf & (n_sibs >= 2),
        "depth >= 2 (excludes children of root)": depth_of_child >= 2,
        "depth >= 2 AND parent has >=2 children": (depth_of_child >= 2) & (n_sibs >= 2),
        # "level-stratified" reading: keep the interquartile depth band [Q1, Q3] = [11, 28].
        # The spec contrasts our band with "HiG2Vec's 6-11", i.e. eligibility IS a depth band.
        "depth in [11, 28] inclusive": (depth_of_child >= 11) & (depth_of_child <= 28),
        "depth in [11, 28) half-open": (depth_of_child >= 11) & (depth_of_child < 28),
        "depth in (11, 28] half-open": (depth_of_child > 11) & (depth_of_child <= 28),
        "depth in [11, 28] AND non-leaf": (depth_of_child >= 11) & (depth_of_child <= 28) & (~is_leaf),
        "depth in [11, 28] AND leaf": (depth_of_child >= 11) & (depth_of_child <= 28) & is_leaf,
        "depth in [11, 28] AND parent has >=2 kids": (depth_of_child >= 11) & (depth_of_child <= 28) & (n_sibs >= 2),
    }

    print(f"{'rule':<48}{'count':>14}{'vs target':>14}")
    print("-" * 76)
    best = None
    for name, mask in rules.items():
        c = int(mask.sum())
        delta = c - TARGET
        flag = "  <== MATCH" if c == TARGET else ""
        if best is None or abs(delta) < abs(best[1] - TARGET):
            best = (name, c)
        print(f"{name:<48}{c:>14,}{delta:>+14,}{flag}")

    print("-" * 76)
    if best and best[1] == TARGET:
        print(f"EXACT: {best[0]}")
    else:
        print(f"NO RULE REPRODUCES {TARGET:,}. Closest: {best[0]} at {best[1]:,} "
              f"({best[1] - TARGET:+,}).")
        print("=> §P2.3's figure is not reproducible from the closure alone; the split builder")
        print("   must DEFINE eligibility explicitly rather than inherit an unsourced number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
