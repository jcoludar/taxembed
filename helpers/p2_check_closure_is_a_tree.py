"""Is the taxonomy closure a TREE, and does its transitive reduction determine the whole closure?

WHY THIS EXISTS (2026-09-24, P2 design gate).

Spec v3 §P2.1 adopts Ganea et al. 2018's split: keep the transitive REDUCTION ("basic" edges)
always in training, hold out 5%/5% of the NON-BASIC edges, and sweep how much of the closure is
visible. Spec v3 §3.2 makes Vendrov's trivial baseline mandatory beside every link-prediction
number: a pair is called positive iff it lies in the transitive closure of the union of the
training and validation edges.

Those two requirements interact. On WordNet (a DAG, multiple parents per node) the reduction does
NOT recover the closure, so the trivial baseline is beatable -- Vendrov measured 88.2%. On a TREE
each node has exactly one parent, so the parent-child edges determine every ancestor-descendant
pair exactly. If our closure is a tree, then "reduction always in training" hands the trivial
baseline 100% of the held-out edges, and the held-out-edge task stops being a prediction task.

This script MEASURES that rather than assuming it, for every clade closure on disk. It reports:
  - n_nodes, n_pairs
  - the basic-edge set (depth_diff == 1) and its size
  - max/modal parents per node, and how many nodes have >1 parent  <- the tree test
  - whether expanding the basic edges to their transitive closure REPRODUCES the stored closure
    exactly (set equality), which is the trivial baseline's accuracy in the limit

No writes outside stdout. Read-only over data/.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path("/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings")
DATA = REPO / "data" / "taxopy"


def load(npz_path: Path):
    z = np.load(npz_path, allow_pickle=True)
    return z, sorted(z.files)


def ancestor_descendant(z, keys):
    """Return (ancestor, descendant, depth_diff|None) using whatever schema this npz uses."""
    lower = {k.lower(): k for k in keys}
    for a_name, d_name in (
        ("ancestor_idx", "descendant_idx"),
        ("ancestor_taxid", "descendant_taxid"),
        ("ancestor", "descendant"),
        ("ancestors", "descendants"),
        ("source", "target"),
        ("parent", "child"),
        ("u", "v"),
    ):
        if a_name in lower and d_name in lower:
            a = np.asarray(z[lower[a_name]])
            d = np.asarray(z[lower[d_name]])
            break
    else:
        # fall back: a single (N,2) edge array
        for k in keys:
            arr = np.asarray(z[k])
            if arr.ndim == 2 and arr.shape[1] == 2:
                a, d = arr[:, 0], arr[:, 1]
                break
        else:
            return None, None, None

    dd = None
    for cand in ("depth_diff", "dd", "distance", "depth_difference"):
        if cand in lower:
            dd = np.asarray(z[lower[cand]])
            break
    return a, d, dd


def closure_from_parents(parent: dict[int, int]) -> set[tuple[int, int]]:
    """Every (ancestor, descendant) reachable by walking parent pointers upward."""
    out: set[tuple[int, int]] = set()
    for node in parent:
        cur = parent.get(node)
        seen = 0
        while cur is not None:
            out.add((cur, node))
            cur = parent.get(cur)
            seen += 1
            if seen > 1000:  # cycle guard; a taxonomy is depth ~40
                raise RuntimeError(f"cycle or runaway chain above node {node}")
    return out


def report(npz_path: Path, expand: bool) -> None:
    z, keys = load(npz_path)
    print(f"\n=== {npz_path.parent.name} ===")
    print(f"  keys: {keys}")

    a, d, dd = ancestor_descendant(z, keys)
    if a is None:
        print("  !! could not identify ancestor/descendant arrays -- schema unknown")
        return

    n_pairs = len(a)
    nodes = np.union1d(np.unique(a), np.unique(d))
    print(f"  n_pairs = {n_pairs:,}   n_nodes = {len(nodes):,}")

    if dd is None:
        print("  !! no depth_diff array; cannot isolate basic edges by dd==1")
        return

    print(f"  depth_diff: min={int(dd.min())} max={int(dd.max())}")
    basic = dd == 1
    n_basic = int(basic.sum())
    print(f"  basic edges (dd==1) = {n_basic:,}   non-basic = {n_pairs - n_basic:,}")
    print(f"  n_nodes - 1         = {len(nodes) - 1:,}   <- equals n_basic iff a tree")

    # the tree test: parents per node, over the basic edges only
    child = d[basic]
    par = a[basic]
    counts = Counter(child.tolist())
    multi = [c for c, k in counts.items() if k > 1]
    hist = Counter(counts.values())
    print(f"  parents-per-node histogram (basic edges): {dict(sorted(hist.items()))}")
    print(f"  nodes with >1 parent: {len(multi):,}")
    if multi:
        print(f"    examples: {multi[:5]}")

    root_anchored = int((par == par[np.argmax(np.bincount(par.astype(np.int64)))]).sum()) if n_basic else 0
    print(f"  (most common parent appears {root_anchored:,} times)")

    if not expand:
        print("  [skipped closure expansion: pass --expand to run it on this clade]")
        return

    if len(multi) > 0:
        print("  NOT a tree -> reduction may not determine the closure; expanding to check")
    parent = {int(c): int(p) for p, c in zip(par, child)}
    derived = closure_from_parents(parent)
    stored = set(zip(a.tolist(), d.tolist()))
    print(f"  closure expanded from basic edges: {len(derived):,} pairs")
    print(f"  stored closure                   : {len(stored):,} pairs")
    only_derived = derived - stored
    only_stored = stored - derived
    print(f"  in derived not stored: {len(only_derived):,}")
    print(f"  in stored not derived: {len(only_stored):,}")
    if not only_derived and not only_stored:
        print("  >>> IDENTICAL. The reduction determines the ENTIRE closure.")
        print("  >>> Vendrov trivial baseline on ANY held-out non-basic edge = 100.00%")
    else:
        recovered = len(stored & derived) / len(stored) * 100
        print(f"  >>> trivial baseline would recover {recovered:.4f}% of stored pairs")


def main() -> int:
    expand_all = "--expand" in sys.argv
    targets = [
        ("mollusca_6447_clean", True),
        ("echinodermata_7586_clean", True),
        ("metazoa_33208_clean", True),
        ("arthropoda_6656_clean", expand_all),
        ("eukaryota_2759_clean", expand_all),
        ("cellular_organisms_131567_clean", expand_all),
    ]
    for name, expand in targets:
        p = DATA / name / f"taxonomy_edges_{name}_transitive.npz"
        if not p.exists():
            print(f"\n=== {name} ===\n  MISSING: {p}")
            continue
        report(p, expand)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
