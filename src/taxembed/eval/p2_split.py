"""Held-out split construction for P2 (spec v3 §P2.3, as corrected 2026-09-24).

WHY A PARENT-EDGE HOLDOUT AND NOT GANEA'S NON-BASIC-EDGE HOLDOUT.

Every NCBI closure in this repo is a strict TREE (one parent per node; measured across all six
clades). On a tree the transitive reduction determines the entire closure, so Ganea's protocol --
"keep the reduction always in training" -- hands Vendrov's trivial baseline 100% of any held-out
non-basic edge. See docs/FINDING_ganea_split_degenerate_on_trees.md.

What IS predictive on a tree is the parent edge itself, withheld for LEAF nodes only. For an
internal node v the parent p stays recoverable, because v's descendants keep their (p, w) edges;
restricting to leaves closes that path. This matches TaxoExpan / Arborist / Octet (spec v3 §3.4),
which all hold out leaves.
"""

from __future__ import annotations

import numpy as np

from taxembed.utils.training_pairs import TrainingPairs

DEFAULT_BAND = (11, 28)  # per-node depth interquartile range, Q1=11 Q3=28 (spec v3 §P2.3)


def leaf_mask(parent: np.ndarray) -> np.ndarray:
    """True for nodes that are nobody's parent.

    `parent_from_closure` makes the root point to itself, so a self-edge must not count as
    parenthood -- otherwise the root would be marked internal for the wrong reason and, in a
    single-node tree, never marked at all.
    """
    parent = np.asarray(parent, dtype=np.int64)
    n = len(parent)
    has_real_parent = np.arange(n, dtype=np.int64) != parent
    is_parent = np.zeros(n, dtype=bool)
    is_parent[parent[has_real_parent]] = True
    return ~is_parent


def depth_from_closure(descendant_idx, descendant_depth, ancestor_idx, ancestor_depth,
                       n_nodes: int) -> np.ndarray:
    """Per-node depth, read off whichever closure column mentions the node."""
    depth = np.full(n_nodes, -1, dtype=np.int64)
    depth[np.asarray(ancestor_idx, dtype=np.int64)] = np.asarray(ancestor_depth, dtype=np.int64)
    depth[np.asarray(descendant_idx, dtype=np.int64)] = np.asarray(descendant_depth, dtype=np.int64)
    return depth


def eligible_nodes(parent: np.ndarray, depth: np.ndarray, band=DEFAULT_BAND,
                   leaves_only: bool = True) -> np.ndarray:
    """Node indices eligible for parent-edge holdout: depth inside `band`, non-root, and leaf.

    `band` is INCLUSIVE at both ends -- that is the reading that reproduces spec §P2.3's
    593,576 eligible nodes exactly (delta +0 against 13 rival rules).
    """
    lo, hi = band
    if lo > hi:
        raise ValueError(f"band lower bound {lo} exceeds upper bound {hi}")
    parent = np.asarray(parent, dtype=np.int64)
    depth = np.asarray(depth, dtype=np.int64)
    n = len(parent)
    mask = (np.arange(n, dtype=np.int64) != parent)      # exclude the root
    mask &= (depth >= lo) & (depth <= hi)
    if leaves_only:
        mask &= leaf_mask(parent)
    return np.flatnonzero(mask).astype(np.int64)


def select_holdout(eligible: np.ndarray, frac_test: float = 0.10, frac_val: float = 0.0,
                   seed: int = 0) -> dict:
    """Partition `eligible` into disjoint test / val / train node sets.

    Uses a dedicated Generator, never the legacy global numpy RNG -- §1.2 records that the
    unseeded global RNG is exactly how this project lost reproducibility before.
    """
    if frac_test < 0 or frac_val < 0 or frac_test + frac_val > 1:
        raise ValueError(f"invalid fractions: test={frac_test} val={frac_val}")
    eligible = np.asarray(eligible, dtype=np.int64)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(eligible))
    n_test = int(len(eligible) * frac_test)
    n_val = int(len(eligible) * frac_val)
    test = np.sort(eligible[perm[:n_test]])
    val = np.sort(eligible[perm[n_test:n_test + n_val]])
    train = np.sort(eligible[perm[n_test + n_val:]])
    return {"test": test, "val": val, "train": train}


def parent_edge_mask(pairs: TrainingPairs, held_out: np.ndarray) -> np.ndarray:
    """True for closure rows to REMOVE: the depth_diff==1 row of each held-out node."""
    held = np.zeros(pairs.n_nodes, dtype=bool)
    held[np.asarray(held_out, dtype=np.int64)] = True
    return (pairs.depth_diff == 1) & held[pairs.descendant_idx]
