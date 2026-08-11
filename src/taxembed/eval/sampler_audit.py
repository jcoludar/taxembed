"""Closed-form audit of the default negative sampler's false-negative rate.

train_hierarchical.py draws negatives uniformly from _depth_to_nodes[depth(descendant)]
(:398, fast path :400-403 -- with replacement, no self-exclusion, no ancestry check),
then scores them as distance from the ANCESTOR (:730-732). A drawn negative is therefore
a FALSE negative -- an actual descendant of the anchor, i.e. a valid positive -- with
probability

    FN(a, dd) = C[a, dd] / N[dd]

C[a, dd] = #descendants of a at depth dd (read off the complete closure)
N[dd]    = #nodes at depth dd

Exact, no sampling. Also censuses pairs whose valid-negative pool is EMPTY (C == N),
which is where a naive "reject descendant negatives" fix silently zeroes the gradient.
"""
from __future__ import annotations

import numpy as np


def false_negative_audit(ancestor_idx, descendant_idx, ancestor_depth, descendant_depth) -> dict:
    anc = np.asarray(ancestor_idx, dtype=np.int64)
    des = np.asarray(descendant_idx, dtype=np.int64)
    anc_depth = np.asarray(ancestor_depth, dtype=np.int64)
    des_depth = np.asarray(descendant_depth, dtype=np.int64)
    n_pairs = len(anc)
    n_nodes = int(max(anc.max(), des.max())) + 1

    depth = np.full(n_nodes, -1, dtype=np.int64)
    depth[des] = des_depth
    depth[anc] = anc_depth              # the root never appears as a descendant
    if (depth < 0).any():
        raise ValueError(f"{int((depth < 0).sum())} nodes have no depth in the closure")
    max_depth = int(depth.max())

    # N[dd]
    nodes_per_depth = np.bincount(depth, minlength=max_depth + 1).astype(np.int64)

    # C[a, dd] via a sparse (ancestor, descendant_depth) group count
    stride = max_depth + 1
    key = anc * stride + des_depth
    uniq_key, counts = np.unique(key, return_counts=True)
    C_pair = counts[np.searchsorted(uniq_key, key)].astype(np.int64)
    N_pair = nodes_per_depth[des_depth]

    fn_pair = C_pair / N_pair
    zero_pool = C_pair >= N_pair

    by_depth = []
    for ad in range(max_depth + 1):
        m = anc_depth == ad
        n = int(m.sum())
        if n == 0:
            continue
        by_depth.append({
            "anchor_depth": ad,
            "n_pairs": n,
            "fn_rate": float(fn_pair[m].mean()),
            "n_zero_pool": int(zero_pool[m].sum()),
        })

    root = anc_depth == 0
    return {
        "n_pairs": int(n_pairs),
        "n_nodes": int(n_nodes),
        "max_depth": max_depth,
        "overall_rate": float(fn_pair.mean()),
        "by_anchor_depth": by_depth,
        "n_zero_pool": int(zero_pool.sum()),
        "frac_zero_pool": float(zero_pool.mean()),
        "n_root_anchored": int(root.sum()),
        "frac_root_anchored": float(root.mean()),
        "nodes_per_depth": nodes_per_depth.tolist(),
    }
