"""Stratified pair sampling by true tree-distance bin (spec §9B: uniform sampling is swamped by
trivially-far cross-kingdom pairs, inflating global correlation). Returns indices into the input
distance array; the caller holds the parallel (a, b) endpoint arrays.
"""
from __future__ import annotations

import numpy as np


def sample_pairs_stratified(tree_dist: np.ndarray, bin_edges, per_bin: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tree_dist = np.asarray(tree_dist)
    inner = list(bin_edges)[1:-1]               # digitize uses interior edges
    which = np.digitize(tree_dist, inner)        # bin id per pair
    n_bins = len(bin_edges) - 1
    picks = []
    for b in range(n_bins):
        members = np.flatnonzero(which == b)
        if members.size == 0:
            continue
        take = min(per_bin, members.size)
        picks.append(rng.choice(members, size=take, replace=False))
    return np.concatenate(picks) if picks else np.array([], dtype=np.int64)
