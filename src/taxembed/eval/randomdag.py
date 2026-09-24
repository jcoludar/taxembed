"""RandomDAG memorization control (spec v3 §P2.4, after GRAM, Choi et al. KDD'17).

Rewire every non-root node to a uniformly random parent at depth-1, keeping its own depth. Because
each node's ancestor count then equals its depth, the closure's pair count is Sum(depth) either
way -- preserved EXACTLY, which is what §P2.4 requires, not approximately.

The control answers: does the recipe's held-out performance depend on the taxonomy's real
structure, or would any tree of the same shape do as well? If a model trained on the randomized
closure scores as well on ITS OWN held-out parents, the geometry is memorizing tree shape rather
than learning taxonomy.
"""

from __future__ import annotations

import numpy as np

from taxembed.utils.training_pairs import TrainingPairs


def randomize_parents(parent: np.ndarray, depth: np.ndarray, seed: int = 0) -> np.ndarray:
    """Uniformly resample each non-root node's parent from the nodes one level above it."""
    parent = np.asarray(parent, dtype=np.int64)
    depth = np.asarray(depth, dtype=np.int64)
    rng = np.random.default_rng(seed)
    out = parent.copy()
    by_depth = {d: np.flatnonzero(depth == d) for d in np.unique(depth)}
    for d in sorted(by_depth):
        if d == 0:
            continue
        above = by_depth.get(d - 1)
        if above is None or len(above) == 0:
            continue                       # no level above: leave these nodes alone
        nodes = by_depth[d]
        out[nodes] = above[rng.integers(0, len(above), size=len(nodes))]
    return out


def closure_from_parent(parent: np.ndarray, depth: np.ndarray) -> TrainingPairs:
    """Expand a parent array into the full ancestor-descendant closure as TrainingPairs."""
    parent = np.asarray(parent, dtype=np.int64)
    depth = np.asarray(depth, dtype=np.int64)
    anc, dsc, ddiff, adep, ddep = [], [], [], [], []
    for node in range(len(parent)):
        if depth[node] == 0:
            continue                       # the root has no ancestors
        cur, step = int(parent[node]), 1
        while True:
            anc.append(cur); dsc.append(node)
            ddiff.append(step); adep.append(int(depth[cur])); ddep.append(int(depth[node]))
            if depth[cur] == 0:            # reached the root; stop before self-looping
                break
            cur, step = int(parent[cur]), step + 1
    if not anc:
        raise ValueError("empty closure")
    to32 = lambda x: np.asarray(x, dtype=np.int32)   # noqa: E731
    to16 = lambda x: np.asarray(x, dtype=np.int16)   # noqa: E731
    return TrainingPairs(
        ancestor_idx=to32(anc), descendant_idx=to32(dsc), depth_diff=to16(ddiff),
        ancestor_depth=to16(adep), descendant_depth=to16(ddep),
        ancestor_taxid=to32(anc), descendant_taxid=to32(dsc),
    )
