"""RandomDAG memorization control (spec v3 §P2.4, after GRAM, Choi et al. KDD'17).

Rewire every non-root node to a uniformly random parent at depth-1, keeping its own depth. Because
each node's ancestor count then equals its depth, the closure's pair count is Sum(depth) either
way -- preserved EXACTLY, which is what §P2.4 requires, not approximately.

The control answers: does the recipe's held-out performance depend on the taxonomy's real
structure, or would any tree of the same shape do as well? If a model trained on the randomized
closure scores as well on ITS OWN held-out parents, the geometry is memorizing tree shape rather
than learning taxonomy.

RETIRED AS P2's CONTROL, 2026-09-24 (`p2_amendment_4_20260924`, results/p2_heldout_preregistration.
json). `randomize_parents` draws each node's new parent UNIFORMLY (i.i.d. with replacement) from
the level above, which does not preserve fan-out: a real taxonomy's fan-out is heavy-tailed (a few
genera with hundreds of children, most nodes with one or two), so uniform reassignment collapses
that tail -- measured on mollusca_6447_clean seed 0 (helpers/p2_randomdag_changes_the_chance_floor.
py): mean fan-out 4.61 -> 1.96, max 187 -> 14, raising P2's chance floor 5.4x
(`p2_amendment_1_20260924`). `degree_matched_shuffle` below is the replacement control: it
preserves the fan-out distribution EXACTLY (a permutation of the real parent-label multiset, never
an i.i.d. resample), so the chance floor and the degree prior match the real tree by construction.
`randomize_parents` is KEPT, unchanged, as the GRAM-style (Choi et al.) uniform-rewire control cited
in the finding note -- it is simply no longer what P2 trains and scores its RandomDAG-class arm
against.
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


def degree_matched_shuffle(parent: np.ndarray, depth: np.ndarray, seed: int = 0) -> np.ndarray:
    """Configuration-model shuffle: permute PARENT LABELS among nodes at each depth, so every
    parent keeps exactly the child count it had, while which children it gets is randomised.

    P2's amendment_4 control (2026-09-24, `p2_amendment_4_20260924`, results/
    p2_heldout_preregistration.json). `randomize_parents` above draws each node's new parent
    UNIFORMLY AT RANDOM (i.i.d., with replacement) from the level above -- which does NOT preserve
    fan-out, since i.i.d. sampling from a heavy-tailed degree distribution regresses toward the
    mean. This function instead takes the ACTUAL multiset of parent labels that depth-`d` nodes
    already carry (one label per depth-`d` node, so a parent with k children contributes k copies
    of its own index to that multiset) and applies `rng.permutation` to it -- a bijection on a
    fixed multiset, not a resample. Every parent that had k children at depth `d` still has
    EXACTLY k children at depth `d` afterward; only WHICH depth-`d` nodes it has changes. Because
    the fan-out multiset is preserved exactly, so is P2's chance floor (`baselines.sibling_chance`,
    1/grandparent-fan-out) and the training-free degree prior (`baselines.degree_prior_rank`) --
    the entire reason this control exists (see the module docstring and `p2_amendment_1_20260924`'s
    5.4x chance-floor-inflation finding, which this construction closes by construction).

    Processed one depth level at a time, each independently: a depth-`d` node's new parent is
    always drawn from the multiset of depth-`d` nodes' OWN (pre-shuffle) parent labels, every one
    of which is, by construction of the input tree, a node at depth `d-1` -- so the result is
    still depth-consistent (every non-root node's parent is exactly one level shallower) and
    therefore still a single-rooted tree with no node its own ancestor, by the same induction
    `randomize_parents` relies on. The root (depth 0) is left untouched.

    Levels with 0 or 1 node are no-ops (nothing to permute): a lone child has only one possible
    parent label to draw from its own multiset, so shuffling it is a no-op by construction, not a
    special case carved out of the general rule.
    """
    parent = np.asarray(parent, dtype=np.int64)
    depth = np.asarray(depth, dtype=np.int64)
    rng = np.random.default_rng(seed)
    out = parent.copy()
    by_depth = {d: np.flatnonzero(depth == d) for d in np.unique(depth)}
    for d in sorted(by_depth):
        if d == 0:
            continue
        nodes = by_depth[d]
        if len(nodes) <= 1:
            continue                       # nothing to permute
        labels = out[nodes].copy()         # the real fan-out multiset contributed at this depth
        out[nodes] = rng.permutation(labels)
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
