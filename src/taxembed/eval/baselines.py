"""No-learning baselines reported beside every P2 link-prediction number (spec v3 §3.2).

Vendrov et al. ICLR 2016 §3.4 classify a pair positive iff it lies in the transitive closure of
the training+validation edges, scoring 88.2% on WordNet. Reporting it is mandatory because a
learned number is only interesting relative to what no learning achieves. On our tree-shaped split
it collapses to 0% recall -- see the module test -- which is itself a reportable property of the
task, not a bug in the baseline.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from taxembed.eval.subtree import euler_intervals, is_descendant


def vendrov_closure_rule(visible_ancestor, visible_descendant, queries, candidates) -> np.ndarray:
    """(queries x candidates) bool: is candidate an ancestor of query in the VISIBLE graph?"""
    visible = set(zip(np.asarray(visible_ancestor).tolist(),
                      np.asarray(visible_descendant).tolist()))
    q = np.asarray(queries).tolist()
    c = np.asarray(candidates).tolist()
    return np.array([[(cand, node) in visible for cand in c] for node in q], dtype=bool)


def sibling_chance(parent: np.ndarray, held_out: np.ndarray) -> np.ndarray:
    """Per held-out node, 1 / (number of children its GRANDPARENT has).

    This is the chance rate of guessing the true parent uniformly among the candidates that the
    retained closure still admits -- the grandparent's children. It is the floor a learned score
    must clear to mean anything.
    """
    parent = np.asarray(parent, dtype=np.int64)
    held_out = np.asarray(held_out, dtype=np.int64)
    fanout = np.bincount(parent[np.arange(len(parent)) != parent], minlength=len(parent))
    grandparent = parent[parent[held_out]]
    n_candidates = np.maximum(fanout[grandparent], 1)
    return 1.0 / n_candidates


def majority_parent_rate(parent: np.ndarray, held_out: np.ndarray) -> float:
    """Accuracy of always answering the commonest true parent among the held-out nodes."""
    parent = np.asarray(parent, dtype=np.int64)
    held_out = np.asarray(held_out, dtype=np.int64)
    if len(held_out) == 0:
        return 0.0
    counts = Counter(parent[held_out].tolist())
    return counts.most_common(1)[0][1] / len(held_out)
