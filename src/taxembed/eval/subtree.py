"""Subtree membership in O(1) per query via Euler-tour intervals.

The negative sampler needs a fast "is X a descendant of A?" test at batch time
(300 negatives x 256 batch = 76,800 queries per batch), which rules out repeated
LCA lifting. An Euler tour gives each node an interval [tin, tout); X descends
from A iff tin[A] <= tin[X] < tout[A]. A node is its own descendant.
"""
from __future__ import annotations

import numpy as np


def parent_from_closure(ancestor_idx, descendant_idx, depth_diff, n_nodes: int) -> np.ndarray:
    """Derive the parent array from a transitive closure, using only depth_diff == 1 rows.

    Nodes with no incoming direct edge (the root) point to themselves.
    """
    parent = np.arange(n_nodes, dtype=np.int64)
    direct = np.asarray(depth_diff) == 1
    parent[np.asarray(descendant_idx)[direct]] = np.asarray(ancestor_idx)[direct]
    return parent


def euler_intervals(parent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Iterative DFS Euler tour. Returns (tin, tout); subtree of v spans [tin[v], tout[v])."""
    parent = np.asarray(parent, dtype=np.int64)
    n = len(parent)
    children: list[list[int]] = [[] for _ in range(n)]
    roots: list[int] = []
    for v in range(n):
        p = int(parent[v])
        if p == v:
            roots.append(v)
        else:
            children[p].append(v)

    tin = np.zeros(n, dtype=np.int64)
    tout = np.zeros(n, dtype=np.int64)
    timer = 0
    for r in roots:
        stack: list[tuple[int, bool]] = [(r, False)]
        while stack:
            v, exiting = stack.pop()
            if exiting:
                tout[v] = timer
                continue
            tin[v] = timer
            timer += 1
            stack.append((v, True))
            for c in reversed(children[v]):
                stack.append((c, False))
    return tin, tout


def is_descendant(anc, node, tin: np.ndarray, tout: np.ndarray) -> np.ndarray:
    """Vectorized subtree membership. Broadcasts; a node is its own descendant."""
    anc = np.asarray(anc)
    node = np.asarray(node)
    return (tin[anc] <= tin[node]) & (tin[node] < tout[anc])
