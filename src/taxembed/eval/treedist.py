"""Exact tree distances over the NCBI taxonomy via binary-lifting LCA.

Operates on integer node ids 0..N-1 (the embedding's own indexing). `parent` is an
int array (root points to itself); `depth` is the root-distance in edges. Vectorized:
all queries take numpy arrays of node ids and return numpy arrays.
"""
from __future__ import annotations

import numpy as np


class TreeDistance:
    def __init__(self, parent: np.ndarray, depth: np.ndarray):
        self.parent = np.asarray(parent, dtype=np.int64)
        self.depth = np.asarray(depth, dtype=np.int64)
        n = len(self.parent)
        max_depth = int(self.depth.max()) if n else 0
        self.maxlog = max(1, int(np.ceil(np.log2(max(2, max_depth + 1)))) + 1)
        # up[k, v] = the (2^k)-th ancestor of v
        self.up = np.zeros((self.maxlog, n), dtype=np.int64)
        self.up[0] = self.parent
        for k in range(1, self.maxlog):
            self.up[k] = self.up[k - 1][self.up[k - 1]]

    def lca(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.int64)
        b = np.asarray(b, dtype=np.int64)
        # ensure a is the deeper-or-equal node
        swap = self.depth[a] < self.depth[b]
        a, b = np.where(swap, b, a), np.where(swap, a, b)
        # lift a up to b's depth
        diff = self.depth[a] - self.depth[b]
        for k in range(self.maxlog):
            move = ((diff >> k) & 1).astype(bool)
            a = np.where(move, self.up[k][a], a)
        # lift both until their ancestors meet
        for k in range(self.maxlog - 1, -1, -1):
            au, bu = self.up[k][a], self.up[k][b]
            move = au != bu
            a = np.where(move, au, a)
            b = np.where(move, bu, b)
        return np.where(a == b, a, self.parent[a])

    def lca_depth(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self.depth[self.lca(a, b)]

    def path_length(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Cophenetic distance = #edges on the a→b path through the LCA (the PRIMARY tree distance)."""
        l = self.lca(a, b)
        return self.depth[np.asarray(a)] + self.depth[np.asarray(b)] - 2 * self.depth[l]
