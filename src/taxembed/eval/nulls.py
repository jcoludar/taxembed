"""Null-model embeddings for baseline deltas (spec §9B: the radial-only null is the key Goodhart guard).

- radial_only_null: keep each node's norm (=depth structure), randomize its direction → fidelity
  attributable to RADIUS alone. The model must beat this.
- shuffled_label_null: permute which embedding vector belongs to which node → destroys all
  label-structure while preserving the marginal point cloud.
- random_ball_null: uniform-ish random points strictly inside the Poincaré ball.
"""
from __future__ import annotations

import numpy as np


def radial_only_null(emb: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    dirs = rng.standard_normal(emb.shape)
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
    return dirs * norms


def shuffled_label_null(emb: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(emb))
    return emb[perm]


def random_ball_null(n: int, dim: int, seed: int = 0, max_norm: float = 0.95) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dirs = rng.standard_normal((n, dim))
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
    r = max_norm * rng.random((n, 1)) ** (1.0 / dim)
    return dirs * r
