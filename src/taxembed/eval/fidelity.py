"""Fidelity metrics for embedded-vs-tree distance (spec §9B: local rank fidelity, NOT global Pearson).

- multiplicative_distortion: the standard metric-embedding quality measure. Scale-invariant: we
  divide out the median ratio first, then report max(r, 1/r) summary stats. 1.0 == perfect.
- knn_retrieval_precision: per-query overlap between the k nearest by embedding and the k nearest by
  tree. The headline LOCAL metric; returns one value per query (feed to taxon_bootstrap_ci).
- within_clade_rank_corr: per-query Spearman of d_emb vs d_tree over a local candidate set.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def multiplicative_distortion(d_emb: np.ndarray, d_tree: np.ndarray) -> dict:
    d_emb = np.asarray(d_emb, float)
    d_tree = np.asarray(d_tree, float)
    m = (d_tree > 0) & (d_emb > 0)
    ratio = d_emb[m] / d_tree[m]
    ratio = ratio / np.median(ratio)            # remove the arbitrary global scale
    dist = np.maximum(ratio, 1.0 / ratio)       # >= 1, symmetric
    return {"median": float(np.median(dist)), "mean": float(np.mean(dist)),
            "p95": float(np.percentile(dist, 95)), "max": float(np.max(dist))}


def knn_retrieval_precision(d_emb_mat: np.ndarray, d_tree_mat: np.ndarray, k: int) -> np.ndarray:
    """d_*_mat: (Q, C) query-to-candidate distances (self-distance must be +inf). Returns (Q,) precision@k."""
    emb_nn = np.argsort(d_emb_mat, axis=1)[:, :k]
    tree_nn = np.argsort(d_tree_mat, axis=1)[:, :k]
    out = np.empty(len(d_emb_mat))
    for i in range(len(d_emb_mat)):
        out[i] = len(set(emb_nn[i]).intersection(tree_nn[i])) / k
    return out


def within_clade_rank_corr(d_emb_mat: np.ndarray, d_tree_mat: np.ndarray) -> np.ndarray:
    """Per-query Spearman rho of embedded vs tree distance over the candidate set. Returns (Q,)."""
    out = np.empty(len(d_emb_mat))
    for i in range(len(d_emb_mat)):
        finite = np.isfinite(d_emb_mat[i]) & np.isfinite(d_tree_mat[i])
        out[i] = spearmanr(d_emb_mat[i][finite], d_tree_mat[i][finite]).correlation
    return out
