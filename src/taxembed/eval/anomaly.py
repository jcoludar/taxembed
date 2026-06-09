"""Pure core for Application #2 — taxonomy QC / anomaly detection (spec §5#2, §9B).

The anomaly score is a SIZE-CONDITIONED kNN-impurity: raw impurity just rediscovers rare/small
clades (spec §9B), so we report it relative to its size-matched expectation (excess_impurity) and,
as the headline, a z-score vs a depth+clade-size-matched random-angle null (matched_null_z). The
score must beat trivial baselines (clade size, depth, degree, distance-to-parent-centroid) on the
synthetic ROC. No I/O here — operates on integer node arrays and float observed-purity arrays.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def excess_impurity(observed_purity: np.ndarray, chance_purity: float) -> np.ndarray:
    """Size-conditioned anomaly score = chance_purity - observed_purity (higher == more anomalous).

    chance_purity is the size-aware expectation Sum_g (n_g/P)^2 (reuse knn_purity_hyperbolic.chance_purity).
    A clean node has observed >> chance -> negative; an impure node has observed << chance -> positive.
    """
    return float(chance_purity) - np.asarray(observed_purity, dtype=np.float64)


def matched_null_z(observed_purity: np.ndarray, null_observed: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Headline score: z of (mu_null - observed) over a depth+clade-size-matched null (higher == anomalous).

    null_observed: (Q, n_null) observed-purity values for matched random-angle null draws per query.
    Returns (Q,) z-scores; a node far BELOW its matched null's purity scores high.
    """
    observed = np.asarray(observed_purity, dtype=np.float64)
    null = np.asarray(null_observed, dtype=np.float64)
    mu = null.mean(axis=1)
    sigma = null.std(axis=1)
    return (mu - observed) / (sigma + eps)


def trivial_baselines(emb: np.ndarray, parent: np.ndarray, depth: np.ndarray,
                      clade_size: np.ndarray, degree: np.ndarray) -> dict:
    """The trivial baselines the score must beat (spec §9B). All (N,), higher == 'more anomalous' guess.

    - clade_size / depth / degree: raw structural quantities (rank by them directly).
    - dist_to_parent_centroid: Euclidean dist from each node's embedding to the centroid of its
      siblings (children of the same parent) — a geometry baseline that ignores neighbour identity.
    """
    emb = np.asarray(emb, dtype=np.float64)
    parent = np.asarray(parent, dtype=np.int64)
    n = len(parent)
    sums = np.zeros((n, emb.shape[1]), dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)
    np.add.at(sums, parent, emb)
    np.add.at(counts, parent, 1.0)
    counts = np.maximum(counts, 1.0)
    parent_centroid = sums[parent] / counts[parent, None]
    dist = np.linalg.norm(emb - parent_centroid, axis=1)
    return {
        "clade_size": np.asarray(clade_size, dtype=np.float64),
        "depth": np.asarray(depth, dtype=np.float64),
        "degree": np.asarray(degree, dtype=np.float64),
        "dist_to_parent_centroid": dist,
    }


def baseline_aucs(labels: np.ndarray, scores: dict) -> dict:
    """ROC-AUC of each score (higher == more anomalous) against binary anomaly labels."""
    labels = np.asarray(labels, dtype=np.int64)
    out = {}
    for name, s in scores.items():
        out[name] = float(roc_auc_score(labels, np.asarray(s, dtype=np.float64)))
    return out


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """BH-FDR adjusted q-values (spec §9B: BH-control per-node significance over ~1.1M nodes)."""
    p = np.asarray(pvals, dtype=np.float64)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(1, n + 1))
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n, dtype=np.float64)
    q[order] = np.minimum(ranked, 1.0)
    return q


def relocate_nodes(parent: np.ndarray, depth: np.ndarray, n: int, seed: int = 0):
    """Synthetic leg-A: move n random NON-root nodes to a random DIFFERENT parent.

    Returns (new_parent, moved_idx). Only nodes with depth>0 are eligible; the new parent is drawn
    uniformly from nodes that are not the node itself nor its current parent.
    """
    parent = np.asarray(parent, dtype=np.int64).copy()
    depth = np.asarray(depth, dtype=np.int64)
    rng = np.random.default_rng(seed)
    eligible = np.flatnonzero(depth > 0)
    moved = rng.choice(eligible, size=min(n, len(eligible)), replace=False)
    new_parent = parent.copy()
    all_nodes = np.arange(len(parent))
    for v in moved:
        choices = all_nodes[(all_nodes != v) & (all_nodes != parent[v])]
        new_parent[v] = rng.choice(choices)
    return new_parent, moved


def displacement_class(orig_parent: np.ndarray, new_parent: np.ndarray,
                       depth: np.ndarray, moved: np.ndarray) -> np.ndarray:
    """Phylogenetic displacement magnitude per moved node (spec §9B: stratify ROC by displacement).

    Defined as the tree distance between the OLD and NEW parent via their depths and LCA-free proxy:
    here we use |depth[old_parent] - depth[new_parent]| + 2 (a monotone proxy for how far the node
    jumped); the CLI replaces this with the exact TreeDistance.path_length(old_parent,new_parent) to
    get the true sister-genus -> cross-kingdom ladder. This pure helper returns the depth-gap proxy so
    the core stays decoupled from TreeDistance; both are monotone in displacement.
    """
    depth = np.asarray(depth, dtype=np.int64)
    op = np.asarray(orig_parent, dtype=np.int64)[moved]
    npar = np.asarray(new_parent, dtype=np.int64)[moved]
    return np.abs(depth[op] - depth[npar]) + 2
