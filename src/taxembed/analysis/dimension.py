"""Intrinsic dimension analysis for taxonomy embeddings.

Three estimators:
1. Angular packing heuristic — theoretical lower bound from information theory
2. PCA participation ratio — effective dimensionality of trained embeddings
3. Recommended dimension — practical recommendation combining estimators
"""

from __future__ import annotations

import numpy as np


def angular_packing_dim(n_items: int, max_cosine: float = 0.2) -> int:
    """Minimum dimension for N unit vectors with max cosine overlap <= epsilon.

    From the Johnson-Lindenstrauss lemma / random projection theory:
        d >= 2 * ln(N) / epsilon^2

    Args:
        n_items: Number of items to embed.
        max_cosine: Maximum allowed pairwise cosine similarity (epsilon).

    Returns:
        Minimum embedding dimension (ceiling).
    """
    if n_items <= 1:
        return 1
    if max_cosine <= 0:
        raise ValueError("max_cosine must be positive")
    return int(np.ceil(2 * np.log(n_items) / (max_cosine ** 2)))


def participation_ratio(embeddings: np.ndarray) -> float:
    """Effective dimensionality via eigenvalue participation ratio.

    d_eff = (sum(lambda_i))^2 / sum(lambda_i^2)

    A value of k means the variance is spread across ~k effective dimensions.

    Args:
        embeddings: Array of shape (n_samples, n_dims).

    Returns:
        Participation ratio (effective dimensionality).
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {embeddings.shape}")
    if embeddings.shape[0] < 2:
        return float(embeddings.shape[1])

    cov = np.cov(embeddings.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 0]

    if len(eigenvalues) == 0:
        return 0.0

    total = float(np.sum(eigenvalues))
    return total ** 2 / float(np.sum(eigenvalues ** 2))


def recommend_dim(
    n_nodes: int,
    max_cosine: float = 0.2,
    embeddings: np.ndarray | None = None,
) -> dict:
    """Return dimension recommendations from multiple estimators.

    Args:
        n_nodes: Number of nodes in the taxonomy.
        max_cosine: Cosine overlap tolerance for angular packing.
        embeddings: Optional trained embeddings for participation ratio analysis.

    Returns:
        Dictionary with estimator results and a recommended value.
    """
    result: dict = {
        "n_nodes": n_nodes,
        "angular_packing": angular_packing_dim(n_nodes, max_cosine),
    }

    # Relaxed estimate at 0.5 for reference
    result["angular_packing_relaxed"] = angular_packing_dim(n_nodes, 0.5)

    if embeddings is not None:
        result["participation_ratio"] = participation_ratio(embeddings)
        result["current_dim"] = embeddings.shape[1]

    # Practical recommendation: angular packing at relaxed eps=0.3, clamped to [10, 200]
    moderate_dim = angular_packing_dim(n_nodes, 0.3)
    result["recommended"] = max(min(moderate_dim, 200), 10)

    return result
