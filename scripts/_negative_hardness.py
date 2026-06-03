"""Pure helpers for the negative-hardness diagnostic (Phase 1, E1c). No I/O.

Measures how much within-clade gradient the softmax/NLL loss receives from the
negatives a sampler draws. See docs/plans/2026-06-03-e1c-phase1-instrumentation.md.
"""
from __future__ import annotations

import numpy as np


def numpy_poincare_distance(u: np.ndarray, v: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Poincaré-ball geodesic distance, numpy re-implementation (broadcasting + arccosh-domain clamp) of model.poincare_distance.

    Supports broadcasting: u,v of shape (..., dim) -> distance of shape (...).
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    sq_u = np.clip(np.sum(u * u, axis=-1), 0.0, 1.0 - eps)
    sq_v = np.clip(np.sum(v * v, axis=-1), 0.0, 1.0 - eps)
    sq_diff = np.sum((u - v) ** 2, axis=-1)
    arg = 1.0 + 2.0 * sq_diff / ((1.0 - sq_u) * (1.0 - sq_v))
    arg = np.maximum(arg, 1.0)  # arccosh domain
    return np.arccosh(arg)


def softmax_pj(d_pos: np.ndarray, d_neg: np.ndarray) -> np.ndarray:
    """Softmax probability mass on each NEGATIVE in the Nickel-Kiela NLL.

    Logits = [-d_pos, -d_neg_1, ..., -d_neg_n]; returns the negative-part
    probabilities, shape (B, n_neg). The loss' gradient on negative j is its p_j,
    so this is the gradient share each negative receives.
    """
    d_pos = np.asarray(d_pos, dtype=np.float64).reshape(-1, 1)   # (B,1)
    d_neg = np.asarray(d_neg, dtype=np.float64)                  # (B,n_neg)
    logits = np.concatenate([-d_pos, -d_neg], axis=1)           # (B,1+n_neg)
    logits -= logits.max(axis=1, keepdims=True)                 # numerical stability
    ex = np.exp(logits)
    p = ex / ex.sum(axis=1, keepdims=True)
    return p[:, 1:]                                             # drop the positive column


def label_negatives(node_class_arr: np.ndarray, node_gp_arr: np.ndarray,
                    descendant_idxs: np.ndarray, negatives: np.ndarray):
    """Boolean masks: is each negative within the anchor's class / grandparent?

    `-1` sentinels (no class / no grandparent) NEVER count as a match — this is the
    guard against the sentinel-collision bug (R2 review finding).
    Returns (within_class, within_gp), each shape (B, n_neg) bool.
    """
    desc_class = node_class_arr[descendant_idxs][:, None]   # (B,1)
    desc_gp = node_gp_arr[descendant_idxs][:, None]         # (B,1)
    neg_class = node_class_arr[negatives]                   # (B,n_neg)
    neg_gp = node_gp_arr[negatives]                         # (B,n_neg)
    within_class = (neg_class == desc_class) & (desc_class != -1) & (neg_class != -1)
    within_gp = (neg_gp == desc_gp) & (desc_gp != -1) & (neg_gp != -1)
    return within_class, within_gp
