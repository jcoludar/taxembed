"""Radius/direction transplants, scored on the TRAINER'S OWN metrics (Task 9 review item 6).

A Poincare embedding is radius (planted: target_radius(depth)) x direction (learned). Swapping the
two between runs isolates which one carries a difference on the metrics the paper already uses:
  x' = |x_B| * x_A / |x_A|      (directions of A, radii of B)
If "prior directions on planted radii" matches canonical on kNN%/Sep, the paper's own metrics
see a RADIAL difference (reading 2); if it stays far below, the difference is ANGULAR (reading 1).

The trainer's metrics (train_small.compute_class_separation / compute_multiscale_knn) draw a
500-node sample from the legacy global numpy RNG. Here they run at a larger n_sample over several
fixed seeds and report mean and SD, so a transplant difference is not sampling noise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def transplant(directions_from: np.ndarray, radii_from: np.ndarray) -> np.ndarray:
    """Directions of one embedding, per-node radii (Poincare norms) of another."""
    a = np.asarray(directions_from, dtype=np.float64)
    r = np.linalg.norm(np.asarray(radii_from, dtype=np.float64), axis=1)
    return a / np.linalg.norm(a, axis=1, keepdims=True) * r[:, None]


def planted(directions_from: np.ndarray, depth: np.ndarray, max_depth: int,
            schedule: str = "log") -> np.ndarray:
    """Directions of an embedding on the planted radius target_radius(depth)."""
    from train_hierarchical import target_radius

    a = np.asarray(directions_from, dtype=np.float64)
    r = target_radius(np.asarray(depth, dtype=np.float64), max_depth, schedule)
    return a / np.linalg.norm(a, axis=1, keepdims=True) * r[:, None]


class _FixedEmbedding:
    """Minimal stand-in for HierarchicalPoincareEmbedding: the trainer's metric functions only
    call get_poincare_embeddings()."""

    def __init__(self, x):
        import torch

        self._x = torch.as_tensor(np.asarray(x, dtype=np.float32))

    def get_poincare_embeddings(self, indices=None):
        return self._x if indices is None else self._x[indices]


def trainer_metrics(x: np.ndarray, pairs, class_info, seeds=(0, 1, 2, 3, 4),
                    n_sample: int = 2000) -> dict:
    """kNN% / Sep (top-level class) and multiscale kNN, exactly as the trainer logs them,
    averaged over fixed seeds at a larger sample."""
    import torch

    from train_small import compute_class_separation, compute_multiscale_knn

    model = _FixedEmbedding(x)
    knn, sep, multi = [], [], []
    for s in seeds:
        np.random.seed(s)
        k_, s_ = compute_class_separation(model, class_info, torch.device("cpu"), n_sample=n_sample)
        knn.append(k_)
        sep.append(s_)
        np.random.seed(s)
        multi.append(compute_multiscale_knn(model, pairs, torch.device("cpu"), n_sample=n_sample))
    levels = sorted(multi[0])
    return {
        "knn_purity_mean": float(np.mean(knn)), "knn_purity_sd": float(np.std(knn, ddof=1)),
        "class_sep_mean": float(np.mean(sep)), "class_sep_sd": float(np.std(sep, ddof=1)),
        "multiscale_knn_mean": {int(lv): float(np.mean([m[lv] for m in multi])) for lv in levels},
        "n_sample": n_sample, "seeds": list(seeds),
    }
