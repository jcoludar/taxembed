"""Taxon-level bootstrap CIs (spec §9B: the unit of analysis is the TAXON, not the pair — pairs are
non-independent, so effective N ≈ #taxa). Caller reduces its analysis to one value per taxon (e.g.
per-taxon kNN-retrieval precision); this resamples taxa with replacement.
"""
from __future__ import annotations

import numpy as np


def taxon_bootstrap_ci(per_taxon_values: np.ndarray, n_boot: int = 1000, seed: int = 0,
                       alpha: float = 0.05):
    """Return (point_mean, lo, hi) where lo/hi are the (alpha/2, 1-alpha/2) percentile-bootstrap CI."""
    v = np.asarray(per_taxon_values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(v)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        boot[i] = v[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi)
