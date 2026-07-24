"""Leakage-hardened CV (spec §5): hold out whole identity-clusters / whole species / whole family|clade;
ortholog rows (same cluster) always move together."""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import GroupKFold

def grouped_holdout(cluster_ids, n_splits: int = 5, seed: int = 0):
    """Yield (train_idx, test_idx) where no cluster_id appears in both. shuffle+random_state make `seed`
    actually do something (sklearn>=1.6 GroupKFold supports it; bare GroupKFold ignores seed)."""
    cluster_ids = np.asarray(cluster_ids)
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(cluster_ids))), shuffle=True, random_state=seed)
    X = np.zeros((len(cluster_ids), 1))
    yield from gkf.split(X, groups=cluster_ids)

def leave_one_group_out(labels):
    """Yield (train_idx, test_idx, held_label) — one fold per unique label. LOSO = this on `Species`
    (CI-bearing, §5); LOFO = this on `family` (multi-family) or `Clade` (PLA2) — the generalization
    stress test reported as a per-fold sign test."""
    labels = np.asarray(labels)
    for g in sorted(set(labels.tolist())):
        te = np.where(labels == g)[0]
        tr = np.where(labels != g)[0]
        yield tr, te, g

def leave_species_out(species):
    """LOSO — alias of leave_one_group_out for the CI-bearing per-species test (§5)."""
    yield from leave_one_group_out(species)

def per_fold_sign_test(deltas):
    """LOFO reporting (§5): deltas = per-fold (f_metric - baseline_metric). Returns (n_pos, n_total, p)
    via a two-sided binomial sign test (too few folds for a pooled-mean CI)."""
    from scipy.stats import binomtest
    deltas = np.asarray(deltas, float)
    nz = deltas[deltas != 0.0]
    n_pos = int((nz > 0).sum())
    p = binomtest(n_pos, len(nz), 0.5).pvalue if len(nz) else 1.0
    return n_pos, len(nz), float(p)
