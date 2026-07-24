import numpy as np
from taxembed.bridge import clean_eval


def _dense_reference(Z, groups, k):
    Z = np.asarray(Z, np.float64)
    norm = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    g = np.asarray(groups)
    sims = norm @ norm.T
    np.fill_diagonal(sims, -np.inf)
    nn = np.argsort(-sims, axis=1)[:, :k]
    pur = [(g[nn[i]] == g[i]).mean() for i in range(len(g))]
    return float(np.mean(pur)), len(pur)


def test_blocked_global_matches_dense():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(200, 16))
    groups = rng.integers(0, 4, size=200)
    ref = _dense_reference(Z, groups, k=clean_eval.KNN_K)
    got = clean_eval.knn_purity_global(Z, groups, block=37)   # odd block to exercise the tail
    assert abs(got[0] - ref[0]) < 1e-9 and got[1] == ref[1]


def _dense_within_strata_reference(Z, groups, strata, k):
    Z = np.asarray(Z, np.float64)
    norm = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    groups = np.asarray(groups); strata = np.asarray(strata)
    purities = []
    for s in np.unique(strata):
        m = np.where(strata == s)[0]
        if len(m) < k + 1:
            continue
        sub = norm[m]; sims = sub @ sub.T
        np.fill_diagonal(sims, -np.inf)
        nn = np.argsort(-sims, axis=1)[:, :k]
        gl = groups[m]
        for i in range(len(m)):
            purities.append(float((gl[nn[i]] == gl[i]).mean()))
    return (float(np.mean(purities)), len(purities)) if purities else (float("nan"), 0)


def test_blocked_within_strata_matches_dense_per_stratum():
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(120, 8))
    groups = rng.integers(0, 3, size=120)
    strata = rng.integers(0, 2, size=120)
    ref = _dense_within_strata_reference(Z, groups, strata, clean_eval.KNN_K)
    got = clean_eval.knn_purity_within_strata(Z, groups, strata, block=29)
    assert abs(got[0] - ref[0]) < 1e-9 and got[1] == ref[1]
