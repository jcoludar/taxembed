"""Cross-check the device-aware torch kNN kernel against the production numpy Poincaré kernel,
and check the vectorized matched-null draws from the correct stratum. Runs on CPU (no GPU in CI)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from _anomaly_knn import observed_purity, matched_null, _safe_batch  # noqa: E402
from knn_purity_hyperbolic import _prep_sqnorms, _batch_distances  # noqa: E402  (numpy reference)


def _clustered(seed=0, n_fam=4, per_fam=12, dim=16):
    """Well-separated angular clusters so the k-NN set is unambiguous (no ties to flip)."""
    rng = np.random.default_rng(seed)
    fam_dir = rng.standard_normal((n_fam, dim)); fam_dir /= np.linalg.norm(fam_dir, axis=1, keepdims=True)
    emb, lab = [], []
    for f in range(n_fam):
        for _ in range(per_fam):
            v = fam_dir[f] + 0.02 * rng.standard_normal(dim); v /= np.linalg.norm(v)
            emb.append(v * 0.85); lab.append(f)
    return np.asarray(emb, dtype=np.float32), np.asarray(lab, dtype=np.int64)


def _numpy_observed_purity(emb, pool_idx, pool_lab, k):
    raw, clip = _prep_sqnorms(emb.astype(np.float64))
    pe, pr, pc = emb[pool_idx], raw[pool_idx], clip[pool_idx]
    keff = min(k, len(pool_idx) - 1)
    d = _batch_distances(pe, pr, pc, pe, pr, pc)
    np.fill_diagonal(d, np.inf)
    nn = np.argpartition(d, keff, axis=1)[:, :keff]
    return (pool_lab[nn] == pool_lab[:, None]).mean(axis=1)


def test_torch_purity_matches_numpy_kernel():
    emb, lab = _clustered()
    pool_idx = np.arange(len(emb))
    ref = _numpy_observed_purity(emb, pool_idx, lab, k=5)
    got = observed_purity(emb, pool_idx, lab, k=5, device="cpu", batch=8)
    assert got.shape == ref.shape
    # neighbour SETS agree on a tie-free fixture -> purities are exactly equal
    assert np.allclose(got, ref, atol=1e-9)
    # and on clean clusters every node's neighbours are same-family -> purity 1.0
    assert np.allclose(got, 1.0)


def test_torch_purity_batch_invariance():
    emb, lab = _clustered(seed=3)
    pool_idx = np.arange(len(emb))
    one = observed_purity(emb, pool_idx, lab, k=4, device="cpu", batch=1000)
    many = observed_purity(emb, pool_idx, lab, k=4, device="cpu", batch=5)
    assert np.allclose(one, many)


def test_safe_batch_caps_to_memory_budget():
    # A million-node pool with the old default flag (2048) is what OOM'd the V100 (job 5674199):
    # the (batch x P) working set must be shrunk so n_buffers simultaneous float32 copies fit budget.
    P = 1_000_000
    budget = 4 * 1024 ** 3
    b = _safe_batch(P, requested=2048, budget_bytes=budget, n_buffers=8)
    assert 1 <= b < 2048
    assert 8 * b * P * 4 <= budget          # working set stays within budget


def test_safe_batch_passthrough_when_pool_small():
    # Small pool (echino 4k regime): the cap must NOT bind -> requested batch preserved unchanged.
    assert _safe_batch(4000, requested=1024, budget_bytes=4 * 1024 ** 3, n_buffers=8) == 1024


def test_safe_batch_floors_at_one():
    # Even a pathologically tiny budget never returns 0 (would zero-step the loop / divide by zero).
    assert _safe_batch(10_000_000, requested=2048, budget_bytes=1024, n_buffers=8) == 1


def test_observed_purity_caps_inner_batch_to_budget():
    # A huge requested batch + a tiny memory budget forces the kernel to internally shrink the
    # query-block; it must STILL return the correct numpy-matching purities (1.0 on clean clusters).
    emb, lab = _clustered(seed=1)
    pool_idx = np.arange(len(emb))
    ref = _numpy_observed_purity(emb, pool_idx, lab, k=5)
    got = observed_purity(emb, pool_idx, lab, k=5, device="cpu", batch=10_000,
                          mem_budget_bytes=4096)      # forces tiny inner query-blocks
    assert np.allclose(got, ref, atol=1e-9)
    assert np.allclose(got, 1.0)


def test_matched_null_shape_and_draws_from_bin():
    rng = np.random.default_rng(0)
    P = 300
    observed = rng.random(P)
    pool_idx = np.arange(P)
    depth = rng.integers(0, 5, P)
    clade_size = rng.integers(1, 200, P)
    null = matched_null(observed, pool_idx, depth, clade_size, n_null=50, n_bins=3, seed=1)
    assert null.shape == (P, 50)
    # every drawn value must be an observed-purity value (drawn from the pool, with replacement)
    assert np.isin(null, observed).all()
    # draws must come from the query's OWN depth x size bin: reconstruct bins and check membership
    db = np.digitize(depth, np.quantile(depth, [1 / 3, 2 / 3]))
    sb = np.digitize(clade_size, np.quantile(clade_size, [1 / 3, 2 / 3]))
    bins = db * 4 + sb
    q = 0
    same_bin_vals = set(observed[bins == bins[q]].tolist())
    assert set(null[q].tolist()).issubset(same_bin_vals)
