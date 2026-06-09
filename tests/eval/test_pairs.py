import numpy as np
from taxembed.eval.pairs import sample_pairs_stratified


def test_stratified_balances_bins_and_counts():
    # true distances 1..6; bin edges [0,2,4,7) -> 3 bins. Ask for 90 pairs.
    rng = np.random.default_rng(0)
    n = 2000
    a = rng.integers(0, 500, n)
    b = rng.integers(0, 500, n)
    dist = rng.integers(1, 7, n)            # pretend tree distances
    idx = sample_pairs_stratified(dist, bin_edges=[0, 2, 4, 7], per_bin=30, seed=0)
    assert len(idx) == 90
    binned = np.digitize(dist[idx], [2, 4])  # 0,1,2
    counts = np.bincount(binned, minlength=3)
    assert (counts == 30).all()             # exactly balanced


def test_handles_small_bin_without_replacement_error():
    dist = np.array([1, 1, 5])              # bin0 has 2, bin2 has 1, bin1 empty
    idx = sample_pairs_stratified(dist, bin_edges=[0, 2, 4, 7], per_bin=30, seed=0)
    # takes all available in undersized bins, never errors
    assert set(idx).issubset({0, 1, 2})
