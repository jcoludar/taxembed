import numpy as np
from taxembed.eval.bootstrap import taxon_bootstrap_ci


def test_ci_brackets_mean_and_is_ordered():
    rng = np.random.default_rng(0)
    vals = rng.normal(0.9, 0.02, 500)       # per-taxon metric values
    mean, lo, hi = taxon_bootstrap_ci(vals, n_boot=1000, seed=0)
    assert lo < mean < hi
    assert abs(mean - vals.mean()) < 1e-9    # point estimate is the plain mean
    assert (hi - lo) < 0.02                   # tight CI for n=500, low variance


def test_ci_widens_with_fewer_units():
    rng = np.random.default_rng(0)
    big = taxon_bootstrap_ci(rng.normal(0.9, 0.05, 1000), n_boot=500, seed=0)
    small = taxon_bootstrap_ci(rng.normal(0.9, 0.05, 30), n_boot=500, seed=0)
    assert (small[2] - small[1]) > (big[2] - big[1])
