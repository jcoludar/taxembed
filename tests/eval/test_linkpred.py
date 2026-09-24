import numpy as np
import pytest

from taxembed.eval.linkpred import (
    candidate_pool,
    linkpred_metrics,
    rank_of_true_parent,
    stratify,
)


def test_candidate_pool_is_the_grandparents_children():
    # 0 -> {1, 2}; 1 -> {3, 4}; node 3's grandparent is 0, whose children are 1 and 2
    parent = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    depth = np.array([0, 1, 1, 2, 2], dtype=np.int64)
    assert sorted(candidate_pool(parent, depth, node=3).tolist()) == [1, 2]


def test_candidate_pool_never_contains_the_query_itself():
    parent = np.array([0, 0, 1, 1], dtype=np.int64)
    depth = np.array([0, 1, 2, 2], dtype=np.int64)
    assert 2 not in candidate_pool(parent, depth, node=2).tolist()


def test_rank_is_one_when_the_true_parent_is_nearest():
    emb = np.array([[0.0, 0.0], [0.1, 0.0], [0.9, 0.0], [0.11, 0.0]])
    r = rank_of_true_parent(emb, node=3, true_parent=1, candidates=np.array([1, 2]))
    assert r == 1


def test_rank_is_last_when_the_true_parent_is_furthest():
    emb = np.array([[0.0, 0.0], [0.9, 0.0], [0.1, 0.0], [0.11, 0.0]])
    r = rank_of_true_parent(emb, node=3, true_parent=1, candidates=np.array([1, 2]))
    assert r == 2


def test_metrics_on_a_hand_computed_case():
    ranks = np.array([1, 2, 4])
    n_cand = np.array([10, 10, 10])
    m = linkpred_metrics(ranks, n_cand)
    assert m["n"] == 3
    assert m["mean_rank"] == pytest.approx(7 / 3)
    assert m["mrr"] == pytest.approx((1 + 0.5 + 0.25) / 3)
    assert m["hits_at_1"] == pytest.approx(1 / 3)
    assert m["hits_at_10"] == pytest.approx(1.0)


def test_normalized_rank_accounts_for_uneven_pool_sizes():
    """A rank of 2 out of 2 is chance; a rank of 2 out of 1000 is near-perfect.
    Un-normalized mean rank would call them similar."""
    m_small = linkpred_metrics(np.array([2]), np.array([2]))
    m_large = linkpred_metrics(np.array([2]), np.array([1000]))
    assert m_small["normalized_rank"] > m_large["normalized_rank"]
    assert m_small["mean_rank"] == m_large["mean_rank"]     # the reason normalization is needed


def test_metrics_on_an_empty_input_do_not_divide_by_zero():
    m = linkpred_metrics(np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    assert m["n"] == 0
    assert np.isnan(m["mrr"])


def test_stratify_groups_ranks_by_key_into_the_named_bins():
    ranks = np.array([1, 5, 1, 9])
    n_candidates = np.array([10, 10, 1000, 1000])
    key = np.array([2, 2, 50, 50])
    out = stratify(ranks, n_candidates, key, bins=[(1, 10), (11, 1000)])
    assert out["1-10"]["n"] == 2
    assert out["11-1000"]["n"] == 2
    assert out["1-10"]["mrr"] == pytest.approx((1 + 0.2) / 2)


def test_stratify_normalized_rank_uses_true_pool_size_not_bin_bounds():
    """Regression test for the 2026-09-24 ruled correction.

    The brief's original `stratify` synthesized each stratum's candidate-pool size from the
    bin's upper bound (`hi`) instead of the real per-query pool size, making `normalized_rank`
    meaningless within a stratum. Here the two bins' real pool sizes (3 and 500) differ sharply
    from their bin bounds (10 and 1000) -- this test would FAIL under that original version,
    which would compute normalized_rank as (2-1)/(10-1) = 1/9 for the first bin (not 0.5) and
    (3-1)/(1000-1) = 2/999 for the second (not 2/499).
    """
    ranks = np.array([2, 2, 3, 3])
    n_candidates = np.array([3, 3, 500, 500])
    key = np.array([2, 2, 50, 50])
    out = stratify(ranks, n_candidates, key, bins=[(1, 10), (11, 1000)])
    assert out["1-10"]["normalized_rank"] == pytest.approx(0.5)
    assert out["11-1000"]["normalized_rank"] == pytest.approx(2 / 499)
