import numpy as np
import pytest

from taxembed.eval.linkpred import (
    _poincare_distance,
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


# --- Finding 1 (fix round 1): exact radius-based path vs the floored ball-coordinate fallback ---


def test_exact_and_ball_paths_agree_deep_inside_the_ball():
    """Comfortably inside the ball, the ball formula's denominator floor never binds, so the
    exact radius-based path and the ball-coordinate formula are the SAME function computed two
    different ways and must agree tightly."""
    u = np.array([0.10, -0.05])
    v = np.array([[0.20, 0.15], [-0.05, 0.30], [0.08, -0.22]])

    d_ball, clip_ball = _poincare_distance(u, v)
    assert clip_ball == 0

    r_u = 2.0 * np.arctanh(np.linalg.norm(u))
    r_v = 2.0 * np.arctanh(np.linalg.norm(v, axis=1))
    d_exact, clip_exact = _poincare_distance(u, v, r_u=r_u, r_v=r_v)
    assert clip_exact == 0

    np.testing.assert_allclose(d_exact, d_ball, atol=1e-8)


def test_exact_path_corrects_the_ball_floors_understatement_near_the_boundary():
    """Two points ~1e-7 inside the boundary, at right angles, make the ball formula's
    denominator (1-|u|^2)(1-|v|^2) ~ 4e-14 -- below the 1e-12 floor -- so the floor silently
    substitutes a denominator ~25x too large and UNDERSTATES the true distance.

    The expected exact distance is re-derived here from the raw hyperbolic law of cosines
    (cosh d = cosh(r_u) cosh(r_v) - sinh(r_u) sinh(r_v) cos(theta)), NOT by calling the function
    under test for its own answer -- so this test fails if the exact path is deleted (the call
    would then fall through to the ball formula and land far short of the independently
    computed value, and the "ball path was floored" assertion would also fail since the two
    paths would coincide).
    """
    eps = 1e-7
    u = np.array([1.0 - eps, 0.0])
    v = np.array([[0.0, 1.0 - eps]])

    d_ball, clip_ball = _poincare_distance(u, v)
    assert clip_ball == 1  # the floor actually bound for this pair

    r_u = 2.0 * np.arctanh(1.0 - eps)
    r_v = 2.0 * np.arctanh(1.0 - eps)
    cos_theta = 0.0  # u and v are orthogonal by construction
    expected_cosh_d = np.cosh(r_u) * np.cosh(r_v) - np.sinh(r_u) * np.sinh(r_v) * cos_theta
    expected_d = np.arccosh(expected_cosh_d)

    d_exact, clip_exact = _poincare_distance(u, v, r_u=np.array(r_u), r_v=np.array([r_v]))
    assert clip_exact == 0
    np.testing.assert_allclose(d_exact, [expected_d], rtol=1e-6)

    # Direction AND magnitude: the floor understated distance by several natural-log units,
    # not by noise-level float drift.
    assert d_exact[0] > d_ball[0] + 1.0


def test_clip_counter_flags_exactly_the_rows_that_hit_the_floor():
    """Non-zero exactly when the floor binds, zero when it does not -- and the COUNT (not just
    a boolean) is accurate: one near-boundary row out of two must report clip_count == 1, not 2."""
    eps = 1e-7
    u = np.array([1.0 - eps, 0.0])
    v = np.array([
        [0.0, 1.0 - eps],  # near boundary -- denominator underflows the 1e-12 floor
        [0.1, 0.1],        # comfortably inside -- floor never binds
    ])
    _d, clip_count = _poincare_distance(u, v)
    assert clip_count == 1

    _d2, clip_count2 = _poincare_distance(
        np.array([0.1, -0.1]), np.array([[0.2, 0.05], [-0.1, 0.3]])
    )
    assert clip_count2 == 0


def test_rank_of_true_parent_radii_fixes_a_real_ranking_flip():
    """This is finding 1's central claim made concrete end-to-end through the public API: the
    ball fallback's floor doesn't just understate a distance, it can flip a ranking between two
    real candidates. Node 0 (query) and candidate 1 both sit within ~1e-9 of the boundary at
    nearly the same angle; candidate 2 sits well inside the ball at a different angle and is the
    TRUE nearest neighbour (verified against the exact law-of-cosines distance). Without radii,
    the floored ball distance ranks candidate 1 first (WRONG: rank 2 for the true parent). With
    radii, the exact path recovers the correct ranking (rank 1)."""
    eps_u = 1e-7
    eps_1 = 1.180938237847287e-09
    theta_1 = 0.0015629306964751483
    n_2 = 0.8396984652461656
    theta_2 = 0.41861718386919283

    emb = np.array([
        [1.0 - eps_u, 0.0],
        [(1 - eps_1) * np.cos(theta_1), (1 - eps_1) * np.sin(theta_1)],
        [n_2 * np.cos(theta_2), n_2 * np.sin(theta_2)],
    ])
    radii = np.array([
        2.0 * np.arctanh(1.0 - eps_u),
        2.0 * np.arctanh(1.0 - eps_1),
        2.0 * np.arctanh(n_2),
    ])

    r_ball = rank_of_true_parent(emb, node=0, true_parent=2, candidates=np.array([1, 2]))
    assert r_ball == 2  # the floor's understatement lets candidate 1 wrongly win

    r_exact = rank_of_true_parent(
        emb, node=0, true_parent=2, candidates=np.array([1, 2]), radii=radii
    )
    assert r_exact == 1  # the exact path recovers the true nearest neighbour


# --- Finding 2 (fix round 1): the seeded tie-break jitter is never exercised without a real tie ---


def test_tie_break_is_deterministic_for_a_given_seed():
    """Two candidates at an EXACT Poincare-distance tie from the query (symmetric placement
    about the origin): the result must be reproducible for a fixed tie_seed."""
    emb = np.array([[0.0, 0.0], [0.3, 0.0], [-0.3, 0.0]])
    r_a = rank_of_true_parent(emb, node=0, true_parent=1, candidates=np.array([1, 2]), tie_seed=5)
    r_b = rank_of_true_parent(emb, node=0, true_parent=1, candidates=np.array([1, 2]), tie_seed=5)
    assert r_a == r_b


def test_tie_break_seed_changes_the_winner_among_tied_candidates():
    """Different tie_seed values must actually be able to change which tied candidate wins --
    proving the tie-break is real randomization, not disguised argsort order (which would
    always favour the lower index regardless of seed)."""
    emb = np.array([[0.0, 0.0], [0.3, 0.0], [-0.3, 0.0]])
    ranks = {
        rank_of_true_parent(emb, node=0, true_parent=1, candidates=np.array([1, 2]), tie_seed=s)
        for s in range(30)
    }
    assert ranks == {1, 2}, f"expected both tie outcomes across seeds, got {ranks}"


def test_jitter_does_not_reorder_candidates_with_a_real_distance_gap():
    """The ~1e-12 jitter scale must never overturn a real (non-tied) distance difference, across
    many different tie_seed values."""
    emb = np.array([[0.0, 0.0], [0.1, 0.0], [0.9, 0.0], [0.11, 0.0]])
    for s in range(30):
        r = rank_of_true_parent(
            emb, node=3, true_parent=1, candidates=np.array([1, 2]), tie_seed=s
        )
        assert r == 1
