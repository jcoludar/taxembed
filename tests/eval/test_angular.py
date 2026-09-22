"""Tests for the radius-free same-depth angular LCA score (Task 9 scorer).

Every closed form is checked against a brute-force oracle built from a DIFFERENT mechanism
(naive parent-walking LCA, not Euler intervals), so a shared bug cannot pass both.
"""
import numpy as np
import pytest

from taxembed.eval.angular import (
    TreeIndex,
    pool_counts,
    null_lca_depth,
    oracle_lca_depth,
    lca_depths,
    select_queries,
    same_depth_neighbors,
    score_embedding,
)


# ---------------------------------------------------------------- fixtures / naive oracles

def _binary_tree_depth3():
    """0 -> {1,2}; 1 -> {3,4}; 2 -> {5,6}; 3 -> {7,8}; 4 -> {9,10}; 5 -> {11,12}; 6 -> {13,14}."""
    parent = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
    depth = np.array([0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3])
    return parent, depth


def _random_tree(n, seed):
    """Random recursive tree: node i attaches to a uniformly chosen earlier node."""
    rng = np.random.default_rng(seed)
    parent = np.zeros(n, dtype=np.int64)
    depth = np.zeros(n, dtype=np.int64)
    for i in range(1, n):
        p = int(rng.integers(0, i))
        parent[i] = p
        depth[i] = depth[p] + 1
    return parent, depth


def _naive_ancestors(parent, v):
    chain = [v]
    while parent[chain[-1]] != chain[-1]:
        chain.append(int(parent[chain[-1]]))
    return chain[::-1]          # root first


def _naive_lca_depth(parent, depth, a, b):
    anc_a = set(_naive_ancestors(parent, a))
    return max(int(depth[x]) for x in _naive_ancestors(parent, b) if x in anc_a)


def _naive_pool_lca(parent, depth, q):
    pool = [v for v in range(len(parent)) if depth[v] == depth[q] and v != q]
    return np.array([_naive_lca_depth(parent, depth, q, v) for v in pool], dtype=float)


def _perfect_directions(parent, depth):
    """u_v = sum of one-hot vectors of v's non-root ancestors (incl. v).

    For two nodes at the same depth d, u_a . u_b = LCA depth, and |u| = sqrt(d), so cosine
    is a strictly increasing function of LCA depth: the ideal cousin ordering.
    """
    n = len(parent)
    u = np.zeros((n, n))
    for v in range(n):
        for a in _naive_ancestors(parent, v)[1:]:
            u[v, a] = 1.0
    u[0, 0] = 1.0               # root: any nonzero direction
    return u


# ---------------------------------------------------------------- closed forms

def test_pool_counts_on_hand_tree():
    parent, depth = _binary_tree_depth3()
    idx = TreeIndex(parent, depth)
    # query 7: pool {8..14}; under 1: {8,9,10}; under 3: {8}; under 7: {}
    assert pool_counts(idx, 7).tolist() == [7, 3, 1, 0]


def test_null_matches_hand_enumeration():
    parent, depth = _binary_tree_depth3()
    idx = TreeIndex(parent, depth)
    # LCA depths of 7 with 8..14: 2,1,1,0,0,0,0 -> mean 4/7
    assert null_lca_depth(pool_counts(idx, 7)) == pytest.approx(4 / 7)


def test_oracle_matches_hand_enumeration():
    parent, depth = _binary_tree_depth3()
    idx = TreeIndex(parent, depth)
    c = pool_counts(idx, 7)
    assert oracle_lca_depth(c, k=1) == pytest.approx(2.0)
    assert oracle_lca_depth(c, k=3) == pytest.approx(4 / 3)
    assert oracle_lca_depth(c, k=5) == pytest.approx(4 / 5)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_null_and_oracle_match_brute_force_on_random_trees(seed):
    parent, depth = _random_tree(300, seed)
    idx = TreeIndex(parent, depth)
    k = 5
    checked = 0
    for q in range(1, 300):
        truth = _naive_pool_lca(parent, depth, q)
        if len(truth) < k:
            continue
        c = pool_counts(idx, q)
        assert null_lca_depth(c) == pytest.approx(truth.mean())
        assert oracle_lca_depth(c, k) == pytest.approx(np.sort(truth)[::-1][:k].mean())
        checked += 1
    assert checked > 50, "fixture too small to exercise the closed forms"


def test_lca_depths_match_naive():
    parent, depth = _random_tree(300, 7)
    idx = TreeIndex(parent, depth)
    rng = np.random.default_rng(0)
    q = rng.integers(0, 300, size=40)
    v = rng.integers(0, 300, size=(40, 6))
    got = lca_depths(idx, q, v)
    want = np.array([[_naive_lca_depth(parent, depth, int(a), int(b)) for b in row]
                     for a, row in zip(q, v)])
    assert (got == want).all()


# ---------------------------------------------------------------- queries

def test_select_queries_is_deterministic_and_filters_unrankable():
    parent, depth = _random_tree(400, 3)
    idx = TreeIndex(parent, depth)
    a = select_queries(idx, n=100, k=5, seed=11)
    b = select_queries(idx, n=100, k=5, seed=11)
    assert (a == b).all()
    for q in a:
        c = pool_counts(idx, int(q))
        assert c[0] >= 5
        assert oracle_lca_depth(c, 5) > null_lca_depth(c)


# ---------------------------------------------------------------- the score itself

def test_perfect_directions_score_one():
    parent, depth = _random_tree(250, 5)
    idx = TreeIndex(parent, depth)
    emb = _perfect_directions(parent, depth)
    q = select_queries(idx, n=10_000, k=5, seed=0)
    out = score_embedding(emb, idx, q, k=5, metric="cosine")
    assert out["S"] == pytest.approx(1.0)


def test_random_directions_score_zero():
    """The initialization state: planted radius, random direction. S must sit at 0."""
    parent, depth = _random_tree(4000, 9)
    idx = TreeIndex(parent, depth)
    rng = np.random.default_rng(1)
    emb = rng.standard_normal((4000, 32))
    q = select_queries(idx, n=3000, k=10, seed=0)
    out = score_embedding(emb, idx, q, k=10, metric="cosine")
    se = out["S_bootstrap_se"]
    assert se > 0
    assert abs(out["S"]) < 3 * se, f"S={out['S']:.4f} vs 3*SE={3 * se:.4f}"


def test_angle_score_is_blind_to_radius_but_poincare_is_not():
    parent, depth = _random_tree(1500, 4)
    idx = TreeIndex(parent, depth)
    rng = np.random.default_rng(2)
    base = _perfect_directions(parent, depth)[:, :]
    base = base + 0.8 * rng.standard_normal(base.shape)       # imperfect but structured
    unit = base / np.linalg.norm(base, axis=1, keepdims=True)

    planted = unit * (0.1 + 0.85 * np.log1p(depth) / np.log1p(depth.max()))[:, None]
    drifted = unit * rng.uniform(0.05, 0.95, size=(len(depth), 1))

    q = select_queries(idx, n=2000, k=10, seed=0)
    a_planted = score_embedding(planted, idx, q, k=10, metric="cosine")
    a_drifted = score_embedding(drifted, idx, q, k=10, metric="cosine")
    assert np.array_equal(a_planted["s"], a_drifted["s"]), "cosine score read the radius"

    p_planted = score_embedding(planted, idx, q, k=10, metric="poincare")
    p_drifted = score_embedding(drifted, idx, q, k=10, metric="poincare")
    # at planted (depth-constant) radius, Poincare ranking within a depth == cosine ranking
    assert p_planted["S"] == pytest.approx(a_planted["S"])
    assert not np.array_equal(p_planted["s"], p_drifted["s"]), "poincare score ignored radius"


def test_collapsed_directions_do_not_score_through_tie_breaking():
    """Every direction identical -> all cosines tie. If ties were broken in tree (tin) order the
    'neighbours' would be the query's closest relatives and S would approach 1. It must sit at 0."""
    parent, depth = _random_tree(4000, 12)
    idx = TreeIndex(parent, depth)
    emb = np.ones((4000, 8))
    q = select_queries(idx, n=3000, k=10, seed=0)
    out = score_embedding(emb, idx, q, k=10, metric="cosine")
    assert abs(out["S"]) < 3 * out["S_bootstrap_se"], f"S={out['S']:.4f}"


def test_forest_is_rejected():
    parent = np.array([0, 0, 2, 2])          # two roots: 0 and 2
    depth = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="single rooted tree"):
        TreeIndex(parent, depth)


def test_paired_difference_of_a_run_with_itself_is_zero():
    from taxembed.eval.angular import paired_S_difference, query_bounds

    parent, depth = _random_tree(2000, 6)
    idx = TreeIndex(parent, depth)
    rng = np.random.default_rng(4)
    emb = _perfect_directions(parent, depth) + 1.5 * rng.standard_normal((2000, 2000))
    q = select_queries(idx, n=1000, k=10, seed=0)
    mu0, mus = query_bounds(idx, q, 10)
    s = score_embedding(emb, idx, q, k=10, bounds=(mu0, mus))["s"]
    out = paired_S_difference(s, s, mu0, mus, idx.ancestor_at(q, 3))
    assert out["diff"] == 0.0 and out["ci95"] == [0.0, 0.0]


def test_poincare_from_hyperbolic_radii_matches_poincare_from_points():
    parent, depth = _random_tree(800, 10)
    idx = TreeIndex(parent, depth)
    rng = np.random.default_rng(5)
    unit = rng.standard_normal((800, 16))
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    x = unit * rng.uniform(0.05, 0.9, size=(800, 1))
    r = 2.0 * np.arctanh(np.linalg.norm(x, axis=1))
    q = select_queries(idx, n=300, k=5, seed=0)
    a = same_depth_neighbors(x, idx, q, 5, metric="poincare")
    b = same_depth_neighbors(x, idx, q, 5, metric="poincare", radii=r)
    assert (a == b).all()


def test_neighbors_exclude_self_and_stay_at_query_depth():
    parent, depth = _random_tree(600, 8)
    idx = TreeIndex(parent, depth)
    rng = np.random.default_rng(3)
    emb = rng.standard_normal((600, 16))
    q = select_queries(idx, n=200, k=5, seed=0)
    nb = same_depth_neighbors(emb, idx, q, k=5, metric="cosine")
    assert nb.shape == (len(q), 5)
    assert (nb != q[:, None]).all()
    assert (depth[nb] == depth[q][:, None]).all()
