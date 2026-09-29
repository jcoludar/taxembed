"""Tests for the P3 placement scorer's machinery, on synthetic trees with known answers.

WHY THESE EXIST. The P2 investigation of 2026-09-29 found an evaluation reading BELOW chance for
200 epochs with nobody able to say whether the cause was the metric, the pool, or a sign flip. The
cheapest guard against repeating that is a test that pins the direction and the chance floor on
data where the right answer is known by construction:

  - a PLANTED embedding, where the true target is nearest, must give normalized_rank == 0.0
  - an ADVERSARIAL embedding, where the true target is farthest, must give 1.0
  - RANDOM directions must give ~0.5

If the ranking direction is ever inverted, test 1 and 2 swap and both fail loudly.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "helpers"))
sys.path.insert(0, str(ROOT / "src"))

import _p3_placement_score as P  # noqa: E402


# ------------------------------------------------------------------ geometry

def test_poincare_dist_is_zero_to_self_and_grows_with_separation():
    u = np.array([0.1, 0.0])
    V = np.array([[0.1, 0.0], [0.2, 0.0], [0.5, 0.0]])
    d = P.poincare_dist(u, V)
    assert d[0] == pytest.approx(0.0, abs=1e-9)
    assert d[1] < d[2]


def test_poincare_dist_grows_without_bound_toward_the_rim():
    """The whole point of the ball: equal euclidean steps cost more near |x|=1."""
    u = np.array([0.0, 0.0])
    near = P.poincare_dist(u, np.array([[0.5, 0.0]]))[0]
    far = P.poincare_dist(u, np.array([[0.99, 0.0]]))[0]
    assert far > 3 * near


# ------------------------------------------------------------------ tree

def _chain_tree():
    """parent map for: 1 -> 2 -> {3,4,5}, and 2 -> 6 -> {7,8}.  1 is root (self-parent)."""
    return {1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 2, 7: 6, 8: 6}


def test_build_tree_depths_and_lca():
    taxids, idx, parent, depth, tin, tout = P.build_tree(_chain_tree())
    assert depth[idx[1]] == 0
    assert depth[idx[2]] == 1
    assert depth[idx[3]] == 2
    assert depth[idx[7]] == 3
    assert P.lca(idx[3], idx[4], parent, depth) == idx[2]
    assert P.lca(idx[3], idx[7], parent, depth) == idx[2]
    assert P.lca(idx[7], idx[8], parent, depth) == idx[6]


def test_pool_index_returns_only_same_depth_nodes_inside_the_subtree():
    taxids, idx, parent, depth, tin, tout = P.build_tree(_chain_tree())
    embedded = np.ones(len(taxids), dtype=bool)
    pi = P.PoolIndex(depth, tin, embedded)

    # depth-2 nodes under node 2 are {3,4,5,6}
    got = set(pi.candidates(2, idx[2], tin, tout).tolist())
    assert got == {idx[3], idx[4], idx[5], idx[6]}
    # depth-3 nodes under 6 are {7,8}
    assert set(pi.candidates(3, idx[6], tin, tout).tolist()) == {idx[7], idx[8]}


def test_a_node_is_its_own_descendant_so_the_pool_degenerates_when_L_is_the_target():
    """When depth(p_new) == depth(L) the pool is {L} alone, and that is CORRECT.

    It is the case where a taxon moved UP to an ancestor of its old parent, so
    L = LCA(p_old, p_new) == p_new. The pre-registration excludes pool size < 2 as trivial
    (rank is always 1 there, for every implementation), so this degenerate pool is dropped
    rather than scored as a free win -- the same rule as p2_amendment_3_20260924.
    """
    taxids, idx, parent, depth, tin, tout = P.build_tree(_chain_tree())
    pi = P.PoolIndex(depth, tin, np.ones(len(taxids), dtype=bool))
    pool = pi.candidates(2, idx[6], tin, tout)
    assert pool.tolist() == [idx[6]]           # L itself, nothing else at that depth below it
    assert len(pool) < 2                       # => excluded upstream as trivial


def test_pool_index_excludes_unembedded_nodes():
    taxids, idx, parent, depth, tin, tout = P.build_tree(_chain_tree())
    embedded = np.ones(len(taxids), dtype=bool)
    embedded[idx[4]] = False
    pi = P.PoolIndex(depth, tin, embedded)
    got = set(pi.candidates(2, idx[2], tin, tout).tolist())
    assert idx[4] not in got
    assert got == {idx[3], idx[5], idx[6]}


# ------------------------------------------------ amendment 2: branch exclusion

def _two_branch_tree():
    """L=1 with two children 10 and 20; each has two depth-2 children.

    10 -> {11, 12}   (p_old's branch)
    20 -> {21, 22}   (p_new's branch)
    """
    return {1: 1, 10: 1, 20: 1, 11: 10, 12: 10, 21: 20, 22: 20}


def test_child_toward_finds_the_branch_of_L_holding_a_node():
    taxids, idx, parent, depth, tin, tout = P.build_tree(_two_branch_tree())
    assert P.child_toward(idx[1], idx[11], parent) == idx[10]
    assert P.child_toward(idx[1], idx[22], parent) == idx[20]
    assert P.child_toward(idx[1], idx[10], parent) == idx[10]
    assert P.child_toward(idx[1], idx[1], parent) is None     # node IS L


def test_amendment_2_removes_p_olds_entire_branch_not_just_p_old():
    """The defect gate (b) caught: p_new can NEVER be in p_old's branch of L, so leaving that
    branch in the pool puts the target only in the far portion and shifts the chance floor
    off 0.5. The whole branch must go, not merely p_old itself.
    """
    pm = _two_branch_tree()
    taxids, idx, parent, depth, tin, tout = P.build_tree(pm)
    n = len(taxids)
    pi = P.PoolIndex(depth, tin, np.ones(n, dtype=bool))
    emb = np.zeros((n, 2), dtype=np.float64)
    rng = np.random.default_rng(0)
    for i in range(n):
        emb[i] = rng.normal(size=2) * 0.1

    # v hangs under 11; p_old = 11 (in branch 10), p_new = 21 (in branch 20). L = 1.
    pm2 = dict(pm)
    pm2[999] = 11
    taxids, idx, parent, depth, tin, tout = P.build_tree(pm2)
    n = len(taxids)
    pi = P.PoolIndex(depth, tin, np.ones(n, dtype=bool))
    emb = np.zeros((n, 2), dtype=np.float64)
    for i in range(n):
        emb[i] = rng.normal(size=2) * 0.1

    q = [(idx[999], idx[21], idx[11])]
    nr, sizes = P.normalized_ranks(q, np.arange(n), emb, pi, parent, depth, tin, tout, None)
    # depth-2 nodes under L=1 are {11,12,21,22}; branch 10 contributes {11,12} and must be gone,
    # leaving exactly {21,22}.
    assert sizes == [2], f"pool should be the 2 nodes of p_new's branch only, got {sizes}"
    assert len(nr) == 1


# ------------------------------------------------------------------ the statistic

def _setup(n_sib=9):
    """v is a child of node 100; candidates are nodes 200..200+n_sib at the same depth."""
    pm = {1: 1, 2: 1}
    L = 2
    cands = list(range(200, 200 + n_sib))
    for c in cands:
        pm[c] = L
    v = 999
    pm[v] = cands[0]                      # v's parent is the FIRST candidate
    return pm, v, cands, L


def _run(emb_vectors, target, exclude):
    pm, v, cands, L = _setup()
    taxids, idx, parent, depth, tin, tout = P.build_tree(pm)
    n = len(taxids)
    emb_rows = np.arange(n, dtype=np.int64)
    embedded = np.ones(n, dtype=bool)
    pi = P.PoolIndex(depth, tin, embedded)
    emb = np.zeros((n, 2), dtype=np.float64)
    for taxid, vec in emb_vectors.items():
        emb[idx[taxid]] = vec
    q = [(idx[v], idx[target], idx[exclude])]
    nr, sizes = P.normalized_ranks(q, emb_rows, emb, pi, parent, depth, tin, tout, None)
    return nr, sizes, cands


def test_planted_embedding_gives_rank_zero():
    """True target nearest => normalized_rank 0.0. Pins the DIRECTION of the ranking."""
    pm, v, cands, L = _setup()
    target = cands[3]
    vecs = {v: np.array([0.0, 0.0])}
    for i, c in enumerate(cands):
        # target sits on top of v; everyone else is far away
        vecs[c] = np.array([0.0, 0.0]) if c == target else np.array([0.5 + 0.01 * i, 0.0])
    nr, sizes, _ = _run(vecs, target=target, exclude=cands[0])
    assert len(nr) == 1
    assert nr[0] == pytest.approx(0.0, abs=1e-9)


def test_adversarial_embedding_gives_rank_one():
    """True target farthest => 1.0. If the sort were inverted this would return 0.0."""
    pm, v, cands, L = _setup()
    target = cands[3]
    vecs = {v: np.array([0.0, 0.0])}
    for i, c in enumerate(cands):
        vecs[c] = np.array([0.9, 0.0]) if c == target else np.array([0.01 * (i + 1), 0.0])
    nr, sizes, _ = _run(vecs, target=target, exclude=cands[0])
    assert nr[0] == pytest.approx(1.0, abs=1e-9)


def test_random_directions_sit_at_the_chance_floor():
    """The floor the whole verdict rests on: E[normalized_rank] = 0.5 under random geometry."""
    rng = np.random.default_rng(0)
    vals = []
    for trial in range(400):
        pm, v, cands, L = _setup()
        target = cands[1 + (trial % 7)]
        vecs = {v: rng.normal(size=2) * 0.2}
        for c in cands:
            vecs[c] = rng.normal(size=2) * 0.2
        nr, _, _ = _run(vecs, target=target, exclude=cands[0])
        if len(nr):
            vals.append(nr[0])
    mean = float(np.mean(vals))
    assert 0.44 < mean < 0.56, f"chance floor drifted: {mean}"


def test_pool_of_one_is_excluded_as_trivial():
    """k<2 is always rank 1 for every implementation; prereg excludes it."""
    pm = {1: 1, 2: 1, 200: 2, 999: 200}
    taxids, idx, parent, depth, tin, tout = P.build_tree(pm)
    n = len(taxids)
    pi = P.PoolIndex(depth, tin, np.ones(n, dtype=bool))
    emb = np.zeros((n, 2))
    q = [(idx[999], idx[200], idx[200])]
    nr, sizes = P.normalized_ranks(q, np.arange(n), emb, pi, parent, depth, tin, tout, None)
    assert len(nr) == 0        # excluded, not scored as a free win


def test_bootstrap_ci_brackets_a_known_mean():
    x = np.full(2000, 0.5)
    lo, hi = P.boot_ci(x, n_boot=500)
    assert lo == pytest.approx(0.5, abs=1e-9) and hi == pytest.approx(0.5, abs=1e-9)
    y = np.concatenate([np.zeros(1000), np.ones(1000)])
    lo, hi = P.boot_ci(y, n_boot=2000)
    assert lo < 0.5 < hi
