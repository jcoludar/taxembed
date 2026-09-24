import numpy as np
import pytest

from taxembed.eval.baselines import (
    chance_mrr_mean, degree_prior_metrics, degree_prior_rank, majority_parent_rate,
    sibling_chance, vendrov_closure_rule,
)


def test_vendrov_rule_is_true_exactly_on_visible_closure_pairs():
    anc = np.array([0, 0, 1])
    dsc = np.array([1, 3, 3])
    got = vendrov_closure_rule(anc, dsc, queries=np.array([3, 3]), candidates=np.array([0, 2]))
    assert got.tolist() == [[True, False], [True, False]]


def test_vendrov_rule_returns_all_false_when_the_parent_edge_was_withheld():
    """The held-out parent is unreachable by construction: the rule cannot recover it."""
    anc = np.array([0])          # only (0 -> 1) visible; (1 -> 3) was withheld
    dsc = np.array([1])
    got = vendrov_closure_rule(anc, dsc, queries=np.array([3]), candidates=np.array([1]))
    assert got.tolist() == [[False]]


def test_sibling_chance_is_one_over_the_grandparents_fanout():
    # 0 -> 1, 0 -> 2 ; 1 -> 3, 1 -> 4. Node 3's parent is 1, whose parent 0 has 2 children.
    parent = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    assert sibling_chance(parent, np.array([3])).tolist() == [0.5]


def test_sibling_chance_is_one_when_the_true_parent_is_an_only_child():
    parent = np.array([0, 0, 1], dtype=np.int64)   # 0 -> 1 -> 2; 0 has one child
    assert sibling_chance(parent, np.array([2])).tolist() == [1.0]


def test_majority_parent_rate_is_the_share_of_the_commonest_true_parent():
    parent = np.array([0, 0, 0, 1, 1, 2], dtype=np.int64)
    # held out 3, 4, 5 -> true parents 1, 1, 2 -> majority share 2/3
    assert majority_parent_rate(parent, np.array([3, 4, 5])) == 2 / 3


# --- C1 (review finding, 2026-09-24, p2_amendment_3_20260924): a training-free degree prior
# (rank candidates by child count, descending) already clears every floor the frozen
# pre-registration checked -- 0.5750 MRR on the real metazoa split. Declared here as a baseline.

# 0 -> {1, 2, 3}; 1 -> {4,5,6,7} (fanout 4); 2 -> {8} (fanout 1); 3 -> {} (fanout 0, a leaf)
_FANOUT_TREE_PARENT = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2], dtype=np.int64)


def test_degree_prior_ranks_the_highest_fanout_candidate_first():
    """LESION CHECK. Node 4's true parent (1) is the HIGHEST-fanout candidate among {1, 2, 3} --
    the degree prior, with no embedding and no training, must rank it first."""
    rank = degree_prior_rank(_FANOUT_TREE_PARENT, node=4, true_parent=1,
                             candidates=np.array([1, 2, 3]))
    assert rank == 1


def test_degree_prior_ranks_the_lowest_fanout_candidate_last():
    """HEALTHY / control direction: same candidate pool, but the true parent (3) has the FEWEST
    children -- the prior must rank it last, not first."""
    rank = degree_prior_rank(_FANOUT_TREE_PARENT, node=4, true_parent=3,
                             candidates=np.array([1, 2, 3]))
    assert rank == 3


def test_degree_prior_tie_break_is_seeded_and_reproducible():
    """Ties (equal fan-out) must be broken by SEEDED jitter, not index order -- mirrors
    `taxembed.eval.linkpred.rank_of_true_parent`'s own tie-break tests."""
    parent = np.array([0, 0, 0], dtype=np.int64)   # nodes 1 and 2 both have 0 children -- a tie
    candidates = np.array([1, 2], dtype=np.int64)
    r_a = degree_prior_rank(parent, node=0, true_parent=1, candidates=candidates, tie_seed=5)
    r_b = degree_prior_rank(parent, node=0, true_parent=1, candidates=candidates, tie_seed=5)
    assert r_a == r_b
    ranks = {degree_prior_rank(parent, node=0, true_parent=1, candidates=candidates, tie_seed=s)
             for s in range(20)}
    assert ranks == {1, 2}, f"expected both tie outcomes across seeds, got {ranks}"


def test_degree_prior_metrics_reduce_to_the_linkpred_metrics_shape():
    """A training-free ranker over several held nodes reduces to the SAME MRR/hits@1/
    normalized_rank shape a learned ranker's `linkpred_metrics` returns, so the two are directly
    comparable as declared baselines in the verdict engine."""
    held = np.array([4, 8], dtype=np.int64)
    candidates_list = [np.array([1, 2, 3]), np.array([1, 2, 3])]
    true_parents = np.array([1, 2], dtype=np.int64)   # node 4 -> fanout-4 parent (rank 1)
    n_candidates = np.array([3, 3], dtype=np.int64)   # node 8 -> fanout-1 parent (rank 2 of 3)
    m = degree_prior_metrics(_FANOUT_TREE_PARENT, held, candidates_list, true_parents, n_candidates)
    assert m["mrr"] == pytest.approx((1.0 + 0.5) / 2)
    assert m["n"] == 2


# --- C2 (review finding, 2026-09-24): sibling_chance_mean (= mean(1/k)) is the chance rate for
# HITS@1, not for MRR. chance_mrr_mean (= mean(H_k/k)) is the correct anchor for an MRR-valued gate.


def test_chance_mrr_mean_matches_a_hand_computed_harmonic_average():
    n_candidates = np.array([2, 5])
    h2 = 1.0 + 1.0 / 2.0
    h5 = 1.0 + 1.0 / 2.0 + 1.0 / 3.0 + 1.0 / 4.0 + 1.0 / 5.0
    expected = float(np.mean([h2 / 2.0, h5 / 5.0]))
    assert chance_mrr_mean(n_candidates) == pytest.approx(expected)


def test_chance_mrr_mean_is_strictly_above_the_hits_at_1_chance_rate():
    """LESION CHECK. H_k > 1 for every k > 1, so a uniformly random ranker's own MRR
    (mean(H_k/k)) must be STRICTLY greater than the hits@1 chance rate (mean(1/k)) whenever any
    pool has k > 1. An implementation that collapsed to `mean(1/k)` (the C2 defect this baseline
    replaces as gate (b)'s anchor) would make the two equal instead."""
    n_candidates = np.array([2, 5, 10, 100])
    mrr_chance = chance_mrr_mean(n_candidates)
    hits1_chance = float(np.mean(1.0 / n_candidates))
    assert mrr_chance > hits1_chance


def test_chance_mrr_mean_equals_the_hits_at_1_rate_only_at_pool_size_one():
    """HEALTHY / degenerate case: at k=1, H_1=1, so H_k/k == 1/k == 1 exactly -- the two chance
    quantities coincide ONLY in this trivial (single-candidate) case, confirming the strict
    inequality above is not an artefact of the formula but genuinely tied to k > 1."""
    n_candidates = np.array([1, 1, 1])
    assert chance_mrr_mean(n_candidates) == pytest.approx(1.0)
    assert chance_mrr_mean(n_candidates) == pytest.approx(float(np.mean(1.0 / n_candidates)))
