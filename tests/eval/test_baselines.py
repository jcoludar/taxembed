import numpy as np

from taxembed.eval.baselines import majority_parent_rate, sibling_chance, vendrov_closure_rule


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
