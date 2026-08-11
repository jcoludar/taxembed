import numpy as np
import pytest

from taxembed.eval.subtree import parent_from_closure, euler_intervals, is_descendant


def _chain_and_fork():
    """Tree:  0 -> 1 -> {2, 3};  3 -> 4
    depths:   0    1     2  2         3
    """
    # closure pairs (ancestor, descendant, depth_diff)
    anc = np.array([0, 0, 0, 0, 1, 1, 1, 3])
    des = np.array([1, 2, 3, 4, 2, 3, 4, 4])
    dd = np.array([1, 2, 2, 3, 1, 1, 2, 1])
    return anc, des, dd


def test_parent_from_closure_uses_only_direct_edges():
    anc, des, dd = _chain_and_fork()
    parent = parent_from_closure(anc, des, dd, n_nodes=5)
    assert parent.tolist() == [0, 0, 1, 1, 3]


def test_euler_intervals_span_subtree_sizes():
    anc, des, dd = _chain_and_fork()
    parent = parent_from_closure(anc, des, dd, n_nodes=5)
    tin, tout = euler_intervals(parent)
    # subtree sizes: 0->5, 1->4, 2->1, 3->2, 4->1
    assert (tout - tin).tolist() == [5, 4, 1, 2, 1]


def test_is_descendant_includes_self_and_excludes_siblings():
    anc, des, dd = _chain_and_fork()
    parent = parent_from_closure(anc, des, dd, n_nodes=5)
    tin, tout = euler_intervals(parent)
    nodes = np.arange(5)
    # everything descends from root 0
    assert is_descendant(np.zeros(5, dtype=int), nodes, tin, tout).all()
    # node 2 and node 3 are siblings
    assert not is_descendant(np.array([2]), np.array([3]), tin, tout)[0]
    # 4 descends from 3, not from 2
    assert is_descendant(np.array([3]), np.array([4]), tin, tout)[0]
    assert not is_descendant(np.array([2]), np.array([4]), tin, tout)[0]
    # self-descendance holds (this is what makes drawing the positive count as a false negative)
    assert is_descendant(np.array([3]), np.array([3]), tin, tout)[0]
