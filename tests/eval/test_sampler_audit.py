import numpy as np

from taxembed.eval.sampler_audit import false_negative_audit


def _two_branch():
    """Tree: 0 -> {1, 2};  1 -> {3, 4};  2 -> {5, 6}
    depths:  0     1  1     2  2          2  2
    Closure pairs (ancestor, descendant):
      (0,1) (0,2) (0,3) (0,4) (0,5) (0,6) (1,3) (1,4) (2,5) (2,6)
    """
    anc = np.array([0, 0, 0, 0, 0, 0, 1, 1, 2, 2])
    des = np.array([1, 2, 3, 4, 5, 6, 3, 4, 5, 6])
    ad = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    dd = np.array([1, 1, 2, 2, 2, 2, 2, 2, 2, 2])
    return anc, des, ad, dd


def test_root_anchored_pairs_are_always_100_percent_false_negative():
    anc, des, ad, dd = _two_branch()
    out = false_negative_audit(anc, des, ad, dd)
    root_rows = [r for r in out["by_anchor_depth"] if r["anchor_depth"] == 0]
    assert len(root_rows) == 1
    # every node at any depth descends from the root
    assert root_rows[0]["fn_rate"] == 1.0
    assert root_rows[0]["n_zero_pool"] == root_rows[0]["n_pairs"]


def test_depth1_anchor_rate_is_half_of_its_depth_layer():
    anc, des, ad, dd = _two_branch()
    out = false_negative_audit(anc, des, ad, dd)
    # anchors 1 and 2 each own 2 of the 4 nodes at depth 2 -> 0.5
    d1 = [r for r in out["by_anchor_depth"] if r["anchor_depth"] == 1][0]
    assert d1["fn_rate"] == 0.5
    assert d1["n_zero_pool"] == 0


def test_overall_rate_is_the_pair_weighted_mean():
    anc, des, ad, dd = _two_branch()
    out = false_negative_audit(anc, des, ad, dd)
    # 6 root-anchored pairs at 1.0, 4 depth-1-anchored pairs at 0.5
    assert out["overall_rate"] == (6 * 1.0 + 4 * 0.5) / 10
    assert out["n_root_anchored"] == 6
    assert out["n_zero_pool"] == 6


def test_nodes_per_depth_counts_every_node_once():
    anc, des, ad, dd = _two_branch()
    out = false_negative_audit(anc, des, ad, dd)
    assert out["nodes_per_depth"] == [1, 2, 4]
