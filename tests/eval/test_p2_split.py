import numpy as np
import pytest

from taxembed.eval.p2_split import (
    depth_from_closure,
    eligible_nodes,
    leaf_mask,
    parent_edge_mask,
    select_holdout,
)
from taxembed.utils.training_pairs import TrainingPairs


def tiny_tree():
    """
    0 (root, depth 0)
    |- 1 (depth 1) -- internal
    |  |- 3 (depth 2) -- leaf
    |  |- 4 (depth 2) -- leaf
    |- 2 (depth 1) -- leaf
    """
    parent = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    depth = np.array([0, 1, 1, 2, 2], dtype=np.int64)
    return parent, depth


def tiny_pairs():
    """Full closure of tiny_tree, as TrainingPairs."""
    anc = np.array([0, 0, 0, 0, 1, 1], dtype=np.int32)
    dsc = np.array([1, 2, 3, 4, 3, 4], dtype=np.int32)
    ad = np.array([0, 0, 0, 0, 1, 1], dtype=np.int16)
    dd_ = np.array([1, 1, 2, 2, 2, 2], dtype=np.int16)
    ddiff = (dd_ - ad).astype(np.int16)
    return TrainingPairs(
        ancestor_idx=anc, descendant_idx=dsc, depth_diff=ddiff,
        ancestor_depth=ad, descendant_depth=dd_,
        ancestor_taxid=anc.copy(), descendant_taxid=dsc.copy(),
    )


def test_leaf_mask_marks_exactly_the_childless_nodes():
    parent, _ = tiny_tree()
    assert leaf_mask(parent).tolist() == [False, False, True, True, True]


def test_root_is_not_a_leaf_even_though_it_self_parents():
    parent, _ = tiny_tree()
    assert not leaf_mask(parent)[0]


def test_depth_from_closure_recovers_every_node_depth():
    pairs = tiny_pairs()
    depth = depth_from_closure(
        pairs.descendant_idx, pairs.descendant_depth,
        pairs.ancestor_idx, pairs.ancestor_depth, n_nodes=5,
    )
    assert depth.tolist() == [0, 1, 1, 2, 2]


def test_eligible_nodes_applies_band_and_leaf_restriction():
    parent, depth = tiny_tree()
    # band [2, 2] keeps only depth-2 nodes; both are leaves
    assert eligible_nodes(parent, depth, band=(2, 2)).tolist() == [3, 4]
    # band [1, 2] with leaves_only drops internal node 1, keeps leaf 2
    assert eligible_nodes(parent, depth, band=(1, 2)).tolist() == [2, 3, 4]
    # without the leaf restriction node 1 returns
    assert eligible_nodes(parent, depth, band=(1, 2), leaves_only=False).tolist() == [1, 2, 3, 4]


def test_eligible_nodes_never_includes_the_root():
    parent, depth = tiny_tree()
    assert 0 not in eligible_nodes(parent, depth, band=(0, 99), leaves_only=False).tolist()


def test_select_holdout_partitions_eligible_exactly_and_is_seed_stable():
    eligible = np.arange(100, dtype=np.int64)
    a = select_holdout(eligible, frac_test=0.10, frac_val=0.10, seed=0)
    b = select_holdout(eligible, frac_test=0.10, frac_val=0.10, seed=0)
    c = select_holdout(eligible, frac_test=0.10, frac_val=0.10, seed=1)

    assert len(a["test"]) == 10 and len(a["val"]) == 10 and len(a["train"]) == 80
    union = np.concatenate([a["test"], a["val"], a["train"]])
    assert np.array_equal(np.sort(union), eligible)      # partition, nothing lost
    assert len(np.unique(union)) == 100                  # and nothing duplicated
    assert np.array_equal(a["test"], b["test"])          # same seed, same split
    assert not np.array_equal(a["test"], c["test"])      # different seed, different split


def test_parent_edge_mask_removes_only_the_dd1_row_of_held_out_nodes():
    pairs = tiny_pairs()
    mask = parent_edge_mask(pairs, held_out=np.array([3], dtype=np.int64))
    # only the (1 -> 3) dd==1 row is marked
    removed = [(int(pairs.ancestor_idx[i]), int(pairs.descendant_idx[i]))
               for i in np.flatnonzero(mask)]
    assert removed == [(1, 3)]


def test_parent_edge_mask_leaves_the_grandparent_edge_intact():
    """The held-out node must keep a coordinate: its dd>=2 ancestry stays in training."""
    pairs = tiny_pairs()
    mask = parent_edge_mask(pairs, held_out=np.array([3], dtype=np.int64))
    kept = pairs[~mask]
    assert (0, 3) in list(zip(kept.ancestor_idx.tolist(), kept.descendant_idx.tolist()))


def test_a_held_out_leaf_parent_is_NOT_recoverable_from_the_retained_closure():
    """The load-bearing property. Node 3 is a leaf; removing (1,3) must hide parent 1."""
    pairs = tiny_pairs()
    mask = parent_edge_mask(pairs, held_out=np.array([3], dtype=np.int64))
    kept = pairs[~mask]
    reachable_to_3 = {int(a) for a, d in zip(kept.ancestor_idx, kept.descendant_idx) if d == 3}
    assert 1 not in reachable_to_3


def test_the_SAME_check_FAILS_for_an_internal_node_the_negative_control():
    """
    A check that could not have failed is not evidence. Holding out internal node 1's
    parent edge does NOT hide parent 0, because 1's descendants keep their (0, 3) and
    (0, 4) edges. This is exactly why the holdout is restricted to leaves.
    """
    pairs = tiny_pairs()
    mask = parent_edge_mask(pairs, held_out=np.array([1], dtype=np.int64))
    kept = pairs[~mask]
    # 0 is no longer a direct ancestor of 1 ...
    reachable_to_1 = {int(a) for a, d in zip(kept.ancestor_idx, kept.descendant_idx) if d == 1}
    assert 0 not in reachable_to_1
    # ... but 0 is still the unique child-side ancestor of 1's whole subtree, so it leaks.
    subtree_of_1 = {3, 4}
    ancestors_of_subtree = {
        int(a) for a, d in zip(kept.ancestor_idx, kept.descendant_idx) if int(d) in subtree_of_1
    }
    assert 0 in ancestors_of_subtree


def test_eligible_nodes_rejects_a_band_whose_bounds_are_inverted():
    parent, depth = tiny_tree()
    with pytest.raises(ValueError):
        eligible_nodes(parent, depth, band=(28, 11))


from pathlib import Path

from taxembed.eval.subtree import parent_from_closure

CELLULAR = Path(__file__).resolve().parents[2] / (
    "data/taxopy/cellular_organisms_131567_clean/"
    "taxonomy_edges_cellular_organisms_131567_clean_transitive.npz"
)


@pytest.mark.skipif(not CELLULAR.exists(), reason="cellular closure not on this machine")
def test_band_eligibility_reproduces_the_spec_figures_on_the_real_closure():
    pairs = TrainingPairs.load(CELLULAR)
    n = pairs.n_nodes
    parent = parent_from_closure(
        pairs.ancestor_idx, pairs.descendant_idx, pairs.depth_diff, n
    )
    depth = depth_from_closure(
        pairs.descendant_idx, pairs.descendant_depth,
        pairs.ancestor_idx, pairs.ancestor_depth, n,
    )
    # spec v3 §P2.3: "593,576 eligible nodes" == depth in [11, 28] inclusive, all nodes
    assert len(eligible_nodes(parent, depth, leaves_only=False)) == 593_576
    # and the leaf-restricted set this plan actually holds out
    assert len(eligible_nodes(parent, depth, leaves_only=True)) == 501_037
