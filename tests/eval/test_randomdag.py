import numpy as np

from taxembed.eval.randomdag import closure_from_parent, randomize_parents


def chain_and_bush():
    # 0 -> {1,2,3}; 1 -> {4,5}; 2 -> {6}
    parent = np.array([0, 0, 0, 0, 1, 1, 2], dtype=np.int64)
    depth = np.array([0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
    return parent, depth


def test_randomization_preserves_every_node_depth():
    parent, depth = chain_and_bush()
    rp = randomize_parents(parent, depth, seed=0)
    assert (depth[rp[depth > 0]] == depth[depth > 0] - 1).all()


def test_randomization_preserves_the_closure_pair_count_exactly():
    parent, depth = chain_and_bush()
    before = len(closure_from_parent(parent, depth))
    after = len(closure_from_parent(randomize_parents(parent, depth, seed=0), depth))
    assert before == after == int(depth.sum())


def test_randomization_actually_moves_something():
    """A control that returns the input is not a control.

    Needs a tree wide enough that a random rewire is unlikely to reproduce the original:
    10 depth-1 nodes and 190 depth-2 nodes, so each depth-2 node picks among 10 parents.
    """
    n = 201
    parent = np.zeros(n, dtype=np.int64)
    depth = np.zeros(n, dtype=np.int64)
    depth[1:11] = 1                      # nodes 1..10 hang off the root
    depth[11:] = 2                       # nodes 11..200 hang off nodes 1..10
    parent[11:] = 1 + (np.arange(n - 11) % 10)
    rp = randomize_parents(parent, depth, seed=0)
    assert not np.array_equal(rp, parent)


def test_the_root_stays_the_root():
    parent, depth = chain_and_bush()
    assert randomize_parents(parent, depth, seed=3)[0] == 0


def test_randomization_is_seed_stable():
    parent, depth = chain_and_bush()
    assert np.array_equal(
        randomize_parents(parent, depth, seed=5), randomize_parents(parent, depth, seed=5)
    )
