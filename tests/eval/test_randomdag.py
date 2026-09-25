import numpy as np

from taxembed.eval.randomdag import closure_from_parent, degree_matched_shuffle, randomize_parents


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


def test_randomization_different_seeds_give_different_results():
    """🛑 Review finding I5, the half fix wave 2 did NOT close. Wave 2 gave
    `degree_matched_shuffle` a different-seeds test plus a lesion check; `randomize_parents` kept
    only same-seed stability, which holds trivially for a function that ignores `seed` outright.

    Measured before writing this: replacing `default_rng(seed)` with `default_rng(0)` inside
    `randomize_parents` passed ALL 16 randomdag tests
    (`helpers/p2_lesion_harness.py --mutation randomize_parents_ignores_seed`). If that lesion were
    real, the frozen/amendment_2 RandomDAG control's three seeds would be ONE draw reported three
    times -- three identical 'independent' control runs.

    Uses the wide fixture, not `chain_and_bush()`: with 3 candidate parents per node a small tree
    can reproduce the same assignment across two seeds by chance, which would make this test
    flaky rather than falsifiable.
    """
    parent, depth = _wide_two_level_tree()
    out5 = randomize_parents(parent, depth, seed=5)
    out6 = randomize_parents(parent, depth, seed=6)
    assert not np.array_equal(out5, out6)


def test_a_hardcoded_internal_seed_fails_the_randomize_parents_different_seeds_check():
    """LESION CHECK for the test above, mirroring the one wave 2 wrote for
    `degree_matched_shuffle`: a `randomize_parents`-alike that accepts `seed` and ignores it passes
    every other assertion in this file, so the different-seeds test must be the one that refuses
    it. Without this, the test above could itself be vacuously satisfiable."""
    parent, depth = _wide_two_level_tree()

    def hardcoded_seed_randomize(parent, depth, seed=0):
        del seed
        return randomize_parents(parent, depth, seed=0)   # always seed 0, ignores the arg

    out5 = hardcoded_seed_randomize(parent, depth, seed=5)
    out6 = hardcoded_seed_randomize(parent, depth, seed=6)
    assert np.array_equal(out5, out6), "fixture must reproduce the lesion for this test to mean anything"


## ------------------------------------------------------------------------------------------
## degree_matched_shuffle (p2_amendment_4_20260924): replaces randomize_parents as P2's control.
## Preserves the fan-out MULTISET exactly (a permutation of the real parent-label list per depth
## level), never an i.i.d. resample -- the fix for randomize_parents inflating the chance floor
## 5.4x by collapsing fan-out toward the mean (p2_amendment_1_20260924).

def _heavy_tailed_tree():
    """One depth-1 hub with 90 children (a heavy fan-out node) beside 9 depth-1 nodes with 1
    child each -- 10 depth-1 parents, 99 depth-2 children, deliberately UNEVEN so a construction
    that regresses toward the mean (randomize_parents' failure mode) is easy to tell apart from
    one that preserves the multiset exactly (degree_matched_shuffle)."""
    n = 1 + 10 + 99
    parent = np.zeros(n, dtype=np.int64)
    depth = np.zeros(n, dtype=np.int64)
    depth[1:11] = 1
    depth[11:] = 2
    # node 1 (the hub) gets 90 children; nodes 2..10 get 1 child each.
    labels = np.array([1] * 90 + list(range(2, 11)), dtype=np.int64)
    assert len(labels) == 99
    parent[11:] = labels
    return parent, depth


def _wide_two_level_tree():
    """10 depth-1 nodes, 190 depth-2 nodes (mirrors the module's existing wide fixture above) --
    wide enough that a genuine shuffle is astronomically unlikely to reproduce the identity."""
    n = 201
    parent = np.zeros(n, dtype=np.int64)
    depth = np.zeros(n, dtype=np.int64)
    depth[1:11] = 1
    depth[11:] = 2
    parent[11:] = 1 + (np.arange(n - 11) % 10)
    return parent, depth


def _fanout_of(parent: np.ndarray) -> np.ndarray:
    parent = np.asarray(parent, dtype=np.int64)
    real = np.arange(len(parent)) != parent
    return np.bincount(parent[real], minlength=len(parent))


def test_degree_matched_shuffle_preserves_every_node_depth():
    parent, depth = _wide_two_level_tree()
    out = degree_matched_shuffle(parent, depth, seed=0)
    assert (depth[out[depth > 0]] == depth[depth > 0] - 1).all()


def test_degree_matched_shuffle_preserves_the_closure_pair_count_exactly():
    parent, depth = _wide_two_level_tree()
    before = len(closure_from_parent(parent, depth))
    after = len(closure_from_parent(degree_matched_shuffle(parent, depth, seed=0), depth))
    assert before == after == int(depth.sum())


def test_degree_matched_shuffle_preserves_the_fanout_multiset_exactly():
    """The load-bearing invariant: not just the MEAN fan-out, the entire SORTED multiset of
    per-node child counts must be identical before and after -- this is what makes the chance
    floor and the degree prior match the real tree by construction (the whole point of the
    amendment). A construction that merely preserves mean fan-out (e.g. randomize_parents' i.i.d.
    resample) would fail this on the heavy-tailed fixture below."""
    parent, depth = _heavy_tailed_tree()
    out = degree_matched_shuffle(parent, depth, seed=0)
    fo_before = np.sort(_fanout_of(parent))
    fo_after = np.sort(_fanout_of(out))
    assert np.array_equal(fo_before, fo_after)
    # not a vacuous check: the real tree's fan-out is genuinely heavy-tailed (one hub at 90).
    assert fo_before.max() == 90


def test_a_construction_that_only_preserves_MEAN_fanout_fails_the_multiset_check():
    """Proves the multiset test above is not vacuously satisfiable: randomize_parents' i.i.d.
    uniform resample preserves depth exactly (same as degree_matched_shuffle) but does NOT
    preserve the fan-out multiset on a heavy-tailed tree -- confirming the multiset assertion
    above is actually discriminating, not just checking something every construction gets for
    free."""
    parent, depth = _heavy_tailed_tree()
    out = randomize_parents(parent, depth, seed=0)
    fo_before = np.sort(_fanout_of(parent))
    fo_after = np.sort(_fanout_of(out))
    assert not np.array_equal(fo_before, fo_after)


def test_degree_matched_shuffle_actually_moves_something():
    """A control that returns its input is not a control (mirrors randomize_parents' own test)."""
    parent, depth = _wide_two_level_tree()
    out = degree_matched_shuffle(parent, depth, seed=0)
    assert not np.array_equal(out, parent)


def test_degree_matched_shuffle_the_root_stays_the_root():
    parent, depth = chain_and_bush()
    assert degree_matched_shuffle(parent, depth, seed=3)[0] == 0


def test_degree_matched_shuffle_is_seed_stable():
    parent, depth = _wide_two_level_tree()
    assert np.array_equal(
        degree_matched_shuffle(parent, depth, seed=5), degree_matched_shuffle(parent, depth, seed=5)
    )


def test_degree_matched_shuffle_different_seeds_give_different_results():
    """The gap the module docstring calls out: the original test_randomdag.py suite never proved
    `seed` was used for anything beyond being accepted as a parameter -- hard-coding
    `default_rng(0)` inside the function would still pass every OTHER test here. This one fails
    on that lesion directly."""
    parent, depth = _wide_two_level_tree()
    out5 = degree_matched_shuffle(parent, depth, seed=5)
    out6 = degree_matched_shuffle(parent, depth, seed=6)
    assert not np.array_equal(out5, out6)


def test_a_hardcoded_internal_seed_fails_the_different_seeds_check():
    """LESION CHECK for the test above: a `degree_matched_shuffle`-alike that ignores `seed` and
    always seeds its RNG with a constant would pass every invariant test above except this one --
    confirming the different-seeds test is not vacuously satisfiable either."""
    parent, depth = _wide_two_level_tree()

    def hardcoded_seed_shuffle(parent, depth, seed=0):
        del seed
        return degree_matched_shuffle(parent, depth, seed=0)   # always seed 0, ignores the arg

    out5 = hardcoded_seed_shuffle(parent, depth, seed=5)
    out6 = hardcoded_seed_shuffle(parent, depth, seed=6)
    assert np.array_equal(out5, out6), "fixture must reproduce the lesion for this test to mean anything"


def _walk_to_root_steps(parent: np.ndarray, depth: np.ndarray, node: int) -> int:
    """Number of hops from `node` up to a depth-0 node via `parent`, raising AssertionError if a
    node is revisited (a cycle -- some node is its own ancestor) or the walk does not terminate
    within `len(parent)` hops."""
    n = len(parent)
    cur = int(node)
    seen: set[int] = set()
    steps = 0
    while depth[cur] != 0:
        assert cur not in seen, f"cycle detected walking up from node {node}"
        seen.add(cur)
        cur = int(parent[cur])
        steps += 1
        assert steps <= n, f"walk from node {node} did not terminate within {n} steps"
    return steps


def test_degree_matched_shuffle_no_node_becomes_its_own_ancestor_and_stays_single_rooted():
    """Walk every node up to the root, following the SHUFFLED parent pointers; each walk must
    terminate at a depth-0 node in exactly `depth[node]` steps, with no repeated node visited
    along the way (a cycle would mean some node became its own ancestor) -- and there must be
    exactly one depth-0 node (single-rooted)."""
    parent, depth = _heavy_tailed_tree()
    out = degree_matched_shuffle(parent, depth, seed=0)
    assert int((depth == 0).sum()) == 1, "fixture must be single-rooted for this test to mean anything"
    for v in range(len(out)):
        steps = _walk_to_root_steps(out, depth, v)
        assert steps == int(depth[v]), (
            f"node {v} reached the root in {steps} steps, expected exactly depth {depth[v]}")


def test_a_construction_that_reassigns_across_depths_fails_the_single_rooted_check():
    """LESION CHECK: a broken shuffle that reassigns a depth-2 node's parent to ANOTHER depth-2
    node (rather than a depth-1 node) breaks the depth-consistency invariant the walk-to-root
    check above relies on -- proving that check is not vacuously satisfiable."""
    parent, depth = _heavy_tailed_tree()
    broken = parent.copy()
    # node 11 and node 12 are both depth-2 siblings; point 11 at 12 instead of a depth-1 node.
    assert depth[11] == 2 and depth[12] == 2
    broken[11] = 12
    steps = _walk_to_root_steps(broken, depth, 11)
    assert steps != int(depth[11]), "the broken fixture must fail the walk-to-root step-count check"
