import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train_hierarchical import seed_everything


def test_seed_everything_makes_legacy_numpy_draws_reproducible():
    seed_everything(1234)
    a = np.random.randint(0, 1000, size=50)
    seed_everything(1234)
    b = np.random.randint(0, 1000, size=50)
    assert (a == b).all()


def test_seed_everything_makes_torch_init_reproducible():
    import torch

    seed_everything(7)
    a = torch.randn(20)
    seed_everything(7)
    b = torch.randn(20)
    assert torch.equal(a, b)


def test_different_seeds_differ():
    seed_everything(1)
    a = np.random.randint(0, 10_000, size=50)
    seed_everything(2)
    b = np.random.randint(0, 10_000, size=50)
    assert not (a == b).all()


# ---------------------------------------------------------------- Task 6: ancestry-aware sampler

def _random_parent(n, seed):
    rng = np.random.default_rng(seed)
    parent = np.zeros(n, dtype=np.int64)
    for i in range(1, n):
        parent[i] = int(rng.integers(max(0, i - 40), i))    # bushy-but-deep random tree
    return parent


def _closure(parent):
    """Full transitive closure (ancestor, descendant, depth_diff) from a parent array."""
    anc, des, dd = [], [], []
    for v in range(1, len(parent)):
        cur, k = v, 0
        while parent[cur] != cur:
            cur = int(parent[cur])
            k += 1
            anc.append(cur)
            des.append(v)
            dd.append(k)
    return np.array(anc), np.array(des), np.array(dd)


def _is_desc(parent, a, x):
    while True:
        if x == a:
            return True
        if parent[x] == x:
            return False
        x = int(parent[x])


def _loader(parent, **kw):
    from train_hierarchical import HierarchicalDataLoader

    anc, des, dd = _closure(parent)
    return HierarchicalDataLoader.from_arrays(ancestor_idx=anc, descendant_idx=des,
                                              depth_diff=dd, **kw)


def test_no_emitted_negative_is_a_descendant_of_its_anchor():
    parent = _random_parent(400, 0)
    seed_everything(0)
    loader = _loader(parent, n_negatives=20, batch_size=64,
                     exclude_descendant_negatives=True, drop_root_anchored=True)
    n_checked = 0
    for ancestors, _d, negatives, _dep in loader:
        for a, row in zip(ancestors.numpy(), negatives.numpy()):
            for x in row:
                assert not _is_desc(parent, int(a), int(x)), f"negative {x} descends from anchor {a}"
                n_checked += 1
    assert n_checked > 10_000
    assert loader._n_replaced > 0, "fixture never produced a false negative -- test is vacuous"


def test_resampled_negatives_keep_the_descendant_depth_when_a_valid_one_exists():
    parent = _random_parent(400, 1)
    depth = np.zeros(400, dtype=np.int64)
    for v in range(1, 400):
        depth[v] = depth[parent[v]] + 1
    seed_everything(1)
    loader = _loader(parent, n_negatives=20, batch_size=64,
                     exclude_descendant_negatives=True, drop_root_anchored=True)
    for ancestors, descendants, negatives, _dep in loader:
        a = ancestors.numpy()
        d = descendants.numpy()
        for i in range(len(a)):
            same_depth = np.flatnonzero(depth == depth[d[i]])
            has_valid = any(not _is_desc(parent, int(a[i]), int(x)) for x in same_depth)
            if has_valid and len(same_depth) > 20:           # fast path: pure same-depth draws
                assert (depth[negatives.numpy()[i]] == depth[d[i]]).all()


def test_zero_pool_rows_relax_depth_not_ancestry_and_are_counted():
    """Tree 0 -> 1 -> {2,3}; 0 -> 4. Anchor 1 owns the whole depth-2 layer {2,3}: no same-depth
    valid negative exists. Spec v3 P1.3: relax the DEPTH constraint, never the ancestry one."""
    parent = np.array([0, 0, 1, 1, 0])
    seed_everything(2)
    loader = _loader(parent, n_negatives=2, batch_size=8,
                     exclude_descendant_negatives=True, drop_root_anchored=True)
    for ancestors, _d, negatives, _dep in loader:
        for a, row in zip(ancestors.numpy(), negatives.numpy()):
            for x in row:
                assert not _is_desc(parent, int(a), int(x))
    assert loader._zero_pool_hits > 0


def test_root_anchored_pairs_are_dropped_without_breaking_indices():
    parent = _random_parent(300, 3)
    seed_everything(3)
    loader = _loader(parent, n_negatives=5, batch_size=32, epoch_fraction=0.5,
                     exclude_descendant_negatives=True, drop_root_anchored=True)
    loader.set_curriculum_phase(3)
    seen = set()
    for ancestors, _d, _n, _dep in loader:                  # must not IndexError
        seen.update(ancestors.numpy().tolist())
    assert 0 not in seen
    assert len(seen) > 5


SHIPPED_SAMPLER_SHA256 = "930ab50b81ced0328ed16228f0a39b4ce159ee75235d0b679e880feda046c371"


def test_flags_off_matches_the_pinned_shipped_sampler():
    """The unfixed arm must BE the historical sampler, not a re-implementation of it.
    The hash was pinned by helpers/pin_default_sampler_hash.py against the committed
    train_hierarchical.py BEFORE the Task 6 change (2026-09-22, 111 batches)."""
    import hashlib

    parent = _random_parent(300, 4)
    seed_everything(5)
    loader = _loader(parent, n_negatives=10, batch_size=32,
                     exclude_descendant_negatives=False, drop_root_anchored=False)
    h = hashlib.sha256()
    n = 0
    for a, d, neg, _ in loader:
        h.update(a.numpy().tobytes())
        h.update(d.numpy().tobytes())
        h.update(neg.numpy().tobytes())
        n += 1
    assert n == 111
    assert h.hexdigest() == SHIPPED_SAMPLER_SHA256


def test_counters_are_bounded_ints():
    parent = _random_parent(200, 6)
    seed_everything(6)
    loader = _loader(parent, n_negatives=5, batch_size=16,
                     exclude_descendant_negatives=True, drop_root_anchored=True)
    for _ in loader:
        pass
    stats = loader.sampler_stats()
    for key in ("n_rows", "n_draws", "n_replaced", "zero_pool_rows"):
        assert isinstance(stats[key], int)
    assert stats["n_rows"] > 0 and 0.0 < stats["observed_fn_rate"] < 1.0


def test_exclude_without_drop_root_is_refused():
    import pytest

    with pytest.raises(ValueError, match="drop_root_anchored"):
        _loader(_random_parent(50, 7), n_negatives=3, batch_size=8,
                exclude_descendant_negatives=True, drop_root_anchored=False)
