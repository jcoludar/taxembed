"""Pin a sha256 of the SHIPPED negative sampler's draws on a fixed fixture.

Run BEFORE the Task 6 sampler change, against the committed train_hierarchical.py. The test
test_flags_off_matches_the_pinned_shipped_sampler then asserts the modified loader, with both new
flags off, reproduces this hash exactly -- i.e. the unfixed arm is the historical sampler, not a
re-implementation of it. Fixture = tests/test_negative_sampling.py::_random_parent(300, 4),
seed_everything(5), n_negatives=10, batch_size=32, full epoch, no curriculum.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))

from taxembed.utils.training_pairs import TrainingPairs  # noqa: E402
from train_hierarchical import HierarchicalDataLoader, seed_everything  # noqa: E402


def random_parent(n, seed):
    rng = np.random.default_rng(seed)
    parent = np.zeros(n, dtype=np.int64)
    for i in range(1, n):
        parent[i] = int(rng.integers(max(0, i - 40), i))
    return parent


def closure(parent):
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


def main() -> None:
    parent = random_parent(300, 4)
    anc, des, dd = closure(parent)
    depth = np.zeros(len(parent), dtype=np.int64)
    for v in range(1, len(parent)):
        depth[v] = depth[parent[v]] + 1
    pairs = TrainingPairs(
        ancestor_idx=anc.astype(np.int32), descendant_idx=des.astype(np.int32),
        depth_diff=dd.astype(np.int16), ancestor_depth=depth[anc].astype(np.int16),
        descendant_depth=depth[des].astype(np.int16),
        ancestor_taxid=anc.astype(np.int32), descendant_taxid=des.astype(np.int32),
    )
    seed_everything(5)
    loader = HierarchicalDataLoader(pairs, n_nodes=len(parent), batch_size=32, n_negatives=10)
    h = hashlib.sha256()
    n = 0
    for a, d, neg, _ in loader:
        h.update(a.numpy().tobytes())
        h.update(d.numpy().tobytes())
        h.update(neg.numpy().tobytes())
        n += 1
    print(f"batches={n} sha256={h.hexdigest()}")


if __name__ == "__main__":
    main()
