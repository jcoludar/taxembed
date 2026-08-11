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
