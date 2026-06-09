import numpy as np
from taxembed.eval.nulls import radial_only_null, shuffled_label_null, random_ball_null


def _emb():
    rng = np.random.default_rng(1)
    e = rng.standard_normal((50, 8)) * 0.1
    return e


def test_radial_only_preserves_norms_changes_direction():
    e = _emb()
    null = radial_only_null(e, seed=0)
    assert np.allclose(np.linalg.norm(e, axis=1), np.linalg.norm(null, axis=1))
    assert not np.allclose(e, null)               # directions scrambled
    assert null.shape == e.shape


def test_shuffled_label_is_a_permutation():
    e = _emb()
    null = shuffled_label_null(e, seed=0)
    # every row of null is some row of e (a permutation of the SAME vectors)
    assert sorted(null.sum(axis=1).round(6)) == sorted(e.sum(axis=1).round(6))


def test_random_ball_inside_ball():
    null = random_ball_null(50, 8, seed=0)
    assert null.shape == (50, 8)
    assert (np.linalg.norm(null, axis=1) < 1.0).all()
