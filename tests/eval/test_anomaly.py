import numpy as np
from taxembed.eval.anomaly import (
    excess_impurity,
    matched_null_z,
    trivial_baselines,
    baseline_aucs,
    benjamini_hochberg,
    relocate_nodes,
    displacement_class,
)


def test_excess_impurity_size_conditioned_sign():
    observed = np.array([1.0, 0.5, 0.0])
    chance = 0.25
    score = excess_impurity(observed, chance)
    assert score[2] > score[1] > score[0]
    assert np.isclose(score[0], chance - 1.0)
    assert np.isclose(score[2], chance - 0.0)


def test_matched_null_z_flags_outlier_not_small_clade():
    rng = np.random.default_rng(0)
    null_obs = rng.normal(0.8, 0.1, size=(4, 200))
    observed = np.array([0.8, 0.79, 0.81, 0.2])
    z = matched_null_z(observed, null_obs)
    assert z[3] > 3.0
    assert abs(z[0]) < 1.0 and abs(z[1]) < 1.0


def test_trivial_baselines_shapes_and_keys():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((20, 4)) * 0.1
    parent = np.array([0] + [0] * 9 + [1] * 10)
    depth = np.array([0] + [1] * 9 + [2] * 10)
    clade_size = np.full(20, 3)
    degree = np.full(20, 2)
    b = trivial_baselines(emb, parent, depth, clade_size, degree)
    assert set(b) == {"clade_size", "depth", "degree", "dist_to_parent_centroid"}
    for v in b.values():
        assert v.shape == (20,)


def test_baseline_aucs_ranks_a_good_score_above_random():
    rng = np.random.default_rng(0)
    labels = np.zeros(200, int); labels[:20] = 1
    good = rng.normal(0, 1, 200); good[:20] += 4
    junk = rng.normal(0, 1, 200)
    aucs = baseline_aucs(labels, {"good": good, "junk": junk})
    assert aucs["good"] > 0.9
    assert abs(aucs["junk"] - 0.5) < 0.15


def test_benjamini_hochberg_monotone_and_bounded():
    p = np.array([0.001, 0.008, 0.02, 0.04, 0.9])
    q = benjamini_hochberg(p)
    assert q.shape == p.shape
    assert (q >= p - 1e-12).all()
    assert (q <= 1.0 + 1e-12).all()
    assert (q[:2] <= 0.05).all()


def test_relocate_and_displacement_class():
    parent = np.array([0, 0, 0, 1, 1])
    depth = np.array([0, 1, 1, 2, 2])
    new_parent, moved = relocate_nodes(parent, depth, n=2, seed=0)
    assert moved.shape[0] == 2
    assert (new_parent[moved] != parent[moved]).all()
    dc = displacement_class(parent, new_parent, depth, moved)
    assert dc.shape == moved.shape
    assert (dc >= 0).all()
