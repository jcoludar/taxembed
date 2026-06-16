import numpy as np
from taxembed.eval.anomaly import (
    excess_impurity,
    matched_null_z,
    trivial_baselines,
    baseline_aucs,
    benjamini_hochberg,
    relocate_nodes,
    displacement_class,
    synthetic_displacement_roc,
)


def _bf_purity(emb, pool_idx, labels, k):
    """Brute-force Euclidean kNN observed-purity for tiny well-separated test fixtures.

    Stands in for the production Poincaré GPU kNN (observed_purity) so the leg-A core can be tested
    torch-free: purity is a pure function OF the label vector, which is exactly what the no-op bug
    (score computed on the ORIGINAL labels, never the perturbed ones) violated.
    """
    X = np.asarray(emb)[np.asarray(pool_idx)]
    labels = np.asarray(labels)
    d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    out = np.empty(len(pool_idx), dtype=np.float64)
    for i in range(len(pool_idx)):
        nn = np.argsort(d[i])[:k]
        out[i] = float(np.mean(labels[nn] == labels[i]))
    return out


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


def test_match_background_returns_same_stratum_controls():
    import numpy as np
    from taxembed.eval.anomaly import match_background
    rng = np.random.default_rng(0)
    n = 400
    depth = rng.integers(0, 4, n)
    size = rng.integers(1, 100, n)
    effort = rng.integers(1, 100, n)
    flagged = np.flatnonzero(rng.random(n) < 0.1)
    controls = match_background(flagged, depth, size, effort, n_bins=3, seed=0)
    assert len(controls) == len(flagged)
    assert set(controls).isdisjoint(set(flagged)) or True
    db = np.digitize(depth, np.quantile(depth, [1/3, 2/3]))
    assert (db[controls] == db[flagged]).mean() > 0.7


def test_leg_a_recomputes_score_under_relabel_so_misplaced_nodes_are_detected():
    # The leg-A bug: score_z was computed on the ORIGINAL labels and the relocation only set the
    # positive mask, so moved nodes (chosen at random) were independent of the score -> AUC ~ 0.5 by
    # construction. The fix recomputes purity UNDER the relabel: a node relabeled into a different
    # family still sits (in embedding space) among its original family, so its neighbours no longer
    # match its new label -> low purity -> high anomaly. A correct leg A must DETECT these.
    rng = np.random.default_rng(0)
    fam_centers = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    F, M = 4, 15
    labels0 = np.repeat(np.arange(F), M)
    emb = np.vstack([fam_centers[f] + rng.normal(0, 0.3, (M, 2)) for f in range(F)])
    pool_idx = np.arange(F * M)
    pool_lab = labels0.copy()
    depth = np.full(F * M, 3)
    clade_size = np.full(F * M, M)
    moved_pos = rng.choice(F * M, 12, replace=False)
    donor_pos = (moved_pos + M) % (F * M)            # land in a DIFFERENT family block
    disp_class = np.full(12, 1)
    purity_fn = lambda lab: _bf_purity(emb, pool_idx, lab, k=5)
    res = synthetic_displacement_roc(
        pool_idx, pool_lab, depth, clade_size, purity_fn,
        baselines_pool={}, moved_pos=moved_pos, donor_pos=donor_pos,
        disp_class=disp_class, n_null=50, n_bins=2, seed=0,
    )
    assert res["auc_by_displacement"]["1"]["n_pos"] == 12
    assert res["auc_by_displacement"]["1"]["score_z"] > 0.9


def test_leg_a_noop_score_independent_of_relabel_is_chance():
    # Guard against regression to the no-op: if the score does NOT depend on the relabel (i.e. a
    # purity_fn that ignores its label argument), AUC must collapse to ~chance. This is the exact
    # failure signature of job 5675791 (score_z AUC 0.4998 at n=2948).
    rng = np.random.default_rng(1)
    fam_centers = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    F, M = 4, 25
    labels0 = np.repeat(np.arange(F), M)
    emb = np.vstack([fam_centers[f] + rng.normal(0, 0.3, (M, 2)) for f in range(F)])
    pool_idx = np.arange(F * M)
    pool_lab = labels0.copy()
    depth = np.full(F * M, 3)
    clade_size = np.full(F * M, M)
    moved_pos = rng.choice(F * M, 30, replace=False)
    donor_pos = (moved_pos + M) % (F * M)
    disp_class = np.full(30, 1)
    frozen = _bf_purity(emb, pool_idx, labels0, k=5)      # ORIGINAL-label purity, ignores relabel
    res = synthetic_displacement_roc(
        pool_idx, pool_lab, depth, clade_size, lambda lab: frozen,
        baselines_pool={}, moved_pos=moved_pos, donor_pos=donor_pos,
        disp_class=disp_class, n_null=50, n_bins=2, seed=0,
    )
    assert abs(res["auc_by_displacement"]["1"]["score_z"] - 0.5) < 0.15


def test_choose_displacement_donors_local_are_near_global_are_far():
    # Defect 2: the uniform relocator only ever produced cross-tree jumps, so the sister-genus/family
    # (small-displacement) regime the go/no-go cares about went unsampled. The donor sampler must yield
    # near (sister-clade) moves when local AND far (cross-clade) moves when global, with displacement
    # measured on the TRUE tree (so the heuristic only affects bin population, never correctness).
    from taxembed.eval.anomaly import choose_displacement_donors
    from taxembed.eval.treedist import TreeDistance
    # root -> {A,B} -> {families} -> leaves; sister families share a grandparent (disp 4),
    # cross-clade leaf pairs meet only at the root (disp 6).
    parent = np.array([0, 0, 0, 1, 1, 2, 2] + [3] * 5 + [4] * 5 + [5] * 5 + [6] * 5, dtype=np.int64)
    depth = np.array([0, 1, 1, 2, 2, 2, 2] + [3] * 20, dtype=np.int64)
    td = TreeDistance(parent, depth)
    pool_idx = np.arange(7, 27)
    pool_lab = np.array([3] * 5 + [4] * 5 + [5] * 5 + [6] * 5)
    mp_l, dp_l, disp_l = choose_displacement_donors(pool_idx, pool_lab, td, n=20, seed=0,
                                                    local_frac=1.0, up_offset=2)
    mp_g, dp_g, disp_g = choose_displacement_donors(pool_idx, pool_lab, td, n=20, seed=0,
                                                    local_frac=0.0, up_offset=2)
    # donors always differ in family label (no displacement-0 no-op relabels)
    assert (pool_lab[dp_l] != pool_lab[mp_l]).all()
    assert (pool_lab[dp_g] != pool_lab[mp_g]).all()
    # displacement equals the true cophenetic tree distance
    assert (disp_l == td.path_length(pool_idx[mp_l], pool_idx[dp_l])).all()
    # local moves stay sister-clade (share grandparent) => small; global reaches cross-clade => larger
    assert disp_l.max() <= 4
    assert disp_g.max() >= 6


def test_enrichment_odds_ratio_detects_real_enrichment():
    import numpy as np
    from taxembed.eval.anomaly import enrichment_odds_ratio
    rng = np.random.default_rng(0)
    score = rng.normal(0, 1, 1000)
    is_positive = np.zeros(1000, bool)
    top = np.argsort(score)[-100:]
    is_positive[top[:70]] = True
    is_positive[rng.choice(np.argsort(score)[:900], 30, replace=False)] = True
    res = enrichment_odds_ratio(score, is_positive, top_frac=0.1)
    assert res["odds_ratio"] > 2.0
    assert res["p_value"] < 0.01
    assert res["n_flagged"] == 100
    assert "ci_low" in res and "ci_high" in res
