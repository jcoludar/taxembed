"""Pure core for Application #2 — taxonomy QC / anomaly detection (spec §5#2, §9B).

The anomaly score is a SIZE-CONDITIONED kNN-impurity: raw impurity just rediscovers rare/small
clades (spec §9B), so we report it relative to its size-matched expectation (excess_impurity) and,
as the headline, a z-score vs a depth+clade-size-matched random-angle null (matched_null_z). The
score must beat trivial baselines (clade size, depth, degree, distance-to-parent-centroid) on the
synthetic ROC. No I/O here — operates on integer node arrays and float observed-purity arrays.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import fisher_exact


def excess_impurity(observed_purity: np.ndarray, chance_purity: float) -> np.ndarray:
    """Size-conditioned anomaly score = chance_purity - observed_purity (higher == more anomalous).

    chance_purity is the size-aware expectation Sum_g (n_g/P)^2 (reuse knn_purity_hyperbolic.chance_purity).
    A clean node has observed >> chance -> negative; an impure node has observed << chance -> positive.
    """
    return float(chance_purity) - np.asarray(observed_purity, dtype=np.float64)


def matched_null_z(observed_purity: np.ndarray, null_observed: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Headline score: z of (mu_null - observed) over a depth+clade-size-matched null (higher == anomalous).

    null_observed: (Q, n_null) observed-purity values for matched random-angle null draws per query.
    Returns (Q,) z-scores; a node far BELOW its matched null's purity scores high.
    """
    observed = np.asarray(observed_purity, dtype=np.float64)
    null = np.asarray(null_observed, dtype=np.float64)
    mu = null.mean(axis=1)
    sigma = null.std(axis=1)
    return (mu - observed) / (sigma + eps)


def trivial_baselines(emb: np.ndarray, parent: np.ndarray, depth: np.ndarray,
                      clade_size: np.ndarray, degree: np.ndarray) -> dict:
    """The trivial baselines the score must beat (spec §9B). All (N,), higher == 'more anomalous' guess.

    - clade_size / depth / degree: raw structural quantities (rank by them directly).
    - dist_to_parent_centroid: Euclidean dist from each node's embedding to the centroid of its
      siblings (children of the same parent) — a geometry baseline that ignores neighbour identity.
    """
    emb = np.asarray(emb, dtype=np.float64)
    parent = np.asarray(parent, dtype=np.int64)
    n = len(parent)
    sums = np.zeros((n, emb.shape[1]), dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)
    np.add.at(sums, parent, emb)
    np.add.at(counts, parent, 1.0)
    counts = np.maximum(counts, 1.0)
    parent_centroid = sums[parent] / counts[parent, None]
    dist = np.linalg.norm(emb - parent_centroid, axis=1)
    return {
        "clade_size": np.asarray(clade_size, dtype=np.float64),
        "depth": np.asarray(depth, dtype=np.float64),
        "degree": np.asarray(degree, dtype=np.float64),
        "dist_to_parent_centroid": dist,
    }


def baseline_aucs(labels: np.ndarray, scores: dict) -> dict:
    """ROC-AUC of each score (higher == more anomalous) against binary anomaly labels."""
    labels = np.asarray(labels, dtype=np.int64)
    out = {}
    for name, s in scores.items():
        out[name] = float(roc_auc_score(labels, np.asarray(s, dtype=np.float64)))
    return out


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """BH-FDR adjusted q-values (spec §9B: BH-control per-node significance over ~1.1M nodes)."""
    p = np.asarray(pvals, dtype=np.float64)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(1, n + 1))
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n, dtype=np.float64)
    q[order] = np.minimum(ranked, 1.0)
    return q


def relocate_nodes(parent: np.ndarray, depth: np.ndarray, n: int, seed: int = 0):
    """Synthetic leg-A: move n random NON-root nodes to a random DIFFERENT parent.

    Returns (new_parent, moved_idx). Only nodes with depth>0 are eligible; the new parent is drawn
    uniformly from nodes that are not the node itself nor its current parent.
    """
    parent = np.asarray(parent, dtype=np.int64).copy()
    depth = np.asarray(depth, dtype=np.int64)
    rng = np.random.default_rng(seed)
    eligible = np.flatnonzero(depth > 0)
    moved = rng.choice(eligible, size=min(n, len(eligible)), replace=False)
    new_parent = parent.copy()
    all_nodes = np.arange(len(parent))
    for v in moved:
        choices = all_nodes[(all_nodes != v) & (all_nodes != parent[v])]
        new_parent[v] = rng.choice(choices)
    return new_parent, moved


def displacement_class(orig_parent: np.ndarray, new_parent: np.ndarray,
                       depth: np.ndarray, moved: np.ndarray) -> np.ndarray:
    """Phylogenetic displacement magnitude per moved node (spec §9B: stratify ROC by displacement).

    Defined as the tree distance between the OLD and NEW parent via their depths and LCA-free proxy:
    here we use |depth[old_parent] - depth[new_parent]| + 2 (a monotone proxy for how far the node
    jumped); the CLI replaces this with the exact TreeDistance.path_length(old_parent,new_parent) to
    get the true sister-genus -> cross-kingdom ladder. This pure helper returns the depth-gap proxy so
    the core stays decoupled from TreeDistance; both are monotone in displacement.
    """
    depth = np.asarray(depth, dtype=np.int64)
    op = np.asarray(orig_parent, dtype=np.int64)[moved]
    npar = np.asarray(new_parent, dtype=np.int64)[moved]
    return np.abs(depth[op] - depth[npar]) + 2


def _qbin(x: np.ndarray, n_bins: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    qs = np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1]) if n_bins > 1 else np.array([])
    return np.digitize(x, qs)


def match_background(flagged_idx: np.ndarray, depth: np.ndarray, clade_size: np.ndarray,
                     study_effort: np.ndarray, n_bins: int = 5, seed: int = 0) -> np.ndarray:
    """For each flagged node draw one control from the SAME depth x size x effort stratum (spec §9B).

    Controls are preferentially non-flagged; if a stratum has no non-flagged member, falls back to any
    member of that stratum. Returns an index array parallel to flagged_idx.
    """
    rng = np.random.default_rng(seed)
    flagged_idx = np.asarray(flagged_idx, dtype=np.int64)
    n = len(depth)
    key = (_qbin(depth, n_bins).astype(np.int64) * (n_bins ** 2)
           + _qbin(clade_size, n_bins).astype(np.int64) * n_bins
           + _qbin(study_effort, n_bins).astype(np.int64))
    flagged_set = set(flagged_idx.tolist())
    controls = np.empty(len(flagged_idx), dtype=np.int64)
    for i, v in enumerate(flagged_idx):
        same = np.flatnonzero(key == key[v])
        pool = np.array([s for s in same if s not in flagged_set], dtype=np.int64)
        if pool.size == 0:
            pool = same[same != v]
        if pool.size == 0:
            pool = same
        controls[i] = rng.choice(pool)
    return controls


def enrichment_odds_ratio(score: np.ndarray, is_positive: np.ndarray,
                          top_frac: float = 0.1) -> dict:
    """Odds ratio + Fisher exact (2x2: flagged-vs-not x positive-vs-not) (spec §9B framing).

    'flagged' = top `top_frac` of nodes by score. Returns OR, Fisher p, the 2x2 counts, and a
    log-OR normal-approx 95% CI (Woolf). Higher score == more anomalous.
    """
    score = np.asarray(score, dtype=np.float64)
    pos = np.asarray(is_positive, dtype=bool)
    n = len(score)
    n_flag = max(1, int(round(top_frac * n)))
    flagged = np.zeros(n, bool)
    flagged[np.argsort(score)[-n_flag:]] = True
    a = int(np.sum(flagged & pos))
    b = int(np.sum(flagged & ~pos))
    c = int(np.sum(~flagged & pos))
    d = int(np.sum(~flagged & ~pos))
    odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative="greater")
    aa, bb, cc, dd = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    log_or = np.log((aa * dd) / (bb * cc))
    se = np.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    return {
        "odds_ratio": float(odds_ratio),
        "p_value": float(p_value),
        "ci_low": float(np.exp(log_or - 1.96 * se)),
        "ci_high": float(np.exp(log_or + 1.96 * se)),
        "n_flagged": int(n_flag),
        "counts": {"flagged_pos": a, "flagged_neg": b, "unflagged_pos": c, "unflagged_neg": d},
    }


def matched_null(observed, pool_idx, depth, clade_size, n_null, n_bins, seed):
    """Per-query null observed-purity drawn from the SAME depth x clade-size stratum (vectorized).

    Canonical pure-core copy (mirrors scripts/_anomaly_knn.matched_null, kept in sync by
    tests/eval/test_anomaly_knn + test_anomaly). For each depth x size bin draw an (m, n_null) index
    matrix and gather, so the only loop is over the <= (n_bins+1)^2 bins. Returns (Q, n_null) float64
    aligned to `pool_idx`.
    """
    rng = np.random.default_rng(seed)
    observed = np.asarray(observed, dtype=np.float64)
    d = np.asarray(depth)[np.asarray(pool_idx, dtype=np.int64)]
    csize = np.asarray(clade_size)[np.asarray(pool_idx, dtype=np.int64)]
    bins = _qbin(d, n_bins) * (n_bins + 1) + _qbin(csize, n_bins)
    Q = len(observed)
    null = np.empty((Q, n_null), dtype=np.float64)
    for b in np.unique(bins):
        members = np.flatnonzero(bins == b)
        m = len(members)
        draw = rng.integers(0, m, size=(m, n_null))
        null[members] = observed[members[draw]]
    return null


def _ancestor_at_depth(td, nodes, target_depth):
    """Lift each node up to `target_depth` via binary lifting (per-node target; clamped at the node)."""
    a = np.asarray(nodes, dtype=np.int64).copy()
    diff = np.maximum(td.depth[a] - np.asarray(target_depth, dtype=np.int64), 0)
    for k in range(td.maxlog):
        move = ((diff >> k) & 1).astype(bool)
        a = np.where(move, td.up[k][a], a)
    return a


def choose_displacement_donors(pool_idx, pool_lab, td, n, seed, local_frac=0.5, up_offset=2):
    """Pick n moved pool positions + a donor pool position each, spanning small->large displacement.

    Fixes defect 2 of the old leg A: the uniform relocator (`relocate_nodes`) only ever produced
    cross-tree jumps, so the sister-genus/family regime (the go/no-go's focus) went unsampled. Here a
    `local` donor shares the moved node's ancestor `up_offset` levels up (a sister/cousin clade -> small
    displacement); a `global` donor is a uniform random pool node of a different family (large
    displacement). Donors always carry a DIFFERENT label (a same-label relabel is a no-op). Displacement
    is returned as the TRUE cophenetic distance, so downstream binning is honest regardless of the
    sampling mix. Returns (moved_pos, donor_pos, disp) — parallel int arrays indexing pool_idx.
    """
    rng = np.random.default_rng(seed)
    pool_idx = np.asarray(pool_idx, dtype=np.int64)
    pool_lab = np.asarray(pool_lab)
    P = len(pool_idx)
    n = min(int(n), P)
    moved_pos = rng.choice(P, size=n, replace=False)

    target = np.maximum(td.depth[pool_idx] - int(up_offset), 0)
    anc = _ancestor_at_depth(td, pool_idx, target)                 # (P,) near-ancestor per pool node
    groups = {}
    for pos in range(P):
        groups.setdefault(int(anc[pos]), []).append(pos)
    groups = {a: np.asarray(v, dtype=np.int64) for a, v in groups.items()}

    want_local = rng.random(n) < float(local_frac)
    donor_pos = np.empty(n, dtype=np.int64)
    for i in range(n):
        mp = int(moved_pos[i])
        lab_g = pool_lab[mp]
        chosen = -1
        if want_local[i]:
            cand = groups.get(int(anc[mp]))
            if cand is not None:
                cand = cand[pool_lab[cand] != lab_g]
                if len(cand):
                    chosen = int(rng.choice(cand))
        if chosen < 0:                                            # wanted global, or local had no sister
            for _ in range(16):
                c = int(rng.integers(0, P))
                if c != mp and pool_lab[c] != lab_g:
                    chosen = c
                    break
            if chosen < 0:                                        # degenerate (monochrome pool) fallback
                alt = np.flatnonzero(pool_lab != lab_g)
                chosen = int(rng.choice(alt)) if len(alt) else int((mp + 1) % P)
        donor_pos[i] = chosen

    disp = td.path_length(pool_idx[moved_pos], pool_idx[donor_pos])
    return moved_pos, donor_pos, np.asarray(disp, dtype=np.int64)


def synthetic_displacement_roc(pool_idx, pool_lab, depth, clade_size, purity_fn, baselines_pool,
                               moved_pos, donor_pos, disp_class, n_null, n_bins, seed):
    """Leg A core (spec §9B), FIXED: recompute the anomaly score UNDER the relabel perturbation.

    The earlier driver scored the ORIGINAL labels and used the relocation only to set the positive
    mask, so moved nodes (chosen at random) were independent of the score -> AUC ~ 0.5 BY CONSTRUCTION
    (job 5675791: score_z 0.4998 at n=2948). Here each moved pool position is relabelled to its
    donor's label, observed purity is RECOMPUTED under those perturbed labels via `purity_fn`
    (injected: prod = Poincare GPU kNN, tests = brute force), turned into the matched-null z-score, and
    we report ROC-AUC of the score (and the trivial baselines) for moved-vs-rest, stratified by
    displacement class. A node relabelled into a far clade still sits in embedding space among its
    ORIGINAL family, so its neighbours stop matching its new label -> low purity -> high z. Displacement
    is measured on the true tree, so the move-sampling heuristic only affects which bins populate, never
    correctness.

    moved_pos / donor_pos / disp_class index into pool_idx (parallel arrays). baselines_pool maps
    name -> (P,) array already gathered onto pool_idx. Returns {auc_by_displacement,
    baseline_auc_by_displacement} matching the driver's roc_by_displacement.json shape.
    """
    pool_lab = np.asarray(pool_lab)
    moved_pos = np.asarray(moved_pos, dtype=np.int64)
    donor_pos = np.asarray(donor_pos, dtype=np.int64)
    disp_class = np.asarray(disp_class, dtype=np.int64)

    new_lab = pool_lab.copy()
    new_lab[moved_pos] = pool_lab[donor_pos]
    obs = np.asarray(purity_fn(new_lab), dtype=np.float64)
    null = matched_null(obs, pool_idx, depth, clade_size, n_null, n_bins, seed)
    score_z = matched_null_z(obs, null)

    P = len(np.asarray(pool_idx))
    is_moved = np.zeros(P, dtype=bool)
    is_moved[moved_pos] = True
    disp_arr = np.full(P, -1, dtype=np.int64)
    disp_arr[moved_pos] = disp_class
    scores_full = {"score_z": score_z,
                   **{n: np.asarray(v, dtype=np.float64) for n, v in baselines_pool.items()}}

    auc_by, base_by = {}, {}
    for dc in sorted(set(disp_class.tolist())):
        pos_mask = is_moved & (disp_arr == dc)
        keep = pos_mask | ~is_moved
        labels = pos_mask[keep].astype(np.int64)
        if labels.sum() < 1 or (labels == 0).sum() < 1:
            continue
        sc = {n: s[keep] for n, s in scores_full.items()}
        aucs = baseline_aucs(labels, sc)
        auc_by[str(dc)] = {"score_z": aucs["score_z"], "n_pos": int(labels.sum())}
        base_by[str(dc)] = {n: aucs[n] for n in baselines_pool}
    return {"auc_by_displacement": auc_by, "baseline_auc_by_displacement": base_by}
