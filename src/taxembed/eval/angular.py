"""Radius-free learned-structure score: same-depth angular LCA (plan v2 Task 9).

Design: docs/specs/2026-09-22-task9-radius-free-scorer-design.md (+ its review addendum)

For a query q at depth d, rank the OTHER nodes at depth d by cosine of direction and score the
mean depth of LCA(q, neighbour) over the top k. Nothing reads a norm, so the planted radius
(norm = target_radius(depth) at init, held there by the radial regularizer) cannot reach it.
Same-depth pairs are never positive training pairs, so a high score means cousins' directions are
ordered by relatedness -- learned structure, not given structure.

Both ends of the scale are exact:
  null   mu0(q) = E[LCA depth] for a uniform pool member = sum_{j>=1} c_j / c_0
         (random directions = the initialization state, so this IS the untrained model)
  oracle mu*(q) = best mean LCA depth any k pool members can reach (greedy from the deepest)
where c_j = # pool members inside the subtree of q's depth-j ancestor.

S = (mean s - mean mu0) / (mean mu* - mean mu0): 0 = random directions, 1 = perfect cousin order.

Uncertainty is a CLUSTER bootstrap over clades (queries in one clade are not independent), and it
is query-sampling uncertainty for ONE trained run -- never run-to-run spread.
"""
from __future__ import annotations

import numpy as np

from taxembed.eval.subtree import euler_intervals

DEPTH_BANDS = {"shallow": (0, 10), "mid": (11, 27), "deep": (28, 10**9)}
CLUSTER_LEVEL = 3          # bootstrap clusters = depth-3 ancestor (the query itself if shallower)


class TreeIndex:
    """Precomputed tree structures shared by every checkpoint scored against one closure."""

    def __init__(self, parent, depth, seed: int = 0):
        self.parent = np.asarray(parent, dtype=np.int64)
        self.depth = np.asarray(depth, dtype=np.int64)
        n = len(self.parent)
        if len(self.depth) != n:
            raise ValueError("parent and depth differ in length")
        roots = np.flatnonzero(self.parent == np.arange(n))
        if len(roots) != 1:
            raise ValueError(f"expected a single rooted tree, found {len(roots)} roots")
        nonroot = self.parent != np.arange(n)
        if (self.depth[self.parent[nonroot]] != self.depth[nonroot] - 1).any():
            raise ValueError("depth is inconsistent with parent (depth[parent[v]] != depth[v]-1)")
        self.n_nodes = n
        self.max_depth = int(self.depth.max())
        self.tin, self.tout = euler_intervals(self.parent)

        # anc[v, j] = ancestor of v at depth j (j <= depth[v]); -1 beyond. anc[v, depth[v]] = v.
        anc = np.full((n, self.max_depth + 1), -1, dtype=np.int64)
        cur = np.arange(n)
        rows = np.arange(n)
        for _ in range(self.max_depth + 1):
            anc[rows, self.depth[cur]] = cur
            cur = self.parent[cur]
        self.anc = anc

        # Per depth: pool sorted by tin (subtrees are contiguous -> counts by searchsorted), and a
        # seeded RANDOM order for the neighbour search, so that ties can never be broken in tree
        # order (a tin-ordered pool would hand collapsed directions their relatives for free).
        rng = np.random.default_rng(seed)
        self.pool_tin: dict[int, np.ndarray] = {}
        self.pool_random: dict[int, np.ndarray] = {}
        self.pool_size = np.zeros(self.max_depth + 1, dtype=np.int64)
        for d in range(self.max_depth + 1):
            nodes = np.flatnonzero(self.depth == d)
            self.pool_tin[d] = np.sort(self.tin[nodes])
            self.pool_random[d] = rng.permutation(nodes)
            self.pool_size[d] = len(nodes)

    def ancestor_at(self, nodes, level: int) -> np.ndarray:
        """Ancestor at depth `level`, or the node itself when it is shallower than that."""
        nodes = np.asarray(nodes, dtype=np.int64)
        lv = np.minimum(level, self.depth[nodes])
        return self.anc[nodes, lv]


def pool_counts(idx: TreeIndex, q: int) -> np.ndarray:
    """c_j for j = 0..depth(q): same-depth nodes (excluding q) inside q's depth-j ancestor."""
    d = int(idx.depth[q])
    a = idx.anc[q, : d + 1]
    pt = idx.pool_tin[d]
    lo = np.searchsorted(pt, idx.tin[a], side="left")
    hi = np.searchsorted(pt, idx.tout[a], side="left")
    return (hi - lo - 1).astype(np.int64)          # -1: q itself sits inside every ancestor


def null_lca_depth(c: np.ndarray) -> float:
    """E[LCA depth] of q with a uniformly drawn same-depth node. Exactly c_j - c_{j+1} pool
    members have LCA depth j, and sum_j j (c_j - c_{j+1}) telescopes to sum_{j>=1} c_j."""
    return float(c[1:].sum() / c[0])


def oracle_lca_depth(c: np.ndarray, k: int) -> float:
    """Best mean LCA depth over any k pool members: take the closest relatives first."""
    remaining, total = k, 0
    for j in range(len(c) - 2, -1, -1):
        take = min(remaining, int(c[j] - c[j + 1]))
        total += take * j
        remaining -= take
        if remaining == 0:
            break
    if remaining:
        raise ValueError(f"pool has {int(c[0])} members, fewer than k={k}")
    return total / k


def lca_depths(idx: TreeIndex, q, v) -> np.ndarray:
    """Depth of LCA(q_i, v_ij). q: (Q,), v: (Q, K). Counts q's ancestors whose subtree holds v."""
    q = np.asarray(q, dtype=np.int64)
    v = np.asarray(v, dtype=np.int64)
    tv = idx.tin[v]
    count = np.zeros(v.shape, dtype=np.int64)
    for j in range(idx.max_depth + 1):
        a = idx.anc[q, j]
        ok = a >= 0
        if not ok.any():
            break
        a_safe = np.where(ok, a, 0)
        inside = (idx.tin[a_safe][:, None] <= tv) & (tv < idx.tout[a_safe][:, None])
        count += inside & ok[:, None]
    return count - 1


def query_bounds(idx: TreeIndex, queries, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-query (mu0, mu*_k); mu*_k is NaN where the pool holds fewer than k members.
    Embedding-independent: compute once per query set."""
    mu0 = np.empty(len(queries))
    mus = np.full(len(queries), np.nan)
    for i, q in enumerate(queries):
        c = pool_counts(idx, int(q))
        mu0[i] = null_lca_depth(c)
        if c[0] >= k:
            mus[i] = oracle_lca_depth(c, k)
    return mu0, mus


def select_queries(idx: TreeIndex, n: int, k: int, seed: int) -> np.ndarray:
    """Fixed-seed uniform sample of rankable queries: pool >= k and mu* > mu0.

    Returned sorted, so the same (n, k, seed) gives the same array on any machine.
    """
    candidates = np.flatnonzero(idx.pool_size[idx.depth] - 1 >= k)
    order = np.random.default_rng(seed).permutation(candidates)
    keep = []
    for q in order:
        c = pool_counts(idx, int(q))
        if oracle_lca_depth(c, k) > null_lca_depth(c) + 1e-12:
            keep.append(int(q))
            if len(keep) == n:
                break
    return np.sort(np.asarray(keep, dtype=np.int64))


def choose_clade_level(idx: TreeIndex, queries, min_n: int = 500, min_groups: int = 3):
    """Shallowest ancestor depth at which >= min_groups clades each hold >= min_n queries.
    Depends only on the tree and the query set, so it is fixed before any embedding is seen."""
    queries = np.asarray(queries, dtype=np.int64)
    for level in range(1, idx.max_depth + 1):
        m = idx.depth[queries] >= level
        if not m.any():
            break
        _, counts = np.unique(idx.anc[queries[m], level], return_counts=True)
        if int((counts >= min_n).sum()) >= min_groups:
            return level
    return None


def same_depth_neighbors(emb, idx: TreeIndex, queries, k: int, metric: str = "cosine",
                         radii=None, chunk: int = 1024, tie_seed: int = 0) -> np.ndarray:
    """Top-k same-depth neighbours of each query, best first (self excluded). (Q, k) node ids.

    metric="cosine"   -- directions only; the primary, radius-free ranking.
    metric="poincare" -- hyperbolic distance, via the hyperbolic law of cosines on the hyperbolic
                         radius r (= |z| under the euclidean parametrization, else 2 artanh|x|):
                         cosh d = cosh r_q cosh r_v - sinh r_q sinh r_v cos t. Dividing by
                         cosh r_q (constant per row) gives the ranking key
                         cosh r_v - tanh r_q sinh r_v cos t, stable near the boundary.

    Exact ties are broken at random PER QUERY (a seeded 1e-12 jitter, far below any real score
    difference). Without it, a collapsed embedding hands every query at a depth the SAME k
    neighbours; that correlates the queries and makes the bootstrap SE too small
    (caught by test_collapsed_directions_do_not_score_through_tie_breaking).
    """
    if metric not in ("cosine", "poincare"):
        raise ValueError(f"unknown metric {metric!r}")
    rng = np.random.default_rng(tie_seed)
    emb = np.asarray(emb, dtype=np.float64)
    queries = np.asarray(queries, dtype=np.int64)
    out = np.empty((len(queries), k), dtype=np.int64)

    norms = np.linalg.norm(emb, axis=1)
    if (norms == 0).any():
        raise ValueError(f"{int((norms == 0).sum())} zero-norm embeddings have no direction")
    unit = emb / norms[:, None]
    if metric == "poincare":
        r = (np.asarray(radii, dtype=np.float64) if radii is not None
             else 2.0 * np.arctanh(np.minimum(norms, 1.0 - 1e-12)))
        cosh_r, sinh_r, tanh_r = np.cosh(r), np.sinh(r), np.tanh(r)

    for d in np.unique(idx.depth[queries]):
        rows = np.flatnonzero(idx.depth[queries] == d)
        pool = idx.pool_random[int(d)]
        pos = {int(v): i for i, v in enumerate(pool)}
        for start in range(0, len(rows), chunk):
            r_ = rows[start:start + chunk]
            qs = queries[r_]
            cos = unit[qs] @ unit[pool].T
            if metric == "cosine":
                score = cos
            else:
                score = -(cosh_r[pool][None, :] - tanh_r[qs][:, None] * sinh_r[pool][None, :] * cos)
            score += rng.uniform(0.0, 1e-12, size=score.shape)
            score[np.arange(len(qs)), [pos[int(x)] for x in qs]] = -np.inf
            top = np.argpartition(-score, k - 1, axis=1)[:, :k]
            order = np.argsort(-np.take_along_axis(score, top, axis=1), axis=1)
            out[r_] = pool[np.take_along_axis(top, order, axis=1)]
    return out


def _S(s, mu0, mus):
    m = ~np.isnan(mus)
    den = mus[m].mean() - mu0[m].mean() if m.any() else 0.0
    return float((s[m].mean() - mu0[m].mean()) / den) if den > 0 else float("nan")


def _cluster_boot(values: list[np.ndarray], mu0, mus, clusters, n_boot, seed):
    """Cluster bootstrap of S for each array in `values` using the SAME resampled clusters.
    S is a ratio of sums, so per-cluster sums make each draw O(#clusters)."""
    m = ~np.isnan(mus)
    uniq, inv = np.unique(clusters[m], return_inverse=True)
    G = len(uniq)
    sum0 = np.bincount(inv, mu0[m], G)
    summ = np.bincount(inv, mus[m], G)
    sums = [np.bincount(inv, v[m], G) for v in values]
    draws = np.random.default_rng(seed).integers(0, G, size=(n_boot, G))
    den = summ[draws].sum(1) - sum0[draws].sum(1)
    return [(sv[draws].sum(1) - sum0[draws].sum(1)) / den for sv in sums], G


def score_embedding(emb, idx: TreeIndex, queries, k: int, metric: str = "cosine",
                    bounds: tuple[np.ndarray, np.ndarray] | None = None, radii=None,
                    extra_ks: tuple[int, ...] = (), clade_level: int | None = None,
                    min_stratum_n: int = 500, n_boot: int = 1000, seed: int = 0) -> dict:
    """Score one embedding at primary k (plus extra_ks). Keeps per-query s for paired use."""
    queries = np.asarray(queries, dtype=np.int64)
    mu0, mus = bounds if bounds is not None else query_bounds(idx, queries, k)
    nb = same_depth_neighbors(emb, idx, queries, k, metric=metric, radii=radii)
    s = lca_depths(idx, queries, nb).mean(axis=1)

    clusters = idx.ancestor_at(queries, CLUSTER_LEVEL)
    (S_b,), n_clusters = _cluster_boot([s], mu0, mus, clusters, n_boot, seed)

    qdepth = idx.depth[queries]
    by_band = {}
    for name, (lo, hi) in DEPTH_BANDS.items():
        m = (qdepth >= lo) & (qdepth <= hi)
        by_band[name] = {"n": int(m.sum()), "S": _S(s[m], mu0[m], mus[m]) if m.any() else None}

    by_clade = {}
    if clade_level is not None:
        m_lv = qdepth >= clade_level
        groups = np.full(len(queries), -1)
        groups[m_lv] = idx.anc[queries[m_lv], clade_level]
        for g in np.unique(groups[m_lv]):
            m = groups == g
            if m.sum() >= min_stratum_n:
                by_clade[int(g)] = {"n": int(m.sum()), "S": _S(s[m], mu0[m], mus[m])}

    extra = {}
    for kk in extra_ks:                 # each k on the queries whose pool can supply k members
        if kk == k:
            continue
        m = idx.pool_size[qdepth] - 1 >= kk
        if not m.any():
            extra[kk] = {"n": 0, "S": None}
            continue
        q_k = queries[m]
        nb_k = same_depth_neighbors(emb, idx, q_k, kk, metric=metric, radii=radii)
        mu0_k, mus_k = query_bounds(idx, q_k, kk)
        extra[kk] = {"n": int(m.sum()), "S": _S(lca_depths(idx, q_k, nb_k).mean(axis=1), mu0_k, mus_k)}

    hits = np.bincount(nb[:, :k].ravel(), minlength=idx.n_nodes)
    return {
        "metric": metric,
        "k": k,
        "n_queries": int(len(queries)),
        "S": _S(s, mu0, mus),
        "S_cluster_se": float(np.nanstd(S_b, ddof=1)),
        "S_cluster_ci95": [float(np.nanpercentile(S_b, 2.5)), float(np.nanpercentile(S_b, 97.5))],
        "n_clusters": int(n_clusters),
        "mean_s": float(s.mean()),
        "mean_mu0": float(mu0.mean()),
        "mean_mustar": float(np.nanmean(mus)),
        "by_band": by_band,
        "by_clade": by_clade,
        "S_at_k": extra,
        "hubness": {"n_distinct_neighbours": int((hits > 0).sum()),
                    "modal_share": float(hits.max() / hits.sum())},
        "s": s,
        # back-compat key used by the unit tests (cluster SE is the reported uncertainty)
        "S_bootstrap_se": float(np.nanstd(S_b, ddof=1)),
    }


def paired_S_difference(s_a, s_b, mu0, mus, clusters, n_boot: int = 1000, seed: int = 0) -> dict:
    """S_a - S_b with the SAME resampled clusters on both sides. Query-sampling uncertainty for
    one pair of runs only; it says nothing about run-to-run spread, which needs seeds."""
    s_a, s_b = np.asarray(s_a), np.asarray(s_b)
    (Sa_b, Sb_b), _ = _cluster_boot([s_a, s_b], mu0, mus, np.asarray(clusters), n_boot, seed)
    diff_b = Sa_b - Sb_b
    return {
        "diff": _S(s_a, mu0, mus) - _S(s_b, mu0, mus),
        "ci95": [float(np.nanpercentile(diff_b, 2.5)), float(np.nanpercentile(diff_b, 97.5))],
        "se": float(np.nanstd(diff_b, ddof=1)),
    }
