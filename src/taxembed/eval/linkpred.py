"""Held-out parent prediction: candidate pools, filtered ranking, MR / MRR / Hits@k.

PROTOCOL. For each held-out LEAF node v whose parent edge was withheld, rank the candidate
parents by embedded distance to v and record the 1-based rank of the true parent p. Candidates
are the children of v's grandparent -- the set the retained closure still admits -- which makes
the task sibling disambiguation and gives a well-defined chance rate of 1/|candidates|
(taxembed.eval.baselines.sibling_chance).

WHY NOT "MAP". With exactly one positive per query, mean average precision is algebraically
identical to mean reciprocal rank. Spec §P2 asks for "MR/MAP"; reporting both would present one
number twice. We report MR, MRR, Hits@k and a pool-size-normalized rank, and say so in Methods.

CORRECTION (2026-09-24, ruled). The original draft's `stratify` synthesized each stratum's
candidate-pool size from the bin's upper bound (`hi`) rather than passing through the real
per-query pool size, which makes `normalized_rank` meaningless inside every stratum. `stratify`
here takes the real `n_candidates` array and slices it alongside `ranks` and `key`.

CORRECTION (2026-09-24, fix round 1). `_poincare_distance`'s ball-coordinate formula floors its
denominator `(1-|u|^2)(1-|v|^2)` at 1e-12 to avoid a NaN. For points legitimately near the ball
boundary -- which this project's embeddings are, by construction, since each node is initialized
at its depth's target radius -- the true denominator can be ~1e-14, so the floor silently
substitutes a value ~100x too large, UNDERSTATING distance there and risking a flipped ranking
between two near-boundary candidates. `_poincare_distance` now takes optional exact hyperbolic
radii (`r_u`, `r_v`) and, when both are given, uses the hyperbolic law of cosines instead, which
never touches `1-|x|^2`. This is the project's established approach: see
`scripts/score_recipe_checkpoints.py:load_checkpoint` (radius = |z| under the euclidean
parametrization `x = tanh(|z|/2) z/|z|`) and `taxembed.eval.angular.same_depth_neighbors`, which
already ranks on the same identity. When radii are not available the ball-coordinate formula
remains as a fallback, but the floor is no longer silent: `_poincare_distance` now returns
`(distances, clip_count)`, where `clip_count` is the number of pairs whose true (pre-floor)
denominator was below 1e-12, so a caller can report the fallback's known blind spot instead of it
passing unnoticed.

CORRECTION (2026-09-24, fix round 1, MINOR #4). `scripts/score_p2_linkpred.py` was calling
`rank_of_true_parent(metric="poincare")` (which computes `_poincare_distance` internally) and
then calling `_poincare_distance` a SECOND time on the same pair, purely to recover
`clip_count` -- doubling the Poincare cost per held node. `rank_of_true_parent` now takes an
optional `return_clip=False` parameter; when True it returns `(rank, clip_count)` from the one
computation it already does. Default behaviour (bare rank) is unchanged for existing callers.
"""

from __future__ import annotations

import numpy as np


def _poincare_distance(u: np.ndarray, v: np.ndarray, r_u=None, r_v=None) -> tuple[np.ndarray, int]:
    """Poincare ball distance from one point u to each row of v.

    Returns `(distances, clip_count)`.

    EXACT PATH -- `r_u` and `r_v` both given (per-node exact hyperbolic radii, e.g. `|z|` from
    the euclidean training parametrization; see `scripts/score_recipe_checkpoints.py`). Uses the
    hyperbolic law of cosines, which never touches `1 - |x|^2` and so never degrades near the
    ball boundary:
        cosh(d) = cosh(r_u) cosh(r_v) - sinh(r_u) sinh(r_v) cos(theta)
    where `theta` is the angle between `u` and `v` at the origin (preserved by the conformal
    Poincare map, so it can be read straight off the Euclidean directions). `clip_count` is
    always 0 on this path -- the floor below is never reached. `cos(theta)` is clamped to
    `[-1, 1]` and the final `cosh(d)` argument to `>= 1.0` so float drift cannot push `arccosh`
    outside its domain.

    BALL-COORDINATE FALLBACK -- `r_u` or `r_v` missing. The textbook formula
        d = arccosh(1 + 2|u-v|^2 / ((1-|u|^2)(1-|v|^2)))
    is used, with its denominator floored at 1e-12 to avoid a NaN from float underflow.
    `clip_count` counts how many of the returned pairs had their TRUE (pre-floor) denominator
    below that floor -- i.e. how many pairs the floor actually understated.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    if r_u is not None and r_v is not None:
        r_u = np.asarray(r_u, dtype=np.float64)
        r_v = np.asarray(r_v, dtype=np.float64)
        nu = np.linalg.norm(u)
        nv = np.linalg.norm(v, axis=-1)
        cos_t = np.clip((v @ u) / np.clip(nu * nv, 1e-12, None), -1.0, 1.0)
        cosh_d = np.cosh(r_u) * np.cosh(r_v) - np.sinh(r_u) * np.sinh(r_v) * cos_t
        cosh_d = np.clip(cosh_d, 1.0, None)
        return np.arccosh(cosh_d), 0

    sq_u = np.sum(u * u)
    sq_v = np.sum(v * v, axis=-1)
    sq_d = np.sum((v - u) ** 2, axis=-1)
    raw_denom = (1.0 - sq_u) * (1.0 - sq_v)
    clip_count = int(np.sum(raw_denom < 1e-12))
    denom = np.clip(raw_denom, 1e-12, None)
    cosh_d = np.clip(1.0 + 2.0 * sq_d / denom, 1.0, None)
    return np.arccosh(cosh_d), clip_count


def _cosine_distance(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v, axis=-1)
    return 1.0 - (v @ u) / np.clip(nu * nv, 1e-12, None)


def candidate_pool(parent: np.ndarray, depth: np.ndarray, node: int,
                   strategy: str = "grandparent_children") -> np.ndarray:
    """Admissible parents for `node` given that its own parent edge was withheld."""
    parent = np.asarray(parent, dtype=np.int64)
    depth = np.asarray(depth, dtype=np.int64)
    if strategy == "grandparent_children":
        grandparent = int(parent[int(parent[node])])
        has_parent = np.arange(len(parent), dtype=np.int64) != parent
        pool = np.flatnonzero(has_parent & (parent == grandparent))
    elif strategy == "same_depth":
        pool = np.flatnonzero(depth == depth[node] - 1)
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    return pool[pool != node].astype(np.int64)


def rank_of_true_parent(emb: np.ndarray, node: int, true_parent: int,
                        candidates: np.ndarray, metric: str = "poincare",
                        tie_seed: int = 0, radii: np.ndarray | None = None,
                        return_clip: bool = False):
    """1-based rank of `true_parent` among `candidates`, nearest first.

    Ties are broken by seeded jitter rather than by array order: index order correlates with
    taxonomic order in this data, so argsort ties would systematically favour low indices.

    `radii`, if given, is a per-node array of exact hyperbolic radii (e.g. `|z|` from the
    euclidean training parametrization -- see `scripts/score_recipe_checkpoints.py`). When
    supplied and `metric="poincare"`, ranking uses `_poincare_distance`'s exact radius-based
    path, which does not degrade near the ball boundary.

    `return_clip` (fix round 1, MINOR #4): when True, returns `(rank, clip_count)` instead of
    the bare rank, where `clip_count` is `_poincare_distance`'s own floor-hit count from the
    SAME distance computation this function already performs -- so a caller that wants both the
    rank and the clip count (e.g. `scripts/score_p2_linkpred.py`) does not need to call
    `_poincare_distance` a second time purely to recover it. `clip_count` is always 0 for
    `metric="cosine"`, since that path never touches `_poincare_distance` at all. Default False
    preserves the old bare-rank return for existing callers.
    """
    candidates = np.asarray(candidates, dtype=np.int64)
    if true_parent not in set(candidates.tolist()):
        raise ValueError(f"true parent {true_parent} absent from candidate pool")
    clip_count = 0
    if metric == "poincare":
        r_u = None if radii is None else radii[node]
        r_v = None if radii is None else radii[candidates]
        d, clip_count = _poincare_distance(emb[node], emb[candidates], r_u=r_u, r_v=r_v)
    else:
        d = _cosine_distance(emb[node], emb[candidates])
    rng = np.random.default_rng(tie_seed + int(node))
    d = d + rng.random(len(d)) * 1e-12
    order = np.argsort(d, kind="stable")
    position = int(np.flatnonzero(candidates[order] == true_parent)[0])
    rank = position + 1
    return (rank, clip_count) if return_clip else rank


def linkpred_metrics(ranks: np.ndarray, n_candidates: np.ndarray) -> dict:
    """MR, MRR, Hits@1, Hits@10 and a pool-size-normalized rank -- computed on the SCORED subset
    only (pool size >= 2).

    CORRECTION (2026-09-24, C3, amendment `p2_amendment_3_20260924`). At pool size k=1 there is
    exactly one candidate, which is therefore always the true parent: rank=1, and
    `normalized_rank = (rank-1)/max(k-1,1) = 0` for EVERY implementation, unconditionally -- the
    best possible value, handed out for free, never a draw. Measured on the real metazoa split
    (seed 0): 3.42% of held-out queries have pool size 1, vs 27.07% on the RandomDAG control (whose
    rewiring collapses fan-out) -- so a free win is handed out roughly 8x more often to the control
    than to the real tree, which is exactly backwards for a control that is supposed to be a fair
    floor. It also drags the real tree's own chance level for normalized_rank away from the clean
    0.5 the amendment's derivation otherwise gets by excluding k=1 (0.5 holds for EVERY k>=2).
    `n_trivial` (pool size 1, excluded) and `n_scored` (pool size >= 2, what every metric below is
    computed over) are now both reported, so a caller can always see how many queries were dropped
    and why -- never a silent shrink.
    """
    ranks = np.asarray(ranks, dtype=np.float64)
    n_candidates = np.asarray(n_candidates, dtype=np.float64)
    n_total = int(len(ranks))
    if n_total == 0:
        return {"n": 0, "n_scored": 0, "n_trivial": 0, "mean_rank": float("nan"),
                "mrr": float("nan"), "hits_at_1": float("nan"), "hits_at_10": float("nan"),
                "normalized_rank": float("nan")}
    trivial = n_candidates <= 1.0
    r = ranks[~trivial]
    nc = n_candidates[~trivial]
    n_trivial = int(trivial.sum())
    n_scored = n_total - n_trivial
    if n_scored == 0:
        return {"n": n_total, "n_scored": 0, "n_trivial": n_trivial, "mean_rank": float("nan"),
                "mrr": float("nan"), "hits_at_1": float("nan"), "hits_at_10": float("nan"),
                "normalized_rank": float("nan")}
    # (rank - 1) / (pool - 1) is 0 for a perfect call and 1 for the worst possible one.
    denom = np.clip(nc - 1.0, 1.0, None)
    return {
        "n": n_total, "n_scored": n_scored, "n_trivial": n_trivial,
        "mean_rank": float(r.mean()),
        "mrr": float((1.0 / r).mean()),
        "hits_at_1": float((r <= 1).mean()),
        "hits_at_10": float((r <= 10).mean()),
        "normalized_rank": float(((r - 1.0) / denom).mean()),
    }


def stratify(ranks, n_candidates, key, bins, assert_full_coverage: bool = False) -> dict:
    """Metrics per inclusive [lo, hi] bin of `key` (e.g. candidate-pool size, node depth).

    `n_candidates` is the REAL per-query candidate-pool size, sliced alongside `ranks` and `key` --
    not synthesized from the bin bounds. Ruled correction 2026-09-24: see module docstring.

    `assert_full_coverage` (2026-09-24, I2): when True, raises if any query with pool size >= 2
    (the SCORED set after the C3 k=1 exclusion -- `linkpred_metrics` above) falls in none of
    `bins`. `POOL_SIZE_BINS` starting at (2, 2) used to silently drop every k=1 query with no trace
    (measured 719 scored vs 740 total on a real run) -- now that k=1 is deliberately excluded
    EVERYWHERE (not just here), a scored query landing in no bin is a genuine gap in `bins`, not an
    expected exclusion, and must fail loudly rather than silently under-count a stratum.
    """
    ranks = np.asarray(ranks)
    n_candidates = np.asarray(n_candidates)
    key = np.asarray(key)
    out = {}
    covered = np.zeros(len(ranks), dtype=bool)
    for lo, hi in bins:
        sel = (key >= lo) & (key <= hi)
        covered |= sel
        out[f"{lo}-{hi}"] = linkpred_metrics(ranks[sel], n_candidates[sel])
    if assert_full_coverage:
        scored = np.asarray(n_candidates, dtype=np.float64) > 1.0
        gap = scored & ~covered
        if np.any(gap):
            n_gap = int(gap.sum())
            raise ValueError(
                f"{n_gap} scored quer{'y' if n_gap == 1 else 'ies'} (pool size >= 2) fall inside "
                f"no bin of {bins} -- bins do not partition the scored set")
    return out
