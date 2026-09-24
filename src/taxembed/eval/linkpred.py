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
                        tie_seed: int = 0, radii: np.ndarray | None = None) -> int:
    """1-based rank of `true_parent` among `candidates`, nearest first.

    Ties are broken by seeded jitter rather than by array order: index order correlates with
    taxonomic order in this data, so argsort ties would systematically favour low indices.

    `radii`, if given, is a per-node array of exact hyperbolic radii (e.g. `|z|` from the
    euclidean training parametrization -- see `scripts/score_recipe_checkpoints.py`). When
    supplied and `metric="poincare"`, ranking uses `_poincare_distance`'s exact radius-based
    path, which does not degrade near the ball boundary. The ball-coordinate fallback's clip
    count is computed but not propagated out of this function -- callers who need it should call
    `_poincare_distance` directly.
    """
    candidates = np.asarray(candidates, dtype=np.int64)
    if true_parent not in set(candidates.tolist()):
        raise ValueError(f"true parent {true_parent} absent from candidate pool")
    if metric == "poincare":
        r_u = None if radii is None else radii[node]
        r_v = None if radii is None else radii[candidates]
        d, _clip_count = _poincare_distance(emb[node], emb[candidates], r_u=r_u, r_v=r_v)
    else:
        d = _cosine_distance(emb[node], emb[candidates])
    rng = np.random.default_rng(tie_seed + int(node))
    d = d + rng.random(len(d)) * 1e-12
    order = np.argsort(d, kind="stable")
    position = int(np.flatnonzero(candidates[order] == true_parent)[0])
    return position + 1


def linkpred_metrics(ranks: np.ndarray, n_candidates: np.ndarray) -> dict:
    """MR, MRR, Hits@1, Hits@10 and a pool-size-normalized rank."""
    ranks = np.asarray(ranks, dtype=np.float64)
    n_candidates = np.asarray(n_candidates, dtype=np.float64)
    if len(ranks) == 0:
        return {"n": 0, "mean_rank": float("nan"), "mrr": float("nan"),
                "hits_at_1": float("nan"), "hits_at_10": float("nan"),
                "normalized_rank": float("nan")}
    # (rank - 1) / (pool - 1) is 0 for a perfect call and 1 for the worst possible one.
    denom = np.clip(n_candidates - 1.0, 1.0, None)
    return {
        "n": int(len(ranks)),
        "mean_rank": float(ranks.mean()),
        "mrr": float((1.0 / ranks).mean()),
        "hits_at_1": float((ranks <= 1).mean()),
        "hits_at_10": float((ranks <= 10).mean()),
        "normalized_rank": float(((ranks - 1.0) / denom).mean()),
    }


def stratify(ranks, n_candidates, key, bins) -> dict:
    """Metrics per inclusive [lo, hi] bin of `key` (e.g. candidate-pool size, node depth).

    `n_candidates` is the REAL per-query candidate-pool size, sliced alongside `ranks` and `key` --
    not synthesized from the bin bounds. Ruled correction 2026-09-24: see module docstring.
    """
    ranks = np.asarray(ranks)
    n_candidates = np.asarray(n_candidates)
    key = np.asarray(key)
    out = {}
    for lo, hi in bins:
        sel = (key >= lo) & (key <= hi)
        out[f"{lo}-{hi}"] = linkpred_metrics(ranks[sel], n_candidates[sel])
    return out
