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
"""

from __future__ import annotations

import numpy as np


def _poincare_distance(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Poincare ball distance from one point u to each row of v."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    sq_u = np.sum(u * u)
    sq_v = np.sum(v * v, axis=-1)
    sq_d = np.sum((v - u) ** 2, axis=-1)
    denom = np.clip((1.0 - sq_u) * (1.0 - sq_v), 1e-12, None)
    return np.arccosh(1.0 + 2.0 * sq_d / denom)


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
                        tie_seed: int = 0) -> int:
    """1-based rank of `true_parent` among `candidates`, nearest first.

    Ties are broken by seeded jitter rather than by array order: index order correlates with
    taxonomic order in this data, so argsort ties would systematically favour low indices.
    """
    candidates = np.asarray(candidates, dtype=np.int64)
    if true_parent not in set(candidates.tolist()):
        raise ValueError(f"true parent {true_parent} absent from candidate pool")
    dist_fn = _poincare_distance if metric == "poincare" else _cosine_distance
    d = dist_fn(emb[node], emb[candidates])
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
