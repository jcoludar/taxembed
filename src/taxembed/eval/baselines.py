"""No-learning baselines reported beside every P2 link-prediction number (spec v3 §3.2).

Vendrov et al. ICLR 2016 §3.4 classify a pair positive iff it lies in the transitive closure of
the training+validation edges, scoring 88.2% on WordNet. Reporting it is mandatory because a
learned number is only interesting relative to what no learning achieves. On our tree-shaped split
it collapses to 0% recall -- see the module test -- which is itself a reportable property of the
task, not a bug in the baseline.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from taxembed.eval.linkpred import linkpred_metrics


def vendrov_closure_rule(visible_ancestor, visible_descendant, queries, candidates) -> np.ndarray:
    """(queries x candidates) bool: is candidate an ancestor of query in the VISIBLE graph?"""
    visible = set(zip(np.asarray(visible_ancestor).tolist(),
                      np.asarray(visible_descendant).tolist()))
    q = np.asarray(queries).tolist()
    c = np.asarray(candidates).tolist()
    return np.array([[(cand, node) in visible for cand in c] for node in q], dtype=bool)


def sibling_chance(parent: np.ndarray, held_out: np.ndarray) -> np.ndarray:
    """Per held-out node, 1 / (number of children its GRANDPARENT has).

    This is the chance rate of guessing the true parent uniformly among the candidates that the
    retained closure still admits -- the grandparent's children. It is the floor a learned score
    must clear to mean anything.
    """
    parent = np.asarray(parent, dtype=np.int64)
    held_out = np.asarray(held_out, dtype=np.int64)
    fanout = np.bincount(parent[np.arange(len(parent)) != parent], minlength=len(parent))
    grandparent = parent[parent[held_out]]
    n_candidates = np.maximum(fanout[grandparent], 1)
    return 1.0 / n_candidates


def majority_parent_rate(parent: np.ndarray, held_out: np.ndarray) -> float:
    """Accuracy of always answering the commonest true parent among the held-out nodes."""
    parent = np.asarray(parent, dtype=np.int64)
    held_out = np.asarray(held_out, dtype=np.int64)
    if len(held_out) == 0:
        return 0.0
    counts = Counter(parent[held_out].tolist())
    return counts.most_common(1)[0][1] / len(held_out)


def _fanout_of(parent: np.ndarray) -> np.ndarray:
    """Per-node child count (fan-out). O(n); callers scoring many held-out nodes against the SAME
    tree must compute this ONCE and pass it to `degree_prior_rank` via `fanout=`, never let it be
    recomputed per node -- see `degree_prior_metrics` below."""
    parent = np.asarray(parent, dtype=np.int64)
    return np.bincount(parent[np.arange(len(parent)) != parent], minlength=len(parent))


def degree_prior_rank(parent: np.ndarray, node: int, true_parent: int, candidates: np.ndarray,
                      tie_seed: int = 0, fanout: np.ndarray | None = None) -> int:
    """1-based rank of `true_parent` among `candidates` when candidates are ranked by DESCENDING
    child count (fan-out) -- the most-connected candidate guessed first.

    C1 (review finding, 2026-09-24, `p2_amendment_3_20260924`). Held-out leaves are sampled
    uniformly from band-eligible leaves, so a candidate parent with more children is more likely to
    BE the true parent, with no embedding and no training involved. Measured on the real metazoa
    split (seed 0, POST the C3 pool-size-1 exclusion this same baseline is itself subject to):
    this training-free ranker alone scores MRR 0.5600 / hits@1 0.4039 / normalized_rank 0.1538 --
    comfortably clearing every floor the frozen pre-registration checked (sibling chance,
    RandomDAG). A model that learned nothing would read GENERALISES under the frozen rule. This is
    now a first-class declared baseline every arm must beat (see `p2_verdict` / `_p2_arm_reading`
    in `taxembed.eval.preregistration`), not an oversight to ignore.

    `fanout`, if given, is the PRECOMPUTED `_fanout_of(parent)` array -- pass it when scoring many
    held-out nodes against the same tree (`degree_prior_metrics` always does) so the O(n) bincount
    runs ONCE instead of once per held-out node (a real metazoa run has ~28,000 of them; the
    per-call bincount cost otherwise dominates wall-clock for no reason). Recomputed from `parent`
    when omitted, for a standalone call.

    Ties are broken by the SAME seeded jitter `taxembed.eval.linkpred.rank_of_true_parent` uses
    (`np.random.default_rng(tie_seed + node)`, added at 1e-12 scale) so index order cannot leak
    into either ranker and the two are comparable apples-to-apples.
    """
    parent = np.asarray(parent, dtype=np.int64)
    candidates = np.asarray(candidates, dtype=np.int64)
    if true_parent not in set(candidates.tolist()):
        raise ValueError(f"true parent {true_parent} absent from candidate pool")
    fanout = _fanout_of(parent) if fanout is None else fanout
    # negate fan-out so an ascending sort (nearest-first, matching rank_of_true_parent's distance
    # convention) puts the HIGHEST fan-out candidate first.
    score = -fanout[candidates].astype(np.float64)
    rng = np.random.default_rng(tie_seed + int(node))
    score = score + rng.random(len(score)) * 1e-12
    order = np.argsort(score, kind="stable")
    position = int(np.flatnonzero(candidates[order] == true_parent)[0])
    return position + 1


def degree_prior_metrics(parent: np.ndarray, held_out: np.ndarray, candidates_list,
                         true_parents: np.ndarray, n_candidates: np.ndarray,
                         tie_seed: int = 0) -> dict:
    """`degree_prior_rank` for every held-out node, reduced to the same MRR/hits@1/normalized_rank
    shape `taxembed.eval.linkpred.linkpred_metrics` returns for a learned ranker, so the two are
    directly comparable. Depends only on tree structure (fan-out), never on any checkpoint -- one
    call per manifest/heldout pair, not per checkpoint (mirrors `sibling_chance`/`vendrov_recall`).

    Computes `_fanout_of(parent)` ONCE and passes it to every `degree_prior_rank` call: on the
    real metazoa split (~28,000 held-out nodes, ~498,000-node tree) recomputing the O(n) bincount
    per node instead cost ~47s of otherwise-avoidable wall-clock (measured,
    `helpers/p2_degree_prior_and_chance_verify.py`) for a quantity that does not change across
    calls.
    """
    return linkpred_metrics(
        degree_prior_ranks(parent, held_out, candidates_list, true_parents, tie_seed=tie_seed),
        n_candidates)


def degree_prior_ranks(parent: np.ndarray, held_out: np.ndarray, candidates_list,
                       true_parents: np.ndarray, tie_seed: int = 0) -> np.ndarray:
    """The training-free degree prior's per-query RANK vector, before any reduction.

    C2 (2026-09-26). `degree_prior_metrics` reduced straight to aggregate metrics, so the prior
    existed only as one number per tree -- and `p2_amendment_6_20260926`'s per-depth-stratum sign
    test consequently divided every stratum by that single AGGREGATE value. Measured on the real
    metazoa seed-0 splits, the per-stratum difficulty ratios are 2.83 / 4.09 / 4.18 against an
    aggregate of 3.81, so the aggregate OVER-ALLOWS by 1.344x in stratum 11-15 (4,751 scored
    queries, 9.5x P2_MIN_STRATUM_N) -- enough to read a genuine per-stratum TIE as a comfortable
    win, in the direction of GENERALISES.

    Exposing the ranks lets the scorer `stratify` them exactly as it stratifies a checkpoint's,
    so the sign test can compare like with like. Nothing is recomputed: `degree_prior_metrics`
    now calls this.
    """
    parent = np.asarray(parent, dtype=np.int64)
    held_out = np.asarray(held_out, dtype=np.int64)
    true_parents = np.asarray(true_parents, dtype=np.int64)
    fanout = _fanout_of(parent)
    return np.array(
        [degree_prior_rank(parent, int(v), int(tp), c, tie_seed=tie_seed, fanout=fanout)
         for v, tp, c in zip(held_out, true_parents, candidates_list)],
        dtype=np.int64)


def chance_mrr_mean(n_candidates: np.ndarray) -> float:
    """Analytic (no simulation) chance-level MRR for a uniformly random ranking: mean(H_k / k)
    over the per-query candidate-pool size k, where H_k is the k-th harmonic number.

    C2 (review finding, 2026-09-24). `sibling_chance_mean` (= mean(1/k)) is the chance rate for
    HITS@1, not for MRR: a uniformly random ranker's own MRR is `mean(H_k/k)`, which is ALWAYS
    larger than `mean(1/k)` for k>1 (measured on the real metazoa split: 0.27107 vs 0.13126, a
    2.07x gap). Gating an MRR-valued quantity against the hits@1 chance rate lets a run sitting
    exactly at chance MRR clear the floor. This is the correct anchor for that gate; see
    `taxembed.eval.preregistration.p2_validity_gate` gate (b) and `_p2_arm_reading`'s
    `margin_above_chance`.
    """
    n_candidates = np.asarray(n_candidates, dtype=np.int64)
    if len(n_candidates) == 0:
        return float("nan")
    max_k = int(n_candidates.max())
    # harmonic numbers 1..max_k via a cumulative sum of 1/i, indexed 0 at H_0 := 0.
    h = np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, max_k + 1, dtype=np.float64))])
    hk = h[n_candidates]
    k = n_candidates.astype(np.float64)
    return float(np.mean(hk / k))
