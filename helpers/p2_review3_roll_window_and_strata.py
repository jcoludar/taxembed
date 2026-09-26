#!/usr/bin/env python3
"""REVIEW WAVE 3 (read-only): two remaining questions about the PUBLISHED per-run number.

(1) `P2_ROLL_WINDOW = 5` is declared in `preregistration.py:273` and described in
    `p2_run_value`'s own docstring as "the mean MRR over the trailing `P2_ROLL_WINDOW` ROLLING
    checkpoints". grep says it is referenced NOWHERE in production code -- only in the docstring
    and in the TEST fixture (`_p2_fanout_arms` slices `ckpts[-P2_ROLL_WINDOW:]`). So the FIXTURE
    guarantees exactly 5 roll checkpoints and production takes whatever the glob
    `${tag}_epoch*.pth` returns. The trainer's rolling queue keeps the last 5 per PROCESS -- but
    `scripts/p2_lrz_train.sh` writes into `/app/artifacts/tags/${TAG}/` without clearing it, so a
    re-run array element leaves the previous attempt's `_epoch*.pth` files behind, and
    `scripts/p2_lrz_score.sh` pre-flights only that `${tag}_epoch200.pth` EXISTS, never how many
    there are. This measures what that does to the published per_run_value and to gate (a).

(2) Will the depth-stratum sign rule have any strata to bind on? `_p2_depth_means` drops a stratum
    whose mean `n` is below `P2_MIN_STRATUM_N = 500`, and `sign_consistent = bool(diffs) and ...`
    is FALSE when every stratum is dropped -- which silently makes GENERALISES unreachable. Counts
    the real held-out nodes per DEPTH_BINS and per POOL_SIZE_BINS on the production splits.

Writes nothing except stdout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))

from taxembed.eval.linkpred import candidate_pool  # noqa: E402
from taxembed.eval.p2_split import depth_from_closure  # noqa: E402
from taxembed.eval.preregistration import (  # noqa: E402
    P2_MIN_STRATUM_N, P2_ROLL_WINDOW, p2_run_value, p2_validity_gate,
)
from taxembed.eval.subtree import parent_from_closure  # noqa: E402
from taxembed.utils.training_pairs import TrainingPairs  # noqa: E402

SPLITS = _REPO / "data" / "p2_splits"
REAL_CLOSURE = (_REPO / "data" / "taxopy" / "metazoa_33208_clean"
                / "taxonomy_edges_metazoa_33208_clean_transitive.npz")
DEPTH_BINS = [(11, 15), (16, 21), (22, 28)]
POOL_SIZE_BINS = [(2, 2), (3, 5), (6, 20), (21, 10**6)]


def ck(epoch, mrr):
    m = {"n": 500, "n_scored": 500, "n_trivial": 0, "mean_rank": 3.0, "mrr": mrr,
         "hits_at_1": 0.7, "hits_at_10": 0.95, "normalized_rank": 1.0 - mrr}
    return {"epoch": epoch, "trainer": {"loss": 4.0}, "metrics": m, "metrics_cosine": m,
            "metrics_poincare": m, "by_depth": {"cosine": {}, "poincare": {}}}


def part1() -> None:
    print("=" * 78)
    print("(1) the _roll glob is unbounded and P2_ROLL_WINDOW is never applied")
    print("=" * 78)
    rng = np.random.default_rng(0)
    ms = [ck(e, 0.82 * min(1.0, e / 100.0)) for e in range(10, 201, 10)]

    clean_roll = [ck(e, 0.82 + rng.normal(0, 4e-4)) for e in (196, 197, 198, 199, 200)]
    # a re-run array element: the FIRST attempt died at epoch 120, leaving its last 5 behind
    stale_roll = [ck(e, 0.55 + rng.normal(0, 4e-4)) for e in (116, 117, 118, 119, 120)]

    for label, roll in (("clean: 5 roll checkpoints (196-200)", clean_roll),
                        ("re-run: 10 roll checkpoints (116-120 stale + 196-200)",
                         stale_roll + clean_roll)):
        res = {"arms": {"vis00_s0_ms": {"checkpoints": ms},
                        "vis00_s0_roll": {"checkpoints": roll}}}
        rv = p2_run_value(res, "vis00", 0)
        g = p2_validity_gate(res, "vis00", 0, 0.13126, 0.27107)
        print(f"\n  {label}")
        print(f"    n_roll               {rv['n_roll']}   (P2_ROLL_WINDOW = {P2_ROLL_WINDOW})")
        print(f"    roll_epochs          {rv['roll_epochs']}")
        print(f"    PUBLISHED per_run_value {rv['per_run_value']:.5f}")
        print(f"    jitter_sd            {rv['jitter_sd']:.6f}")
        print(f"    gate_a {g['gate_a_pass']}  gate_b {g['gate_b_pass']}  gate_c {g['gate_c_pass']}"
              f"  VALID {g['valid']}")
    print("\n  => the two per_run_values differ; the second is a mean over a mid-training value, "
          "\n     and its inflated jitter fails gate (a) => UNINFORMATIVE for the whole array.")


def part2() -> None:
    print("\n" + "=" * 78)
    print("(2) will the depth-stratum sign rule have strata to bind on? (min n = "
          f"{P2_MIN_STRATUM_N})")
    print("=" * 78)
    pairs = TrainingPairs.load(REAL_CLOSURE)
    n = pairs.n_nodes
    parent = parent_from_closure(pairs.ancestor_idx, pairs.descendant_idx, pairs.depth_diff, n)
    depth = depth_from_closure(pairs.descendant_idx, pairs.descendant_depth,
                               pairs.ancestor_idx, pairs.ancestor_depth, n)
    for seed in (0, 1, 2):
        held = np.asarray(np.load(SPLITS / f"p2_metazoa_33208_clean_seed{seed}_heldout.npz")["test"],
                          dtype=np.int64)
        dh = depth[held]
        n_cand = np.array([len(candidate_pool(parent, depth, node=int(v))) for v in held],
                          dtype=np.int64)
        depth_counts = {f"{lo}-{hi}": int(((dh >= lo) & (dh <= hi)).sum()) for lo, hi in DEPTH_BINS}
        # n_scored per depth stratum -- what the metric is actually computed over
        depth_scored = {f"{lo}-{hi}": int(((dh >= lo) & (dh <= hi) & (n_cand >= 2)).sum())
                        for lo, hi in DEPTH_BINS}
        pool_counts = {f"{lo}-{hi}": int(((n_cand >= lo) & (n_cand <= hi)).sum())
                       for lo, hi in POOL_SIZE_BINS}
        covered = sum(pool_counts.values())
        print(f"\n  seed {seed}: n_held {len(held)}")
        print(f"    by_depth n (total)   {json.dumps(depth_counts)}")
        print(f"    by_depth n_scored    {json.dumps(depth_scored)}")
        print(f"    strata kept at n>={P2_MIN_STRATUM_N}: "
              f"{[k for k, v in depth_counts.items() if v >= P2_MIN_STRATUM_N]}")
        print(f"    by_pool_size n       {json.dumps(pool_counts)}")
        print(f"    pool bins cover {covered} of {int((n_cand >= 2).sum())} scored "
              f"(assert_full_coverage would raise: {covered != int((n_cand >= 2).sum())})")
    print("\n  NOTE `_p2_depth_means` thresholds on `n` (TOTAL, including the k=1 queries every "
          "metric\n  excludes), not on `n_scored`.")


def main() -> int:
    part1()
    part2()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
