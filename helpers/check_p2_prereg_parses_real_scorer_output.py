"""Replay the P2 pre-registration engine's PARSER against REAL scorer output.

Mirrors `helpers/check_prereg_parses_real_scorer_output.py` (tasks 8/9) for exactly the same
reason: the engine's own test suite runs on synthetic fixtures written by the same person who
wrote the engine -- "a verifier you wrote shares your blind spot". This script instead loads one
or more ACTUAL `scripts/score_p2_linkpred.py` output files (merging them first if more than one
is given, via `taxembed.eval.preregistration.merge_p2_scorer_outputs` -- production
(`scripts/p2_lrz_score.sh`) always writes 9 separate files, never one combined file) and runs
`p2_group_stats` / `p2_validity_gate` / `p2_verdict` against them, so a field that is named
differently, nested differently, or absent on real data fails HERE rather than tomorrow morning
with the array banked and a verdict owed.

This is the C4 fix (2026-09-24, `p2_amendment_3_20260924`): before it, the engine's `_p2_roll_key`
/`_p2_ms_key` lookup, the required `chance_mrr_mean`/`degree_prior` baseline keys, and the
`--baselines-key`-tagged control blocks had NO test running them against anything
`score_p2_linkpred.py` itself produced -- only hand-built fixtures that shared the engine's own
assumptions about the JSON shape.

It deliberately does NOT print a final verdict for a partial/smoke input: the pre-registration
admits only the complete design (3 seeds per arm, every arm's own baselines block present). This
script reports what it can parse and flags what is missing, same posture as its task 8/9 sibling.

  <python> helpers/check_p2_prereg_parses_real_scorer_output.py <scorer.json> [<scorer2.json> ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from taxembed.eval.preregistration import (  # noqa: E402
    P2_DATA_ARMS, merge_p2_scorer_outputs, p2_group_stats, p2_run_value, p2_validity_gate,
)

REQUIRED_BASELINE_FIELDS = ("sibling_chance_mean", "chance_mrr_mean", "chance_hits_at_1_mean",
                           "degree_prior", "vendrov_recall", "majority_parent_rate")
REQUIRED_DEGREE_PRIOR_FIELDS = ("mrr", "hits_at_1", "hits_at_10", "normalized_rank", "n")
REQUIRED_CKPT_FIELDS = ("epoch", "trainer", "metrics", "metrics_cosine", "metrics_poincare",
                        "by_depth")


def _check_baselines(name: str, block: dict) -> list[str]:
    missing = [f"{name}.{f}" for f in REQUIRED_BASELINE_FIELDS if f not in block]
    if "degree_prior" in block:
        missing += [f"{name}.degree_prior.{f}" for f in REQUIRED_DEGREE_PRIOR_FIELDS
                   if f not in block["degree_prior"]]
    return missing


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        raise SystemExit(__doc__)
    results = [json.loads(p.read_text()) for p in paths]
    print(f"files: {[str(p) for p in paths]}")

    result = merge_p2_scorer_outputs(results) if len(results) > 1 else results[0]
    if len(results) > 1:
        print(f"merged {len(results)} files -> {len(result['arms'])} arm+seed+kind entries")
        if "_baselines_disagreements" in result:
            print(f"  BASELINES DISAGREEMENTS across inputs: {result['_baselines_disagreements']}")

    print(f"arms present: {sorted(result['arms'])}")

    # Every REQUIRED baseline field the engine reads must exist on every baselines* block.
    missing: list[str] = []
    for key, block in result.items():
        if not key.startswith("baselines"):
            continue
        missing += _check_baselines(key, block)
    if missing:
        print("\nMISSING BASELINE FIELDS the engine reads:")
        for m in missing:
            print(f"  {m}")
        raise SystemExit(1)
    print(f"all required baseline fields present on every 'baselines*' block")

    # Every REQUIRED per-checkpoint field the engine reads must exist on every real checkpoint.
    ckpt_missing: list[str] = []
    for arm_seed_key, run in result["arms"].items():
        for c in run.get("checkpoints", []):
            for f in REQUIRED_CKPT_FIELDS:
                if f not in c:
                    ckpt_missing.append(f"{arm_seed_key} ep{c.get('epoch')}: {f}")
    if ckpt_missing:
        print("\nMISSING CHECKPOINT FIELDS the engine reads:")
        for m in ckpt_missing:
            print(f"  {m}")
        raise SystemExit(1)
    print(f"all {len(REQUIRED_CKPT_FIELDS)} engine-read checkpoint fields present")

    # by_depth keys, as real data actually uses them (the engine coerces stratum names to str).
    any_run = next(iter(result["arms"].values()), None)
    if any_run and any_run.get("checkpoints"):
        by_depth = any_run["checkpoints"][0].get("by_depth", {}).get("cosine", {})
        print(f"\nby_depth keys: {list(by_depth)}")

    # Parse whichever arm+seed pairs the engine can actually see (both _ms and _roll present).
    seeds = []
    arms_to_check = set(P2_DATA_ARMS) | {a for a in sys.argv[2:]}
    for arm in sorted(arms_to_check) or ["vis00", "vis50"]:
        for s in (0, 1, 2):
            if f"{arm}_s{s}_ms" in result["arms"] and f"{arm}_s{s}_roll" in result["arms"]:
                seeds.append((arm, s))
    print(f"\nparsing {len(seeds)} run(s) the engine can see (both _ms and _roll present): {seeds}")

    baselines = result.get("baselines")
    for arm, s in seeds:
        rv = p2_run_value(result, arm, s)
        print(f"  {arm}_s{s}: per_run_value {rv['per_run_value']:.4f} over {rv['n_roll']} roll "
              f"ckpt(s) {rv['roll_epochs']}, {rv['n_milestones']} milestone(s) | "
              f"milestone_mrr_max {rv['milestone_mrr_max']:.4f} | "
              f"final_epoch_present {rv['final_epoch_present']}")
        if baselines is not None:
            g = p2_validity_gate(result, arm, s, float(baselines["sibling_chance_mean"]),
                                 float(baselines["chance_mrr_mean"]))
            print(f"    gate_a {g['gate_a_pass']} (rise {g['gate_a_rise']:+.4f} vs "
                  f"{g['gate_a_threshold']:.4f}) | gate_b {g['gate_b_pass']} | "
                  f"gate_c {g['gate_c_pass']} | valid {g['valid']}")

    # A full p2_group_stats pass for any arm with all 3 seeds present -- exercises the exact call
    # path p2_verdict uses (including the REQUIRED chance_mrr_mean/degree_prior baseline keys).
    if baselines is not None:
        for arm in P2_DATA_ARMS:
            if all((arm, s) in seeds for s in (0, 1, 2)):
                gs = p2_group_stats(result, arm, (0, 1, 2), baselines)
                print(f"\n{arm} group stats: mean {gs['mean']:.4f}, min {gs['min']:.4f}, "
                      f"invalid_runs {gs['invalid_runs']}, "
                      f"degree_prior.mrr {gs['degree_prior']['mrr']:.4f}")

    print("\nPARSER OK on real scorer output. No verdict computed: the pre-registration admits "
          "only the complete design (3 seeds per arm, every baselines block present).")


if __name__ == "__main__":
    main()
