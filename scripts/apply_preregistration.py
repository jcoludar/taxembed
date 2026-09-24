"""Apply a frozen pre-registration to a scorer output and print the verdict.

  <python> scripts/apply_preregistration.py --task 9 --json <seeds_*.json> --out <verdict.json>
  <python> scripts/apply_preregistration.py --task 8 --json <task8_seeds_*.json> --out <verdict.json>
  <python> scripts/apply_preregistration.py --task p2 --json <p2_scores_*.json> --out <verdict.json>

The decision logic lives in src/taxembed/eval/preregistration.py and was written and tested
BEFORE any array result was visible (tests/eval/test_preregistration.py). This script only routes
a file into it and renders the result, so that reading the verdict cannot become an occasion for
re-deciding what the verdict rule was.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from taxembed.eval.preregistration import (  # noqa: E402
    p2_verdict, task8_verdict, task9_verdict,
)


def _render(v: dict) -> None:
    cmp = v["comparison"]
    a, b = cmp["arm_a"], cmp["arm_b"]
    print(f"\n=== {v['task']} ===")
    print(f"pre-registration: {'; '.join(v['preregistration'])}")

    print(f"\nvalidity gates (either failing => UNINFORMATIVE, never a pass, never a fail)")
    for side in ("a", "b"):
        for g in cmp["gates"][side]:
            mark = "PASS" if g["valid"] else "FAIL"
            print(f"  [{mark}] {g['arm']}_s{g['seed']}  "
                  f"rise {g['gate_a_rise']:+.4f} vs {g['gate_a_threshold']:.4f} "
                  f"({'ok' if g['gate_a_pass'] else 'FAILED'})  |  "
                  f"final-phase loss drop {g['gate_b_loss_drop']:+.4f} vs jitter "
                  f"{g['gate_b_loss_jitter']:.4f} ({'ok' if g['gate_b_pass'] else 'FAILED'})")

    s = cmp["S_angle"]
    print(f"\nS_angle (per run = mean over rolling epochs 196-200)")
    print(f"  {a:>10}: {['%.4f' % x for x in s['a']]}  mean {s['a_mean']:.4f}")
    print(f"  {b:>10}: {['%.4f' % x for x in s['b']]}  mean {s['b_mean']:.4f}")
    print(f"  difference {a} - {b} = {s['mean_diff_a_minus_b']:+.4f}   "
          f"equivalence band {cmp['equivalence']['band']:.4f} "
          f"(pooled within-arm seed SD {cmp['equivalence']['pooled_within_arm_seed_sd']:.5f})")
    print(f"  all {a} above all {b}: {s['all_a_above_all_b']}   "
          f"all {b} above all {a}: {s['all_b_above_all_a']}   "
          f"ranges overlap: {s['ranges_overlap']}")

    la = cmp["level_auc"]
    print(f"\nlevel AUC (amendment 1: must agree in sign with S_angle, else MIXED)")
    print(f"  {a:>10}: {['%.4f' % x for x in la['a']]}")
    print(f"  {b:>10}: {['%.4f' % x for x in la['b']]}")
    print(f"  all {a} above all {b}: {la['all_a_above_all_b']}   "
          f"all {b} above all {a}: {la['all_b_above_all_a']}")

    print(f"\nS_poincare arm means: {a} {cmp['S_poincare']['a_mean']:.4f}, "
          f"{b} {cmp['S_poincare']['b_mean']:.4f}")

    print(f"\nstrata (sign of {a} - {b} must be consistent across ALL of them)")
    for field, d in cmp["strata"].items():
        print(f"  {field}: {d['n_strata']} strata, "
              f"{d['n_positive']} positive / {d['n_negative']} negative")
        if d["dropped_below_min_n"]:
            print(f"    dropped below n>=500: {d['dropped_below_min_n']}")
        for key, diff in sorted(d["diffs"].items(), key=lambda kv: kv[1]):
            print(f"    {key:>12}: {diff:+.4f}")

    if cmp["flags_poincare_above_angle"]:
        print(f"\nFLAGGED (S_poincare > S_angle -- structure carried by radius that S_angle "
              f"cannot see): {', '.join(cmp['flags_poincare_above_angle'])}")

    print(f"\n>>> VERDICT: {v['verdict']}")
    print(f"    {v['meaning']}")
    if "attribution" in v:
        print(f"    {v['attribution']}")


def _render_p2(v: dict) -> None:
    print(f"\n=== {v['task']} ===")
    print(f"pre-registration: {'; '.join(v['preregistration'])}")
    randomdag_chance = v["randomdag_sibling_chance_mean"]
    if isinstance(randomdag_chance, dict):
        chance_str = ", ".join(f"{name} {val:.4f}" for name, val in randomdag_chance.items())
    else:
        chance_str = f"{randomdag_chance:.4f}"
    print(f"\nsibling_chance_mean (real tree) {v['sibling_chance_mean']:.4f}   "
          f"(RandomDAG) {chance_str}")

    # amendment_2 (2026-09-24): "controls" (plural, one per matched RandomDAG arm) replaces the
    # single "control" key -- fall back to the old shape so this renders either output unchanged.
    controls = v.get("controls", {"randomdag": v.get("control")})
    print(f"\nvalidity gates (a) learning (b) floor (c) completion -- "
          f"any failing => UNINFORMATIVE, no direction read")
    for group_name, g in {**v["groups"], **controls}.items():
        for gate in g["gates"]:
            mark = "PASS" if gate["valid"] else "FAIL"
            print(f"  [{mark}] {gate['arm']}  per_run_value {gate['per_run_value']:.4f}  "
                  f"rise {gate['gate_a_rise']:+.4f} vs {gate['gate_a_threshold']:.4f} "
                  f"({'ok' if gate['gate_a_pass'] else 'FAILED'})  |  "
                  f"floor ({'ok' if gate['gate_b_pass'] else 'FAILED'})  |  "
                  f"epoch200 scored ({'ok' if gate['gate_c_pass'] else 'FAILED'})")

    if v["verdict"] != "UNINFORMATIVE":
        for name, r in v["by_arm_verdict"].items():
            print(f"\n{name}: {r['verdict']}  (cross-tree metric: {r['cross_tree_metric']})")
            print(f"  margin above OWN chance {r['margin_above_chance']:+.4f} "
                  f"vs threshold {r['margin_threshold']:.4f} "
                  f"({'ok' if r['above_chance_margin'] else 'FAILED'})")
            if r["amendment_1_applied"]:
                print(f"  below every RandomDAG seed (normalized_rank): "
                      f"{r['below_control_all_seeds']}   "
                      f"below chance level 0.5: {r['below_chance_level']}   "
                      f"equivalent to RandomDAG: {r['equivalent_to_control']}")
            else:
                print(f"  above every RandomDAG seed: {r['above_control_all_seeds']}   "
                      f"equivalent to RandomDAG: {r['equivalent_to_control']}")
            d = r["depth_strata"]
            print(f"  depth strata ({name} - randomdag, on {r['cross_tree_metric']}): "
                  f"{d['n_strata']} strata, sign consistent {r['sign_consistent']}")
            if d["dropped_below_min_n"]:
                print(f"    dropped below n>=500: {d['dropped_below_min_n']}")

    print(f"\n>>> VERDICT: {v['verdict']}")
    print(f"    {v['meaning']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="apply a frozen pre-registration to a scorer output")
    ap.add_argument("--task", choices=["8", "9", "p2"], required=True)
    ap.add_argument("--json", required=True, help="scorer output (score_recipe_checkpoints.py "
                    "for tasks 8/9, score_p2_linkpred.py for p2)")
    ap.add_argument("--out", help="write the verdict JSON here")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--amendment-2", action="store_true",
                    help="tasks 8/9: apply preregistration_v2_amendment_2_20260924 (gate b: loss "
                         "must not RISE). --task p2: apply p2_amendment_2_20260924 instead -- the "
                         "12-arm matched-control design (vis00 vs randomdag_vis00, vis50 vs "
                         "randomdag_vis50, never crossed; USER DESIGN decision, not an "
                         "outcome-driven revision). Same flag, task-specific meaning, mirroring "
                         "how --amendment-1 is p2-only.")
    ap.add_argument("--amendment-1", action="store_true",
                    help="--task p2 only: apply p2_amendment_1_20260924 (RandomDAG's rewiring "
                         "measurably inflates its own chance floor ~5.4x, so the cross-tree "
                         "comparison uses normalized_rank, chance=0.5 for every pool size, "
                         "instead of raw MRR). No-op for --task 8/9.")
    args = ap.parse_args()

    result = json.loads(Path(args.json).read_text())
    seeds = tuple(int(x) for x in args.seeds.split(","))

    if args.task == "p2":
        verdict = p2_verdict(result, seeds, amendment_1=args.amendment_1,
                             amendment_2=args.amendment_2)
        render = _render_p2
    else:
        verdict = (task9_verdict if args.task == "9" else task8_verdict)(
            result, seeds, args.amendment_2)
        verdict["amendment_2_applied"] = bool(args.amendment_2)
        render = _render
    verdict["scored_from"] = str(Path(args.json).resolve())
    render(verdict)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(verdict, indent=2))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
