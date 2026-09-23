"""Replay the pre-registration engine's PARSER against REAL scorer output.

The engine's 20 tests run on synthetic fixtures I wrote, which share my assumptions about the
scorer's JSON shape -- a verifier you wrote shares your blind spot. This runs run_value() and
validity_gate() against an actual score_recipe_checkpoints.py file, so a field that is named
differently, nested differently, or absent on real data fails HERE rather than tomorrow morning
with both arrays banked and a verdict owed.

It deliberately does NOT produce a verdict: the smoke covers one arm, and the pre-registration
admits only the complete 3v3.

  <python> helpers/check_prereg_parses_real_scorer_output.py <scorer.json> [arm ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from taxembed.eval.preregistration import run_value, validity_gate  # noqa: E402

REQUIRED_CKPT_FIELDS = ("epoch", "S_angle", "S_poincare", "level_auc_mean",
                        "S_angle_by_band", "S_angle_by_clade", "trainer")


def main() -> None:
    path = Path(sys.argv[1])
    result = json.loads(path.read_text())
    print(f"file: {path}")
    print(f"runs present: {sorted(result['runs'])}")
    print(f"init_null S_angle: {result['init_null']['S_angle']:+.6f}")

    # Every field the engine reads must exist on every real checkpoint.
    missing = []
    for arm, run in result["runs"].items():
        for c in run["checkpoints"]:
            for f in REQUIRED_CKPT_FIELDS:
                if f not in c:
                    missing.append(f"{arm} ep{c.get('epoch')}: {f}")
    if missing:
        print("\nMISSING FIELDS the engine reads:")
        for m in missing:
            print(f"  {m}")
        raise SystemExit(1)
    print(f"all {len(REQUIRED_CKPT_FIELDS)} engine-read fields present on every checkpoint")

    # Stratum keys: the engine coerces to str, so confirm what real data actually uses.
    a_ckpt = next(iter(result["runs"].values()))["checkpoints"][0]
    print(f"\nby_band keys:  {list(a_ckpt['S_angle_by_band'])}")
    print(f"by_clade keys: {list(a_ckpt['S_angle_by_clade'])[:6]}"
          f"{' …' if len(a_ckpt['S_angle_by_clade']) > 6 else ''}"
          f"  ({len(a_ckpt['S_angle_by_clade'])} strata)")
    print(f"by_clade n range: "
          f"{min(d['n'] for d in a_ckpt['S_angle_by_clade'].values())}"
          f"-{max(d['n'] for d in a_ckpt['S_angle_by_clade'].values())}")
    print(f"trainer keys:  {sorted(a_ckpt['trainer'])}")

    seeds = []
    for arm in sys.argv[2:] or ["canonical"]:
        for s in (0, 1, 2):
            if f"{arm}_s{s}_roll" in result["runs"] and f"{arm}_s{s}_ms" in result["runs"]:
                seeds.append((arm, s))
    print(f"\nparsing {len(seeds)} run(s) the engine can see: {seeds}")
    for arm, s in seeds:
        rv = run_value(result, arm, s)
        g = validity_gate(result, arm, s)
        print(f"  {arm}_s{s}: S_angle {rv['S_angle']:+.4f} over {rv['n_roll']} roll ckpt(s) "
              f"{rv['roll_epochs']}, {rv['n_milestones']} milestone(s) | "
              f"level AUC {rv['level_auc']:.4f} | S_poincare {rv['S_poincare']:+.4f} | "
              f"gate_a rise {g['gate_a_rise']:+.4f} vs {g['gate_a_threshold']:.4f}")

    print("\nPARSER OK on real scorer output. No verdict computed: the pre-registration admits "
          "only the complete 3v3.")


if __name__ == "__main__":
    main()
