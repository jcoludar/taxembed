"""READ-ONLY PROBE (review 4, 2026-09-26).

C-A asks the question in both directions. The commit's own probe
(`helpers/p2_amendment6_reachability.py`) asks "is GENERALISES reachable?" and "is it refused
when the arm does not out-improve its control?". It never asks the third question:

    can an arm read GENERALISES while its DEGREE-MATCHED CONTROL predicts the held-out parent
    BETTER THAN IT DOES, on every seed, on the raw metric?

Under the ratio reading the answer is yes, and it is not a corner case: the two trees' priors
differ ~3.8x (measured), so any arm/control pair whose raw normalized_rank ratio sits between 1
and ~3.6 gets GENERALISES while losing outright. Driven through the REAL p2_verdict with the
REAL measured metazoa priors. The point is not that the ratio is the wrong statistic -- it is the
USER's ruling -- but that the verdict PROSE and the renderer say nothing about it.

Nothing is written. Nothing under src/, scripts/, tests/ or results/ is touched.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from taxembed.eval.preregistration import p2_verdict  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "tp", REPO / "tests" / "eval" / "test_preregistration.py")
tp = importlib.util.module_from_spec(spec)
sys.modules["tp"] = tp
spec.loader.exec_module(tp)

LIVE = dict(amendment_1=True, amendment_2=True, amendment_4=True)
R, C = tp._A6_REAL_PRIOR_NR, tp._A6_CTRL_PRIOR_NR       # 0.153805 / 0.040407, measured metazoa


def case(label: str, arm_nr: float, ctrl_nr: float) -> None:
    res = tp._p2_amendment6_result(real_factor=arm_nr / R, ctrl_factor=ctrl_nr / C)
    v = p2_verdict(res, **LIVE, amendment_6=True)
    a = v["by_arm_verdict"]["vis00"]
    print(f"\n{label}")
    print(f"  raw normalized_rank (LOWER IS BETTER):  arm {arm_nr:.5f}   control {ctrl_nr:.5f}")
    print(f"  -> the CONTROL ranks the true parent better on every seed: {ctrl_nr < arm_nr}")
    print(f"  ratios to own prior:  arm {arm_nr / R:.4f}   control {ctrl_nr / C:.4f}")
    print(f"  below_control_all_seeds={a['below_control_all_seeds']}  "
          f"equivalent_to_control={a['equivalent_to_control']}  "
          f"band={a['equivalence_band_vs_control']:.5f}  mean_diff={a['mean_diff_vs_control']:+.5f}")
    print(f"  VERDICT: {v['verdict']}")
    print(f"  published prose: \"{a['meaning'][:150]}...\"")


def main() -> int:
    print("measured metazoa priors: real degree_prior.normalized_rank "
          f"{R:.5f}, degmatch {C:.5f}  (ratio {R / C:.3f})")
    case("A. the control BEATS the arm outright, 2.2x, on the raw metric",
         arm_nr=0.045, ctrl_nr=0.020)
    case("B. the arm and the control score IDENTICALLY on the raw metric",
         arm_nr=0.030, ctrl_nr=0.030)
    case("C. the boundary: the raw ratio at which GENERALISES stops",
         arm_nr=0.045, ctrl_nr=0.045 / (0.95 * R / C))
    print("\n----------------------------------------------------------------------------")
    print("For reference, the same three read under the CURRENT PRODUCTION instruction")
    print("(`--amendment-1 --amendment-2 --amendment-4`, which is still what BOTH job")
    print("scripts' headers tell the operator to run -- neither mentions --amendment-6):")
    for label, arm_nr, ctrl_nr in (("A", 0.045, 0.020), ("B", 0.030, 0.030)):
        res = tp._p2_amendment6_result(real_factor=arm_nr / R, ctrl_factor=ctrl_nr / C)
        v = p2_verdict(res, **LIVE)
        print(f"  {label}: {v['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
