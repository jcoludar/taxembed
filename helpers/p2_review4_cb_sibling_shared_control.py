"""READ-ONLY PROBE (review 4, 2026-09-26).

C-B's fix landed on `_p2_matched_controls` only. `_p2_single_shared_control`
(preregistration.py:942) still ends the shared RandomDAG control's fallback chain at the REAL
tree's bare `"baselines"` key. This probe measures whether the exact defect C-B closed is still
reachable through the sibling function, and by how much the published control floor moves.

Nothing is written. Nothing under src/, scripts/, tests/ or results/ is touched.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests" / "eval"))

from taxembed.eval.preregistration import (  # noqa: E402
    _p2_matched_controls, _p2_single_shared_control, p2_verdict,
)

spec = importlib.util.spec_from_file_location(
    "tp", REPO / "tests" / "eval" / "test_preregistration.py")
tp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tp)


def main() -> None:
    # A 9-arm result: vis00/vis50/randomdag, ONE shared control, and NO
    # `baselines_randomdag*` key anywhere -- exactly review scenario S5's shape
    # (the control's own scorer JSON never reached the merge).
    res = tp._p2_nine_arm_result(vis00_final=0.55, vis50_final=0.50,
                                 randomdag_final=0.30, sibling_chance=0.05)
    assert "baselines_randomdag" not in res, "fixture must not supply the control's own block"
    real_floor = res["baselines"]["sibling_chance_mean"]
    real_prior = res["baselines"]["degree_prior"]

    controls, _ = _p2_single_shared_control(res, (0, 1, 2))
    got = controls["randomdag"]
    print("== _p2_single_shared_control (amendment_2=False, the frozen default path) ==")
    print(f"  control block resolved                : NO ERROR RAISED")
    print(f"  control sibling_chance_mean published : {got['sibling_chance_mean']}")
    print(f"  the REAL tree's sibling_chance_mean   : {real_floor}")
    print(f"  identical?                            : "
          f"{got['sibling_chance_mean'] == real_floor}")
    print(f"  control degree_prior published        : {got['degree_prior']}")
    print(f"  the REAL tree's degree_prior          : {real_prior}")
    print(f"  identical?                            : {got['degree_prior'] == real_prior}")

    v = p2_verdict(res, (0, 1, 2))
    print(f"  p2_verdict(default) completes         : verdict={v['verdict']}")
    print(f"  published randomdag_sibling_chance_mean: {v['randomdag_sibling_chance_mean']}")

    # The sibling that WAS fixed, same input shape, for contrast.
    print()
    print("== _p2_matched_controls (amendment_2/4 path, the one C-B fixed) ==")
    res2 = tp._p2_twelve_arm_result_named(
        vis00_final=0.55, vis50_final=0.50,
        control_vis00_final=0.30, control_vis50_final=0.30,
        control_vis00_name="degmatch_vis00", control_vis50_name="degmatch_vis50",
        sibling_chance=0.05)
    del res2["baselines_degmatch_vis00"]
    del res2["baselines_degmatch_vis50"]
    try:
        _p2_matched_controls(res2, (0, 1, 2), {"vis00": "degmatch_vis00",
                                               "vis50": "degmatch_vis50"})
        print("  NO ERROR RAISED  <- would be the defect")
    except KeyError as exc:
        print(f"  RAISES as intended: {str(exc)[:90]}...")

    # ... but the chain does NOT "end inside the control's own family", as the fix's own
    # comment (preregistration.py:984-989) states. Its second link is "baselines_randomdag",
    # which for a degmatch control is a THIRD tree, not its own.
    print()
    print("== the chain's SECOND link, `baselines_randomdag`, for a DEGMATCH control ==")
    res3 = tp._p2_twelve_arm_result_named(
        vis00_final=0.55, vis50_final=0.50,
        control_vis00_final=0.30, control_vis50_final=0.30,
        control_vis00_name="degmatch_vis00", control_vis50_name="degmatch_vis50",
        sibling_chance=0.05)
    del res3["baselines_degmatch_vis00"]
    del res3["baselines_degmatch_vis50"]
    # a stale RandomDAG scorer JSON from the retired 9-arm design reaches the same merge
    res3["baselines_randomdag"] = tp._p2_baselines(
        0.18833, degree_prior={"mrr": 0.30, "hits_at_1": 0.2, "hits_at_10": 0.6,
                               "normalized_rank": 0.2487, "n": 500, "n_scored": 500,
                               "n_trivial": 0})
    controls3, _ = _p2_matched_controls(res3, (0, 1, 2), {"vis00": "degmatch_vis00",
                                                          "vis50": "degmatch_vis50"})
    got3 = controls3["degmatch_vis00"]
    print(f"  NO ERROR RAISED. degmatch_vis00's degree_prior.normalized_rank resolved to "
          f"{got3['degree_prior']['normalized_rank']}")
    print(f"  that is the RETIRED RandomDAG tree's value (0.2487), not the degree-matched "
          f"tree's (~0.0404) -- a 6.2x wrong amendment_6 DENOMINATOR")
    print(f"  sibling_chance_mean published for this control: {got3['sibling_chance_mean']}")


if __name__ == "__main__":
    main()
