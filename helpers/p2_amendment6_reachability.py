#!/usr/bin/env python3
"""amendment_6 (C-A) reachability probe -- read-only, PRE-RUN, no GPU.

The question amendment_6 has to answer is NOT "can we now get GENERALISES" -- any loosening does
that. It is the pair of questions:

  (1) REACHABILITY.  Does a real arm that genuinely improves on its own tree's training-free
      degree prior MORE than the control improves on ITS own prior now read GENERALISES, where
      the unamended reading returned MEMORISES on the SAME data?
  (2) IT COULD STILL HAVE DISAGREED.  Does an arm that improves by the SAME factor as the
      control, or by LESS, still read MEMORISES under amendment_6?

If only (1) held, amendment_6 would be a thumb on the scale. Both together are what make it a
correction of a confound rather than a licence.

Every baseline number below is MEASURED -- imported from `p2_review3_verdict_scenarios`, which
took them from the 12 real metazoa splits (`p2_review3_measure_production_splits.py`,
re-derived 2026-09-26) and from a real trained checkpoint through the production scorer path.
The engine is the real `p2_verdict`; only the checkpoint JSON is synthesised, in the shape
`scripts/score_p2_linkpred.py` writes.

Writes nothing except stdout. Touches no production file.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "helpers"))

from p2_review3_verdict_scenarios import (  # noqa: E402
    DEGMATCH_BASE, LEAKY_MRR, LEAKY_NR, REAL_BASE, build,
)
from taxembed.eval.preregistration import p2_verdict  # noqa: E402

LIVE = dict(amendment_1=True, amendment_2=True, amendment_4=True)
REAL_DP = REAL_BASE[0]["dp_nr"]
CTRL_DP = DEGMATCH_BASE[0]["dp_nr"]


def read(merged: dict, **flags) -> tuple[str, dict]:
    v = p2_verdict(merged, (0, 1, 2), **flags)
    if v["verdict"] == "UNINFORMATIVE":
        return v["verdict"], {}
    return v["verdict"], v["by_arm_verdict"]["vis00"]


def prose_is_consistent(reading: dict) -> bool:
    """The verdict STRING is what gets quoted into the manuscript, so it must not assert
    something the reading's own booleans deny. Checks the two claims that were wrong before
    amendment_6's second and third corrections: GENERALISES while `equivalent_to_control`, and
    a MIXED sentence blaming a depth-stratum sign flip that did not happen."""
    if not reading:
        return True
    verdict, meaning = reading["verdict"], reading["meaning"]
    if verdict == "GENERALISES" and reading.get("equivalent_to_control"):
        print("    PROSE DEFECT: GENERALISES published while equivalent_to_control=True")
        return False
    if verdict == "MIXED" and reading.get("sign_consistent") and "sign" in meaning:
        print("    PROSE DEFECT: MIXED blames a sign flip, but sign_consistent=True")
        return False
    if verdict == "MIXED" and not reading.get("below_control_all_seeds"):
        if "does not beat its control" not in meaning:
            print(f"    PROSE DEFECT: MIXED does not name the binding clause -- {meaning!r}")
            return False
    return True


def scenario(label: str, real_factor: float, ctrl_factor: float, expect_a6: str) -> bool:
    """One arm/control pair expressed as IMPROVEMENT FACTORS on each tree's own degree prior.

    factor < 1.0 means "better than the training-free prior on my own tree" (normalized_rank is
    lower-is-better). factor == 1.0 means "exactly the prior, i.e. learned nothing beyond fan-out".
    """
    real_nr, ctrl_nr = REAL_DP * real_factor, CTRL_DP * ctrl_factor
    merged = build(LEAKY_MRR, real_nr, 0.95, ctrl_nr, "degmatch")

    old_verdict, old = read(merged, **LIVE)
    new_verdict, new = read(merged, **LIVE, amendment_6=True)

    ok = new_verdict == expect_a6
    print(f"\n===== {label} =====")
    print(f"  real arm improves on ITS prior by {real_factor:.3f}  (nr {real_nr:.5f} "
          f"vs own prior {REAL_DP:.5f})")
    print(f"  control    improves on ITS prior by {ctrl_factor:.3f}  (nr {ctrl_nr:.5f} "
          f"vs own prior {CTRL_DP:.5f})")
    print(f"  amendment_6=False -> {old_verdict:<12} "
          f"below_control_all_seeds={old.get('below_control_all_seeds')} "
          f"equivalent_to_control={old.get('equivalent_to_control')} "
          f"band={old.get('equivalence_band_vs_control'):.6f} "
          f"|mean_diff|={abs(old.get('mean_diff_vs_control', float('nan'))):.6f}")
    print(f"  amendment_6=True  -> {new_verdict:<12} "
          f"below_control_all_seeds={new.get('below_control_all_seeds')} "
          f"equivalent_to_control={new.get('equivalent_to_control')} "
          f"band={new.get('equivalence_band_vs_control'):.6f} "
          f"|mean_diff|={abs(new.get('mean_diff_vs_control', float('nan'))):.6f}")
    print(f"  meaning: {new.get('meaning', '(none)')}")
    prose_ok = prose_is_consistent(new)
    print(f"  expected under amendment_6: {expect_a6}   -> "
          f"{'OK' if ok else 'MISMATCH'}{'' if prose_ok else ' + PROSE DEFECT'}")
    return ok and prose_ok


def main() -> int:
    print("Measured own-tree training-free degree priors (normalized_rank, seed 0):")
    print(f"  real tree      {REAL_DP:.5f}")
    print(f"  degmatch control {CTRL_DP:.5f}   -> the control's task is "
          f"{REAL_DP / CTRL_DP:.2f}x easier for a ranker that learned NOTHING")

    results = []
    # (1) REACHABILITY: the real arm genuinely learns more, relative to its own tree's prior.
    results.append(scenario(
        "R1  real arm improves 4x on its own prior, control only 1.3x  (expect GENERALISES)",
        real_factor=0.25, ctrl_factor=0.75, expect_a6="GENERALISES"))

    # (2) IT COULD HAVE DISAGREED -- three independent ways.
    results.append(scenario(
        "R2  BOTH improve by the SAME factor -- no relative gain  (expect MEMORISES)",
        real_factor=0.242, ctrl_factor=0.242, expect_a6="MEMORISES"))
    # R3 reads MIXED, not MEMORISES, and that is the DESIGNED semantics, not a defect: MEMORISES
    # means "indistinguishable from chance / from the prior / from the control", and an arm
    # uniformly WORSE than its control is none of those three. What R3 originally exposed was the
    # MIXED branch's PROSE, which blamed a depth-stratum sign flip that never happened -- fixed
    # by amendment_6's third correction and asserted by `prose_is_consistent` above.
    results.append(scenario(
        "R3  the CONTROL improves MORE than the real arm  (expect MIXED, naming the real cause)",
        real_factor=0.60, ctrl_factor=0.20, expect_a6="MIXED"))
    results.append(scenario(
        "R4  real arm learns NOTHING beyond fan-out (exactly its own prior)  (expect MEMORISES)",
        real_factor=1.0, ctrl_factor=1.0, expect_a6="MEMORISES"))

    # (3) The leaky upper bound -- the review's S3/S4, now read under amendment_6.
    leak_factor = LEAKY_NR / REAL_DP
    print(f"\n[leaky upper bound] a checkpoint that SAW every held-out edge improved on its own "
          f"prior by {leak_factor:.4f}")
    results.append(scenario(
        "R5  real arm at its LEAKY ceiling vs a control that learned nothing  "
        "(expect GENERALISES: the ceiling must at least be reachable)",
        real_factor=leak_factor, ctrl_factor=1.0, expect_a6="GENERALISES"))

    print("\n================ SUMMARY ================")
    print(f"  {sum(results)} of {len(results)} scenarios matched expectation")
    if not all(results):
        print("  FAILED -- amendment_6 does not behave as the amendment block claims")
        return 1
    print("  PASS -- GENERALISES is reachable when the real arm out-improves its control on the")
    print("         own-prior scale, and is still REFUSED when it does not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
