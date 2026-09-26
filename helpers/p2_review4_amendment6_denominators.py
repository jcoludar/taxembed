"""READ-ONLY PROBE (review 4, 2026-09-26).

Two questions about amendment_6's denominator:

  A. PER-SEED. The Part-2 fix established that `degree_prior` is a PER-SEED quantity (each seed
     draws its own held-out set). `p2_group_stats` still surfaces only `seeds[0]`'s block as
     `stats["degree_prior"]`, defending it as "cosmetic ... which value is DISPLAYED".
     amendment_6 makes that value the DENOMINATOR of the published cross-tree statistic, so it is
     no longer cosmetic. Measure whether all three seeds are divided by seed 0's prior, and
     whether a per-seed denominator would change `below_control_all_seeds`.

  B. PER-STRATUM. `_p2_depth_means` values are divided by the AGGREGATE prior. Construct the case
     where a per-stratum prior gives the opposite sign test, i.e. a different verdict.

Nothing is written. Nothing under src/, scripts/, tests/ or results/ is touched.
"""
from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from taxembed.eval.preregistration import p2_verdict  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "tp", REPO / "tests" / "eval" / "test_preregistration.py")
tp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tp)

LIVE = dict(amendment_1=True, amendment_2=True, amendment_4=True)
# The production metazoa figures the amendment itself records.
REAL_PRIOR = [0.15381, 0.15317, 0.15320]
CTRL_PRIOR = [0.04041, 0.03824, 0.03851]


def _seed_tag(res: dict) -> dict:
    """Turn the fixture's bare baselines keys into the SEED-TAGGED shape production writes,
    giving each seed its OWN measured degree prior (the real per-split numbers above)."""
    out = {k: v for k, v in res.items() if not k.startswith("baselines")}
    for s in (0, 1, 2):
        b = copy.deepcopy(res["baselines"])
        b["degree_prior"]["normalized_rank"] = REAL_PRIOR[s]
        out[f"baselines_s{s}"] = b
        for c in ("degmatch_vis00", "degmatch_vis50"):
            bc = copy.deepcopy(res[f"baselines_{c}"])
            bc["degree_prior"]["normalized_rank"] = CTRL_PRIOR[s]
            out[f"baselines_{c}_s{s}"] = bc
    return out


def part_a() -> None:
    print("=" * 94)
    print("A. amendment_6 divides EVERY seed by SEED 0's prior, not by that seed's own")
    print("=" * 94)
    # An arm sitting a hair better than its control on the own-prior scale, on seed-0's
    # denominators.  The three seeds' real priors differ by 0.42%, the control's by 5.4% --
    # the spreads the wave-3 review measured on the production splits.
    # Chosen so the two denominator conventions straddle the RELATIVE equivalence band:
    # raw arm/control normalized_rank ratio 3.70, between 0.95*mean(1/pC)/mean(1/pR) = 3.734
    # (the per-seed equivalence threshold) and 0.95*pR0/pC0 = 3.616 (the as-coded one).
    res = tp._p2_amendment6_result(real_factor=0.500, ctrl_factor=0.500 / 3.70
                                   * (tp._A6_REAL_PRIOR_NR / tp._A6_CTRL_PRIOR_NR))
    res = _seed_tag(res)
    v = p2_verdict(res, **LIVE, amendment_6=True)
    arm = v["by_arm_verdict"]["vis00"]
    print(f"  denominator the engine used for the ARM     : "
          f"{arm['arm_degree_prior_normalized_rank']!r}")
    print(f"  the three seeds' OWN real-tree priors        : {REAL_PRIOR}")
    print(f"  denominator the engine used for the CONTROL : "
          f"{arm['control_degree_prior_normalized_rank']!r}")
    print(f"  the three seeds' OWN control priors          : {CTRL_PRIOR}")
    print(f"  => seed 1 and seed 2 are divided by seed 0's prior: "
          f"{arm['arm_degree_prior_normalized_rank'] == REAL_PRIOR[0]}")

    # Now compute what a PER-SEED denominator would have given, from the very same run values.
    g_nr = np.array(arm["arm_values_as_compared"]) * arm["arm_degree_prior_normalized_rank"]
    c_nr = np.array(arm["control_values_as_compared"]) * arm["control_degree_prior_normalized_rank"]
    as_coded = bool(max(arm["arm_values_as_compared"]) < min(arm["control_values_as_compared"]))
    per_seed_g = g_nr / np.array(REAL_PRIOR)
    per_seed_c = c_nr / np.array(CTRL_PRIOR)
    per_seed = bool(per_seed_g.max() < per_seed_c.min())
    print()
    print(f"  raw normalized_rank, arm      : {[round(x, 6) for x in g_nr]}")
    print(f"  raw normalized_rank, control  : {[round(x, 6) for x in c_nr]}")
    print(f"  arm ratios AS CODED (seed-0 denominator)  : "
          f"{[round(x, 6) for x in arm['arm_values_as_compared']]}")
    print(f"  arm ratios with PER-SEED denominators     : {[round(x, 6) for x in per_seed_g]}")
    print(f"  ctrl ratios AS CODED (seed-0 denominator) : "
          f"{[round(x, 6) for x in arm['control_values_as_compared']]}")
    print(f"  ctrl ratios with PER-SEED denominators    : {[round(x, 6) for x in per_seed_c]}")
    print()
    print(f"  below_control_all_seeds  AS CODED : {as_coded}   (engine reports "
          f"{arm['below_control_all_seeds']})")
    print(f"  below_control_all_seeds  PER-SEED : {per_seed}")

    # the equivalence clause, the one that actually binds here
    band_coded = 0.05 * abs(np.mean(arm["control_values_as_compared"]))
    diff_coded = float(np.mean(arm["arm_values_as_compared"])
                       - np.mean(arm["control_values_as_compared"]))
    band_ps = 0.05 * abs(float(per_seed_c.mean()))
    diff_ps = float(per_seed_g.mean() - per_seed_c.mean())
    print()
    print(f"  equivalence AS CODED : |{diff_coded:+.6f}| vs band {band_coded:.6f} -> "
          f"equivalent={abs(diff_coded) < band_coded}  (engine reports "
          f"{arm['equivalent_to_control']})")
    print(f"  equivalence PER-SEED : |{diff_ps:+.6f}| vs band {band_ps:.6f} -> "
          f"equivalent={abs(diff_ps) < band_ps}")
    verdict_ps = ("GENERALISES" if (per_seed and not abs(diff_ps) < band_ps) else "not GENERALISES")
    print()
    print(f"  VERDICT as coded (engine)     : {v['verdict']}")
    print(f"  VERDICT with per-seed priors  : {verdict_ps}")
    print(f"  => the denominator convention ALONE moves the headline verdict: "
          f"{(v['verdict'] == 'GENERALISES') != (verdict_ps == 'GENERALISES')}")


def part_b() -> None:
    print()
    print("=" * 94)
    print("B. the depth-stratum sign test uses the AGGREGATE prior; a per-stratum one flips it")
    print("=" * 94)
    # Aggregate priors: real 0.15381, control 0.04041 -> ratio 3.807.
    # Suppose the true PER-STRATUM priors in the deepest bin are real 0.060, control 0.050
    # (ratio 1.20, not 3.81) -- deep bins have small candidate pools on BOTH trees, so the
    # difficulty gap that exists in aggregate largely vanishes there.
    # MEASURED on the production splits by helpers/p2_review4_per_stratum_prior.py, seed 0:
    # aggregate real 0.15381 / control 0.04041 (ratio 3.806), but stratum 11-15's own priors are
    # real 0.17640 / control 0.06229 (ratio 2.832) -- the aggregate OVER-ALLOWS by 1.344x there.
    agg_real, agg_ctrl = 0.15381, 0.04041
    strat_real, strat_ctrl = 0.17640, 0.06229
    # arm and control each halve THEIR OWN per-stratum prior -- a genuine tie in that stratum.
    arm_nr, ctrl_nr = strat_real * 0.5, strat_ctrl * 0.5
    diff_aggregate = arm_nr / agg_real - ctrl_nr / agg_ctrl
    diff_per_stratum = arm_nr / strat_real - ctrl_nr / strat_ctrl
    print(f"  stratum '11-15': arm normalized_rank {arm_nr:.5f}, control {ctrl_nr:.5f} "
          f"-- each exactly HALF its own tree's per-stratum prior, i.e. a genuine tie")
    print(f"  aggregate priors : real {agg_real}, control {agg_ctrl}  (ratio "
          f"{agg_real / agg_ctrl:.3f})")
    print(f"  per-stratum priors: real {strat_real}, control {strat_ctrl}  (ratio "
          f"{strat_real / strat_ctrl:.3f})")
    print(f"  diff AS CODED (aggregate)  : {diff_aggregate:+.6f}  -> sign_consistent needs < 0: "
          f"{diff_aggregate < 0}")
    print(f"  diff with PER-STRATUM prior: {diff_per_stratum:+.6f}  -> "
          f"{diff_per_stratum < 0}")
    print(f"  => the two give OPPOSITE answers for this stratum: "
          f"{(diff_aggregate < 0) != (diff_per_stratum < 0)}")

    # And drive it through the real engine so the verdict itself moves.
    print()
    print("  through the real engine (same numbers, one stratum):")
    for label, depth_scale in (("arm and control identical in every stratum", 1.0),):
        res = tp._p2_amendment6_result(
            real_factor=arm_nr / agg_real * depth_scale,
            ctrl_factor=ctrl_nr / agg_ctrl * depth_scale,
            real_prior_nr=agg_real, ctrl_prior_nr=agg_ctrl)
        v = p2_verdict(res, **LIVE, amendment_6=True)
        a = v["by_arm_verdict"]["vis00"]
        print(f"    {label}: verdict={v['verdict']} sign_consistent={a['sign_consistent']} "
              f"diffs={ {k: round(x, 5) for k, x in a['depth_strata']['diffs'].items()} }")


if __name__ == "__main__":
    part_a()
    part_b()
