#!/usr/bin/env python3
"""REVIEW WAVE 3 (read-only): run `p2_verdict` on scorer-shaped inputs built from MEASURED numbers.

Not fixtures invented to make a point -- every arm/baseline value below comes from
`helpers/p2_review3_measure_production_splits.py` (the 12 real metazoa splits) or
`helpers/p2_review3_can_a_real_model_clear_the_control.py` (a real trained checkpoint through the
production scorer path). The engine is the real one; only the checkpoint JSON is synthesised, in
exactly the shape `scripts/score_p2_linkpred.py` writes and `scripts/p2_lrz_score.sh` registers.

Scenarios
---------
S1  C1's exploit under the RETIRED RandomDAG control -- must read GENERALISES (the defect).
S2  C1's exploit under the LIVE degree-matched control -- must NOT read GENERALISES (the fix).
S3  Real arm at the LEAKY UPPER BOUND (a checkpoint trained on every held-out edge), control merely
    matching its OWN training-free degree prior. The most generous case a real run can hope for.
S4  Same real arm, but the control arm IMPROVES on its own degree prior by the same factor the real
    checkpoint improved on ITS own prior (0.0405/0.1672 = 0.242). The expected case.
S5  The degmatch baselines key is ABSENT from the merged output: does anything refuse, or does the
    control silently inherit the REAL tree's floor (C4 #3's defect, one rename later)?
S6  Every baselines key is BARE (no _s<seed> suffix) -- the pre-Part-2 shape: does
    `assert_baselines_agree_across_seeds` notice, or is the Part 2 fix silently unenforced?

Writes nothing except stdout.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from taxembed.eval.preregistration import (  # noqa: E402
    merge_p2_scorer_outputs, p2_verdict,
)

MS_EPOCHS = list(range(10, 201, 10))
ROLL_EPOCHS = [196, 197, 198, 199, 200]

# ---- MEASURED, metazoa_33208_clean, seeds 0/1/2 (p2_review3_measure_production_splits.py) -------
REAL_BASE = {
    0: dict(sibling_chance_mean=0.13126037008135205, chance_mrr_mean=0.2710689315319428,
            dp_mrr=0.5599955004571145, dp_nr=0.15380546762948136, dp_h1=0.403883990381112),
    1: dict(sibling_chance_mean=0.13097682861393373, chance_mrr_mean=0.26989929083662373,
            dp_mrr=0.5606058018829624, dp_nr=0.15316559423653356, dp_h1=0.4078367049389466),
    2: dict(sibling_chance_mean=0.13222451994760923, chance_mrr_mean=0.2724753627464091,
            dp_mrr=0.5618369713678991, dp_nr=0.153195527408201, dp_h1=0.406400349905234),
}
DEGMATCH_BASE = {
    0: dict(sibling_chance_mean=0.1883306075428152, chance_mrr_mean=0.3229176300761225,
            dp_mrr=0.7526577030633812, dp_nr=0.04040666865910578, dp_h1=0.6386396748715589),
    1: dict(sibling_chance_mean=0.1816582037945914, chance_mrr_mean=0.31649041596545757,
            dp_mrr=0.7571252075114502, dp_nr=0.038240010365341026, dp_h1=0.64624145785877),
    2: dict(sibling_chance_mean=0.1900589977372855, chance_mrr_mean=0.326323989923558,
            dp_mrr=0.7573066986607998, dp_nr=0.03851134146187385, dp_h1=0.6459686664119221),
}
# RandomDAG's own figures, from p2_amendment_3's measured block (the retired control).
RANDOMDAG_BASE = {s: dict(sibling_chance_mean=0.55816, chance_mrr_mean=0.5960,
                          dp_mrr=0.42, dp_nr=0.24868839545805835, dp_h1=0.30) for s in (0, 1, 2)}

# a real trained checkpoint, mollusca, through the production scorer (leaky upper bound)
LEAKY_MRR, LEAKY_NR = 0.817220903289968, 0.040492621226291055


def baselines_block(b: dict) -> dict:
    return {
        "sibling_chance_mean": b["sibling_chance_mean"],
        "chance_hits_at_1_mean": b["sibling_chance_mean"],
        "chance_mrr_mean": b["chance_mrr_mean"],
        "vendrov_recall": 0.0,
        "majority_parent_rate": 0.002,
        "degree_prior": {"n": 28418, "n_scored": 27446, "n_trivial": 972,
                         "mean_rank": 6.0, "mrr": b["dp_mrr"], "hits_at_1": b["dp_h1"],
                         "hits_at_10": 0.86, "normalized_rank": b["dp_nr"]},
    }


def ckpt(epoch: int, mrr: float, nr: float, jitter: float = 0.0) -> dict:
    m = {"n": 28418, "n_scored": 27446, "n_trivial": 972, "mean_rank": 3.0,
         "mrr": mrr + jitter, "hits_at_1": 0.7, "hits_at_10": 0.95,
         "normalized_rank": nr + jitter * 0.01}
    depth = {name: {"n": 9000, "mrr": mrr + jitter, "normalized_rank": nr + jitter * 0.01}
             for name in ("11-15", "16-21", "22-28")}
    return {"epoch": epoch, "path": f"fake_epoch{epoch}.pth",
            "trainer": {"loss": 4.0, "epoch": epoch},
            "metrics": m, "metrics_cosine": m, "metrics_poincare": m,
            "by_pool_size": {"cosine": {}, "poincare": {}},
            "by_depth": {"cosine": depth, "poincare": depth}}


def arm_entry(mrr: float, nr: float) -> tuple[dict, dict]:
    """(_ms, _roll). Milestones rise from chance to the target so gate (a) can pass honestly;
    the roll window carries a small real jitter so gate (a)'s `jitter > 0` guard is satisfied."""
    ms = [ckpt(e, mrr * min(1.0, e / 100.0), nr) for e in MS_EPOCHS]
    roll = [ckpt(e, mrr, nr, jitter=(i - 2) * 1e-4) for i, e in enumerate(ROLL_EPOCHS)]
    return {"checkpoints": ms}, {"checkpoints": roll}


def build(real_mrr, real_nr, ctrl_mrr, ctrl_nr, control_prefix: str,
          seed_tagged: bool = True, emit_control_baselines: bool = True) -> dict:
    files = []
    for s in (0, 1, 2):
        arms = {}
        for arm in ("vis00", "vis50"):
            ms, roll = arm_entry(real_mrr, real_nr)
            arms[f"{arm}_s{s}_ms"], arms[f"{arm}_s{s}_roll"] = ms, roll
        key = f"baselines_s{s}" if seed_tagged else "baselines"
        files.append({"arms": arms, key: baselines_block(REAL_BASE[s])})

        ctrl_base = DEGMATCH_BASE[s] if control_prefix == "degmatch" else RANDOMDAG_BASE[s]
        for vis in ("vis00", "vis50"):
            name = f"{control_prefix}_{vis}"
            ms, roll = arm_entry(ctrl_mrr, ctrl_nr)
            f = {"arms": {f"{name}_s{s}_ms": ms, f"{name}_s{s}_roll": roll}}
            if emit_control_baselines:
                ck = f"baselines_{name}_s{s}" if seed_tagged else f"baselines_{name}"
                f[ck] = baselines_block(ctrl_base)
            files.append(f)
    return merge_p2_scorer_outputs(files)


def report(name: str, merged: dict, **flags) -> dict:
    v = p2_verdict(merged, (0, 1, 2), **flags)
    print(f"\n===== {name} =====   flags {flags}")
    print(f"  VERDICT: {v['verdict']}")
    if v["verdict"] == "UNINFORMATIVE":
        print(f"  {v['meaning']}")
        return v
    for arm, r in v["by_arm_verdict"].items():
        keys = [k for k in ("above_chance_margin", "above_degree_prior", "below_degree_prior",
                            "above_control_all_seeds", "below_control_all_seeds",
                            "below_chance_level", "equivalent_to_control", "sign_consistent")
                if k in r]
        print(f"  {arm}: {r['verdict']}  " + "  ".join(f"{k}={r[k]}" for k in keys))
    print(f"  published control floor (randomdag_sibling_chance_mean) = "
          f"{json.dumps(v['randomdag_sibling_chance_mean'])}")
    return v


def main() -> int:
    dp0 = REAL_BASE[0]
    # ---- S1: C1's exploit under the RETIRED RandomDAG control ----------------------------------
    m = build(dp0["dp_mrr"], dp0["dp_nr"], RANDOMDAG_BASE[0]["dp_mrr"],
              RANDOMDAG_BASE[0]["dp_nr"], "randomdag")
    report("S1  fan-out-only model vs RETIRED RandomDAG control (C1's defect: expect GENERALISES)",
           m, amendment_1=True, amendment_2=True)

    # ---- S2: C1's exploit under the LIVE degree-matched control --------------------------------
    m = build(dp0["dp_mrr"], dp0["dp_nr"], DEGMATCH_BASE[0]["dp_mrr"],
              DEGMATCH_BASE[0]["dp_nr"], "degmatch")
    report("S2  fan-out-only model vs LIVE degmatch control (C1's fix: expect NOT GENERALISES)",
           m, amendment_1=True, amendment_2=True, amendment_4=True)

    # ---- S3: real arm at its LEAKY ceiling, control frozen at its own training-free prior ------
    m = build(LEAKY_MRR, LEAKY_NR, DEGMATCH_BASE[0]["dp_mrr"], DEGMATCH_BASE[0]["dp_nr"],
              "degmatch")
    report("S3  real arm at a LEAKY UPPER BOUND (nr 0.04049) vs a control that learned NOTHING "
           "beyond fan-out (nr 0.04041-0.04)", m,
           amendment_1=True, amendment_2=True, amendment_4=True)

    # ---- S4: control improves on its own prior by the same factor the real model did -----------
    factor = LEAKY_NR / REAL_BASE[0]["dp_nr"]
    ctrl_nr = DEGMATCH_BASE[0]["dp_nr"] * factor
    print(f"\n[S4] real model improved on its own degree prior by factor "
          f"{factor:.4f} ({REAL_BASE[0]['dp_nr']:.5f} -> {LEAKY_NR:.5f}); applying the SAME factor "
          f"to the control gives control nr {ctrl_nr:.5f}")
    m = build(LEAKY_MRR, LEAKY_NR, 0.95, ctrl_nr, "degmatch")
    report("S4  real arm at its LEAKY ceiling vs a control that learns as well as the real arm did",
           m, amendment_1=True, amendment_2=True, amendment_4=True)

    # ---- S5: the control's own baselines key is ABSENT -----------------------------------------
    m = build(LEAKY_MRR, LEAKY_NR, 0.95, ctrl_nr, "degmatch", emit_control_baselines=False)
    print("\n[S5] merged keys:", sorted(k for k in m if k.startswith("baselines")))
    v = report("S5  control baselines key ABSENT -- does anything refuse?", m,
               amendment_1=True, amendment_2=True, amendment_4=True)
    got = v["randomdag_sibling_chance_mean"]
    print(f"  CONTROL's published floor with its own key missing: {json.dumps(got)}")
    print(f"  the REAL tree's floor is                          : "
          f"{REAL_BASE[0]['sibling_chance_mean']:.5f}")
    print(f"  the CONTROL's TRUE measured floor is              : "
          f"{DEGMATCH_BASE[0]['sibling_chance_mean']:.5f}")
    print(f"  => silently inherited the real tree's floor? "
          f"{all(abs(x - REAL_BASE[0]['sibling_chance_mean']) < 1e-9 for x in got.values())}")
    print(f"  _baselines_disagreements present? {'_baselines_disagreements' in m}")

    # ---- S6: all-bare baselines keys (pre-Part-2 shape) ----------------------------------------
    m = build(LEAKY_MRR, LEAKY_NR, 0.95, ctrl_nr, "degmatch", seed_tagged=False)
    print("\n[S6] merged keys:", sorted(k for k in m if k.startswith("baselines")))
    print(f"  _baselines_cross_seed_spread (families checked) = "
          f"{json.dumps(m.get('_baselines_cross_seed_spread'))}")
    # make seed 1's real baseline WILDLY wrong under the bare shape and see if the merge notices
    files = []
    for s in (0, 1, 2):
        arms = {}
        for arm in ("vis00", "vis50"):
            ms, roll = arm_entry(LEAKY_MRR, LEAKY_NR)
            arms[f"{arm}_s{s}_ms"], arms[f"{arm}_s{s}_roll"] = ms, roll
        b = copy.deepcopy(REAL_BASE[s])
        if s == 1:
            b["dp_mrr"] = 0.99     # a wrong split landing under this seed: 77% off the others
            b["chance_mrr_mean"] = 0.60
        files.append({"arms": arms, "baselines": baselines_block(b)})
    try:
        merged = merge_p2_scorer_outputs(files)
        print(f"  a 77%-off seed-1 baseline under the BARE key: merge RAISED? False  "
              f"(families checked: {json.dumps(merged.get('_baselines_cross_seed_spread'))})")
        print(f"  value the engine will use for ALL seeds: chance_mrr_mean="
              f"{merged['baselines']['chance_mrr_mean']}, "
              f"degree_prior.mrr={merged['baselines']['degree_prior']['mrr']}")
        print(f"  _baselines_disagreements recorded (not raised)? "
              f"{'_baselines_disagreements' in merged}")
    except ValueError as exc:
        print(f"  merge RAISED: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
