#!/usr/bin/env python3
"""REVIEW WAVE 3 (read-only): the flag combination PRODUCTION will use is never tested.

`scripts/p2_lrz_train.sh` and `scripts/p2_lrz_score.sh` both instruct, in their own headers:

    go through src/taxembed/eval/preregistration.py::p2_verdict(..., amendment_1=True,
    amendment_2=True, amendment_4=True)

Grep of `tests/eval/test_preregistration.py`: `amendment_1=True` appears on lines 679, 713, 958;
`amendment_4=True` on lines 1155, 1185, 1198, 1209, 1223. The two sets are DISJOINT. So every
TestP2Amendment4 assertion is computed with `amendment_1=False`, i.e. on the RAW-MRR cross-tree
comparison that `p2_amendment_1_20260924` rule_1 forbids and that amendment_4's own block calls
still-load-bearing-to-avoid.

This script asks what the suite's OWN amendment_4 fixtures read under the production flags, using
the test module's own helper functions -- not a fixture I invented.

Writes nothing except stdout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))

from taxembed.eval.preregistration import p2_verdict, _p2_depth_means  # noqa: E402
from tests.eval import test_preregistration as T  # noqa: E402


def show(label: str, v: dict) -> None:
    print(f"\n----- {label}")
    print(f"  VERDICT {v['verdict']}")
    if "by_arm_verdict" in v:
        for arm, r in v["by_arm_verdict"].items():
            keys = [k for k in ("above_chance_margin", "above_degree_prior", "below_degree_prior",
                                "above_control_all_seeds", "below_control_all_seeds",
                                "below_chance_level", "equivalent_to_control", "sign_consistent")
                    if k in r]
            print(f"  {arm}: {r['verdict']}  metric={r['cross_tree_metric']}  "
                  + "  ".join(f"{k}={r[k]}" for k in keys))
            print(f"      depth_strata n={r['depth_strata']['n_strata']} "
                  f"diffs={json.dumps({k: round(x, 5) for k, x in r['depth_strata']['diffs'].items()})} "
                  f"dropped={json.dumps(r['depth_strata']['dropped_below_min_n'])}")


def main() -> int:
    # The suite's own amendment_4 fixture, verbatim from
    # TestP2Amendment4::test_matched_degmatch_controls_read_each_real_arm_against_its_own_visibility
    res = T._p2_twelve_arm_result_named(
        vis00_final=0.55, vis50_final=0.50,
        control_vis00_final=0.10, control_vis50_final=0.10,
        control_vis00_name="degmatch_vis00", control_vis50_name="degmatch_vis50",
        sibling_chance=0.05)

    print("what the checkpoint dicts this fixture builds actually carry in by_depth['cosine']:")
    c = res["arms"]["vis00_s0_ms"]["checkpoints"][-1]
    print(f"  metrics keys   : {sorted(c['metrics_cosine'])}")
    print(f"  by_depth stratum '16-21' keys: {sorted(c['by_depth']['cosine']['16-21'])}")

    print("\nwhat _p2_depth_means can read out of it:")
    for field in ("mrr", "normalized_rank"):
        d = _p2_depth_means(res, "vis00", (0, 1, 2), field=field)
        print(f"  field={field:<16} means={json.dumps({k: round(v, 4) for k, v in d['means'].items()})} "
              f"dropped={json.dumps(d['dropped_below_min_n'])}")

    show("AS TESTED:  p2_verdict(res, amendment_4=True)   <- the only reading the suite asserts",
         p2_verdict(res, amendment_4=True))
    show("AS PRODUCTION WILL RUN IT:  p2_verdict(res, amendment_1=True, amendment_2=True, "
         "amendment_4=True)",
         p2_verdict(res, amendment_1=True, amendment_2=True, amendment_4=True))

    # Now the same fixture with normalized_rank ADDED to by_depth, to isolate the cause.
    patched = json.loads(json.dumps(res))
    for key, run in patched["arms"].items():
        for ck in run["checkpoints"]:
            nr = ck["metrics_cosine"]["normalized_rank"]
            for stratum in ck["by_depth"]["cosine"].values():
                stratum["normalized_rank"] = nr
    show("SAME FIXTURE + normalized_rank present in by_depth, production flags",
         p2_verdict(patched, amendment_1=True, amendment_2=True, amendment_4=True))

    print("\n=> If the two production-flag readings above DIFFER, the cause is a by_depth field the")
    print("   fixtures never populate: `sign_consistent` is structurally False whenever")
    print("   by_depth['cosine'][stratum] lacks 'normalized_rank', and GENERALISES is then")
    print("   unreachable under amendment_1=True for EVERY fixture built on `_p2_ckpt`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
