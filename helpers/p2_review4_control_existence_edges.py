"""READ-ONLY PROBE (review 4, 2026-09-26).

Edges of `assert_control_baselines_exist` (preregistration.py:1155-1199):

  1. the `key.rsplit("_s", 1)` arm-name parse, on the real production keys and on adversarial ones
  2. the `range(16)` seed probe, against `p2_verdict`/`merge_p2_scorer_outputs`' `seeds` parameter
     (which the function is never given)
  3. ONE-OF-SIXTEEN: the check passes when a control has a family for ANY seed, so one of the nine
     scorer JSONs missing still merges clean
  4. the real-tree family: is `baselines_s{seed}` existence asserted anywhere?

Nothing is written. Nothing under src/, scripts/, tests/ or results/ is touched.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from taxembed.eval.preregistration import (  # noqa: E402
    P2_DATA_ARMS, assert_control_baselines_exist, merge_p2_scorer_outputs,
)


def parsed_arm(key: str):
    parts = key.rsplit("_s", 1)
    return parts[0] if len(parts) == 2 else None


def part1() -> None:
    print("1. arm-name parse, `key.rsplit('_s', 1)[0]`")
    cases = [
        ("vis00_s0_roll", "vis00"),
        ("vis50_s2_ms", "vis50"),
        ("randomdag_s1_roll", "randomdag"),
        ("degmatch_vis00_s0_ms", "degmatch_vis00"),
        ("degmatch_vis50_s10_roll", "degmatch_vis50"),
        # adversarial: a KIND containing "_s"
        ("degmatch_vis00_s0_scored", "degmatch_vis00"),
        ("vis00_s0_shuffled", "vis00"),
        # adversarial: an arm name ending in a bare "s"
        ("degmatch_s_s0_roll", "degmatch_s"),
        # no "_s" at all
        ("perfect", None),
    ]
    bad = 0
    for key, want in cases:
        got = parsed_arm(key)
        ok = got == want
        bad += (not ok)
        print(f"   {key:<30} -> {str(got):<18} expected {str(want):<18} "
              f"{'ok' if ok else 'WRONG'}")
    print(f"   wrong parses: {bad}")
    print(f"   P2_DATA_ARMS = {P2_DATA_ARMS}  (anything else is treated as a CONTROL)")


def _mk(arms_keys, baselines_keys) -> dict:
    ck = [{"epoch": e, "metrics_cosine": {"mrr": 0.8, "normalized_rank": 0.04}}
          for e in range(196, 201)]
    return {"arms": {k: {"checkpoints": ck} for k in arms_keys},
            **{k: {"sibling_chance_mean": 0.1, "chance_mrr_mean": 0.2,
                   "degree_prior": {"mrr": 0.5, "normalized_rank": 0.04}} for k in baselines_keys}}


def part2() -> None:
    print("\n2. the range(16) seed probe vs a non-default `seeds`")
    arms = [f"{a}_s{s}_{k}" for a in ("vis00", "degmatch_vis00")
            for s in (20, 21, 22) for k in ("ms", "roll")]
    res = _mk(arms, [f"baselines_s{s}" for s in (20, 21, 22)]
              + [f"baselines_degmatch_vis00_s{s}" for s in (20, 21, 22)])
    try:
        out = assert_control_baselines_exist(res)
        print(f"   seeds 20/21/22, every control block PRESENT -> {out}")
    except ValueError as exc:
        print(f"   seeds 20/21/22, every control block PRESENT -> FALSE ALARM: "
              f"{str(exc)[:120]}")
    print("   (merge_p2_scorer_outputs passes `seeds` to assert_baselines_agree_across_seeds "
          "but NOT to assert_control_baselines_exist -- preregistration.py:1149-1151)")


def part3() -> None:
    print("\n3. ONE-OF-SIXTEEN: a control family present for seed 0 only")
    files = []
    for s in (0, 1, 2):
        files.append(_mk([f"vis00_s{s}_ms", f"vis00_s{s}_roll",
                          f"vis50_s{s}_ms", f"vis50_s{s}_roll"], [f"baselines_s{s}"]))
        for arm in ("degmatch_vis00", "degmatch_vis50"):
            bl = [f"baselines_{arm}_s{s}"] if s == 0 else []     # seeds 1 and 2 never arrived
            files.append(_mk([f"{arm}_s{s}_ms", f"{arm}_s{s}_roll"], bl))
    try:
        merged = merge_p2_scorer_outputs(files)
        print(f"   merge SUCCEEDS. _control_baselines_present = "
              f"{merged['_control_baselines_present']}")
        print(f"   baselines keys actually present: "
              f"{sorted(k for k in merged if k.startswith('baselines'))}")
        print("   => 4 of the 6 control baselines families are MISSING and the merge is silent;")
        print("      the failure is deferred to _p2_matched_controls, three steps later.")
    except ValueError as exc:
        print(f"   merge RAISES: {str(exc)[:160]}")


def part4() -> None:
    print("\n4. the REAL-TREE family: is `baselines_s{seed}` existence asserted?")
    files = []
    for s in (0, 1, 2):
        bl = [f"baselines_s{s}"] if s != 2 else []      # seed 2's real file never arrived
        files.append(_mk([f"vis00_s{s}_ms", f"vis00_s{s}_roll",
                          f"vis50_s{s}_ms", f"vis50_s{s}_roll"], bl))
        for arm in ("degmatch_vis00", "degmatch_vis50"):
            files.append(_mk([f"{arm}_s{s}_ms", f"{arm}_s{s}_roll"],
                             [f"baselines_{arm}_s{s}"]))
    # plus one legacy bare "baselines" block, e.g. one re-score run without --baselines-key
    files[0]["baselines"] = files[0]["baselines_s0"]
    merged = merge_p2_scorer_outputs(files)
    print(f"   merge SUCCEEDS with baselines_s2 ABSENT and a legacy bare `baselines` present.")
    print(f"   keys: {sorted(k for k in merged if k.startswith('baselines'))}")
    from taxembed.eval.preregistration import _p2_baselines_for_seed
    got = _p2_baselines_for_seed(merged, ["baselines"], 2)
    print(f"   seed 2 resolves to the bare legacy block: {got is merged['baselines']}")
    print("   => seed 2's gate is computed against seed 0's floor, silently -- the exact defect")
    print("      the Part-2 seed-tagging fix exists to prevent, still reachable when ONE file")
    print("      of the nine is produced without --baselines-key.")


if __name__ == "__main__":
    part1()
    part2()
    part3()
    part4()
