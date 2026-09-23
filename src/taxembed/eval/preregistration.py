"""Apply the Task 8 / Task 9 pre-registrations to a scorer output, mechanically.

WHY THIS IS CODE AND NOT A JUDGEMENT CALL. Both pre-registrations were frozen before their
arrays ran (`results/recipe_angular_comparison.json` -> `preregistration_v2_20260922` plus
`preregistration_v2_amendment_1_20260922`; `results/objective_integrity_delta_preregistration.json`).
A frozen rule that is applied by hand, after the numbers are visible, is a rule with a degree of
freedom left in it. This module was written while array 5802007 was still running and no arm's
S_angle had been seen, so the mapping from numbers to verdict could not be tuned to the numbers.

Both tasks share one comparison engine because their mechanics are identical -- three seeds per
arm, per-run value = mean over the rolling epoch 196-200 checkpoints, an all-above-all seed rule,
an equivalence band, and a per-stratum sign condition. Only the verdict LABELS differ
(Task 9: reading 1 / 2 / 3; Task 8: MATERIAL / ROBUST / MIXED).

The verdict is UNINFORMATIVE if either arm fails its validity gate -- never a pass, never a fail.
That gate exists because the echinodermata run mapped "the canonical arm never trained" onto
"prior beats canonical" by default.
"""
from __future__ import annotations

import numpy as np

FINAL_PHASE_MIN_EPOCH = 130     # "loss is compared only within the final curriculum phase"
GATE_A_JITTER_MULTIPLE = 10     # max_t S_angle(t) - S_angle(init) > 10 x within-run jitter SD
EQUIV_FLOOR = 0.01              # |mean difference| < max(0.01, 2 x pooled within-arm seed SD)
EQUIV_SD_MULTIPLE = 2
MIN_STRATUM_N = 500


def _roll_key(arm: str, seed: int) -> str:
    return f"{arm}_s{seed}_roll"


def _ms_key(arm: str, seed: int) -> str:
    return f"{arm}_s{seed}_ms"


def run_value(result: dict, arm: str, seed: int) -> dict:
    """The per-run value and its within-run jitter, exactly as pre-registered.

    "per_run_value": mean of S_angle over the rolling epoch 196-200 checkpoints (annealing noise
    averaged, within-run jitter reported as their SD).
    """
    roll = result["runs"][_roll_key(arm, seed)]["checkpoints"]
    ms = result["runs"][_ms_key(arm, seed)]["checkpoints"]
    s = np.array([c["S_angle"] for c in roll], dtype=float)
    auc = np.array([c["level_auc_mean"] for c in roll], dtype=float)
    poi = np.array([c["S_poincare"] for c in roll], dtype=float)
    return {
        "arm": arm, "seed": seed,
        "n_roll": len(roll), "n_milestones": len(ms),
        "roll_epochs": [c["epoch"] for c in roll],
        "S_angle": float(s.mean()),
        "S_angle_jitter_sd": float(s.std(ddof=1)) if len(s) > 1 else float("nan"),
        "level_auc": float(auc.mean()),
        "S_poincare": float(poi.mean()),
        "flag_poincare_above_angle": bool(poi.mean() > s.mean()),
        "milestone_S_angle_max": float(max(c["S_angle"] for c in ms)),
    }


def validity_gate(result: dict, arm: str, seed: int) -> dict:
    """(a) training happened, on S_angle; (b) final-phase loss fell. Either failing => UNINFORMATIVE.

    (a) deliberately uses max over milestones, not the final value: a rise-then-collapse
    trajectory passes. Monotonicity is NOT required.
    """
    rv = run_value(result, arm, seed)
    roll = result["runs"][_roll_key(arm, seed)]["checkpoints"]
    ms = result["runs"][_ms_key(arm, seed)]["checkpoints"]
    init_s = float(result["init_null"]["S_angle"])

    jitter = rv["S_angle_jitter_sd"]
    rise = rv["milestone_S_angle_max"] - init_s
    gate_a = bool(np.isfinite(jitter) and rise > GATE_A_JITTER_MULTIPLE * jitter)

    final = [c for c in ms if c["epoch"] >= FINAL_PHASE_MIN_EPOCH
             and c["trainer"].get("loss") is not None]
    losses = [float(c["trainer"]["loss"]) for c in final]
    roll_losses = [float(c["trainer"]["loss"]) for c in roll
                   if c["trainer"].get("loss") is not None]
    loss_jitter = float(np.std(roll_losses, ddof=1)) if len(roll_losses) > 1 else float("nan")
    if len(losses) >= 2 and np.isfinite(loss_jitter):
        loss_drop = losses[0] - losses[-1]
        gate_b = bool(loss_drop > loss_jitter)
    else:
        loss_drop, gate_b = float("nan"), False

    return {
        **rv,
        "gate_a_rise": float(rise), "gate_a_threshold": float(GATE_A_JITTER_MULTIPLE * jitter),
        "gate_a_pass": gate_a,
        "gate_b_final_phase_epochs": [c["epoch"] for c in final],
        "gate_b_loss_drop": float(loss_drop), "gate_b_loss_jitter": loss_jitter,
        "gate_b_pass": gate_b,
        "valid": bool(gate_a and gate_b),
    }


def _stratum_means(result: dict, arm: str, seeds, field: str) -> dict:
    """Arm-level mean of a stratum's S, averaged over the rolling window then over seeds.

    Strata below MIN_STRATUM_N are dropped and reported, never silently skipped: the sign rule
    binds on strata the pre-registration admits, and which ones those are is part of the verdict.
    """
    per_stratum: dict[str, list] = {}
    dropped: dict[str, int] = {}
    for seed in seeds:
        roll = result["runs"][_roll_key(arm, seed)]["checkpoints"]
        acc: dict[str, list] = {}
        for c in roll:
            for name, d in c[field].items():
                if d is None or d.get("S") is None:
                    continue
                if int(d.get("n", 0)) < MIN_STRATUM_N:
                    dropped[str(name)] = int(d.get("n", 0))
                    continue
                acc.setdefault(str(name), []).append(float(d["S"]))
        for name, vals in acc.items():
            per_stratum.setdefault(name, []).append(float(np.mean(vals)))
    return {
        "means": {k: float(np.mean(v)) for k, v in per_stratum.items() if len(v) == len(seeds)},
        "dropped_below_min_n": dropped,
    }


def compare_arms(result: dict, arm_a: str, arm_b: str, seeds=(0, 1, 2)) -> dict:
    """The shared engine. `arm_a` is the arm a positive difference favours."""
    gates_a = [validity_gate(result, arm_a, s) for s in seeds]
    gates_b = [validity_gate(result, arm_b, s) for s in seeds]
    invalid = [f"{g['arm']}_s{g['seed']}" for g in gates_a + gates_b if not g["valid"]]

    a_s = np.array([g["S_angle"] for g in gates_a])
    b_s = np.array([g["S_angle"] for g in gates_b])
    a_auc = np.array([g["level_auc"] for g in gates_a])
    b_auc = np.array([g["level_auc"] for g in gates_b])
    a_poi = np.array([g["S_poincare"] for g in gates_a])
    b_poi = np.array([g["S_poincare"] for g in gates_b])

    a_above = bool(a_s.min() > b_s.max())          # all 3 of A above all 3 of B
    b_above = bool(b_s.min() > a_s.max())
    auc_a_above = bool(a_auc.min() > b_auc.max())
    auc_b_above = bool(b_auc.min() > a_auc.max())

    mean_diff = float(a_s.mean() - b_s.mean())
    pooled_sd = float(np.sqrt(np.mean([a_s.var(ddof=1), b_s.var(ddof=1)])))
    equiv_band = max(EQUIV_FLOOR, EQUIV_SD_MULTIPLE * pooled_sd)
    ranges_overlap = bool(a_s.min() <= b_s.max() and b_s.min() <= a_s.max())

    strata = {}
    sign_consistent_pos = True
    sign_consistent_neg = True
    for field in ("S_angle_by_band", "S_angle_by_clade"):
        ma = _stratum_means(result, arm_a, seeds, field)
        mb = _stratum_means(result, arm_b, seeds, field)
        shared = sorted(set(ma["means"]) & set(mb["means"]))
        diffs = {k: ma["means"][k] - mb["means"][k] for k in shared}
        strata[field] = {
            "n_strata": len(shared),
            "diffs": diffs,
            "n_positive": int(sum(v > 0 for v in diffs.values())),
            "n_negative": int(sum(v < 0 for v in diffs.values())),
            "dropped_below_min_n": {**ma["dropped_below_min_n"], **mb["dropped_below_min_n"]},
        }
        if diffs:
            sign_consistent_pos &= all(v > 0 for v in diffs.values())
            sign_consistent_neg &= all(v < 0 for v in diffs.values())
        else:
            sign_consistent_pos = sign_consistent_neg = False

    a_wins = bool(a_above and a_poi.mean() >= b_poi.mean() and sign_consistent_pos and auc_a_above)
    b_wins = bool(b_above and b_poi.mean() >= a_poi.mean() and sign_consistent_neg and auc_b_above)
    equivalent = bool(ranges_overlap and abs(mean_diff) < equiv_band)

    return {
        "arm_a": arm_a, "arm_b": arm_b, "seeds": list(seeds),
        "gates": {"a": gates_a, "b": gates_b, "invalid_runs": invalid},
        "S_angle": {"a": a_s.tolist(), "b": b_s.tolist(),
                    "a_mean": float(a_s.mean()), "b_mean": float(b_s.mean()),
                    "mean_diff_a_minus_b": mean_diff,
                    "all_a_above_all_b": a_above, "all_b_above_all_a": b_above,
                    "ranges_overlap": ranges_overlap},
        "level_auc": {"a": a_auc.tolist(), "b": b_auc.tolist(),
                      "all_a_above_all_b": auc_a_above, "all_b_above_all_a": auc_b_above},
        "S_poincare": {"a_mean": float(a_poi.mean()), "b_mean": float(b_poi.mean())},
        "equivalence": {"pooled_within_arm_seed_sd": pooled_sd, "band": equiv_band,
                        "within_band": bool(abs(mean_diff) < equiv_band)},
        "strata": strata,
        "sign_consistent_positive": sign_consistent_pos,
        "sign_consistent_negative": sign_consistent_neg,
        "flags_poincare_above_angle": [f"{g['arm']}_s{g['seed']}"
                                       for g in gates_a + gates_b
                                       if g["flag_poincare_above_angle"]],
        "_a_wins": a_wins, "_b_wins": b_wins, "_equivalent": equivalent,
        "_uninformative": bool(invalid),
    }


def task9_verdict(result: dict, seeds=(0, 1, 2)) -> dict:
    """canonical vs prior -> pre-registered reading 1 / 2 / 3, MIXED, or UNINFORMATIVE."""
    cmp = compare_arms(result, "canonical", "prior", seeds)
    if cmp["_uninformative"]:
        verdict, meaning = "UNINFORMATIVE", (
            "A validity gate failed; the contrast is not read in either direction. "
            f"Invalid runs: {', '.join(cmp['gates']['invalid_runs'])}")
    elif cmp["_a_wins"]:
        verdict, meaning = "READING_1_CANONICAL_BETTER", (
            "Figure 4's claim HOLDS on learned angular structure, and the figure should be "
            "RE-PLOTTED on a learned metric because its current y-axis measures the planted axis.")
    elif cmp["_b_wins"]:
        verdict, meaning = "READING_3_PRIOR_BETTER", (
            "ESCALATE. The recipe claim does not survive; the manuscript needs restructuring.")
    elif cmp["_equivalent"]:
        verdict, meaning = "READING_2_EQUIVALENT", (
            "The recipe's demonstrated benefit is PRESERVATION OF THE RADIAL PRIOR, not "
            "acquisition of structure. Figure 4's caption and the 'collapse' language need "
            "rewriting.")
    else:
        verdict, meaning = "MIXED", (
            "Reported per stratum and per metric; no aggregate claim.")
    return {
        "task": "plan v2 Task 9 -- Figure 4 recipe claim on the radius-free metric",
        "preregistration": ["results/recipe_angular_comparison.json :: preregistration_v2_20260922",
                            "results/recipe_angular_comparison.json :: "
                            "preregistration_v2_amendment_1_20260922"],
        "verdict": verdict, "meaning": meaning,
        "attribution": ("Any difference is attributed to the 4-factor package (effective batch, "
                        "n_negatives, lr, lr schedule), never to the batch lever alone."),
        "comparison": cmp,
    }


def task8_verdict(result: dict, seeds=(0, 1, 2)) -> dict:
    """fixed vs unfixed sampler -> MATERIAL / ROBUST / MIXED, or UNINFORMATIVE."""
    cmp = compare_arms(result, "fixed", "unfixed", seeds)
    material = cmp["_a_wins"] or cmp["_b_wins"]
    if cmp["_uninformative"]:
        verdict, meaning = "UNINFORMATIVE", (
            f"A validity gate failed. Invalid runs: {', '.join(cmp['gates']['invalid_runs'])}")
    elif material and abs(cmp["S_angle"]["mean_diff_a_minus_b"]) >= cmp["equivalence"]["band"]:
        direction = "fixed > unfixed" if cmp["_a_wins"] else "fixed < unfixed"
        verdict = "MATERIAL"
        meaning = (f"The sampler fix materially changes learned structure ({direction}). "
                   + ("The shipped artifact under-learned because of the defect; Methods must say "
                      "so and a cellular retrain with the fix is scoped."
                      if cmp["_a_wins"] else
                      "Report it; the false negatives acted as a regularizer, or the "
                      "depth-relaxed negatives are too easy -- investigate before any retrain."))
    elif cmp["_equivalent"] or cmp["S_angle"]["ranges_overlap"]:
        verdict, meaning = "ROBUST", (
            "The defect does not materially change learned structure at this scale. Methods "
            "still states the deviation (C1) and reports this as a robustness result.")
    else:
        verdict, meaning = "MIXED", "Reported per stratum; no aggregate claim."
    return {
        "task": "plan v2 Task 8 -- does the ancestry-aware sampler change what the recipe learns?",
        "preregistration": ["results/objective_integrity_delta_preregistration.json"],
        "verdict": verdict, "meaning": meaning,
        "comparison": cmp,
    }
