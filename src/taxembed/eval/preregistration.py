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


def validity_gate(result: dict, arm: str, seed: int, amendment_2: bool = False) -> dict:
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
        # amendment 2 (2026-09-24): the frozen rule required the loss to FALL, which a CONVERGED
        # arm cannot do -- and converged vs never-trained are opposite situations with an identical
        # measurement. The amended rule only requires that the loss did not materially RISE, so a
        # diverging arm still fails. Gate (a) is what establishes that training happened.
        gate_b = bool(loss_drop > -loss_jitter) if amendment_2 else bool(loss_drop > loss_jitter)
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


# The n>=500 qualifier attaches to CLADE strata only, not to depth bands.
#
# "the canonical-minus-prior sign is positive in every depth band and every clade stratum with
# n >= 500 queries" is ambiguous read alone, but score_embedding() settles it: its
# min_stratum_n=500 filters `by_clade` and never touches `by_band`, which always carries all
# three bands. Applying the threshold to bands as well would silently drop `shallow`, which is
# n=327 on the real metazoa query set -- weakening the sign condition from 5 strata to 4 and
# making a verdict easier to reach than the pre-registration allows.
MIN_N_BY_FIELD = {"S_angle_by_band": 0, "S_angle_by_clade": MIN_STRATUM_N}


def _stratum_means(result: dict, arm: str, seeds, field: str) -> dict:
    """Arm-level mean of a stratum's S, averaged over the rolling window then over seeds.

    Strata below this field's minimum are dropped and reported, never silently skipped: the sign
    rule binds on strata the pre-registration admits, and which ones those are is part of the
    verdict.
    """
    min_n = MIN_N_BY_FIELD.get(field, MIN_STRATUM_N)
    per_stratum: dict[str, list] = {}
    dropped: dict[str, int] = {}
    for seed in seeds:
        roll = result["runs"][_roll_key(arm, seed)]["checkpoints"]
        acc: dict[str, list] = {}
        for c in roll:
            for name, d in c[field].items():
                if d is None or d.get("S") is None:
                    continue
                if int(d.get("n", 0)) < min_n:
                    dropped[str(name)] = int(d.get("n", 0))
                    continue
                acc.setdefault(str(name), []).append(float(d["S"]))
        for name, vals in acc.items():
            per_stratum.setdefault(name, []).append(float(np.mean(vals)))
    return {
        "means": {k: float(np.mean(v)) for k, v in per_stratum.items() if len(v) == len(seeds)},
        "dropped_below_min_n": dropped,
    }


def compare_arms(result: dict, arm_a: str, arm_b: str, seeds=(0, 1, 2), amendment_2: bool = False) -> dict:
    """The shared engine. `arm_a` is the arm a positive difference favours."""
    gates_a = [validity_gate(result, arm_a, s, amendment_2) for s in seeds]
    gates_b = [validity_gate(result, arm_b, s, amendment_2) for s in seeds]
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


def task9_verdict(result: dict, seeds=(0, 1, 2), amendment_2: bool = False) -> dict:
    """canonical vs prior -> pre-registered reading 1 / 2 / 3, MIXED, or UNINFORMATIVE."""
    cmp = compare_arms(result, "canonical", "prior", seeds, amendment_2)
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


## ---------------------------------------------------------------------------------------------
## P2: held-out link prediction, real taxonomy vs RandomDAG (results/p2_heldout_preregistration.json)
##
## P2's gates are DIFFERENT IN KIND from task8/9's, and deliberately so (see the JSON's
## `validity_gate` for the full reasoning): there is NO training-loss gate here at all. The
## engine above (`validity_gate`, `compare_arms`) exists to gate a training-loss proxy that failed
## twice for two different reasons -- Task 9 could not tell a converged arm from a never-trained
## one on flat loss, and amendment 2's fix was then blind to Task 8's fixed arm, whose loss ROSE
## while its S_angle simultaneously rose (the model was improving). P2 gates directly on the
## quantity of interest (MRR) and on a fact about the run (did epoch 200 get scored), never on
## loss, so it needs its own gate and comparison engine below rather than reusing `validity_gate`.
##
## Consumes the Task 6 JSON (scripts/score_p2_linkpred.py): `result["arms"][f"{group}_s{seed}"]
## ["checkpoints"]` is a list of per-checkpoint dicts (epoch, trainer.loss, metrics/metrics_cosine
## with an "mrr" field, by_depth["cosine"][stratum] with "n"/"mrr"), and `result["baselines"]
## ["sibling_chance_mean"]` is the chance floor for the REAL-tree arms (vis00, vis50). The
## RandomDAG arm rewires parent assignments (Task 5), which changes grandparent fan-out and
## therefore CAN change its own sibling-chance floor even though depth is preserved exactly -- if
## RandomDAG was scored against its own randomised manifest in a separate invocation, its own
## floor belongs under `result["baselines_randomdag"]["sibling_chance_mean"]`; when that key is
## absent, `p2_group_stats` falls back to the shared `baselines` block. That fallback is a stated
## simplifying assumption, not a silent one -- flag it if RandomDAG's real floor differs.

P2_ROLL_WINDOW = 5                # trailing scored checkpoints define the per-run value + its jitter
P2_FINAL_EPOCH = 200              # the declared final checkpoint; gate (c) requires it be present
P2_GATE_A_JITTER_MULTIPLE = 10    # max_t MRR(t) - MRR(init) > 10 x within-run jitter SD
P2_FLOOR_SD_MULTIPLE = 2          # "above sibling_chance_mean by >= 2x pooled within-arm seed SD"
P2_EQUIV_FLOOR = 0.01             # equivalence-vs-RandomDAG band floor, mirrors EQUIV_FLOOR above
P2_MIN_STRATUM_N = 500
P2_DATA_ARMS = ("vis00", "vis50")
P2_CONTROL_ARM = "randomdag"


def _p2_checkpoints(result: dict, arm_seed_key: str) -> list[dict]:
    ckpts = result["arms"][arm_seed_key]["checkpoints"]
    return sorted(ckpts, key=lambda c: int(c["epoch"]))


def _p2_mrr(ckpt: dict) -> float:
    """The primary (cosine) MRR for one checkpoint.

    Prefers the explicit `metrics_cosine` block over the un-suffixed `metrics` alias so this
    reads correctly even if a JSON was produced with `--metric poincare` by mistake (Ruling 3:
    P2's primary metric is cosine BY DESIGN, not by whatever flag the invocation happened to use --
    see the candidate-pool depth-homogeneity argument in the pre-registration's `metrics.
    primary_reason`).
    """
    block = ckpt.get("metrics_cosine", ckpt["metrics"])
    return float(block["mrr"])


def p2_run_value(result: dict, arm_seed_key: str) -> dict:
    """One run's MRR trajectory, its per-run value, and its within-run jitter.

    `mrr_init` is the primary MRR at the EARLIEST scored checkpoint in this run's own trajectory.
    P2's scorer (unlike the S_angle engine's `init_null`) records no separate untrained-embedding
    baseline, so the run's own first checkpoint stands in for it -- gate (a) below is defined on
    the run's OWN trajectory for exactly this reason.

    `per_run_value` is the mean MRR over the trailing `P2_ROLL_WINDOW` scored checkpoints (mirrors
    the S_angle rolling-window convention), with `jitter_sd` its SD -- the noise floor gate (a)
    measures a rise against.
    """
    ckpts = _p2_checkpoints(result, arm_seed_key)
    if not ckpts:
        raise ValueError(f"{arm_seed_key}: no checkpoints scored")
    epochs = [int(c["epoch"]) for c in ckpts]
    mrrs = np.array([_p2_mrr(c) for c in ckpts], dtype=float)
    roll = mrrs[-P2_ROLL_WINDOW:]
    return {
        "arm": arm_seed_key,
        "n_checkpoints": len(ckpts),
        "epochs": epochs,
        "mrr_series": mrrs.tolist(),
        "mrr_init": float(mrrs[0]),
        "mrr_max": float(mrrs.max()),
        "per_run_value": float(roll.mean()),
        "jitter_sd": float(roll.std(ddof=1)) if len(roll) > 1 else float("nan"),
        "final_epoch_present": bool(P2_FINAL_EPOCH in epochs),
    }


def p2_validity_gate(result: dict, arm_seed_key: str, sibling_chance_mean: float) -> dict:
    """(a) learning, on MRR's own trajectory; (b) floor, vs sibling chance; (c) completion, epoch
    200 scored. NO loss gate -- see the module-level comment above and the JSON's `validity_gate`.

    Gate (a) explicitly FAILS on a zero or non-finite jitter SD rather than letting it collapse
    the `10 x jitter` threshold to zero and pass any positive rise: a real run always jitters, and
    a fixture (or a genuinely dead arm) pinned to exactly one value is the degenerate case this
    project's own test suite was once fooled by.
    """
    rv = p2_run_value(result, arm_seed_key)
    rise = rv["mrr_max"] - rv["mrr_init"]
    jitter = rv["jitter_sd"]
    gate_a = bool(np.isfinite(jitter) and jitter > 0.0
                  and rise > P2_GATE_A_JITTER_MULTIPLE * jitter)
    gate_b = bool(rv["per_run_value"] > sibling_chance_mean)
    gate_c = bool(rv["final_epoch_present"])
    return {
        **rv,
        "sibling_chance_mean": float(sibling_chance_mean),
        "gate_a_rise": float(rise),
        "gate_a_threshold": (float(P2_GATE_A_JITTER_MULTIPLE * jitter)
                             if np.isfinite(jitter) else float("nan")),
        "gate_a_pass": gate_a,
        "gate_b_pass": gate_b,
        "gate_c_pass": gate_c,
        "valid": bool(gate_a and gate_b and gate_c),
    }


def p2_group_stats(result: dict, group: str, seeds, sibling_chance_mean: float) -> dict:
    """Gates + per-run values for one arm's seeds (e.g. group='vis00' -> vis00_s0/_s1/_s2)."""
    gates = [p2_validity_gate(result, f"{group}_s{s}", sibling_chance_mean) for s in seeds]
    invalid = [g["arm"] for g in gates if not g["valid"]]
    vals = np.array([g["per_run_value"] for g in gates], dtype=float)
    return {
        "group": group, "seeds": list(seeds), "gates": gates, "invalid_runs": invalid,
        "per_run_values": vals.tolist(),
        "mean": float(vals.mean()), "min": float(vals.min()), "max": float(vals.max()),
        "pooled_within_arm_seed_sd": float(vals.std(ddof=1)) if len(vals) > 1 else float("nan"),
    }


def _p2_depth_means(result: dict, group: str, seeds) -> dict:
    """Per-depth-stratum mean MRR at the FINAL (epoch 200) checkpoint, averaged over seeds.

    Strata whose mean `n` (averaged over seeds) is below `P2_MIN_STRATUM_N` are dropped and
    reported, never silently skipped -- mirrors `_stratum_means` above.
    """
    per_stratum: dict[str, list] = {}
    ns: dict[str, list] = {}
    for s in seeds:
        ckpts = _p2_checkpoints(result, f"{group}_s{s}")
        final = next((c for c in ckpts if int(c["epoch"]) == P2_FINAL_EPOCH), None)
        if final is None:
            continue
        for stratum, d in final.get("by_depth", {}).get("cosine", {}).items():
            if d is None or d.get("mrr") is None:
                continue
            per_stratum.setdefault(stratum, []).append(float(d["mrr"]))
            ns.setdefault(stratum, []).append(int(d.get("n", 0)))
    means, dropped = {}, {}
    for stratum, vals in per_stratum.items():
        mean_n = float(np.mean(ns[stratum]))
        if mean_n < P2_MIN_STRATUM_N:
            dropped[stratum] = mean_n
            continue
        means[stratum] = float(np.mean(vals))
    return {"means": means, "dropped_below_min_n": dropped}


def _p2_arm_reading(result: dict, name: str, seeds, group: dict, control: dict,
                    sibling_chance_mean: float) -> dict:
    """GENERALISES / MEMORISES / MIXED for one real-tree arm against sibling chance + RandomDAG."""
    margin = group["min"] - sibling_chance_mean
    pooled_sd = group["pooled_within_arm_seed_sd"]
    threshold = P2_FLOOR_SD_MULTIPLE * pooled_sd if np.isfinite(pooled_sd) else float("nan")
    above_chance_margin = bool(np.isfinite(threshold) and margin >= threshold)
    above_control_all_seeds = bool(group["min"] > control["max"])

    ranges_overlap = bool(group["min"] <= control["max"] and control["min"] <= group["max"])
    g_var = np.var(group["per_run_values"], ddof=1) if len(group["per_run_values"]) > 1 else 0.0
    c_var = np.var(control["per_run_values"], ddof=1) if len(control["per_run_values"]) > 1 else 0.0
    pooled_vs_control = float(np.sqrt(np.mean([g_var, c_var])))
    equiv_band = max(P2_EQUIV_FLOOR, P2_FLOOR_SD_MULTIPLE * pooled_vs_control)
    mean_diff = group["mean"] - control["mean"]
    equivalent_to_control = bool(ranges_overlap or abs(mean_diff) < equiv_band)

    depth = _p2_depth_means(result, name, seeds)
    depth_control = _p2_depth_means(result, P2_CONTROL_ARM, seeds)
    shared = sorted(set(depth["means"]) & set(depth_control["means"]))
    diffs = {k: depth["means"][k] - depth_control["means"][k] for k in shared}
    sign_consistent = bool(diffs) and all(v > 0 for v in diffs.values())

    if above_chance_margin and above_control_all_seeds and sign_consistent:
        verdict, meaning = "GENERALISES", (
            "MRR clears sibling chance by the declared margin, beats every RandomDAG seed, and "
            "the sign holds in every depth stratum with n>=500: the model predicts relations it "
            "never saw. This is the answer to the overfitting challenge.")
    elif (not above_chance_margin) or equivalent_to_control:
        verdict, meaning = "MEMORISES", (
            "MRR is not meaningfully separated from sibling chance, or is statistically "
            "indistinguishable from the RandomDAG control. The structure is in-sample only; the "
            "manuscript must say so.")
    else:
        verdict, meaning = "MIXED", "Sign flips across depth strata; report per stratum, no aggregate claim."

    return {
        "arm": name, "verdict": verdict, "meaning": meaning,
        "margin_above_chance": float(margin), "margin_threshold": float(threshold),
        "above_chance_margin": above_chance_margin,
        "above_control_all_seeds": above_control_all_seeds,
        "equivalent_to_control": equivalent_to_control,
        "mean_diff_vs_control": float(mean_diff), "equivalence_band_vs_control": equiv_band,
        "depth_strata": {"diffs": diffs, "n_strata": len(shared),
                         "dropped_below_min_n": {**depth["dropped_below_min_n"],
                                                  **depth_control["dropped_below_min_n"]}},
        "sign_consistent": sign_consistent,
    }


def p2_verdict(result: dict, seeds=(0, 1, 2)) -> dict:
    """vis00/vis50 vs sibling chance and RandomDAG -> GENERALISES / MEMORISES / MIXED /
    UNINFORMATIVE, per results/p2_heldout_preregistration.json.

    UNINFORMATIVE beats every other reading: if ANY seed of ANY arm (including RandomDAG) fails
    gate (a)/(b)/(c), the whole verdict is UNINFORMATIVE and neither data arm is read. Otherwise
    each of vis00/vis50 gets its own GENERALISES/MEMORISES/MIXED reading (`by_arm_verdict`); the
    top-level `verdict` is that shared reading if both arms agree, else MIXED.
    """
    sibling_chance_mean = float(result["baselines"]["sibling_chance_mean"])
    randomdag_baselines = result.get("baselines_randomdag", result["baselines"])
    randomdag_chance = float(randomdag_baselines["sibling_chance_mean"])

    control = p2_group_stats(result, P2_CONTROL_ARM, seeds, randomdag_chance)
    groups = {g: p2_group_stats(result, g, seeds, sibling_chance_mean) for g in P2_DATA_ARMS}

    invalid = list(control["invalid_runs"])
    for g in groups.values():
        invalid += g["invalid_runs"]

    base = {
        "task": "P2 -- held-out link prediction, real taxonomy vs RandomDAG",
        "preregistration": ["results/p2_heldout_preregistration.json"],
        "seeds": list(seeds),
        "sibling_chance_mean": sibling_chance_mean,
        "randomdag_sibling_chance_mean": randomdag_chance,
        "control": control, "groups": groups,
    }

    if invalid:
        base["verdict"] = "UNINFORMATIVE"
        base["meaning"] = ("A validity gate failed; no direction is read for either arm. "
                           f"Invalid runs: {', '.join(invalid)}")
        return base

    by_arm = {name: _p2_arm_reading(result, name, seeds, groups[name], control,
                                    sibling_chance_mean)
             for name in P2_DATA_ARMS}
    verdicts = {v["verdict"] for v in by_arm.values()}
    if len(verdicts) == 1:
        overall = verdicts.pop()
    else:
        overall = "MIXED"

    base["by_arm_verdict"] = by_arm
    base["verdict"] = overall
    base["meaning"] = "; ".join(f"{name}: {by_arm[name]['meaning']}" for name in P2_DATA_ARMS)
    return base


def task8_verdict(result: dict, seeds=(0, 1, 2), amendment_2: bool = False) -> dict:
    """fixed vs unfixed sampler -> MATERIAL / ROBUST / MIXED, or UNINFORMATIVE."""
    cmp = compare_arms(result, "fixed", "unfixed", seeds, amendment_2)
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
