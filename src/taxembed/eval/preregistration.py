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
P2_FLOOR_SD_MULTIPLE = 2          # "above chance_mrr_mean by >= 2x pooled within-arm seed SD"
P2_EQUIV_FLOOR = 0.01             # equivalence-vs-RandomDAG band floor, mirrors EQUIV_FLOOR above
P2_EQUIV_REL_FRACTION = 0.05      # amendment_6=True: equivalence band as a FRACTION of the control's
# own mean, replacing the absolute MRR-scale `P2_EQUIV_FLOOR` on the normalized_rank reading.
# WHY A FRACTION AT ALL (2026-09-26, `p2_amendment_6_20260926`, C-A's second mechanism): 0.01 was
# calibrated on MRR, whose production values are ~0.8, so it is a ~1.2% band there. Applied
# unchanged to `normalized_rank` at the amendment_1 branch it became ~25% of the quantity compared
# (production normalized_rank ~0.04), and the measured consequence is not hypothetical: in review
# scenario S3 two arms differing by 0.2% read `equivalent_to_control=True` on a |mean_diff| of
# 0.00008 -- an automatic MEMORISES. 5% of the control's own mean reproduces the ~1.2% - 5% order
# the MRR reading always had, on whatever scale the compared quantity actually lives at, and the
# `P2_FLOOR_SD_MULTIPLE * pooled_sd` term is unchanged and still dominates whenever seed noise is
# the larger effect. Chosen for scale-consistency with the existing MRR band BEFORE any P2 array
# ran and before any P2 outcome data existed -- not tuned against a result.
P2_MIN_STRATUM_N = 500
P2_DATA_ARMS = ("vis00", "vis50")
P2_CONTROL_ARM = "randomdag"                     # amendment_2=False (default): one shared control
P2_MATCHED_CONTROL = {                            # amendment_2=True (p2_amendment_2_20260924):
    "vis00": "randomdag_vis00",                   # each real arm reads against its OWN
    "vis50": "randomdag_vis50",                   # visibility-matched control -- never crossed.
}
P2_MATCHED_CONTROL_DEGMATCH = {                   # amendment_4=True (p2_amendment_4_20260924):
    "vis00": "degmatch_vis00",                    # RandomDAG RETIRED as P2's control -- the
    "vis50": "degmatch_vis50",                    # degree-matched shuffle (randomdag.py::
}                                                  # degree_matched_shuffle) replaces it, same
                                                   # matched-pair shape amendment_2 established.

P2_BASELINES_SEED_TOLERANCE = 0.25                # Part 2 fix (2026-09-24, no amendment flag --
# a correctness fix, not a design choice with two legitimate readings, same posture as
# amendment_3's C1-C5): declared relative-spread tolerance `assert_baselines_agree_across_seeds`
# enforces across seeds 0/1/2's OWN sibling_chance_mean / chance_mrr_mean / degree_prior blocks.
# Measured on the real mollusca_6447_clean closure, 3 seeds, visibility-independent (helpers/
# p2_cross_seed_baseline_spread.py): sibling_chance_mean relative spread 1.19%, chance_mrr_mean
# 1.63%, degree_prior.mrr 10.00%, degree_prior.normalized_rank 12.00% (740 held-out nodes/seed --
# the production metazoa split has ~28k, so genuine sampling noise is expected to be considerably
# smaller there, per the usual 1/sqrt(n) scaling). 25% is comfortably above every measured figure
# yet far below what an actual bug produces (RandomDAG's own chance-floor inflation vs the real
# tree measured 440%; the degree-matched control's residual gap, still real after amendment_4,
# measured 84% -- see p2_amendment_4_20260924 below). A tighter tolerance would risk flagging
# ordinary held-out-sample noise on the smaller degree_prior fields as a "disagreement"; a looser
# one would fail to catch a genuinely wrong split/closure landing under the wrong seed's key.


def _p2_roll_key(arm: str, seed: int) -> str:
    """*_roll: rolling checkpoints nearest epoch 200 -- THE per-run value and its within-run
    jitter (mirrors `_roll_key` above). `scripts/p2_lrz_score.sh` already registers arms under
    this suffix; the C4 fix (2026-09-24, `p2_amendment_3_20260924`) is that the engine did not
    look for it and raised KeyError on real scorer output looking up the bare `{arm}_s{seed}`."""
    return f"{arm}_s{seed}_roll"


def _p2_ms_key(arm: str, seed: int) -> str:
    """*_ms: milestone checkpoints (epoch 10..200) -- the trajectory gate (a)'s rise-check and
    gate (c)'s completion-check read (mirrors `_ms_key` above and `scripts/p2_lrz_score.sh`'s own
    comment: "*_ms ... -> completion check (gate c) and the trajectory used for the learning gate
    (gate a)")."""
    return f"{arm}_s{seed}_ms"


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


def _p2_normalized_rank(ckpt: dict) -> float:
    """The primary (cosine) `normalized_rank` for one checkpoint -- `(rank-1)/(pool-1)`.

    amendment_1_20260924 (see the JSON's `p2_amendment_1_20260924` block): RandomDAG's rewiring
    preserves each node's DEPTH but not its FAN-OUT, so its own `sibling_chance_mean` measured
    ~5.4x higher than the real tree's on mollusca_6447_clean seed 0
    (helpers/p2_randomdag_changes_the_chance_floor.py) -- RandomDAG is a different-difficulty
    task, not merely a scrambled one. `normalized_rank`'s chance level is exactly 0.5 for EVERY
    pool size (uniform rank on 1..pool under a random ranking), so it is the quantity every
    CROSS-TREE comparison (an arm vs the RandomDAG control) must use instead of raw MRR. MRR
    remains the metric for WITHIN-tree comparisons (an arm vs its own sibling_chance_mean; vis00
    vs vis50), which this function does not touch.

    Falls back to NaN when the field is absent (fix round 2, 2026-09-24): `p2_run_value` computes
    this for EVERY checkpoint unconditionally, regardless of whether the caller ever asked for an
    amended (cross-tree) reading. Before this fix a JSON lacking `normalized_rank` (e.g. one scored
    before this field existed) raised KeyError even on the plain `amendment_1=False,
    amendment_2=False` frozen path, which never reads it -- coupling the frozen reading's validity
    to a field it does not use. NaN propagates harmlessly into the roll-window mean/SD (both become
    NaN) unless and until an amended reading actually consumes it, at which point a NaN comparison
    is uniformly False -- silent-safe rather than a crash, but still visibly wrong if someone reads
    an amended verdict off data that was never scored for it.
    """
    block = ckpt.get("metrics_cosine", ckpt["metrics"])
    nr = block.get("normalized_rank")
    return float(nr) if nr is not None else float("nan")


def p2_run_value(result: dict, arm: str, seed: int) -> dict:
    """One run's MRR trajectory, its per-run value, and its within-run jitter.

    CORRECTION (2026-09-24, C4 #1, `p2_amendment_3_20260924`). `scripts/p2_lrz_score.sh` registers
    TWO separate checkpoint groups per run, mirroring Task 9's `_roll_key`/`_ms_key` convention:
    `*_ms` (milestone checkpoints, epoch 10..200 every 10) is the trajectory gate (a)'s rise-check
    and gate (c)'s completion-check read; `*_roll` (rolling checkpoints nearest epoch 200) is THE
    per-run value and its jitter. The old single-trajectory version of this function looked up the
    bare `{arm}_s{seed}` key, which the real scorer output never has -- `_p2_roll_key`/`_p2_ms_key`
    now read the keys production actually writes.

    `mrr_init` is the primary MRR at the EARLIEST scored MILESTONE checkpoint (epoch 10) -- kept
    for backward-compatible reporting, but see I3: gate (a)'s rise is now measured against
    `chance_mrr_mean`, not this epoch-10 value, since epoch 10 already reflects some training and
    is not an untrained baseline.

    `per_run_value` is the mean MRR over the trailing `P2_ROLL_WINDOW` ROLLING checkpoints (mirrors
    the S_angle rolling-window convention), with `jitter_sd` its SD -- the noise floor gate (a)
    measures a rise against. C-C (2026-09-26): that sentence was TRUE OF THE DOCSTRING ONLY until
    this date -- the window is now actually applied, and a roll set that is not exactly
    `P2_ROLL_WINDOW` long RAISES. See the comment in the body.

    `roll_epochs` records which epochs the window actually covered, so a reader can see the set
    the published value was computed over instead of inferring it.
    """
    all_roll = _p2_checkpoints(result, _p2_roll_key(arm, seed))
    ms = _p2_checkpoints(result, _p2_ms_key(arm, seed))
    if not all_roll:
        raise ValueError(f"{_p2_roll_key(arm, seed)}: no checkpoints scored")
    if not ms:
        raise ValueError(f"{_p2_ms_key(arm, seed)}: no checkpoints scored")
    # C-C (2026-09-26). `P2_ROLL_WINDOW` was DECLARED, described in this docstring, and applied
    # only by a TEST FIXTURE -- nothing in src/ or scripts/ ever applied it, so `per_run_value`
    # was the mean over EVERY checkpoint the scorer's glob returned
    # (`p2_lrz_score.sh`: --checkpoints "${arm}_s${s}_roll=${dir}/${tag}_epoch*.pth").
    #
    # This is not a tidy-up. The trainer keeps the last 5 rolling checkpoints PER PROCESS and
    # `p2_lrz_train.sh` writes into /app/artifacts/tags/${TAG}/ WITHOUT clearing it, while the
    # array runs at --time=1-00:00:00 against runs the ledger sizes at 11-19 h -- so a resubmitted
    # element is likely, and a first attempt that died at epoch 120 leaves _epoch116..120.pth
    # orphaned beside the second attempt's _epoch196..200.pth. Measured on exactly that 10-element
    # roll set (helpers/p2_review3_roll_window_and_strata.py): published per_run_value 0.68503
    # instead of 0.82002 (-16% on the headline number) and jitter_sd 0.142284 instead of 0.000171,
    # which FAILS gate (a) and makes the WHOLE ARRAY UNINFORMATIVE, since one invalid arm-seed
    # invalidates everything.
    #
    # The count is asserted rather than silently trimmed: taking the trailing 5 of a 10-element
    # set would quietly average the right checkpoints while concealing that the tag directory was
    # never cleared, and a run with the wrong number of rolling checkpoints is a fact worth
    # failing on. The test fixture guaranteed exactly 5, so no test could ever see this.
    roll = all_roll[-P2_ROLL_WINDOW:]
    if len(all_roll) != P2_ROLL_WINDOW:
        raise ValueError(
            f"{_p2_roll_key(arm, seed)}: {len(all_roll)} rolling checkpoints "
            f"(epochs {[int(c['epoch']) for c in all_roll]}), expected exactly "
            f"{P2_ROLL_WINDOW}. More than {P2_ROLL_WINDOW} usually means the tag directory was "
            f"not cleared between training attempts and this run's glob picked up a previous "
            f"attempt's orphaned checkpoints; fewer means the run did not complete its rolling "
            f"window. Either way the per-run value and its jitter would be computed over the "
            f"wrong set -- clear the tag directory and rescore rather than reading this.")
    roll_epochs = [int(c["epoch"]) for c in roll]
    ms_epochs = [int(c["epoch"]) for c in ms]
    roll_mrrs = np.array([_p2_mrr(c) for c in roll], dtype=float)
    roll_nrs = np.array([_p2_normalized_rank(c) for c in roll], dtype=float)
    ms_mrrs = np.array([_p2_mrr(c) for c in ms], dtype=float)
    return {
        "arm": f"{arm}_s{seed}",
        "n_roll": len(roll), "n_milestones": len(ms),
        "roll_epochs": roll_epochs,
        "milestone_epochs": ms_epochs,
        "mrr_series": roll_mrrs.tolist(),
        "mrr_init": float(ms_mrrs[0]),
        "mrr_max": float(ms_mrrs.max()),
        "milestone_mrr_max": float(ms_mrrs.max()),
        "per_run_value": float(roll_mrrs.mean()),
        "jitter_sd": float(roll_mrrs.std(ddof=1)) if len(roll_mrrs) > 1 else float("nan"),
        # amendment_1_20260924: the cross-tree (arm vs RandomDAG) equivalent of the fields
        # above, computed on normalized_rank instead of MRR -- see `_p2_normalized_rank`.
        "normalized_rank_series": roll_nrs.tolist(),
        "per_run_value_normalized_rank": float(roll_nrs.mean()),
        "normalized_rank_jitter_sd": (float(roll_nrs.std(ddof=1)) if len(roll_nrs) > 1
                                      else float("nan")),
        "final_epoch_present": bool(P2_FINAL_EPOCH in ms_epochs),
    }


def p2_validity_gate(result: dict, arm: str, seed: int, sibling_chance_mean: float,
                     chance_mrr_mean: float) -> dict:
    """(a) learning, on MRR's own trajectory; (b) floor, vs chance MRR; (c) completion, epoch
    200 scored. NO loss gate -- see the module-level comment above and the JSON's `validity_gate`.

    CORRECTION (2026-09-24, C2/I3, `p2_amendment_3_20260924`). Both the rise anchor (a) and the
    floor (b) used to be measured against `sibling_chance_mean` (= mean(1/k), the chance rate for
    HITS@1) or against the run's own epoch-10 value -- neither is the right anchor for an
    MRR-valued quantity. `chance_mrr_mean` (= mean(H_k/k), a uniformly-random ranker's own MRR) is
    the correct chance level for MRR (measured 0.27107 vs sibling_chance_mean's 0.13126 on the real
    metazoa split -- a 2.07x gap) and is free once computed, so I3 uses it as gate (a)'s anchor too
    instead of the run's own epoch-10 milestone. `sibling_chance_mean` is still recorded (kept
    under its own name, per C2) for anyone who wants the hits@1-chance reading, but it no longer
    gates anything here.

    Gate (a) explicitly FAILS on a zero or non-finite jitter SD rather than letting it collapse
    the `10 x jitter` threshold to zero and pass any positive rise: a real run always jitters, and
    a fixture (or a genuinely dead arm) pinned to exactly one value is the degenerate case this
    project's own test suite was once fooled by.
    """
    rv = p2_run_value(result, arm, seed)
    rise = rv["milestone_mrr_max"] - chance_mrr_mean
    jitter = rv["jitter_sd"]
    gate_a = bool(np.isfinite(jitter) and jitter > 0.0
                  and rise > P2_GATE_A_JITTER_MULTIPLE * jitter)
    gate_b = bool(rv["per_run_value"] > chance_mrr_mean)
    gate_c = bool(rv["final_epoch_present"])
    return {
        **rv,
        "sibling_chance_mean": float(sibling_chance_mean),
        "chance_mrr_mean": float(chance_mrr_mean),
        "gate_a_rise": float(rise),
        "gate_a_threshold": (float(P2_GATE_A_JITTER_MULTIPLE * jitter)
                             if np.isfinite(jitter) else float("nan")),
        "gate_a_pass": gate_a,
        "gate_b_pass": gate_b,
        "gate_c_pass": gate_c,
        "valid": bool(gate_a and gate_b and gate_c),
    }


def _p2_baselines_for_seed(result: dict, prefixes: list[str], seed: int) -> dict:
    """Resolve ONE seed's own baselines block via a fallback chain of prefixes (Part 2 fix,
    2026-09-24 -- a correctness fix, not a design choice with two legitimate readings, same
    posture as amendment_3's C1-C5).

    Each seed draws its own held-out node set (`build_p2_split.py`/`build_p2_degmatch_split.py`
    write one manifest/heldout PER SEED), so `sibling_chance_mean`/`chance_mrr_mean`/
    `degree_prior` are themselves PER-SEED quantities, not one shared value silently taken from
    seed 0 and reused for seeds 1/2's runs. For each `prefix` in order, tries the SEED-SPECIFIC
    key (`{prefix}_s{seed}`) first, then the bare `{prefix}` key, before moving to the next
    prefix -- extending the fallback chain amendment_1/amendment_2 already established
    (`baselines_<control>` -> `baselines_randomdag` -> `baselines`) by one more link at the front
    of each stage. A caller passing `prefixes=["baselines"]` (the real-tree pair) therefore checks
    `baselines_s{seed}` then `baselines`; a caller passing
    `prefixes=["baselines_degmatch_vis00", "baselines_randomdag", "baselines"]` (a matched
    control under amendment_4) checks `baselines_degmatch_vis00_s{seed}`,
    `baselines_degmatch_vis00`, `baselines_randomdag_s{seed}`, `baselines_randomdag`,
    `baselines_s{seed}`, `baselines` in that order. On a scorer output that predates Part 2 (no
    seed-tagged key exists anywhere in the chain), every seed resolves to the SAME bare-prefix
    dict -- reproducing the pre-Part-2 single-shared-baseline behaviour exactly, not a new one.
    """
    for prefix in prefixes:
        seed_key = f"{prefix}_s{seed}"
        if seed_key in result:
            return result[seed_key]
        if prefix in result:
            return result[prefix]
    raise KeyError(f"none of {[f'{p}(_s{seed})' for p in prefixes]} present in scorer output")


def p2_group_stats(result: dict, group: str, seeds, baselines_by_seed: dict) -> dict:
    """Gates + per-run values for one arm's seeds (e.g. group='vis00' -> vis00_s0/_s1/_s2).

    Carries BOTH the MRR aggregates (unchanged, used for the own-tree floor margin and for the
    pre-amendment_1 reading) and the normalized_rank aggregates (amendment_1_20260924, used for
    the cross-tree-vs-RandomDAG reading) -- so both `_p2_arm_reading` branches can be computed
    from the same `p2_group_stats` call without re-scanning the checkpoints.

    `baselines_by_seed` (2026-09-24, Part 2 fix -- signature change from a single shared
    `baselines: dict`, C1/C2's original `p2_amendment_3_20260924` shape) maps EACH seed to ITS
    OWN baselines block: `{seed: {"sibling_chance_mean": ..., "chance_mrr_mean": ...,
    "degree_prior": {...}}}`, resolved per seed by `_p2_baselines_for_seed` (callers below always
    go through it). Every seed's OWN gate (`p2_validity_gate`) is computed against ITS OWN
    dict -- never seed 0's value reused for seed 1/2's runs, the defect Part 2 fixes. Required
    directly (`baselines_by_seed[s][k]`, never `.get(k, default)`): every scorer invocation now
    always writes `sibling_chance_mean`, `chance_mrr_mean` and `degree_prior`, so a KeyError here
    means the JSON predates the C1/C2 fix and must be rescored, not silently patched over with a
    guessed value.

    The GROUP-level summary fields (`degree_prior`, `chance_mrr_mean`, `sibling_chance_mean`
    returned below) come from `seeds[0]`'s own dict, never an average -- deliberately, so that on
    a pre-Part-2 fixture (where every seed's dict is the SAME object via the fallback above) this
    function's output is BYTE-IDENTICAL to the pre-Part-2 implementation's, with no floating-point
    drift from averaging. This is safe specifically because `p2_verdict` calls
    `assert_baselines_agree_across_seeds` on the merged scorer output before any gate here is
    read -- by the time this executes, seed 0's value is already known to agree with every other
    seed's within the declared tolerance, so which one is DISPLAYED is cosmetic. What is NOT
    cosmetic -- which value GATES each seed's own run -- already uses that seed's own dict, above.
    """
    gates = [p2_validity_gate(result, group, s,
                              float(baselines_by_seed[s]["sibling_chance_mean"]),
                              float(baselines_by_seed[s]["chance_mrr_mean"]))
            for s in seeds]
    invalid = [g["arm"] for g in gates if not g["valid"]]
    vals = np.array([g["per_run_value"] for g in gates], dtype=float)
    vals_nr = np.array([g["per_run_value_normalized_rank"] for g in gates], dtype=float)
    first = baselines_by_seed[seeds[0]]
    return {
        "group": group, "seeds": list(seeds), "gates": gates, "invalid_runs": invalid,
        "per_run_values": vals.tolist(),
        "mean": float(vals.mean()), "min": float(vals.min()), "max": float(vals.max()),
        "pooled_within_arm_seed_sd": float(vals.std(ddof=1)) if len(vals) > 1 else float("nan"),
        "per_run_values_normalized_rank": vals_nr.tolist(),
        "mean_normalized_rank": float(vals_nr.mean()),
        "min_normalized_rank": float(vals_nr.min()),
        "max_normalized_rank": float(vals_nr.max()),
        "pooled_within_arm_seed_sd_normalized_rank": (
            float(vals_nr.std(ddof=1)) if len(vals_nr) > 1 else float("nan")),
        "degree_prior": first["degree_prior"],
        "chance_mrr_mean": float(first["chance_mrr_mean"]),
        "sibling_chance_mean": float(first["sibling_chance_mean"]),
        "baselines_by_seed": {s: baselines_by_seed[s] for s in seeds},
    }


def _p2_depth_means(result: dict, group: str, seeds, field: str = "mrr") -> dict:
    """Per-depth-stratum mean of `field` ('mrr', the pre-amendment_1 default; or
    'normalized_rank', amendment_1_20260924's cross-tree field) at the FINAL (epoch 200)
    checkpoint, averaged over seeds.

    Strata whose mean `n` (averaged over seeds) is below `P2_MIN_STRATUM_N` are dropped and
    reported, never silently skipped -- mirrors `_stratum_means` above.

    CORRECTION (2026-09-24, I2, `p2_amendment_3_20260924`). A stratum whose `d.get(field)` is
    `None` for EVERY seed (e.g. `n=0` there, or -- after the C3 exclusion -- a depth bin whose
    every query happened to have pool size 1) used to be skipped on every iteration and therefore
    never recorded in `per_stratum`/`ns` at all: it silently vanished from BOTH `means` and
    `dropped_below_min_n`, rather than showing up in the latter with `n=0`. `seen_strata` tracks
    every stratum LABEL the raw data mentions (regardless of whether its value was usable), so a
    stratum that never contributed a value still appears in `dropped_below_min_n`, never missing.

    Reads `_p2_ms_key` (2026-09-24, C4 #1): the FINAL epoch-200 checkpoint lives in the milestone
    trajectory (`*_ms`, epoch 10..200) production writes, not a bare `{group}_s{s}` key.
    """
    per_stratum: dict[str, list] = {}
    ns: dict[str, list] = {}
    seen_strata: set[str] = set()
    for s in seeds:
        ckpts = _p2_checkpoints(result, _p2_ms_key(group, s))
        final = next((c for c in ckpts if int(c["epoch"]) == P2_FINAL_EPOCH), None)
        if final is None:
            continue
        for stratum, d in final.get("by_depth", {}).get("cosine", {}).items():
            seen_strata.add(stratum)
            if d is None or d.get(field) is None:
                continue
            per_stratum.setdefault(stratum, []).append(float(d[field]))
            ns.setdefault(stratum, []).append(int(d.get("n", 0)))
    means, dropped = {}, {}
    for stratum in seen_strata:
        vals = per_stratum.get(stratum)
        stratum_ns = ns.get(stratum)
        mean_n = float(np.mean(stratum_ns)) if stratum_ns else 0.0
        if not vals or mean_n < P2_MIN_STRATUM_N:
            dropped[stratum] = mean_n
            continue
        means[stratum] = float(np.mean(vals))
    return {"means": means, "dropped_below_min_n": dropped}


def _p2_own_prior_nr(stats: dict, which: str) -> float:
    """The training-free degree prior's `normalized_rank` for ONE tree, as the amendment_6
    denominator -- raising rather than returning a value that would silently divide.

    `stats` is a `p2_group_stats` dict (a real arm's or a control's); `which` names it for the
    error message. A prior of 0.0 means the training-free ranker put the true parent first for
    every single held-out node, and a negative one is impossible for `(rank-1)/(pool-1)`: either
    is a broken baselines block, not a tree that is merely very easy. Dividing by it would yield
    `inf`/`nan` and propagate into a comparison where NaN is uniformly False -- i.e. a silent
    MEMORISES. Measured production values are ~0.153 (real) and ~0.038-0.040 (degmatch), so this
    raise is a structural guard, never a threshold anything is expected to approach.
    """
    prior = float(stats["degree_prior"]["normalized_rank"])
    if not np.isfinite(prior) or prior <= 0.0:
        raise ValueError(
            f"amendment_6: {which} ({stats['group']}) has degree_prior.normalized_rank={prior!r}, "
            "which cannot be an own-prior denominator. The baselines block for this arm is wrong "
            "or missing -- rescore it; do not read a verdict off this result.")
    return prior


def _p2_arm_reading(result: dict, name: str, seeds, group: dict, control: dict,
                    amendment_1: bool = False, control_group: str = P2_CONTROL_ARM,
                    amendment_6: bool = False) -> dict:
    """GENERALISES / MEMORISES / MIXED for one real-tree arm against chance, the degree prior,
    and RandomDAG.

    `sibling_chance_mean` no longer arrives as a separate parameter (2026-09-24, C1/C2,
    `p2_amendment_3_20260924`): the own-tree chance anchor and the degree-prior baseline are now
    read straight off `group` (`group["chance_mrr_mean"]`, `group["degree_prior"]`), which
    `p2_group_stats` already populated from THIS group's own baselines block -- one source of
    truth instead of a value threaded separately through every call site.

    `control_group` (2026-09-24, `p2_amendment_2_20260924` in the pre-registration JSON): the name
    of the RandomDAG arm `control` was built from -- defaults to the single shared `P2_CONTROL_ARM`
    ("randomdag", the frozen 9-arm design), but under amendment_2 the caller passes the CALLER's own
    visibility-matched control ("randomdag_vis00" for `name="vis00"`, "randomdag_vis50" for
    `name="vis50"`) so the depth-stratum comparison below reads that arm's own control, never the
    other real arm's. `control` (the group stats dict) must already be `p2_group_stats`'d for
    whichever group `control_group` names -- this parameter only controls which arm's checkpoints
    `_p2_depth_means` re-reads for the per-depth-stratum sign check.

    `amendment_1` (2026-09-24, `p2_amendment_1_20260924` in the pre-registration JSON):
    `randomdag.randomize_parents` preserves each node's DEPTH exactly but not its FAN-OUT --
    measured on mollusca_6447_clean seed 0 (helpers/p2_randomdag_changes_the_chance_floor.py),
    mean sibling_chance is 5.403x HIGHER on the randomised tree than the real one, so RandomDAG
    is a different-difficulty (easier) task, not merely a scrambled one. Raw MRR is therefore NOT
    comparable across the arm and the RandomDAG control; a false MEMORISES verdict is the direct
    consequence, since the easier control can out-score a genuinely-generalising real arm on raw
    MRR almost regardless of what the model learned.

    `amendment_1=False` (default) reproduces the ORIGINAL 2026-09-24 freeze reading, unchanged,
    comparing raw MRR head-to-head against RandomDAG's raw MRR -- kept reachable so the frozen
    reading remains reproducible (never silently rewritten). `amendment_1=True` instead compares
    normalized_rank -- pool-size (and so fan-out) invariant, chance level exactly 0.5 for every
    pool size -- for every CROSS-TREE quantity (the arm vs RandomDAG); the arm's own MRR-vs-its-
    own-sibling_chance_mean margin is WITHIN-tree and is unchanged by the flag either way.
    """
    # own-tree margin: an arm's own MRR vs its own chance_mrr_mean (C2) -- unaffected by the
    # RandomDAG fan-out defect either way, since RandomDAG never enters this comparison.
    chance_mrr = group["chance_mrr_mean"]
    degree_prior = group["degree_prior"]
    margin = group["min"] - chance_mrr
    pooled_sd = group["pooled_within_arm_seed_sd"]
    threshold = P2_FLOOR_SD_MULTIPLE * pooled_sd if np.isfinite(pooled_sd) else float("nan")
    above_chance_margin = bool(np.isfinite(threshold) and margin >= threshold)

    if not amendment_1:
        # C1 (2026-09-24, `p2_amendment_3_20260924`): the arm's own MRR must ALSO beat the
        # training-free degree prior computed on this SAME tree/heldout set -- a candidate parent
        # with more children is more likely to be the true parent with no embedding and no
        # training at all (measured MRR 0.5600 on the real metazoa split, clearing every OTHER
        # floor below). Without this clause a model that learned nothing but happens to embed
        # high-fanout nodes centrally (or is never even trained, since GENERALISES never checked
        # this) can still read GENERALISES.
        above_degree_prior = bool(group["min"] > degree_prior["mrr"])

        above_control_all_seeds = bool(group["min"] > control["max"])

        ranges_overlap = bool(group["min"] <= control["max"] and control["min"] <= group["max"])
        g_var = np.var(group["per_run_values"], ddof=1) if len(group["per_run_values"]) > 1 else 0.0
        c_var = np.var(control["per_run_values"], ddof=1) if len(control["per_run_values"]) > 1 else 0.0
        pooled_vs_control = float(np.sqrt(np.mean([g_var, c_var])))
        equiv_band = max(P2_EQUIV_FLOOR, P2_FLOOR_SD_MULTIPLE * pooled_vs_control)
        mean_diff = group["mean"] - control["mean"]
        equivalent_to_control = bool(ranges_overlap or abs(mean_diff) < equiv_band)

        depth = _p2_depth_means(result, name, seeds, field="mrr")
        depth_control = _p2_depth_means(result, control_group, seeds, field="mrr")
        shared = sorted(set(depth["means"]) & set(depth_control["means"]))
        diffs = {k: depth["means"][k] - depth_control["means"][k] for k in shared}
        sign_consistent = bool(diffs) and all(v > 0 for v in diffs.values())

        if (above_chance_margin and above_degree_prior and above_control_all_seeds
                and sign_consistent):
            verdict, meaning = "GENERALISES", (
                "MRR clears chance MRR by the declared margin, beats the training-free degree "
                "prior, beats every RandomDAG seed, and the sign holds in every depth stratum "
                "with n>=500: the model predicts relations it never saw. This is the answer to "
                "the overfitting challenge.")
        elif (not above_chance_margin) or (not above_degree_prior) or equivalent_to_control:
            verdict, meaning = "MEMORISES", (
                "MRR is not meaningfully separated from chance, does not beat the training-free "
                "degree prior, or is statistically indistinguishable from the RandomDAG control. "
                "The structure is in-sample only, or explained by a training-free prior; the "
                "manuscript must say so.")
        else:
            verdict, meaning = "MIXED", ("Sign flips across depth strata; report per stratum, "
                                         "no aggregate claim.")

        return {
            "arm": name, "verdict": verdict, "meaning": meaning,
            # R6 (2026-09-26): the control this arm was ACTUALLY read against. The renderer used
            # to hard-code "randomdag" in its depth-strata line and printed it even under
            # amendment_4, where RandomDAG is retired -- a sentence that gets quoted.
            "control_group": control_group,
            "amendment_1_applied": False, "cross_tree_metric": "mrr",
            "margin_above_chance": float(margin), "margin_threshold": float(threshold),
            "above_chance_margin": above_chance_margin,
            "chance_mrr_mean": float(chance_mrr),
            "degree_prior_mrr": float(degree_prior["mrr"]),
            "above_degree_prior": above_degree_prior,
            "above_control_all_seeds": above_control_all_seeds,
            "equivalent_to_control": equivalent_to_control,
            "mean_diff_vs_control": float(mean_diff), "equivalence_band_vs_control": equiv_band,
            "depth_strata": {"diffs": diffs, "n_strata": len(shared),
                             "dropped_below_min_n": {**depth["dropped_below_min_n"],
                                                      **depth_control["dropped_below_min_n"]}},
            "sign_consistent": sign_consistent,
        }

    # amendment_1=True: the cross-tree comparison, on normalized_rank. Lower is better; chance
    # is exactly 0.5 for every pool size, so it needs no per-arm floor the way MRR does.
    g_nr = np.asarray(group["per_run_values_normalized_rank"], dtype=float)
    c_nr = np.asarray(control["per_run_values_normalized_rank"], dtype=float)

    # amendment_6_20260926 (C-A): every CROSS-TREE quantity below is read as a RATIO TO THE
    # ARM'S OWN TREE'S training-free degree prior, never as a raw normalized_rank head-to-head.
    # amendment_1 made the metric pool-size invariant, which removes the FAN-OUT confound; it
    # does not remove the TASK-DIFFICULTY confound, because the two trees have different
    # training-free floors. Measured on the 12 production metazoa splits
    # (helpers/p2_review3_measure_production_splits.py, re-derived 2026-09-26): degree-prior
    # normalized_rank is 0.15381/0.15317/0.15320 on the real tree and 0.04041/0.03824/0.03851
    # on the degree-matched control -- the control's task is ~4x easier for a ranker that
    # learned NOTHING. Under the raw reading a real arm must beat a number its control gets for
    # free, so `below_control_all_seeds` is structurally unreachable and GENERALISES cannot be
    # returned however good the geometry is. Dividing each side by its own tree's prior asks the
    # question the design intends -- "which model improved more on what its tree hands out for
    # free" -- and is invariant to the difficulty gap by construction. 1.0 is exactly "no better
    # than the training-free prior" on either tree.
    if amendment_6:
        g_prior_nr = _p2_own_prior_nr(group, "real arm")
        c_prior_nr = _p2_own_prior_nr(control, "control")
        g_cmp, c_cmp = g_nr / g_prior_nr, c_nr / c_prior_nr
    else:
        g_prior_nr = float(degree_prior["normalized_rank"])
        # None, not NaN: "no denominator was used" is a fact, whereas NaN is a number that is
        # not equal to itself -- which silently breaks the byte-identity check that keeps the
        # unamended reading reproducible (`p2_verdict(...) == p2_verdict(..., amendment_6=False)`
        # compares False on a NaN field however identical the two readings are).
        c_prior_nr = None
        g_cmp, c_cmp = g_nr, c_nr

    below_control_all_seeds = bool(g_cmp.max() < c_cmp.min())    # mirrors above_control_all_seeds
    below_chance_level = bool(g_nr.max() < 0.5)                  # weakest (highest) seed still <0.5
    # C1, normalized_rank form: the arm's own normalized_rank must ALSO sit below the degree
    # prior's normalized_rank on this SAME tree (own-tree, mirrors above_degree_prior on MRR).
    # Untouched by amendment_6 -- it was ALREADY an own-tree comparison, and under amendment_6 it
    # is exactly the statement `g_cmp.max() < 1.0`.
    below_degree_prior = bool(g_nr.max() < degree_prior["normalized_rank"])

    ranges_overlap_nr = bool(g_cmp.min() <= c_cmp.max() and c_cmp.min() <= g_cmp.max())
    g_var_nr = float(np.var(g_cmp, ddof=1)) if len(g_cmp) > 1 else 0.0
    c_var_nr = float(np.var(c_cmp, ddof=1)) if len(c_cmp) > 1 else 0.0
    pooled_vs_control_nr = float(np.sqrt(np.mean([g_var_nr, c_var_nr])))
    if amendment_6:
        # RELATIVE band (see `P2_EQUIV_REL_FRACTION`): a fraction of the control's own mean
        # instead of the absolute MRR-scale floor, which was ~25% of a normalized_rank value.
        equiv_band_nr = max(P2_EQUIV_REL_FRACTION * abs(float(c_cmp.mean())),
                            P2_FLOOR_SD_MULTIPLE * pooled_vs_control_nr)
    else:
        equiv_band_nr = max(P2_EQUIV_FLOOR, P2_FLOOR_SD_MULTIPLE * pooled_vs_control_nr)
    mean_diff_nr = float(g_cmp.mean() - c_cmp.mean())
    equivalent_to_control_nr = bool(ranges_overlap_nr or abs(mean_diff_nr) < equiv_band_nr)

    depth_nr = _p2_depth_means(result, name, seeds, field="normalized_rank")
    depth_control_nr = _p2_depth_means(result, control_group, seeds, field="normalized_rank")
    shared_nr = sorted(set(depth_nr["means"]) & set(depth_control_nr["means"]))
    # amendment_6 applies the SAME own-prior scaling per stratum. The scorer emits `degree_prior`
    # only as an aggregate (scripts/score_p2_linkpred.py writes one block per `--baselines-key`,
    # with no by_depth breakdown), so each side is divided by its tree's AGGREGATE prior rather
    # than by a per-stratum one. That is a uniform positive rescaling of each side, so it cannot
    # reorder strata within an arm; what it does is relax the cross-tree sign test by exactly the
    # measured difficulty ratio (~4x), which is the confound amendment_6 exists to remove. A
    # per-stratum prior would be sharper and is NOT claimed here -- recorded as a stated
    # limitation in the amendment block, not a silent approximation.
    if amendment_6:
        diffs_nr = {k: depth_nr["means"][k] / g_prior_nr - depth_control_nr["means"][k] / c_prior_nr
                    for k in shared_nr}
    else:
        diffs_nr = {k: depth_nr["means"][k] - depth_control_nr["means"][k] for k in shared_nr}
    sign_consistent_nr = bool(diffs_nr) and all(v < 0 for v in diffs_nr.values())

    # The cross-tree clause reads differently under amendment_6, and the verdict's own prose is
    # what ends up quoted in the manuscript -- so it states WHICH comparison was made, never a
    # generic "beats the control".
    _cross = ("its improvement on its OWN tree's training-free degree prior exceeds the "
              "control's improvement on ITS own tree's prior, for every seed "
              "(amendment_6_20260926)" if amendment_6 else
              "normalized_rank sits below every control seed's normalized_rank")
    _amendments = ("amendment_1_20260924, amendment_3_20260924, amendment_6_20260926."
                   if amendment_6 else "amendment_1_20260924, amendment_3_20260924.")
    # amendment_6_20260926, second correction (found by helpers/p2_amendment6_reachability.py
    # scenario R2, PRE-RUN): GENERALISES did not require `not equivalent_to_control_nr`, so an arm
    # STATISTICALLY INDISTINGUISHABLE from its control could still be published as generalising --
    # the MEMORISES branch tests that boolean, but it is an `elif` and never runs once GENERALISES
    # has fired. `below_control_all_seeds` is a strict ordering (`max < min`) and settles a
    # mathematical TIE on the last bit of floating point, so under amendment_6's ratio scale the
    # case the review calls the EXPECTED one -- real arm and control improving on their own priors
    # by the SAME factor -- read GENERALISES with |mean_diff| = 0.000000 and
    # equivalent_to_control = True in the same dict. A verdict that contradicts its own reported
    # booleans is not a reading. Adding the clause makes the two branches consistent: an arm must
    # be better than its control AND distinguishable from it.
    #
    # Scope: the amendment_1 branch only. The frozen `amendment_1=False` raw-MRR reading above
    # carries the same latent structure and is deliberately NOT changed -- it is kept reachable
    # solely so the original 2026-09-24 freeze stays byte-identically reproducible, and it is
    # never the published reading. Recorded in the amendment block rather than silently fixed.
    if (above_chance_margin and below_degree_prior and below_control_all_seeds
            and below_chance_level and sign_consistent_nr
            and not equivalent_to_control_nr):
        verdict, meaning = "GENERALISES", (
            "MRR clears its own chance MRR by the declared margin, AND normalized_rank sits "
            f"below the training-free degree prior's normalized_rank AND {_cross} AND below the "
            "pool-size-invariant chance level of 0.5, AND the sign holds in every depth stratum "
            "with n>=500: the model predicts relations it never saw, read on a metric the "
            "control's own chance floor cannot confound, and clears a training-free prior. "
            + _amendments)
    elif (not above_chance_margin) or (not below_degree_prior) or equivalent_to_control_nr:
        verdict, meaning = "MEMORISES", (
            "MRR is not meaningfully separated from its own chance MRR, normalized_rank does not "
            "beat the training-free degree prior, or the arm is statistically indistinguishable "
            "from the control on the cross-tree quantity. The structure is in-sample only, or "
            "explained by a training-free prior; the manuscript must say so. " + _amendments)
    else:
        # amendment_6_20260926, third correction (found by p2_amendment6_reachability.py scenario
        # R3, PRE-RUN): this branch is reached for THREE different reasons -- a depth-stratum sign
        # flip, the arm failing to beat its control on every seed, or the arm not clearing the
        # 0.5 chance level -- but its prose asserted the first one unconditionally. On R3 (the arm
        # uniformly WORSE than its control, sign_consistent False only because the diffs are
        # positive throughout, no flip anywhere) the published sentence would have been simply
        # untrue. A verdict string is quoted into the manuscript, so it now names what actually
        # happened.
        _why = []
        if not below_control_all_seeds:
            _why.append("the arm does not beat its control on every seed")
        if not below_chance_level:
            _why.append("normalized_rank does not clear the pool-size-invariant chance level 0.5")
        if not sign_consistent_nr:
            _why.append("the sign of (arm - control) is not negative in every depth stratum "
                        "with n>=500" if diffs_nr else
                        "no depth stratum with n>=500 is shared by the arm and its control")
        verdict, meaning = "MIXED", ("No aggregate claim; report per stratum. Binding: "
                                     + "; ".join(_why) + ". " + _amendments)

    return {
        "arm": name, "verdict": verdict, "meaning": meaning,
        "control_group": control_group,            # R6, see the frozen branch above
        "amendment_1_applied": True,
        "amendment_6_applied": bool(amendment_6),
        "cross_tree_metric": ("normalized_rank_over_own_degree_prior" if amendment_6
                              else "normalized_rank"),
        "margin_above_chance": float(margin), "margin_threshold": float(threshold),
        "above_chance_margin": above_chance_margin,
        "chance_mrr_mean": float(chance_mrr),
        "degree_prior_normalized_rank": float(degree_prior["normalized_rank"]),
        "below_degree_prior": below_degree_prior,
        "below_control_all_seeds": below_control_all_seeds,
        "below_chance_level": below_chance_level,
        "equivalent_to_control": equivalent_to_control_nr,
        "mean_diff_vs_control": mean_diff_nr, "equivalence_band_vs_control": equiv_band_nr,
        # amendment_6's own audit trail: the two denominators and both sides of the comparison
        # AS COMPARED, so a reader never has to reconstruct which scale a number lives on.
        # Under amendment_6=False these are the raw normalized_rank values and
        # `control_degree_prior_normalized_rank` is None (no denominator was used).
        "arm_degree_prior_normalized_rank": float(g_prior_nr),
        "control_degree_prior_normalized_rank": (None if c_prior_nr is None
                                                 else float(c_prior_nr)),
        "arm_values_as_compared": [float(v) for v in g_cmp],
        "control_values_as_compared": [float(v) for v in c_cmp],
        "depth_strata": {"diffs": diffs_nr, "n_strata": len(shared_nr),
                         "dropped_below_min_n": {**depth_nr["dropped_below_min_n"],
                                                  **depth_control_nr["dropped_below_min_n"]}},
        "sign_consistent": sign_consistent_nr,
    }


def _p2_single_shared_control(result: dict, seeds) -> dict:
    """amendment_2=False (default, the frozen 9-arm design): ONE RandomDAG control, read by
    BOTH vis00 and vis50 -- exactly the original `p2_verdict` body, unchanged, so the default
    call path stays byte-identical to before amendment_2 existed.

    Resolves the control's baselines PER SEED (2026-09-24, Part 2 fix) via
    `_p2_baselines_for_seed`, fallback chain `baselines_randomdag_s{seed}` -> `baselines_randomdag`
    -> `baselines_s{seed}` -> `baselines` -- extending the pre-Part-2 chain
    (`baselines_randomdag` -> `baselines`) by one seed-specific link at each stage. On a
    pre-Part-2 scorer output (no seed-tagged key anywhere in the chain) every seed resolves to
    the SAME dict, reproducing the old single-shared-baseline behaviour exactly.
    """
    prefixes = ["baselines_randomdag", "baselines"]
    baselines_by_seed = {s: _p2_baselines_for_seed(result, prefixes, s) for s in seeds}
    control = p2_group_stats(result, P2_CONTROL_ARM, seeds, baselines_by_seed)
    return {P2_CONTROL_ARM: control}, {g: P2_CONTROL_ARM for g in P2_DATA_ARMS}


def _p2_matched_controls(result: dict, seeds, control_map: dict | None = None) -> dict:
    """amendment_2=True (`p2_amendment_2_20260924`) or amendment_4=True
    (`p2_amendment_4_20260924`): one p2_group_stats per matched control (randomdag_vis00/
    randomdag_vis50 under amendment_2, degmatch_vis00/degmatch_vis50 under amendment_4) -- vis00
    is read ONLY against its own matched control, vis50 ONLY against ITS own, never crossed
    (rule_1 of amendment_2, unchanged in meaning by amendment_4's control swap).

    `control_map` (2026-09-24, amendment_4 -- generalises what was a hard-coded
    `P2_MATCHED_CONTROL` reference): defaults to `P2_MATCHED_CONTROL`
    (randomdag_vis00/randomdag_vis50, amendment_2's original mapping) when omitted, so every
    existing amendment_2 call site is unchanged; `p2_verdict` passes
    `P2_MATCHED_CONTROL_DEGMATCH` here under amendment_4.

    Each control's own baselines are resolved PER SEED (2026-09-24, Part 2 fix) via
    `_p2_baselines_for_seed`, fallback chain `baselines_<control_name>_s{seed}` ->
    `baselines_<control_name>` -> `baselines_randomdag_s{seed}` -> `baselines_randomdag` ->
    `baselines_s{seed}` -> `baselines` -- the same three-step (now six-step, with the seed-tagged
    link added at each stage) fallback amendment_1 established for the single control, applied per
    matched pair since there are two controls (rule_3). C4 (2026-09-24): `--baselines-key` in
    `scripts/score_p2_linkpred.py` lets the scorer populate the specific key directly, instead of
    every invocation writing the same plain `baselines` and this fallback silently handing every
    control the REAL tree's floor; Part 2 additionally seed-tags that key so seeds 1/2 are never
    silently handed seed 0's floor either.
    """
    control_map = P2_MATCHED_CONTROL if control_map is None else control_map
    controls = {}
    for control_name in sorted(set(control_map.values())):
        # C-B (2026-09-26): `"baselines"` -- the REAL TREE's block -- is NO LONGER the last link
        # in a control's fallback chain. It was, and the consequence was measured (review
        # scenario S5): with `baselines_degmatch_vis{00,50}_s{n}` merely ABSENT from the merge --
        # a `--baselines-key` typo, one of the 9 JSONs not passed to `apply_preregistration.py`,
        # or a partially-failed scoring job -- every matched control silently resolved the REAL
        # tree's floor and published `randomdag_sibling_chance_mean` 0.13126 instead of its own
        # 0.18833. No error, no warning, and `assert_baselines_agree_across_seeds` cannot see it
        # because the real and control blocks are DIFFERENT FAMILIES and an absent family has
        # nothing to disagree with. The control's gate (b) was then tested against a floor 23%
        # too low, i.e. the control became EASIER to validate than it should be. A control
        # reading the real tree's floor is never the right answer, so the chain now ends inside
        # the control's own family and a miss RAISES, naming the key that is missing. The
        # seed-tagged -> bare-prefix fallback WITHIN the control's own family is kept: that one
        # is the documented pre-Part-2 compatibility path, and it can only ever resolve to a
        # block measured on this control's own tree.
        prefixes = [f"baselines_{control_name}", "baselines_randomdag"]
        try:
            baselines_by_seed = {s: _p2_baselines_for_seed(result, prefixes, s) for s in seeds}
        except KeyError as exc:
            raise KeyError(
                f"control '{control_name}' has no baselines block of its own: {exc}. Falling "
                f"back to the real tree's 'baselines' would publish the REAL tree's chance floor "
                f"and degree prior as this control's, which is never correct -- rescore this "
                f"control with --baselines-key baselines_{control_name}_s<seed>, or pass the "
                f"missing file to the merge.") from exc
        controls[control_name] = p2_group_stats(result, control_name, seeds, baselines_by_seed)
    return controls, dict(control_map)


def _p2_relative_spread(values: list[float]) -> float:
    """(max-min)/mean -- the relative-spread statistic `assert_baselines_agree_across_seeds`
    checks. 0.0 when every value is exactly equal (including all-zero). When the mean is
    (numerically) zero but the values are NOT all equal, returns `inf` rather than dividing by
    zero -- any non-zero spread around a zero mean is a genuine disagreement, never a false
    negative that a division guard would otherwise silently wave through."""
    arr = np.asarray(values, dtype=float)
    spread = float(arr.max() - arr.min())
    mean = float(arr.mean())
    if abs(mean) < 1e-12:
        return 0.0 if spread == 0.0 else float("inf")
    return spread / abs(mean)


def assert_baselines_agree_across_seeds(result: dict, seeds=(0, 1, 2),
                                        tolerance: float = P2_BASELINES_SEED_TOLERANCE) -> dict:
    """Fail loudly if any seed-tagged baselines FAMILY's own sibling_chance_mean/chance_mrr_mean/
    degree_prior.{mrr,normalized_rank} disagrees across seeds by more than `tolerance` (relative
    spread, `(max-min)/mean`) -- Part 2 fix (2026-09-24, no amendment flag: a correctness fix, not
    a design choice with two legitimate readings, same posture as amendment_3's C1-C5). 'They
    should agree closely' is the word that precedes every silent defect in this build, so this
    measures it instead of assuming it -- called from `merge_p2_scorer_outputs` per Part 2's
    ruling ('have the merge step assert they agree, failing loudly if not').

    A baselines FAMILY is any prefix P such that `{P}_s{s}` exists in `result` for at least two of
    `seeds` -- e.g. prefix `"baselines"` (the real-tree pair, shared by vis00/vis50) or
    `"baselines_degmatch_vis00"` (one matched control). A merged result with NO seed-tagged keys
    at all (every pre-Part-2 fixture and every pre-Part-2 production JSON, which wrote one bare
    `"baselines"`/`"baselines_<control>"` key shared by all 3 seeds) has no families to check and
    this is a silent no-op -- additive, never a behaviour change for an existing caller.

    Returns a report dict (`family -> field -> relative spread`) for a caller that wants the
    measured numbers even when nothing failed; raises `ValueError`, naming every offending
    family+field and the tolerance, when any exceeds it.
    """
    families: set[str] = set()
    for key in result:
        if not key.startswith("baselines"):
            continue
        for s in seeds:
            suffix = f"_s{s}"
            if key.endswith(suffix):
                families.add(key[: -len(suffix)])
                break

    report: dict[str, dict[str, float]] = {}
    violations: list[str] = []
    for family in sorted(families):
        present = {s: result[f"{family}_s{s}"] for s in seeds if f"{family}_s{s}" in result}
        if len(present) < 2:
            continue
        field_report: dict[str, float] = {}
        for field in ("sibling_chance_mean", "chance_mrr_mean"):
            vals = [float(b[field]) for b in present.values() if field in b]
            if len(vals) < 2:
                continue
            spread = _p2_relative_spread(vals)
            field_report[field] = spread
            if spread > tolerance:
                violations.append(
                    f"{family}.{field}: relative spread {spread:.3f} exceeds tolerance "
                    f"{tolerance:.3f} across seeds {sorted(present)}")
        for dp_field in ("mrr", "normalized_rank"):
            vals = [float(b["degree_prior"][dp_field]) for b in present.values()
                   if "degree_prior" in b and dp_field in b["degree_prior"]]
            if len(vals) < 2:
                continue
            spread = _p2_relative_spread(vals)
            field_report[f"degree_prior.{dp_field}"] = spread
            if spread > tolerance:
                violations.append(
                    f"{family}.degree_prior.{dp_field}: relative spread {spread:.3f} exceeds "
                    f"tolerance {tolerance:.3f} across seeds {sorted(present)}")
        report[family] = field_report

    if violations:
        raise ValueError(
            "baselines disagree across seeds beyond the declared tolerance "
            f"({tolerance:.0%}) -- a real per-seed measurement should not vary this much; this "
            "usually means the wrong split/closure landed under the wrong seed's key, not "
            "ordinary sampling noise:\n  " + "\n  ".join(violations))
    return report


def merge_p2_scorer_outputs(results: list[dict], seeds=(0, 1, 2),
                            baselines_tolerance: float = P2_BASELINES_SEED_TOLERANCE) -> dict:
    """Merge N `scripts/score_p2_linkpred.py` output dicts into the single dict `p2_verdict`
    expects (2026-09-24, C4 #2, `p2_amendment_3_20260924`).

    `scripts/p2_lrz_score.sh` writes 9 SEPARATE JSON files per array (3 seeds x {real vis00/vis50
    pair, randomdag_vis00, randomdag_vis50}), but `p2_verdict` reads one `result["arms"]` dict and
    up to three top-level `baselines*` blocks. This merges without re-deriving anything:

    `arms` is the UNION of every input's `arms` dict, already namespaced by arm+seed+kind
    (`"vis00_s0_roll"`, `"randomdag_vis00_s1_ms"`, ...) by the scorer itself. A key claimed by TWO
    inputs is an ERROR, never a silent overwrite -- that would mean one file's run is being
    silently dropped from the merged verdict.

    Every top-level key starting with `"baselines"` (`"baselines"`, `"baselines_randomdag_vis00"`,
    `"baselines_randomdag_vis50"`, ...) is copied from the FIRST input that defines it. Inputs are
    expected to be passed in a stable order (e.g. seed order) so "first" means "seed 0's own
    measurement" -- a stated simplifying assumption, not a silent one: P2's baselines
    (sibling_chance_mean, chance_mrr_mean, degree_prior) are properties of a SPECIFIC heldout
    split, which nominally differs per seed (each seed has its own manifest/heldout pair), but
    `p2_verdict` has always read ONE value shared across all seeds of an arm -- this merge
    preserves that pre-existing assumption rather than introducing a new one. A LATER input
    disagreeing with an already-recorded baselines block is not silently dropped either: it is
    recorded under `_baselines_disagreements` (keyed by the baselines key) for a reader to check,
    since a few-percent difference across seeds' own splits is expected and not on its own fatal,
    but a large one would be worth knowing about before trusting the merged floor.

    PART 2 FIX (2026-09-24, no amendment flag -- a correctness fix, not a design choice with two
    legitimate readings, same posture as amendment_3's C1-C5). The paragraph above describes the
    PRE-Part-2 world, where every seed wrote the SAME bare key (`"baselines"`) and this merge
    picked whichever file's value merged first ("seed 0 wins", recording only that a later one
    disagreed). `scripts/p2_lrz_score.sh` now writes SEED-TAGGED keys (`"baselines_s0"`,
    `"baselines_degmatch_vis00_s1"`, ...), which never collide across seeds -- each survives the
    union above distinctly, with no "first wins" step involved for them at all. What this merge
    step ADDS for that seed-tagged shape (the task's own ruling: 'have the merge step assert they
    agree, failing loudly if not') is calling `assert_baselines_agree_across_seeds` on the fully
    merged result before returning: it raises `ValueError` if any seed's own sibling_chance_mean/
    chance_mrr_mean/degree_prior sits more than `baselines_tolerance` (relative spread) from its
    sibling seeds' -- a genuinely wrong split/closure landing under the wrong seed's key is caught
    HERE, not three verdict computations later. A merged result with no seed-tagged keys at all
    (every pre-Part-2 fixture) has nothing to check and this is a silent no-op.
    """
    merged: dict = {"arms": {}}
    disagreements: dict[str, list] = {}
    for result in results:
        for key, value in result.get("arms", {}).items():
            if key in merged["arms"]:
                raise ValueError(f"duplicate arm key {key!r} across merged scorer outputs -- "
                                 f"two input files claim the same arm+seed+kind")
            merged["arms"][key] = value
        for key, value in result.items():
            if key == "arms" or not key.startswith("baselines"):
                continue
            if key not in merged:
                merged[key] = value
            elif merged[key] != value:
                disagreements.setdefault(key, []).append(value)
    if disagreements:
        merged["_baselines_disagreements"] = disagreements
    # PART 2: raises loudly if any seed-tagged family disagrees beyond tolerance; a no-op when no
    # seed-tagged baselines keys are present (every pre-Part-2 shape).
    merged["_baselines_cross_seed_spread"] = assert_baselines_agree_across_seeds(
        merged, seeds=seeds, tolerance=baselines_tolerance)
    merged["_control_baselines_present"] = assert_control_baselines_exist(merged)
    return merged


def assert_control_baselines_exist(result: dict) -> dict:
    """Every CONTROL arm present under `arms` must have a baselines family of its own.

    C-B (2026-09-26). `assert_baselines_agree_across_seeds` measures AGREEMENT, which is a
    question you can only ask about a family that is present -- an ABSENT family has nothing to
    disagree with, so a missing control block sailed through it. That is not hypothetical:
    review scenario S5 measured the whole chain, with `baselines_degmatch_vis{00,50}_s{n}`
    simply absent from the merge, producing a verdict with the REAL tree's floor published as
    the control's, `_baselines_disagreements` absent, and no error. EXISTENCE, not agreement,
    is the check nothing performed.

    Raising here rather than in `_p2_matched_controls` alone is deliberate: the merge is where
    the 9 (now 12) separate scorer JSONs come together, so it is the first moment at which
    "this file was never passed in" is knowable, and it fails before any verdict is computed
    rather than partway through one.

    Returns `{control_arm: resolving_key}` for a caller that wants to see WHICH key each control
    resolved to; raises `ValueError` naming every control that has none.
    """
    control_arms = set()
    for key in result.get("arms", {}):
        # keys are "<arm>_s<seed>_<kind>"; the arm may itself contain underscores
        parts = key.rsplit("_s", 1)
        if len(parts) != 2:
            continue
        arm = parts[0]
        if arm not in P2_DATA_ARMS:
            control_arms.add(arm)

    present, missing = {}, []
    for arm in sorted(control_arms):
        candidates = [f"baselines_{arm}"] + [f"baselines_{arm}_s{s}" for s in range(16)]
        found = next((k for k in candidates if k in result), None)
        if found is None:
            missing.append(arm)
        else:
            present[arm] = found
    if missing:
        raise ValueError(
            "control arm(s) present under 'arms' with NO baselines family of their own: "
            f"{missing}. Without one, each would resolve the REAL tree's floor and degree prior "
            "-- a control judged against the wrong tree's difficulty. Rescore with "
            "--baselines-key baselines_<arm>_s<seed>, or pass the missing scorer JSON to the "
            "merge.")
    return present


def p2_verdict(result: dict, seeds=(0, 1, 2), amendment_1: bool = False,
               amendment_2: bool = False, amendment_4: bool = False,
               amendment_6: bool = False) -> dict:
    """vis00/vis50 vs sibling chance and its control (RandomDAG, or the degree-matched shuffle
    under amendment_4) -> GENERALISES / MEMORISES / MIXED / UNINFORMATIVE, per results/
    p2_heldout_preregistration.json.

    UNINFORMATIVE beats every other reading: if ANY seed of ANY arm (including every control arm)
    fails gate (a)/(b)/(c), the whole verdict is UNINFORMATIVE and neither data arm is read.
    Otherwise each of vis00/vis50 gets its own GENERALISES/MEMORISES/MIXED reading
    (`by_arm_verdict`); the top-level `verdict` is that shared reading if both arms agree, else
    MIXED. The validity gates themselves (a/b/c) are untouched by `amendment_1`/`amendment_2`/
    `amendment_4` -- they are within-tree facts about a run, never a cross-tree comparison.

    `amendment_1` (2026-09-24, `p2_amendment_1_20260924`): default False reproduces the ORIGINAL
    2026-09-24 freeze reading unchanged (raw MRR compared head-to-head against the control). True
    applies the amended reading -- RandomDAG's rewiring measurably inflates its own chance floor
    (~5.4x on mollusca_6447_clean seed 0), so the cross-tree comparison uses normalized_rank
    (chance = 0.5 for every pool size) instead. Still the correct pool-size-invariant quantity
    under amendment_4's degree-matched control too (see that amendment's own block: the residual
    fan-out/chance-floor gap there is smaller, ~1.8x, but not zero, so this rule stays
    load-bearing, not merely belt-and-braces). See `_p2_arm_reading` and the JSON amendment block
    for the full derivation. Both readings are always computable from the same result.

    `amendment_2` (2026-09-24, `p2_amendment_2_20260924`, USER DESIGN decision -- the arm structure
    changed from 9 arms to 12, not an outcome-driven revision): default False reproduces the
    ORIGINAL 9-arm reading unchanged -- ONE shared RandomDAG control (`P2_CONTROL_ARM`, "randomdag")
    read against BOTH vis00 and vis50, exactly as before this flag existed. True switches to the
    12-arm matched-control design: vis00 is read only against `randomdag_vis00`, vis50 only against
    `randomdag_vis50` -- each real arm against a control trained at its OWN visibility, so the
    visibility knob never leaks into the real-vs-scrambled contrast.

    `amendment_4` (2026-09-24, `p2_amendment_4_20260924`, USER DESIGN decision, BEFORE any array
    was submitted and BEFORE any P2 outcome data existed): RETIRES RandomDAG as P2's control and
    substitutes the DEGREE-MATCHED shuffle (`randomdag.degree_matched_shuffle`) -- same 12-arm
    matched shape amendment_2 established, sole substitution: `degmatch_vis00`/`degmatch_vis50`
    read in place of `randomdag_vis00`/`randomdag_vis50` (`P2_MATCHED_CONTROL_DEGMATCH` in place
    of `P2_MATCHED_CONTROL`). Default False leaves amendment_2's own control selection exactly as
    it was. When True, amendment_4 takes precedence over `amendment_2` for CONTROL SELECTION (the
    degree-matched arms only exist in the matched-per-visibility shape, so `amendment_4=True`
    always reads the matched design, whatever `amendment_2` is passed as) -- `amendment_1` and
    `amendment_4` are independent and combine freely, exactly as `amendment_1`/`amendment_2` do.
    See `_p2_matched_controls` and the JSON amendment block for the full derivation, including the
    MEASURED (not assumed) residual chance-floor gap.

    `amendment_6` (2026-09-26, `p2_amendment_6_20260926`, USER DESIGN decision of 2026-09-25,
    BEFORE any array was submitted and BEFORE any P2 outcome data existed): every CROSS-TREE
    quantity is read as a RATIO TO EACH ARM'S OWN TREE'S training-free degree prior, and the
    normalized_rank equivalence band becomes RELATIVE (`P2_EQUIV_REL_FRACTION` of the control's
    own mean) instead of the absolute MRR-scale `P2_EQUIV_FLOOR`. Default False leaves
    amendment_1's reading exactly as it was. REQUIRES `amendment_1=True` -- it amends that
    branch's cross-tree comparison and has no meaning on the raw-MRR frozen reading, so passing
    it alone RAISES rather than being silently ignored (a flag that does nothing is how a
    pre-registration stops describing the computation it names). See `_p2_arm_reading`'s
    amendment_6 block for the measured justification: the degree-matched control's task is ~4x
    easier for a training-free ranker, which makes GENERALISES structurally unreachable under
    the unamended reading.

    PART 2 (2026-09-24, no amendment flag -- a correctness fix): each group's/control's own
    baselines are now resolved PER SEED (`p2_group_stats`'s `baselines_by_seed`), never one value
    shared across all 3 seeds taken from whichever seed's file merged first. See
    `_p2_baselines_for_seed` and `assert_baselines_agree_across_seeds`.
    """
    if amendment_6 and not amendment_1:
        raise ValueError(
            "amendment_6=True requires amendment_1=True: amendment_6 amends the normalized_rank "
            "cross-tree comparison that only the amendment_1 branch performs. On the raw-MRR "
            "frozen reading it would silently do nothing.")
    baselines_by_seed = {s: _p2_baselines_for_seed(result, ["baselines"], s) for s in seeds}
    sibling_chance_mean = float(baselines_by_seed[seeds[0]]["sibling_chance_mean"])
    groups = {g: p2_group_stats(result, g, seeds, baselines_by_seed) for g in P2_DATA_ARMS}

    if amendment_4:
        controls, control_for_arm = _p2_matched_controls(result, seeds, P2_MATCHED_CONTROL_DEGMATCH)
    elif amendment_2:
        controls, control_for_arm = _p2_matched_controls(result, seeds, P2_MATCHED_CONTROL)
    else:
        controls, control_for_arm = _p2_single_shared_control(result, seeds)

    invalid = []
    for c in controls.values():
        invalid += c["invalid_runs"]
    for g in groups.values():
        invalid += g["invalid_runs"]

    matched_controls = bool(amendment_4 or amendment_2)
    base = {
        "task": ("P2 -- held-out link prediction, real taxonomy vs degree-matched control"
                 if amendment_4 else
                 "P2 -- held-out link prediction, real taxonomy vs RandomDAG"),
        "preregistration": ["results/p2_heldout_preregistration.json"],
        "seeds": list(seeds),
        "amendment_1_applied": bool(amendment_1),
        "amendment_2_applied": bool(amendment_2),
        "amendment_4_applied": bool(amendment_4),
        "amendment_6_applied": bool(amendment_6),
        "sibling_chance_mean": sibling_chance_mean,
        "chance_mrr_mean": float(baselines_by_seed[seeds[0]]["chance_mrr_mean"]),
        "degree_prior": baselines_by_seed[seeds[0]]["degree_prior"],
        "groups": groups,
    }
    # "randomdag_sibling_chance_mean" is a legacy key name kept UNCHANGED regardless of which
    # control is active (amendment_4's degmatch control included) -- scripts/apply_
    # preregistration.py::_render_p2 and tests/eval/test_score_p2_cli.py read it by this name; a
    # rename here would silently break both for no reading-accuracy benefit, since the key's
    # VALUE is always "whichever control(s) are actually in use"'s own floor, not literally
    # RandomDAG's.
    if matched_controls:
        base["controls"] = controls
        base["randomdag_sibling_chance_mean"] = {
            name: c["gates"][0]["sibling_chance_mean"] for name, c in controls.items()}
    else:
        base["control"] = controls[P2_CONTROL_ARM]
        base["randomdag_sibling_chance_mean"] = (
            controls[P2_CONTROL_ARM]["gates"][0]["sibling_chance_mean"])

    if invalid:
        base["verdict"] = "UNINFORMATIVE"
        base["meaning"] = ("A validity gate failed; no direction is read for either arm. "
                           f"Invalid runs: {', '.join(invalid)}")
        return base

    by_arm = {name: _p2_arm_reading(result, name, seeds, groups[name], controls[control_for_arm[name]],
                                    amendment_1=amendment_1, control_group=control_for_arm[name],
                                    amendment_6=amendment_6)
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
