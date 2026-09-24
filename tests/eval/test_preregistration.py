"""The pre-registration engine must be able to return every verdict it can return.

A decision rule that only ever emits one label is not a rule. Each test below drives the engine
to a DIFFERENT outcome from synthetic scorer output, so a change that collapses the rule onto a
single verdict fails here rather than in the manuscript.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from taxembed.eval.preregistration import (
    compare_arms, p2_validity_gate, p2_verdict, task8_verdict, task9_verdict, validity_gate,
)

_REPO = Path(__file__).resolve().parents[2]


def _ckpt(epoch, s_angle, loss, auc=0.9, poincare=None, band=None, clade=None):
    return {
        "epoch": epoch,
        "S_angle": s_angle,
        "S_poincare": s_angle - 0.05 if poincare is None else poincare,
        "level_auc_mean": auc,
        "trainer": {"loss": loss},
        "S_angle_by_band": band if band is not None else {
            "shallow": {"n": 4000, "S": s_angle + 0.01},
            "mid": {"n": 5000, "S": s_angle - 0.01},
            "deep": {"n": 1000, "S": s_angle},
        },
        "S_angle_by_clade": clade if clade is not None else {
            "101": {"n": 3000, "S": s_angle + 0.02},
            "202": {"n": 2000, "S": s_angle - 0.02},
        },
    }


def _run(s_angle, loss_start=5.0, loss_end=4.0, auc=None, jitter=1e-4, rise=True,
         band_offset=0.0, clade_offset=0.0):
    """One run: 20 milestones + 5 rolling checkpoints, in the scorer's output shape.

    `auc` defaults to a monotone function of S_angle, because in a real run the two lenses move
    together; a fixture that pins both arms to the SAME level AUC makes amendment 1's
    all-above-all condition unsatisfiable and quietly turns every test into MIXED. Tests that
    want the two lenses to disagree set `auc` explicitly.
    """
    auc = 0.5 + 0.4 * s_angle if auc is None else auc
    rng = np.random.default_rng(0)
    roll, ms = [], []
    for i, ep in enumerate(range(196, 201)):
        s = s_angle + (i - 2) * jitter
        roll.append(_ckpt(ep, s, loss_end + i * 1e-6, auc,
                          band={"shallow": {"n": 4000, "S": s + 0.01 + band_offset},
                                "mid": {"n": 5000, "S": s - 0.01 + band_offset},
                                "deep": {"n": 1000, "S": s + band_offset}},
                          clade={"101": {"n": 3000, "S": s + 0.02 + clade_offset},
                                 "202": {"n": 2000, "S": s - 0.02 + clade_offset}}))
    for j, ep in enumerate(range(10, 201, 10)):
        frac = (j + 1) / 20
        # A dead run wanders at its OWN jitter scale around its own level -- not at some
        # unrelated larger scale, which would manufacture a rise above the init null and let
        # gate (a) pass a run that never trained.
        s = (s_angle * frac) if rise else (s_angle + rng.normal(0.0, jitter))
        loss = loss_start - (loss_start - loss_end) * frac
        ms.append(_ckpt(ep, s, loss, auc))
    return roll, ms


def _result(specs: dict, init_s=-0.0015):
    """specs: {(arm, seed): kwargs for _run}"""
    runs = {}
    for (arm, seed), kw in specs.items():
        roll, ms = _run(**kw)
        runs[f"{arm}_s{seed}_roll"] = {"checkpoints": roll}
        runs[f"{arm}_s{seed}_ms"] = {"checkpoints": ms}
    return {"init_null": {"S_angle": init_s}, "runs": runs}


def _both_arms(a_s, b_s, arm_a="canonical", arm_b="prior", **kw):
    specs = {}
    for seed in (0, 1, 2):
        specs[(arm_a, seed)] = {"s_angle": a_s + seed * 1e-4, **kw}
        specs[(arm_b, seed)] = {"s_angle": b_s + seed * 1e-4, **kw}
    return _result(specs)


class TestValidityGate:
    def test_a_trained_run_passes_both_gates(self):
        g = validity_gate(_result({("canonical", 0): {"s_angle": 0.90}}), "canonical", 0)
        assert g["gate_a_pass"] and g["gate_b_pass"] and g["valid"]

    def test_a_run_that_never_trained_fails_gate_a(self):
        """The echinodermata failure: no rise above the init null, so it is UNINFORMATIVE
        rather than evidence for the other arm."""
        g = validity_gate(_result({("canonical", 0): {"s_angle": -0.0015, "rise": False}}),
                          "canonical", 0)
        assert not g["gate_a_pass"]
        assert not g["valid"]

    def test_a_flat_final_phase_loss_fails_gate_b_under_the_FROZEN_rule(self):
        res = _result({("canonical", 0): {"s_angle": 0.90, "loss_start": 4.0, "loss_end": 4.0}})
        g = validity_gate(res, "canonical", 0)
        assert not g["gate_b_pass"]
        assert not g["valid"]

    def test_a_CONVERGED_arm_passes_gate_b_under_amendment_2(self):
        """🧨 The defect amendment 2 fixes, and the reason the original suite could not catch it.

        A converged arm and an arm that never trained produce the SAME flat final-phase loss.
        The frozen rule failed both; gate (a) is what actually distinguishes them. The earlier
        version of this test asserted only that flat loss FAILS — encoding the defect as correct
        behaviour and then verifying it.
        """
        res = _result({("canonical", 0): {"s_angle": 0.90, "loss_start": 4.0, "loss_end": 4.0}})
        g = validity_gate(res, "canonical", 0, amendment_2=True)
        assert g["gate_b_pass"]
        assert g["gate_a_pass"]        # structure rose: this arm demonstrably trained
        assert g["valid"]

    def test_a_DIVERGING_arm_still_fails_gate_b_under_amendment_2(self):
        """Amendment 2 must not make gate (b) vacuous: a rising loss is still a failure."""
        res = _result({("canonical", 0): {"s_angle": 0.90, "loss_start": 4.0, "loss_end": 4.5}})
        g = validity_gate(res, "canonical", 0, amendment_2=True)
        assert not g["gate_b_pass"]
        assert not g["valid"]

    def test_amendment_2_does_not_reopen_the_echinodermata_hole(self):
        """The failure the gates exist for: an arm whose loss fell but which learned nothing.

        It always passed gate (b); gate (a) is what excludes it. Amendment 2 touches only (b),
        so this must still be UNINFORMATIVE.
        """
        specs = {}
        for seed in (0, 1, 2):
            specs[("canonical", seed)] = {"s_angle": 0.90 + seed * 1e-4}
            specs[("prior", seed)] = {"s_angle": 0.60 + seed * 1e-4}
        # prior_s1 never acquired structure: it sits at the init null and wanders at its own
        # jitter scale. NOT pinned to exactly 0.0 — a perfectly constant arm has zero jitter, so
        # gate (a)'s 10x-jitter threshold collapses to zero and any rise clears it. Real runs
        # always jitter; a fixture that does not is testing a degenerate case.
        specs[("prior", 1)] = {"s_angle": -0.0015, "rise": False}
        res = _result(specs)
        v = task9_verdict(res, amendment_2=True)
        assert v["verdict"] == "UNINFORMATIVE"
        assert "prior_s1" in v["comparison"]["gates"]["invalid_runs"]

    def test_gate_a_admits_rise_then_collapse(self):
        """Monotonicity is explicitly NOT required: max over milestones is what counts."""
        res = _result({("canonical", 0): {"s_angle": 0.30}})
        ms = res["runs"]["canonical_s0_ms"]["checkpoints"]
        for c in ms[10:]:
            c["S_angle"] = 0.01           # collapses after peaking mid-run
        g = validity_gate(res, "canonical", 0)
        assert g["gate_a_pass"] and g["valid"]

    def test_only_final_phase_milestones_enter_gate_b(self):
        res = _result({("canonical", 0): {"s_angle": 0.90}})
        g = validity_gate(res, "canonical", 0)
        assert min(g["gate_b_final_phase_epochs"]) >= 130


class TestTask9Readings:
    def test_reading_1_when_canonical_is_above_on_every_lens(self):
        v = task9_verdict(_both_arms(0.90, 0.60))
        assert v["verdict"] == "READING_1_CANONICAL_BETTER"

    def test_reading_3_is_reachable_and_mirrors_reading_1(self):
        v = task9_verdict(_both_arms(0.60, 0.90))
        assert v["verdict"] == "READING_3_PRIOR_BETTER"
        assert "ESCALATE" in v["meaning"]

    def test_reading_2_when_the_arms_are_within_the_equivalence_band(self):
        v = task9_verdict(_both_arms(0.9000, 0.9002))
        assert v["verdict"] == "READING_2_EQUIVALENT"

    def test_uninformative_beats_every_other_verdict(self):
        """A failed gate must win even when the aggregate looks like a clean reading 1."""
        res = _both_arms(0.90, 0.60)
        for c in res["runs"]["prior_s1_ms"]["checkpoints"]:
            c["S_angle"] = 0.0
        for c in res["runs"]["prior_s1_roll"]["checkpoints"]:
            c["trainer"]["loss"] = 4.0
        v = task9_verdict(res)
        assert v["verdict"] == "UNINFORMATIVE"
        assert "prior_s1" in v["comparison"]["gates"]["invalid_runs"]

    def test_level_auc_disagreement_forces_mixed(self):
        """Amendment 1: S_angle and level AUC must agree in sign, else MIXED."""
        specs = {}
        for seed in (0, 1, 2):
            specs[("canonical", seed)] = {"s_angle": 0.90 + seed * 1e-4, "auc": 0.80}
            specs[("prior", seed)] = {"s_angle": 0.60 + seed * 1e-4, "auc": 0.95}
        v = task9_verdict(_result(specs))
        assert v["verdict"] == "MIXED"

    def test_a_stratum_sign_flip_forces_mixed(self):
        """A seed-separated aggregate whose sign flips in one stratum is not reading 1."""
        res = _both_arms(0.90, 0.60)
        for seed in (0, 1, 2):
            for c in res["runs"][f"canonical_s{seed}_roll"]["checkpoints"]:
                c["S_angle_by_clade"]["202"]["S"] = 0.10      # below prior in this clade only
        v = task9_verdict(res)
        assert v["verdict"] == "MIXED"

    def test_small_clade_strata_are_dropped_and_reported_not_silently_skipped(self):
        res = _both_arms(0.90, 0.60)
        for arm in ("canonical", "prior"):
            for seed in (0, 1, 2):
                for c in res["runs"][f"{arm}_s{seed}_roll"]["checkpoints"]:
                    c["S_angle_by_clade"]["202"]["n"] = 12
                    c["S_angle_by_clade"]["202"]["S"] = -5.0   # would flip the sign if counted
        v = task9_verdict(res)
        assert v["verdict"] == "READING_1_CANONICAL_BETTER"
        assert v["comparison"]["strata"]["S_angle_by_clade"]["dropped_below_min_n"]["202"] == 12

    def test_a_small_DEPTH_BAND_still_binds_the_sign_rule(self):
        """The n>=500 qualifier attaches to clade strata, not to depth bands.

        `shallow` is n=327 on the real metazoa query set. Dropping it would weaken the sign
        condition and make a verdict easier to reach than the pre-registration allows -- so a
        small band with a flipped sign must still force MIXED.
        """
        res = _both_arms(0.90, 0.60)
        for seed in (0, 1, 2):
            for c in res["runs"][f"canonical_s{seed}_roll"]["checkpoints"]:
                c["S_angle_by_band"]["shallow"]["n"] = 327
                c["S_angle_by_band"]["shallow"]["S"] = 0.10     # below prior in this band only
            for c in res["runs"][f"prior_s{seed}_roll"]["checkpoints"]:
                c["S_angle_by_band"]["shallow"]["n"] = 327
        v = task9_verdict(res)
        assert v["verdict"] == "MIXED"
        assert not v["comparison"]["strata"]["S_angle_by_band"]["dropped_below_min_n"]

    def test_poincare_above_angle_is_flagged(self):
        specs = {}
        for seed in (0, 1, 2):
            specs[("canonical", seed)] = {"s_angle": 0.90 + seed * 1e-4}
            specs[("prior", seed)] = {"s_angle": 0.60 + seed * 1e-4}
        res = _result(specs)
        for c in res["runs"]["prior_s0_roll"]["checkpoints"]:
            c["S_poincare"] = c["S_angle"] + 0.2
        v = task9_verdict(res)
        assert "prior_s0" in v["comparison"]["flags_poincare_above_angle"]


class TestTask8Readings:
    def test_material_when_the_fix_moves_the_result(self):
        v = task8_verdict(_both_arms(0.90, 0.60, arm_a="fixed", arm_b="unfixed"))
        assert v["verdict"] == "MATERIAL"
        assert "fixed > unfixed" in v["meaning"]

    def test_material_reports_the_other_direction_too(self):
        v = task8_verdict(_both_arms(0.60, 0.90, arm_a="fixed", arm_b="unfixed"))
        assert v["verdict"] == "MATERIAL"
        assert "fixed < unfixed" in v["meaning"]
        assert "regularizer" in v["meaning"]

    def test_robust_when_the_arms_overlap(self):
        v = task8_verdict(_both_arms(0.9000, 0.9002, arm_a="fixed", arm_b="unfixed"))
        assert v["verdict"] == "ROBUST"

    def test_uninformative_when_a_gate_fails(self):
        res = _both_arms(0.90, 0.60, arm_a="fixed", arm_b="unfixed")
        for c in res["runs"]["fixed_s2_ms"]["checkpoints"]:
            c["S_angle"] = 0.0
        assert task8_verdict(res)["verdict"] == "UNINFORMATIVE"


class TestEngineInvariants:
    def test_the_equivalence_band_never_falls_below_its_floor(self):
        cmp = compare_arms(_both_arms(0.90, 0.90), "canonical", "prior")
        assert cmp["equivalence"]["band"] >= 0.01

    def test_arm_order_flips_the_sign_but_not_the_decision(self):
        res = _both_arms(0.90, 0.60)
        fwd = compare_arms(res, "canonical", "prior")
        rev = compare_arms(res, "prior", "canonical")
        assert fwd["S_angle"]["mean_diff_a_minus_b"] == pytest.approx(
            -rev["S_angle"]["mean_diff_a_minus_b"])
        assert fwd["_a_wins"] and rev["_b_wins"]

    def test_per_run_value_is_the_rolling_window_mean(self):
        res = _result({("canonical", 0): {"s_angle": 0.5, "jitter": 0.01}})
        g = validity_gate(res, "canonical", 0)
        roll = [c["S_angle"] for c in res["runs"]["canonical_s0_roll"]["checkpoints"]]
        assert g["S_angle"] == pytest.approx(float(np.mean(roll)))
        assert g["roll_epochs"] == [196, 197, 198, 199, 200]


## ------------------------------------------------------------------------------------------
## P2: held-out link prediction (results/p2_heldout_preregistration.json). P2 HAS NO TRAINING-
## LOSS GATE -- three gates instead, (a) learning on MRR's own trajectory, (b) floor vs sibling
## chance, (c) completion (epoch 200 scored). Every gate below is tested BOTH ways: it fires on
## the lesion, and it stays silent on the healthy-but-unusual case -- the pair Task 9's original
## suite was missing.

P2_EPOCHS = [10, 50, 100, 150, 180, 190, 195, 198, 200]   # last 5 = the P2_ROLL_WINDOW


def _p2_ckpt(epoch, mrr, loss=4.0, depth=None):
    m = {"n": 500, "mean_rank": 3.0, "mrr": mrr, "hits_at_1": mrr,
         "hits_at_10": min(1.0, 3 * mrr), "normalized_rank": 1.0 - mrr}
    return {
        "epoch": epoch,
        "trainer": {"loss": loss},
        "metrics": m, "metrics_cosine": m, "metrics_poincare": m,
        "by_depth": {"cosine": depth if depth is not None else {
            "11-15": {"n": 2000, "mrr": mrr},
            "16-21": {"n": 3000, "mrr": mrr},
            "22-28": {"n": 800, "mrr": mrr},
        }, "poincare": {}},
    }


def _p2_series(mrrs, epochs=None, losses=None, depths=None):
    epochs = epochs if epochs is not None else P2_EPOCHS[: len(mrrs)]
    losses = losses if losses is not None else [4.0] * len(mrrs)
    depths = depths if depths is not None else [None] * len(mrrs)
    return [_p2_ckpt(e, m, l, d) for e, m, l, d in zip(epochs, mrrs, losses, depths)]


def _p2_result(groups: dict, sibling_chance_mean=0.20, randomdag_chance=None):
    """groups: {arm_seed_key: [ckpt, ...]}"""
    out = {"baselines": {"sibling_chance_mean": sibling_chance_mean},
          "arms": {k: {"checkpoints": v} for k, v in groups.items()}}
    if randomdag_chance is not None:
        out["baselines_randomdag"] = {"sibling_chance_mean": randomdag_chance}
    return out


def _p2_healthy(final_mrr, init_mrr=0.05, jitter=0.004, rng_seed=0, plateau_from_epoch=180):
    """Ramps from init_mrr toward final_mrr over the PRE-plateau epochs, then holds flat at
    final_mrr (plus jitter) from plateau_from_epoch onward, so the trailing P2_ROLL_WINDOW
    checkpoints (epochs 180-200) measure steady-state noise rather than the ramp's own trend --
    mirroring the existing suite's separate roll/milestone fixture lists (see `_run` above),
    adapted to P2's single unified checkpoint trajectory (Task 6's JSON has no roll/ms split)."""
    rng = np.random.default_rng(rng_seed)
    pre = [e for e in P2_EPOCHS if e < plateau_from_epoch]
    mrrs = []
    for ep in P2_EPOCHS:
        if ep >= plateau_from_epoch:
            base = final_mrr
        else:
            frac = pre.index(ep) / len(pre)
            base = init_mrr + (final_mrr - init_mrr) * frac
        mrrs.append(float(np.clip(base + rng.normal(0.0, jitter), 1e-6, 1.0)))
    return mrrs


def _p2_dead(level=0.05, jitter=0.01, rng_seed=0):
    """A run that never trained: wanders at ITS OWN realistic jitter scale around a low level --
    never pinned to exactly one value. A zero-SD fixture would collapse gate (a)'s `10 x jitter`
    threshold to zero and let any rise through; that exact bug shipped once in this project."""
    rng = np.random.default_rng(rng_seed)
    return [float(np.clip(level + rng.normal(0.0, jitter), 1e-6, 1.0)) for _ in P2_EPOCHS]


def _p2_group(final_by_seed_fn, base_seed, seed_offset=0.001, **kw):
    """Build 3 seeds' checkpoint lists for one arm, e.g. `_p2_group(0.55, base_seed=100)`."""
    if callable(final_by_seed_fn):
        raise TypeError("pass a base final_mrr, not a callable")
    return {s: _p2_series(_p2_healthy(final_by_seed_fn + i * seed_offset,
                                      rng_seed=base_seed + s, **kw))
           for i, s in enumerate((0, 1, 2))}


def _p2_nine_arm_result(vis00_final, vis50_final, randomdag_final, sibling_chance=0.05):
    groups = {}
    for name, final, base_seed in (("vis00", vis00_final, 100),
                                   ("vis50", vis50_final, 200),
                                   ("randomdag", randomdag_final, 300)):
        for seed, ckpts in _p2_group(final, base_seed=base_seed).items():
            groups[f"{name}_s{seed}"] = ckpts
    return _p2_result(groups, sibling_chance_mean=sibling_chance)


class TestP2ValidityGate:
    def test_gate_a_fires_on_a_dead_arm_with_realistic_jitter(self):
        """🛑 The Task 9 fixture bug, guarded against here: jitter must NOT be pinned to 0.0."""
        mrrs = _p2_dead(level=0.05, jitter=0.01, rng_seed=1)
        res = _p2_result({"vis00_s0": _p2_series(mrrs)})
        g = p2_validity_gate(res, "vis00_s0", sibling_chance_mean=0.02)
        assert g["jitter_sd"] > 0.0, "fixture must genuinely jitter, not sit at exactly one value"
        assert not g["gate_a_pass"]
        assert not g["valid"]

    def test_gate_a_stays_silent_on_a_CONVERGED_arm(self):
        """The Task 9 lesion: a trajectory that rose early and then plateaued is convergence,
        not failure to train, and must not fail gate (a)."""
        mrrs = [0.05, 0.30, 0.55, 0.70, 0.702, 0.699, 0.701, 0.698, 0.700]
        res = _p2_result({"vis00_s0": _p2_series(mrrs)})
        g = p2_validity_gate(res, "vis00_s0", sibling_chance_mean=0.02)
        assert g["gate_a_pass"]
        assert g["valid"]

    def test_gate_a_zero_jitter_is_not_trivially_passed_by_any_rise(self):
        """A degenerate zero-SD roll window (the exact historical bug) must FAIL gate (a), not
        let a trivial rise clear a threshold that collapsed to zero."""
        mrrs = [0.05, 0.05, 0.05, 0.06, 0.05, 0.05, 0.05, 0.05, 0.05]   # spike OUTSIDE the roll
        res = _p2_result({"vis00_s0": _p2_series(mrrs)})
        g = p2_validity_gate(res, "vis00_s0", sibling_chance_mean=0.02)
        assert g["jitter_sd"] == 0.0
        assert g["gate_a_rise"] > 0.0                  # there IS a rise
        assert not g["gate_a_pass"]                    # but it must not pass on that alone
        assert not g["valid"]

    def test_gate_b_fires_when_mrr_sits_at_sibling_chance(self):
        mrrs = _p2_healthy(final_mrr=0.20, init_mrr=0.05, rng_seed=2)
        chance = float(np.mean(mrrs[-5:]))              # per_run_value lands EXACTLY at chance
        res = _p2_result({"vis00_s0": _p2_series(mrrs)})
        g = p2_validity_gate(res, "vis00_s0", sibling_chance_mean=chance)
        assert g["per_run_value"] == pytest.approx(chance)
        assert not g["gate_b_pass"]
        assert not g["valid"]

    def test_gate_b_stays_silent_when_clearly_above_chance(self):
        mrrs = _p2_healthy(final_mrr=0.60, init_mrr=0.05, rng_seed=2)
        res = _p2_result({"vis00_s0": _p2_series(mrrs)})
        g = p2_validity_gate(res, "vis00_s0", sibling_chance_mean=0.20)
        assert g["gate_b_pass"]

    def test_gate_c_fires_when_the_final_checkpoint_is_missing(self):
        mrrs = _p2_healthy(final_mrr=0.60, rng_seed=2)
        ckpts = [c for c in _p2_series(mrrs) if c["epoch"] != 200]
        res = _p2_result({"vis00_s0": ckpts})
        g = p2_validity_gate(res, "vis00_s0", sibling_chance_mean=0.05)
        assert not g["gate_c_pass"]
        assert not g["valid"]

    def test_gate_c_stays_silent_when_the_final_checkpoint_is_present(self):
        mrrs = _p2_healthy(final_mrr=0.60, rng_seed=2)
        res = _p2_result({"vis00_s0": _p2_series(mrrs)})
        g = p2_validity_gate(res, "vis00_s0", sibling_chance_mean=0.05)
        assert g["gate_c_pass"]

    def test_a_rising_training_loss_alone_does_NOT_invalidate_a_run(self):
        """🛑 The Task 8 lesion, encoded so a future editor cannot quietly reintroduce a loss
        gate: loss rises throughout (3.90 -> 4.02, the fixed arm's real trajectory) while MRR
        improves throughout, and the run must still be VALID -- P2 has no loss gate at all."""
        mrrs = _p2_healthy(final_mrr=0.60, init_mrr=0.05, rng_seed=2)
        n = len(mrrs)
        rising_loss = [3.90 + (4.02 - 3.90) * (i / (n - 1)) for i in range(n)]
        res = _p2_result({"vis00_s0": _p2_series(mrrs, losses=rising_loss)})
        g = p2_validity_gate(res, "vis00_s0", sibling_chance_mean=0.05)
        assert g["gate_a_pass"] and g["gate_b_pass"] and g["gate_c_pass"]
        assert g["valid"]
        # and the rising loss is nowhere in the gate's own reasoning:
        assert not any("loss" in key for key in g), (
            f"a loss-derived field crept back into the gate: {[k for k in g if 'loss' in k]}")


class TestP2Verdict:
    def test_generalises_when_both_real_arms_clear_chance_and_randomdag(self):
        res = _p2_nine_arm_result(vis00_final=0.55, vis50_final=0.50, randomdag_final=0.10,
                                  sibling_chance=0.05)
        v = p2_verdict(res)
        assert v["verdict"] == "GENERALISES"
        assert v["by_arm_verdict"]["vis00"]["verdict"] == "GENERALISES"
        assert v["by_arm_verdict"]["vis50"]["verdict"] == "GENERALISES"
        assert not v["control"]["invalid_runs"]

    def test_memorises_when_arms_are_indistinguishable_from_randomdag(self):
        res = _p2_nine_arm_result(vis00_final=0.11, vis50_final=0.115, randomdag_final=0.11,
                                  sibling_chance=0.05)
        v = p2_verdict(res)
        assert v["verdict"] == "MEMORISES"
        assert v["by_arm_verdict"]["vis00"]["verdict"] == "MEMORISES"
        assert v["by_arm_verdict"]["vis50"]["verdict"] == "MEMORISES"

    def test_uninformative_when_any_single_seed_fails_a_gate(self):
        """UNINFORMATIVE must beat every other reading, even for a clean-looking pair of arms --
        including a failure on the RandomDAG control, not only on a data arm."""
        res = _p2_nine_arm_result(vis00_final=0.55, vis50_final=0.50, randomdag_final=0.10,
                                  sibling_chance=0.05)
        res["arms"]["randomdag_s1"]["checkpoints"] = [
            c for c in res["arms"]["randomdag_s1"]["checkpoints"] if c["epoch"] != 200]
        v = p2_verdict(res)
        assert v["verdict"] == "UNINFORMATIVE"
        assert "randomdag_s1" in v["control"]["invalid_runs"]
        assert "by_arm_verdict" not in v            # no direction read for either arm

    def test_a_depth_stratum_sign_flip_forces_MIXED_for_that_arm_and_overall(self):
        """A seed-separated aggregate whose sign flips in one depth stratum is not GENERALISES
        for that arm -- mirrors the Task 9 engine's stratum sign-consistency rule."""
        res = _p2_nine_arm_result(vis00_final=0.55, vis50_final=0.50, randomdag_final=0.10,
                                  sibling_chance=0.05)
        for s in (0, 1, 2):
            ckpts = res["arms"][f"vis00_s{s}"]["checkpoints"]
            final = next(c for c in ckpts if c["epoch"] == 200)
            # below EVERY randomdag seed's own "22-28" value (~0.10-0.11) in this stratum only
            final["by_depth"]["cosine"]["22-28"] = {"n": 800, "mrr": 0.02}
        v = p2_verdict(res)
        assert v["by_arm_verdict"]["vis00"]["verdict"] == "MIXED"
        assert not v["by_arm_verdict"]["vis00"]["sign_consistent"]
        assert v["by_arm_verdict"]["vis50"]["verdict"] == "GENERALISES"   # untouched
        assert v["verdict"] == "MIXED"                                    # arms disagree

    def test_small_depth_strata_are_dropped_and_reported_not_silently_skipped(self):
        res = _p2_nine_arm_result(vis00_final=0.55, vis50_final=0.50, randomdag_final=0.10,
                                  sibling_chance=0.05)
        for s in (0, 1, 2):
            ckpts = res["arms"][f"vis00_s{s}"]["checkpoints"]
            final = next(c for c in ckpts if c["epoch"] == 200)
            final["by_depth"]["cosine"]["22-28"] = {"n": 12, "mrr": -5.0}   # would flip the sign
        v = p2_verdict(res)
        assert v["by_arm_verdict"]["vis00"]["verdict"] == "GENERALISES"
        dropped = v["by_arm_verdict"]["vis00"]["depth_strata"]["dropped_below_min_n"]
        assert dropped.get("22-28") == 12

    def test_randomdag_arm_can_use_its_own_sibling_chance_mean(self):
        """RandomDAG's rewiring changes grandparent fan-out, so its own floor can genuinely
        differ from the real-tree arms' -- `baselines_randomdag` must be read when present."""
        res = _p2_nine_arm_result(vis00_final=0.55, vis50_final=0.50, randomdag_final=0.10,
                                  sibling_chance=0.05)
        res["baselines_randomdag"] = {"sibling_chance_mean": 0.30}   # randomdag now AT its floor
        v = p2_verdict(res)
        assert v["verdict"] == "UNINFORMATIVE"
        assert any(a.startswith("randomdag_") for a in v["control"]["invalid_runs"])


class TestP2PreregistrationArtifact:
    """The frozen JSON itself must state what Ruling 1 requires, not just this session's prose."""

    @pytest.fixture(scope="class")
    def prereg(self):
        path = _REPO / "results" / "p2_heldout_preregistration.json"
        return json.loads(path.read_text())

    def test_required_top_level_keys_are_present(self, prereg):
        for key in ("status", "task", "why", "arms", "declared_confounds", "metrics",
                   "validity_gate", "readings", "manuscript_hooks"):
            assert key in prereg

    def test_status_declares_frozen_before_any_run_and_amendment_only_revision(self, prereg):
        assert "frozen" in prereg["status"].lower()
        assert "amendment" in prereg["status"].lower()

    def test_nine_arms_declared_including_randomdag_from_the_outset(self, prereg):
        for name in ("vis00", "vis50", "randomdag"):
            assert name in prereg["arms"]
        assert "randomised" in prereg["arms"]["randomdag"].lower()

    def test_exactly_four_declared_confounds(self, prereg):
        assert len(prereg["declared_confounds"]) == 4

    def test_all_four_readings_declared(self, prereg):
        assert set(prereg["readings"]) == {"GENERALISES", "MEMORISES", "MIXED", "UNINFORMATIVE"}

    def test_validity_gate_states_there_is_no_loss_gate(self, prereg):
        gate_text = prereg["validity_gate"].lower()
        assert "no training-loss gate" in gate_text
        assert "loss" in gate_text                  # still discussed, just never gates
        assert "epoch 200" in prereg["validity_gate"]

    def test_primary_metric_is_cosine_with_the_planted_radius_reasoning(self, prereg):
        assert "cosine" in prereg["metrics"]["primary"].lower()
        assert "planted" in prereg["metrics"]["primary_reason"].lower()
