"""The pre-registration engine must be able to return every verdict it can return.

A decision rule that only ever emits one label is not a rule. Each test below drives the engine
to a DIFFERENT outcome from synthetic scorer output, so a change that collapses the rule onto a
single verdict fails here rather than in the manuscript.
"""
from __future__ import annotations

import numpy as np
import pytest

from taxembed.eval.preregistration import (
    compare_arms, task8_verdict, task9_verdict, validity_gate,
)


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
