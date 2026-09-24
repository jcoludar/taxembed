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

from taxembed.eval.linkpred import linkpred_metrics
from taxembed.eval.preregistration import (
    P2_ROLL_WINDOW, compare_arms, merge_p2_scorer_outputs, p2_group_stats, p2_run_value,
    p2_validity_gate, p2_verdict, task8_verdict, task9_verdict, validity_gate,
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


# C1/C2 default: a WEAK training-free baseline, comfortably below every existing fixture's
# healthy final MRR (>= 0.10) -- so it is a no-op for old fixtures unless a test overrides it to
# specifically exercise the new degree-prior clause (TestP2DegreePriorClause below).
_DEFAULT_DEGREE_PRIOR = {"mrr": 0.03, "hits_at_1": 0.02, "hits_at_10": 0.06,
                         "normalized_rank": 0.45, "n": 500, "n_scored": 500, "n_trivial": 0}


def _p2_fanout_arms(groups: dict) -> dict:
    """{arm_seed_key: [ckpt, ...]} -> {f'{key}_ms': {...}, f'{key}_roll': {...}}.

    C4 #1 (2026-09-24, `p2_amendment_3_20260924`): production (`scripts/p2_lrz_score.sh`) splits
    each run into TWO registered checkpoint groups, `*_ms` (milestones, epoch 10..200) and `*_roll`
    (rolling checkpoints nearest epoch 200) -- mirrors Task 9's `_roll_key`/`_ms_key` convention.
    Every P2 fixture builder below still describes ONE trajectory per arm+seed (unchanged); this
    is the single place that fans it out into the two suffixed keys the engine now reads
    (`_p2_roll_key`/`_p2_ms_key`), so no individual builder needed to change its own logic.
    """
    arms = {}
    for key, ckpts in groups.items():
        arms[f"{key}_ms"] = {"checkpoints": ckpts}
        arms[f"{key}_roll"] = {"checkpoints": ckpts[-P2_ROLL_WINDOW:]}
    return arms


def _p2_baselines(sibling_chance_mean, chance_mrr_mean=None, degree_prior=None) -> dict:
    """One baselines block, with the C1/C2 defaults described above `_DEFAULT_DEGREE_PRIOR`.

    `chance_mrr_mean` defaults to `sibling_chance_mean` itself (2026-09-24): a caller not
    exercising the C1/C2 distinction gets gate (b) behaving EXACTLY as before this fix (`per_run_
    value > sibling_chance_mean`, since the two values are then numerically identical) -- the
    default is a no-op, not a silent behaviour change, for every pre-existing fixture.
    """
    return {
        "sibling_chance_mean": sibling_chance_mean,
        "chance_mrr_mean": chance_mrr_mean if chance_mrr_mean is not None else sibling_chance_mean,
        "degree_prior": degree_prior if degree_prior is not None else dict(_DEFAULT_DEGREE_PRIOR),
    }


def _p2_result(groups: dict, sibling_chance_mean=0.20, randomdag_chance=None,
              chance_mrr_mean=None, degree_prior=None,
              randomdag_chance_mrr_mean=None, randomdag_degree_prior=None):
    """groups: {arm_seed_key: [ckpt, ...]} -- ONE trajectory per arm+seed, as before; fanned out
    into `_ms`/`_roll` by `_p2_fanout_arms` (C4 #1) and given the C1/C2 baseline defaults above."""
    out = {
        "baselines": _p2_baselines(sibling_chance_mean, chance_mrr_mean, degree_prior),
        "arms": _p2_fanout_arms(groups),
    }
    if randomdag_chance is not None:
        out["baselines_randomdag"] = _p2_baselines(
            randomdag_chance, randomdag_chance_mrr_mean, randomdag_degree_prior)
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
    """`p2_validity_gate(result, arm, seed, sibling_chance_mean, chance_mrr_mean)` (2026-09-24,
    C4 #1: `arm_seed_key` split into `arm, seed`; I3: gate (a)'s anchor is now `chance_mrr_mean`,
    not the run's own epoch-10 value, so every test below passes it explicitly, chosen per test
    to reproduce the ORIGINAL intent (a self-referential anchor close to the fixture's own level
    for the gate-a-focused tests; a low external anchor for the gate-b-focused/healthy-run tests)."""

    def test_gate_a_fires_on_a_dead_arm_with_realistic_jitter(self):
        """🛑 The Task 9 fixture bug, guarded against here: jitter must NOT be pinned to 0.0."""
        mrrs = _p2_dead(level=0.05, jitter=0.01, rng_seed=1)
        res = _p2_result({"vis00_s0": _p2_series(mrrs)})
        g = p2_validity_gate(res, "vis00", 0, sibling_chance_mean=0.02, chance_mrr_mean=0.05)
        assert g["jitter_sd"] > 0.0, "fixture must genuinely jitter, not sit at exactly one value"
        assert not g["gate_a_pass"]
        assert not g["valid"]

    def test_gate_a_stays_silent_on_a_CONVERGED_arm(self):
        """The Task 9 lesion: a trajectory that rose early and then plateaued is convergence,
        not failure to train, and must not fail gate (a)."""
        mrrs = [0.05, 0.30, 0.55, 0.70, 0.702, 0.699, 0.701, 0.698, 0.700]
        res = _p2_result({"vis00_s0": _p2_series(mrrs)})
        g = p2_validity_gate(res, "vis00", 0, sibling_chance_mean=0.02, chance_mrr_mean=0.02)
        assert g["gate_a_pass"]
        assert g["valid"]

    def test_gate_a_zero_jitter_is_not_trivially_passed_by_any_rise(self):
        """A degenerate zero-SD roll window (the exact historical bug) must FAIL gate (a), not
        let a trivial rise clear a threshold that collapsed to zero."""
        mrrs = [0.05, 0.05, 0.05, 0.06, 0.05, 0.05, 0.05, 0.05, 0.05]   # spike OUTSIDE the roll
        res = _p2_result({"vis00_s0": _p2_series(mrrs)})
        g = p2_validity_gate(res, "vis00", 0, sibling_chance_mean=0.02, chance_mrr_mean=0.05)
        assert g["jitter_sd"] == 0.0
        assert g["gate_a_rise"] > 0.0                  # there IS a rise
        assert not g["gate_a_pass"]                    # but it must not pass on that alone
        assert not g["valid"]

    def test_gate_b_fires_when_mrr_sits_at_chance_mrr_mean(self):
        mrrs = _p2_healthy(final_mrr=0.20, init_mrr=0.05, rng_seed=2)
        chance = float(np.mean(mrrs[-5:]))              # per_run_value lands EXACTLY at chance
        res = _p2_result({"vis00_s0": _p2_series(mrrs)})
        g = p2_validity_gate(res, "vis00", 0, sibling_chance_mean=0.05, chance_mrr_mean=chance)
        assert g["per_run_value"] == pytest.approx(chance)
        assert not g["gate_b_pass"]
        assert not g["valid"]

    def test_gate_b_stays_silent_when_clearly_above_chance(self):
        mrrs = _p2_healthy(final_mrr=0.60, init_mrr=0.05, rng_seed=2)
        res = _p2_result({"vis00_s0": _p2_series(mrrs)})
        g = p2_validity_gate(res, "vis00", 0, sibling_chance_mean=0.20, chance_mrr_mean=0.20)
        assert g["gate_b_pass"]

    def test_gate_c_fires_when_the_final_checkpoint_is_missing(self):
        mrrs = _p2_healthy(final_mrr=0.60, rng_seed=2)
        ckpts = [c for c in _p2_series(mrrs) if c["epoch"] != 200]
        res = _p2_result({"vis00_s0": ckpts})
        g = p2_validity_gate(res, "vis00", 0, sibling_chance_mean=0.05, chance_mrr_mean=0.05)
        assert not g["gate_c_pass"]
        assert not g["valid"]

    def test_gate_c_stays_silent_when_the_final_checkpoint_is_present(self):
        mrrs = _p2_healthy(final_mrr=0.60, rng_seed=2)
        res = _p2_result({"vis00_s0": _p2_series(mrrs)})
        g = p2_validity_gate(res, "vis00", 0, sibling_chance_mean=0.05, chance_mrr_mean=0.05)
        assert g["gate_c_pass"]

    def test_a_rising_training_loss_alone_does_NOT_invalidate_a_run(self):
        """🛑 The Task 8 lesion, encoded so a future editor cannot quietly reintroduce a loss
        gate: loss rises throughout (3.90 -> 4.02, the fixed arm's real trajectory) while MRR
        improves throughout, and the run must still be VALID -- P2 has no loss gate at all."""
        mrrs = _p2_healthy(final_mrr=0.60, init_mrr=0.05, rng_seed=2)
        n = len(mrrs)
        rising_loss = [3.90 + (4.02 - 3.90) * (i / (n - 1)) for i in range(n)]
        res = _p2_result({"vis00_s0": _p2_series(mrrs, losses=rising_loss)})
        g = p2_validity_gate(res, "vis00", 0, sibling_chance_mean=0.05, chance_mrr_mean=0.05)
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
        # gate (c) reads the milestone (`_ms`) trajectory (C4 #1) -- remove epoch 200 from there.
        res["arms"]["randomdag_s1_ms"]["checkpoints"] = [
            c for c in res["arms"]["randomdag_s1_ms"]["checkpoints"] if c["epoch"] != 200]
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
            # `_p2_depth_means` reads the epoch-200 checkpoint off the `_ms` trajectory (C4 #1).
            ckpts = res["arms"][f"vis00_s{s}_ms"]["checkpoints"]
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
            ckpts = res["arms"][f"vis00_s{s}_ms"]["checkpoints"]
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
        # randomdag now AT its floor -- chance_mrr_mean/degree_prior are REQUIRED keys (C1/C2:
        # p2_group_stats reads them directly, no silent fallback), so a hand-authored override
        # must supply them too, not just sibling_chance_mean.
        res["baselines_randomdag"] = {"sibling_chance_mean": 0.30, "chance_mrr_mean": 0.30,
                                      "degree_prior": dict(_DEFAULT_DEGREE_PRIOR)}
        v = p2_verdict(res)
        assert v["verdict"] == "UNINFORMATIVE"
        assert any(a.startswith("randomdag_") for a in v["control"]["invalid_runs"])


## ------------------------------------------------------------------------------------------
## p2_amendment_1_20260924 -- RandomDAG's rewiring measurably inflates its own chance floor
## (~5.4x, helpers/p2_randomdag_changes_the_chance_floor.py, mollusca seed 0), so a CROSS-TREE
## comparison (an arm vs RandomDAG) must use normalized_rank (chance=0.5 for every pool size),
## never raw MRR. Below, `mrr` and `normalized_rank` are set INDEPENDENTLY per checkpoint (unlike
## `_p2_ckpt` above, which couples them as `1.0 - mrr`) so a fixture can express "RandomDAG's raw
## MRR is higher only because its task is easier, while its normalized_rank shows it is still at
## chance" -- the exact shape of the defect.

def _p2_decoupled_ckpt(epoch, mrr, nr, loss=4.0):
    m = {"n": 500, "mean_rank": 3.0, "mrr": mrr, "hits_at_1": mrr,
         "hits_at_10": min(1.0, 3 * mrr), "normalized_rank": nr}
    depth = {k: {"n": 2000, "mrr": mrr, "normalized_rank": nr}
             for k in ("11-15", "16-21", "22-28")}
    return {
        "epoch": epoch, "trainer": {"loss": loss},
        "metrics": m, "metrics_cosine": m, "metrics_poincare": m,
        "by_depth": {"cosine": depth, "poincare": {}},
    }


def _p2_decoupled_series(mrr_final, nr_final, mrr_init=0.02, nr_init=0.95,
                         jitter_mrr=0.004, jitter_nr=0.01, rng_seed=0, plateau_from_epoch=180):
    """Ramps mrr and normalized_rank INDEPENDENTLY from their own init toward their own final
    value, then holds flat (plus jitter) from `plateau_from_epoch` -- same shape as `_p2_healthy`
    above, decoupled so the two metrics can disagree, as the amendment's defect requires."""
    rng = np.random.default_rng(rng_seed)
    pre = [e for e in P2_EPOCHS if e < plateau_from_epoch]
    out = []
    for ep in P2_EPOCHS:
        if ep >= plateau_from_epoch:
            mrr = mrr_final + rng.normal(0.0, jitter_mrr)
            nr = nr_final + rng.normal(0.0, jitter_nr)
        else:
            frac = pre.index(ep) / len(pre)
            mrr = mrr_init + (mrr_final - mrr_init) * frac
            nr = nr_init + (nr_final - nr_init) * frac
        out.append(_p2_decoupled_ckpt(ep, float(np.clip(mrr, 1e-6, 1.0)),
                                      float(np.clip(nr, 1e-6, 1.0))))
    return out


def _p2_decoupled_result(vis00, vis50, randomdag, sibling_chance, randomdag_chance,
                         seed_offset_mrr=0.03, seed_offset_nr=0.01):
    """vis00/vis50/randomdag are each (mrr_final, nr_final) pairs. Seeds 0/1/2 spread the MRR
    UP (so ranges can be made to overlap the control's, as the frozen rule's `ranges_overlap`
    needs to land on MEMORISES) and spread normalized_rank DOWN i.e. better (so amendment_1's
    all-below-all condition is comfortably clean, not a coin flip on which seed lands where)."""
    groups = {}
    for name, (mrr_final, nr_final), base_seed in (
        ("vis00", vis00, 100), ("vis50", vis50, 200), ("randomdag", randomdag, 300),
    ):
        for i, s in enumerate((0, 1, 2)):
            groups[f"{name}_s{s}"] = _p2_decoupled_series(
                mrr_final=mrr_final + i * seed_offset_mrr,
                nr_final=max(1e-6, nr_final - i * seed_offset_nr),
                rng_seed=base_seed + s)
    return {
        "baselines": _p2_baselines(sibling_chance),
        "baselines_randomdag": _p2_baselines(randomdag_chance),
        "arms": _p2_fanout_arms(groups),
    }


class TestNormalizedRankChanceLevel:
    """The amendment's central premise: normalized_rank's chance level is 0.5 for EVERY pool
    size, unlike raw MRR (chance ~ 1/pool, pool-size-dependent). If this were false, chance-
    normalising the cross-tree comparison would not fix anything."""

    @pytest.mark.parametrize("pool", [2, 500])
    def test_random_ranks_average_to_one_half_regardless_of_pool_size(self, pool):
        rng = np.random.default_rng(20260924)
        n_queries = 20000
        ranks = rng.integers(1, pool + 1, size=n_queries)      # uniform on 1..pool inclusive
        n_candidates = np.full(n_queries, pool, dtype=np.int64)
        m = linkpred_metrics(ranks, n_candidates)
        assert m["normalized_rank"] == pytest.approx(0.5, abs=0.02)


class TestP2Amendment1:
    def test_amendment_1_prevents_a_false_memorises_when_randomdag_is_an_easier_task(self):
        """THE test amendment_1 exists for. RandomDAG's raw MRR (~0.40) is HIGHER than vis00's
        (~0.38) -- an easier task scoring better on the confounded metric, ranges overlapping --
        while vis00's normalized_rank (~0.21, well under 0.5) is far BETTER than RandomDAG's
        (~0.54, at/above chance). Under the FROZEN rule (raw MRR, amendment_1=False) this reads
        MEMORISES: this test must fail if the amendment is reverted."""
        res = _p2_decoupled_result(
            vis00=(0.38, 0.21), vis50=(0.39, 0.19), randomdag=(0.40, 0.54),
            sibling_chance=0.05, randomdag_chance=0.27)   # ~5.4x, the measured ratio

        frozen = p2_verdict(res, amendment_1=False)
        assert frozen["by_arm_verdict"]["vis00"]["verdict"] == "MEMORISES", (
            "fixture must reproduce the defect on the frozen rule for this test to mean anything")

        amended = p2_verdict(res, amendment_1=True)
        assert amended["by_arm_verdict"]["vis00"]["verdict"] != "MEMORISES"
        assert amended["by_arm_verdict"]["vis00"]["verdict"] == "GENERALISES"
        assert amended["by_arm_verdict"]["vis00"]["cross_tree_metric"] == "normalized_rank"
        assert amended["by_arm_verdict"]["vis00"]["below_control_all_seeds"]
        assert amended["by_arm_verdict"]["vis00"]["below_chance_level"]

    def test_default_call_still_reproduces_the_original_frozen_reading(self):
        """A bare `p2_verdict(res)` call must still reproduce the pre-amendment_1 verdicts on the
        pre-amendment_1 fixtures -- the amendment must not silently rewrite history.

        (fix round 2, 2026-09-24: this test used to also assert
        `p2_verdict(res) == p2_verdict(res, amendment_1=False)`. Since `amendment_1` already
        defaults to False, that compared two calls with IDENTICAL effective arguments -- it could
        not fail for any implementation of `p2_verdict`'s logic, correct or not, and so carried no
        regression signal. The assertions kept below already do the real job: they pin the
        default call's verdict against known fixtures, which a change to the actual reading WOULD
        break.)"""
        res_generalises = _p2_nine_arm_result(vis00_final=0.55, vis50_final=0.50,
                                              randomdag_final=0.10, sibling_chance=0.05)
        assert p2_verdict(res_generalises)["verdict"] == "GENERALISES"
        assert p2_verdict(res_generalises)["amendment_1_applied"] is False

        res_memorises = _p2_nine_arm_result(vis00_final=0.11, vis50_final=0.115,
                                            randomdag_final=0.11, sibling_chance=0.05)
        assert p2_verdict(res_memorises, amendment_1=False)["verdict"] == "MEMORISES"

    def test_own_floor_margin_is_untouched_by_the_amendment(self):
        """The own-tree MRR-vs-own-sibling_chance_mean margin (rule_2: within-tree, unaffected)
        must be numerically IDENTICAL whether or not amendment_1 is applied."""
        res = _p2_decoupled_result(
            vis00=(0.38, 0.21), vis50=(0.39, 0.19), randomdag=(0.40, 0.54),
            sibling_chance=0.05, randomdag_chance=0.27)
        frozen = p2_verdict(res, amendment_1=False)
        amended = p2_verdict(res, amendment_1=True)
        for name in ("vis00", "vis50"):
            assert (frozen["by_arm_verdict"][name]["margin_above_chance"]
                   == pytest.approx(amended["by_arm_verdict"][name]["margin_above_chance"]))
            assert frozen["by_arm_verdict"][name]["above_chance_margin"] == (
                amended["by_arm_verdict"][name]["above_chance_margin"])


## ------------------------------------------------------------------------------------------
## p2_amendment_2_20260924 -- USER DESIGN decision: run RandomDAG at BOTH visibilities, so every
## real arm (vis00, vis50) has a matched control trained the same way (randomdag_vis00,
## randomdag_vis50). 12 arms, not 9. vis00 is read ONLY against randomdag_vis00, vis50 ONLY
## against randomdag_vis50 -- never cross-matched.

def _p2_twelve_arm_result(vis00_final, vis50_final, randomdag_vis00_final, randomdag_vis50_final,
                          sibling_chance=0.05):
    groups = {}
    for name, final, base_seed in (
        ("vis00", vis00_final, 100), ("vis50", vis50_final, 200),
        ("randomdag_vis00", randomdag_vis00_final, 300),
        ("randomdag_vis50", randomdag_vis50_final, 400),
    ):
        for seed, ckpts in _p2_group(final, base_seed=base_seed).items():
            groups[f"{name}_s{seed}"] = ckpts
    return _p2_result(groups, sibling_chance_mean=sibling_chance)


class TestP2Amendment2:
    def test_matched_controls_read_each_real_arm_against_its_own_visibility(self):
        """The 12-arm reading with matched controls produces the expected verdict on a synthetic
        fixture: both real arms clear their own chance floor and their own matched (weak)
        control, so the overall reading is GENERALISES, exactly as the 9-arm design would give
        for the same numbers."""
        res = _p2_twelve_arm_result(vis00_final=0.55, vis50_final=0.50,
                                    randomdag_vis00_final=0.10, randomdag_vis50_final=0.10,
                                    sibling_chance=0.05)
        v = p2_verdict(res, amendment_2=True)
        assert v["amendment_2_applied"] is True
        assert v["verdict"] == "GENERALISES"
        assert v["by_arm_verdict"]["vis00"]["verdict"] == "GENERALISES"
        assert v["by_arm_verdict"]["vis50"]["verdict"] == "GENERALISES"
        assert set(v["controls"]) == {"randomdag_vis00", "randomdag_vis50"}
        assert not v["controls"]["randomdag_vis00"]["invalid_runs"]
        assert not v["controls"]["randomdag_vis50"]["invalid_runs"]
        assert "control" not in v

    def test_amendment_2_default_false_reproduces_the_original_nine_arm_reading(self):
        """amendment_2 defaults to False: the 9-arm reading (one shared control) must be produced
        unchanged, on the pre-amendment_2 fixtures -- the new flag must not rewrite the old one's
        behaviour."""
        res = _p2_nine_arm_result(vis00_final=0.55, vis50_final=0.50, randomdag_final=0.10,
                                  sibling_chance=0.05)
        default = p2_verdict(res)
        explicit_false = p2_verdict(res, amendment_2=False)
        assert default["verdict"] == explicit_false["verdict"] == "GENERALISES"
        assert default["amendment_2_applied"] is False
        assert "controls" not in default
        assert default["control"]["invalid_runs"] == []

    def test_cross_matching_the_controls_changes_the_verdict(self):
        """Load-bearing: if the matched pairing were decorative, swapping which control's data
        sits under which label would leave the verdict unchanged. Here `randomdag_vis00` is a
        WEAK control (final MRR 0.10, far below vis50) and `randomdag_vis50` is a STRONG one
        (final MRR 0.499, seed ranges overlapping vis50's own 0.50) -- read against its correctly
        matched (strong, overlapping) control, vis50 must NOT clear the all-seeds-above condition
        (MEMORISES); read against the other (weak) one, it clears it easily (GENERALISES). The
        pairing being load-bearing means swapping the two controls' underlying data between their
        labels must change vis50's verdict."""
        matched_source = _p2_twelve_arm_result(
            vis00_final=0.55, vis50_final=0.50,
            randomdag_vis00_final=0.10, randomdag_vis50_final=0.499, sibling_chance=0.05)
        v_matched = p2_verdict(matched_source, amendment_2=True)
        assert v_matched["by_arm_verdict"]["vis50"]["verdict"] == "MEMORISES", (
            "fixture must reproduce the close-control MEMORISES reading for this test to mean "
            "anything")

        # cross-matched: swap the two RandomDAG arms' checkpoint data between their labels, so
        # `randomdag_vis50` (still matched to vis50 by NAME) now carries the WEAK data and
        # `randomdag_vis00` carries the STRONG data -- the wrong pairing, by construction.
        cross = json.loads(json.dumps(matched_source))
        for s in (0, 1, 2):
            for suffix in ("_ms", "_roll"):     # C4 #1: swap BOTH registered checkpoint groups
                key_a = f"randomdag_vis00_s{s}{suffix}"
                key_b = f"randomdag_vis50_s{s}{suffix}"
                cross["arms"][key_a], cross["arms"][key_b] = cross["arms"][key_b], cross["arms"][key_a]
        v_cross = p2_verdict(cross, amendment_2=True)

        assert v_cross["by_arm_verdict"]["vis50"]["verdict"] == "GENERALISES"
        assert (v_matched["by_arm_verdict"]["vis50"]["verdict"]
               != v_cross["by_arm_verdict"]["vis50"]["verdict"])

    def test_uninformative_when_a_single_control_seed_fails_a_gate(self):
        """UNINFORMATIVE must beat every other reading even under the 12-arm design, including a
        failure confined to ONE of the two matched controls."""
        res = _p2_twelve_arm_result(vis00_final=0.55, vis50_final=0.50,
                                    randomdag_vis00_final=0.10, randomdag_vis50_final=0.10,
                                    sibling_chance=0.05)
        res["arms"]["randomdag_vis50_s1_ms"]["checkpoints"] = [
            c for c in res["arms"]["randomdag_vis50_s1_ms"]["checkpoints"] if c["epoch"] != 200]
        v = p2_verdict(res, amendment_2=True)
        assert v["verdict"] == "UNINFORMATIVE"
        assert "randomdag_vis50_s1" in v["meaning"]
        assert "by_arm_verdict" not in v


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

    def test_frozen_block_is_untouched_by_the_amendment(self, prereg):
        """p2_amendment_1_20260924 must be an ADDED top-level key, never an edit of the frozen
        block -- declared_confounds stays at its originally-frozen count of 4."""
        assert len(prereg["declared_confounds"]) == 4
        assert set(prereg["readings"]) == {"GENERALISES", "MEMORISES", "MIXED", "UNINFORMATIVE"}

    def test_amendment_1_block_present_and_predates_any_p2_run(self, prereg):
        assert "p2_amendment_1_20260924" in prereg
        status = prereg["p2_amendment_1_20260924"]["status"].lower()
        assert "before any p2 array was submitted" in status
        assert "before any p2 outcome data" in status
        assert "2026-09-24" in prereg["p2_amendment_1_20260924"]["status"]

    def test_amendment_1_records_the_measured_chance_floor_ratio(self, prereg):
        blob = json.dumps(prereg["p2_amendment_1_20260924"])
        assert "5.403" in blob
        assert "normalized_rank" in blob

    def test_amendment_2_block_present_and_predates_any_p2_run(self, prereg):
        assert "p2_amendment_2_20260924" in prereg
        status = prereg["p2_amendment_2_20260924"]["status"].lower()
        assert "before any p2 array was submitted" in status
        assert "before any p2 outcome data" in status
        assert "user design" in status
        assert "2026-09-24" in prereg["p2_amendment_2_20260924"]["status"]

    def test_amendment_2_declares_twelve_arms_with_matched_never_crossed_controls(self, prereg):
        blob = json.dumps(prereg["p2_amendment_2_20260924"])
        assert "randomdag_vis00" in blob and "randomdag_vis50" in blob
        assert "12" in blob
        assert "never" in blob.lower() and "cross" in blob.lower()

    def test_amendment_1_and_2_blocks_leave_each_other_and_the_frozen_block_untouched(self, prereg):
        """Rule 7 of the task: amendment_2 is additive only. declared_confounds and the readings
        set are the frozen block's own invariants (checked identically by the amendment_1 tests
        above); re-asserting them here guards against amendment_2 having edited the frozen block
        or amendment_1 in place instead of adding a new top-level key."""
        assert len(prereg["declared_confounds"]) == 4
        assert set(prereg["readings"]) == {"GENERALISES", "MEMORISES", "MIXED", "UNINFORMATIVE"}
        assert "5.403" in json.dumps(prereg["p2_amendment_1_20260924"])


## ------------------------------------------------------------------------------------------
## p2_amendment_3_20260924 review-finding fixes: C1 (degree prior), C2 (chance_mrr_mean anchor),
## I3 (gate (a)'s anchor), C4 (merge_p2_scorer_outputs). Each class below fires on the lesion
## (the exact old behaviour the fix replaces) and stays silent on the healthy/control case.


class TestP2DegreePriorClause:
    """C1: GENERALISES must require beating the training-free degree prior, not merely chance."""

    def test_a_strong_degree_prior_blocks_generalises(self):
        """LESION CHECK. The SAME fixture the pre-C1 suite already reads as GENERALISES -- but
        with the degree prior's OWN MRR set above the arms' MRR, so a model doing no better than
        the prior must NOT read GENERALISES. Reverting the C1 clause (dropping `above_degree_
        prior` from the verdict condition) makes this test fail."""
        res = _p2_nine_arm_result(vis00_final=0.55, vis50_final=0.50, randomdag_final=0.10,
                                  sibling_chance=0.05)
        res["baselines"]["degree_prior"] = {"mrr": 0.60, "hits_at_1": 0.5, "hits_at_10": 0.8,
                                            "normalized_rank": 0.1, "n": 500}
        v = p2_verdict(res)
        assert v["by_arm_verdict"]["vis00"]["verdict"] == "MEMORISES"
        assert v["by_arm_verdict"]["vis00"]["above_degree_prior"] is False

    def test_a_weak_degree_prior_does_not_block_generalises(self):
        """HEALTHY CASE. The default weak degree prior (well below the arms' MRR) must not
        interfere with an otherwise-clean GENERALISES reading."""
        res = _p2_nine_arm_result(vis00_final=0.55, vis50_final=0.50, randomdag_final=0.10,
                                  sibling_chance=0.05)
        v = p2_verdict(res)
        assert v["by_arm_verdict"]["vis00"]["verdict"] == "GENERALISES"
        assert v["by_arm_verdict"]["vis00"]["above_degree_prior"] is True

    def test_amendment_1_normalized_rank_branch_also_requires_beating_the_degree_prior(self):
        """The SAME clause applies to the amendment_1 (normalized_rank, cross-tree) reading --
        C1 is not specific to raw MRR."""
        res = _p2_decoupled_result(
            vis00=(0.38, 0.21), vis50=(0.39, 0.19), randomdag=(0.40, 0.54),
            sibling_chance=0.05, randomdag_chance=0.27)
        res["baselines"]["degree_prior"] = {"mrr": 0.03, "hits_at_1": 0.02, "hits_at_10": 0.06,
                                            "normalized_rank": 0.05,  # BELOW vis00's own ~0.21
                                            "n": 500}
        v = p2_verdict(res, amendment_1=True)
        assert v["by_arm_verdict"]["vis00"]["verdict"] != "GENERALISES"
        assert v["by_arm_verdict"]["vis00"]["below_degree_prior"] is False


class TestP2ChanceMRRGateAnchor:
    """C2: gate (b) must compare against chance_mrr_mean (= mean(H_k/k), the correct chance
    level for MRR), not sibling_chance_mean (= mean(1/k), the chance level for hits@1)."""

    def test_gate_b_fails_a_run_between_the_two_chance_rates(self):
        """LESION CHECK. per_run_value (~0.15) clears the OLD anchor (sibling_chance_mean=0.10)
        but sits BELOW the correct one (chance_mrr_mean=0.25). Reverting C2 (gate (b) reading
        sibling_chance_mean again) makes gate_b wrongly PASS this run."""
        mrrs = _p2_healthy(final_mrr=0.15, init_mrr=0.05, jitter=0.001, rng_seed=3)
        res = _p2_result({"vis00_s0": _p2_series(mrrs)})
        g = p2_validity_gate(res, "vis00", 0, sibling_chance_mean=0.10, chance_mrr_mean=0.25)
        assert g["per_run_value"] > 0.10          # clears the OLD (wrong) anchor
        assert g["per_run_value"] < 0.25          # but not the CORRECT one
        assert not g["gate_b_pass"]

    def test_gate_b_passes_a_run_above_both_chance_rates(self):
        """HEALTHY CASE: a run comfortably above BOTH chance rates passes regardless of which
        one gates it -- proving the fix doesn't just tighten every gate indiscriminately."""
        mrrs = _p2_healthy(final_mrr=0.60, init_mrr=0.05, jitter=0.001, rng_seed=3)
        res = _p2_result({"vis00_s0": _p2_series(mrrs)})
        g = p2_validity_gate(res, "vis00", 0, sibling_chance_mean=0.10, chance_mrr_mean=0.25)
        assert g["gate_b_pass"]


class TestP2GateAAnchorIsChanceMRR:
    """I3: gate (a)'s rise is anchored on chance_mrr_mean, not the run's own epoch-10 value --
    epoch 10 already reflects SOME training and is not a genuine 'untrained' baseline."""

    def test_a_warm_started_run_passes_gate_a_only_once_anchored_on_chance(self):
        """LESION CHECK. Every milestone checkpoint sits close together (0.50-0.55) -- a
        'warm-started' trajectory with almost no rise from ITS OWN epoch-10 value, which would
        give only a small rise under the OLD (epoch-10) anchor despite the model clearly
        performing far above chance throughout."""
        mrrs = [0.50, 0.51, 0.515, 0.52, 0.525, 0.53, 0.535, 0.54, 0.55]
        res = _p2_result({"vis00_s0": _p2_series(mrrs)})
        g = p2_validity_gate(res, "vis00", 0, sibling_chance_mean=0.05, chance_mrr_mean=0.05)
        old_style_rise = max(mrrs) - mrrs[0]              # the OLD (epoch-10) anchor's rise
        assert g["gate_a_rise"] > old_style_rise * 5       # the NEW anchor gives a far bigger one
        assert g["gate_a_pass"]

    def test_a_run_with_no_rise_above_chance_still_fails_gate_a(self):
        """HEALTHY / control direction: a run that never separates from chance_mrr_mean at all
        must still fail gate (a) even under the new anchor -- the anchor changed, not the logic
        that a genuine lack of rise fails it."""
        mrrs = _p2_dead(level=0.05, jitter=0.005, rng_seed=7)
        res = _p2_result({"vis00_s0": _p2_series(mrrs)})
        g = p2_validity_gate(res, "vis00", 0, sibling_chance_mean=0.02, chance_mrr_mean=0.05)
        assert not g["gate_a_pass"]


class TestMergeP2ScorerOutputs:
    """C4 #2/#3: `scripts/p2_lrz_score.sh` writes 9 SEPARATE JSON files; `merge_p2_scorer_
    outputs` combines them into the single dict `p2_verdict` expects, keeping each control's OWN
    baselines key distinct (never falling back to the real tree's floor for a control that has
    one of its own)."""

    def test_merge_combines_disjoint_arms_from_multiple_files(self):
        a = {"baselines": {"sibling_chance_mean": 0.1}, "arms": {"vis00_s0_ms": {"checkpoints": []}}}
        b = {"baselines": {"sibling_chance_mean": 0.1}, "arms": {"vis50_s0_ms": {"checkpoints": []}}}
        merged = merge_p2_scorer_outputs([a, b])
        assert set(merged["arms"]) == {"vis00_s0_ms", "vis50_s0_ms"}

    def test_merge_raises_on_a_duplicate_arm_key(self):
        """LESION CHECK. Two files claiming the SAME arm+seed+kind key must never silently drop
        one -- that would mean a real run vanishes from the merged verdict without a trace."""
        a = {"baselines": {}, "arms": {"vis00_s0_ms": {"checkpoints": [1]}}}
        b = {"baselines": {}, "arms": {"vis00_s0_ms": {"checkpoints": [2]}}}
        with pytest.raises(ValueError, match="duplicate arm key"):
            merge_p2_scorer_outputs([a, b])

    def test_merge_keeps_each_controls_own_baselines_key_distinct(self):
        """C4 #3, LESION CHECK direction if reverted: a real-tree file and a matched-control
        file, each carrying its OWN baselines key (as `--baselines-key` now writes), must both
        survive the merge distinctly -- never collapsing the control onto the real tree's floor."""
        real = {"baselines": {"sibling_chance_mean": 0.05}, "arms": {"vis00_s0_ms": {}}}
        control = {"baselines_randomdag_vis00": {"sibling_chance_mean": 0.30},
                  "arms": {"randomdag_vis00_s0_ms": {}}}
        merged = merge_p2_scorer_outputs([real, control])
        assert merged["baselines"]["sibling_chance_mean"] == 0.05
        assert merged["baselines_randomdag_vis00"]["sibling_chance_mean"] == 0.30

    def test_merge_records_a_disagreement_instead_of_silently_picking_one(self):
        """HEALTHY CASE: two files' baselines blocks under the SAME key disagreeing (e.g. two
        seeds' own splits) does not raise, but is recorded, not silently overwritten or averaged."""
        a = {"baselines": {"sibling_chance_mean": 0.05}, "arms": {}}
        b = {"baselines": {"sibling_chance_mean": 0.06}, "arms": {}}
        merged = merge_p2_scorer_outputs([a, b])
        assert merged["baselines"]["sibling_chance_mean"] == 0.05      # first (seed 0) wins
        assert merged["_baselines_disagreements"]["baselines"] == [{"sibling_chance_mean": 0.06}]

    def test_merged_output_feeds_p2_verdict_without_a_keyerror(self):
        """End-to-end: build a merged result from SEPARATE per-file baselines blocks (mirroring
        scripts/p2_lrz_score.sh's real-pair-file + matched-control-file split), and confirm
        p2_verdict reads it straight through -- the exact shape C4 exists to make loadable."""
        real = _p2_nine_arm_result(vis00_final=0.55, vis50_final=0.50, randomdag_final=0.10,
                                   sibling_chance=0.05)
        real_only = {"baselines": real["baselines"],
                    "arms": {k: v for k, v in real["arms"].items()
                             if not k.startswith("randomdag")}}
        control_only = {"baselines_randomdag": real["baselines"],
                       "arms": {k: v for k, v in real["arms"].items()
                                if k.startswith("randomdag")}}
        merged = merge_p2_scorer_outputs([real_only, control_only])
        v = p2_verdict(merged)
        assert v["verdict"] in {"GENERALISES", "MEMORISES", "MIXED", "UNINFORMATIVE"}


class TestP2Amendment3Artifact:
    """The frozen JSON's p2_amendment_3_20260924 block must state what the task requires, not
    just this session's prose -- mirrors TestP2PreregistrationArtifact's pattern for amendments
    1/2."""

    @pytest.fixture(scope="class")
    def prereg(self):
        path = _REPO / "results" / "p2_heldout_preregistration.json"
        return json.loads(path.read_text())

    def test_block_present_and_predates_any_p2_run(self, prereg):
        assert "p2_amendment_3_20260924" in prereg
        status = prereg["p2_amendment_3_20260924"]["status"].lower()
        assert "before any p2 array was submitted" in status
        assert "before any p2 outcome data" in status
        assert "review finding" in status
        assert "2026-09-24" in prereg["p2_amendment_3_20260924"]["status"]

    def test_states_amendment_1_rule_1_was_wrong_at_k_equals_one(self, prereg):
        blob = json.dumps(prereg["p2_amendment_3_20260924"]["c3_pool_size_one_exclusion"])
        assert "rule_1" in blob
        assert "0.5" in blob
        assert "k=1" in blob or "pool-size-1" in blob or "pool size 1" in blob

    def test_states_gate_b_compared_mrr_to_a_hits_at_1_chance_rate(self, prereg):
        blob = json.dumps(prereg["p2_amendment_3_20260924"]["c2_chance_mrr_mean"]).lower()
        assert "hits@1" in blob or "hits_at_1" in blob
        assert "chance_mrr_mean" in blob

    def test_states_the_degree_prior_is_a_declared_baseline_the_arm_must_beat(self, prereg):
        blob = json.dumps(prereg["p2_amendment_3_20260924"]["c1_degree_prior_baseline"]).lower()
        assert "degree_prior" in blob
        assert "generalises" in blob
        assert "memorises" in blob

    def test_states_metrics_poincare_is_not_independent_corroboration(self, prereg):
        block = prereg["p2_amendment_3_20260924"]["metrics_poincare_is_not_independent_corroboration"]
        blob = json.dumps(block).lower()
        assert "not independent corroboration" in blob or "not merely correlated" in blob
        assert "monoton" in blob
        # honesty: this session made no training run, so it must say so, not imply otherwise.
        assert "not on a real trained checkpoint" in blob or "no training run" in blob

    def test_frozen_block_and_earlier_amendments_are_untouched(self, prereg):
        """Additive-only: re-check the frozen block's own invariants (mirrors the equivalent
        amendment_1/amendment_2 tests) to guard against amendment_3 having edited anything in
        place instead of adding a new top-level key."""
        assert len(prereg["declared_confounds"]) == 4
        assert set(prereg["readings"]) == {"GENERALISES", "MEMORISES", "MIXED", "UNINFORMATIVE"}
        assert "5.403" in json.dumps(prereg["p2_amendment_1_20260924"])
        assert "randomdag_vis00" in json.dumps(prereg["p2_amendment_2_20260924"])
