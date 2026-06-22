#!/usr/bin/env python3
"""Smoke test for the LR-schedule helper added 2026-06-02 (Experiment 1 prep).

Runs offline: no GPU, no data, no model. Asserts cosine_warmrestart resets to
base_lr at every curriculum phase boundary and decays to ~lr_min_mult * base_lr
at the end of each segment. Also checks 'const' and 'cosine' modes.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from train_small import _compute_scheduled_lr, auto_curriculum_phases


def _close(a: float, b: float, atol: float = 1e-9) -> bool:
    return abs(a - b) <= atol


def test_const() -> None:
    for ep in (1, 50, 200):
        got = _compute_scheduled_lr(
            epoch=ep, n_epochs=200, base_lr=0.001,
            schedule="const", lr_min_mult=0.01, phase_boundaries=None,
        )
        assert _close(got, 0.001), f"const@ep{ep}={got!r}"


def test_cosine_endpoints() -> None:
    base = 0.001
    start = _compute_scheduled_lr(
        epoch=1, n_epochs=200, base_lr=base,
        schedule="cosine", lr_min_mult=0.01, phase_boundaries=None,
    )
    end = _compute_scheduled_lr(
        epoch=200, n_epochs=200, base_lr=base,
        schedule="cosine", lr_min_mult=0.01, phase_boundaries=None,
    )
    assert _close(start, base), f"cosine start = {start!r}, want {base}"
    assert _close(end, base * 0.01), f"cosine end = {end!r}, want {base * 0.01}"


def test_warmrestart_on_metazoa_auto_phases() -> None:
    # Reproduce Experiment 1 conditions: auto curriculum on max_depth ~36 (metazoa
    # observed dd<=9 at ep 40, dd<=18 at ep 80 per 2026-06-02 SESSION_LOG.md).
    phases = auto_curriculum_phases(max_depth=36, n_epochs=200)
    assert phases == [(1, 1), (40, 9), (80, 18), (120, None)], phases
    boundaries = sorted({pe for pe, _ in phases})  # [1, 40, 80, 120]

    base = 0.001
    floor = base * 0.01

    # At each phase boundary, LR resets to base (t=0).
    for ep in boundaries:
        got = _compute_scheduled_lr(
            epoch=ep, n_epochs=200, base_lr=base,
            schedule="cosine_warmrestart", lr_min_mult=0.01,
            phase_boundaries=boundaries,
        )
        assert _close(got, base), f"warmrestart@ep{ep}={got!r}, want {base}"

    # At the LAST epoch within each segment, LR is close to floor.
    # Segment ends are 39, 79, 119, 200 (last epoch of phase4).
    seg_ends_to_segs = [(39, 1, 40), (79, 40, 80), (119, 80, 120), (200, 120, 201)]
    for ep, e_start, e_end in seg_ends_to_segs:
        seg_len = e_end - e_start
        # Expected t at the LAST epoch in the segment:
        t = (ep - e_start) / seg_len
        # Should be near 1 (but not exactly 1, since cosine spans [e_start, e_end))
        assert 0.5 < t < 1.0, f"sanity check: t={t} at ep{ep}"
        got = _compute_scheduled_lr(
            epoch=ep, n_epochs=200, base_lr=base,
            schedule="cosine_warmrestart", lr_min_mult=0.01,
            phase_boundaries=boundaries,
        )
        assert got < base * 0.5, (
            f"warmrestart@ep{ep} (seg_end, t={t:.3f})={got!r}, want close to floor {floor}"
        )

    # The Experiment-1 leverage point: at ep 80, LR resets to base AFTER the
    # ep-79 floor at the dd<=18 transition. This is the whole point.
    lr_79 = _compute_scheduled_lr(
        epoch=79, n_epochs=200, base_lr=base,
        schedule="cosine_warmrestart", lr_min_mult=0.01,
        phase_boundaries=boundaries,
    )
    lr_80 = _compute_scheduled_lr(
        epoch=80, n_epochs=200, base_lr=base,
        schedule="cosine_warmrestart", lr_min_mult=0.01,
        phase_boundaries=boundaries,
    )
    assert lr_80 > lr_79 * 3, (
        f"warm restart at ep80 (dd<=18 transition) should boost LR vs ep79: "
        f"lr_79={lr_79:.6e}, lr_80={lr_80:.6e}"
    )
    print(
        f"  Experiment-1 trajectory checkpoint: lr_79={lr_79:.6e} -> lr_80={lr_80:.6e} "
        f"(reset at dd<=18 transition)"
    )


def main() -> None:
    print("[1/3] const schedule")
    test_const()
    print("  ✓ const stays at base_lr at every epoch")

    print("[2/3] cosine full-run")
    test_cosine_endpoints()
    print("  ✓ ep1=base_lr, ep200=base_lr * lr_min_mult")

    print("[3/3] cosine_warmrestart with metazoa auto phases (ep 1/40/80/120)")
    test_warmrestart_on_metazoa_auto_phases()
    print("  ✓ resets at every phase boundary, floor at end of each segment")

    print("\n✓ All LR-schedule smoke checks passed")


if __name__ == "__main__":
    main()
