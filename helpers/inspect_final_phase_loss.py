"""Print the final-curriculum-phase loss trajectory per run, i.e. exactly what gate (b) reads.

Written when Task 9's pre-registered verdict came back UNINFORMATIVE on a single arm
(`prior_s2`), to establish WHY the gate fired before deciding what the verdict means. The gate was
written to catch "this arm never trained". A flat final-phase loss is also the signature of an arm
that CONVERGED, and those are opposite situations with the same measurement.

  <python> helpers/inspect_final_phase_loss.py <scorer.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

FINAL_PHASE_MIN_EPOCH = 130


def main() -> None:
    result = json.loads(Path(sys.argv[1]).read_text())
    print(f"{'run':>14}  {'ep130':>10} {'ep200':>10} {'drop':>10} {'jitter':>8}  "
          f"{'gate_b':>6}   final-phase milestones")
    for name, run in result["runs"].items():
        if not name.endswith("_ms"):
            continue
        base = name[:-3]
        ms = [c for c in run["checkpoints"]
              if c["epoch"] >= FINAL_PHASE_MIN_EPOCH and c["trainer"].get("loss") is not None]
        roll = result["runs"][base + "_roll"]["checkpoints"]
        rl = [c["trainer"]["loss"] for c in roll if c["trainer"].get("loss") is not None]
        jitter = float(np.std(rl, ddof=1))
        losses = [c["trainer"]["loss"] for c in ms]
        drop = losses[0] - losses[-1]
        traj = " ".join(f"{c['epoch']}:{c['trainer']['loss']:.4f}" for c in ms)
        print(f"{base:>14}  {losses[0]:10.4f} {losses[-1]:10.4f} {drop:+10.5f} {jitter:8.5f}  "
              f"{'PASS' if drop > jitter else 'FAIL':>6}   {traj}")

    print("\nS_angle trajectory over the SAME final-phase milestones (did structure still move?)")
    for name, run in result["runs"].items():
        if not name.endswith("_ms"):
            continue
        ms = [c for c in run["checkpoints"] if c["epoch"] >= FINAL_PHASE_MIN_EPOCH]
        traj = " ".join(f"{c['epoch']}:{c['S_angle']:.4f}" for c in ms)
        print(f"{name[:-3]:>14}  {traj}")


if __name__ == "__main__":
    main()
