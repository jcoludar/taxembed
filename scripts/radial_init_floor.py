"""Compute the depth<->norm correlation AT INITIALIZATION over a real closure.

_initialize_by_depth sets every node's norm to target_radius(depth) at step 0, and
radial_regularizer holds it there. So the headline depth-norm correlation has a floor
that owes nothing to training. This reports that floor for both radial schedules; the
trained value must be quoted against it.

Usage (single absolute-path invocation, per CLAUDE.md shell hygiene):
  <venv-python> scripts/radial_init_floor.py --npz <closure.npz> --out results/radial_init_floor.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from taxembed.eval.radial import initialization_depth_norm_r  # noqa: E402


def node_depths_from_closure(npz) -> np.ndarray:
    """Per-node depth array derived from the closure's ancestor/descendant depth columns."""
    anc = np.asarray(npz["ancestor_idx"], dtype=np.int64)
    des = np.asarray(npz["descendant_idx"], dtype=np.int64)
    n_nodes = int(max(anc.max(), des.max())) + 1
    depth = np.full(n_nodes, -1, dtype=np.int64)
    depth[des] = np.asarray(npz["descendant_depth"], dtype=np.int64)
    depth[anc] = np.asarray(npz["ancestor_depth"], dtype=np.int64)
    if (depth < 0).any():
        raise ValueError(f"{int((depth < 0).sum())} nodes have no depth in the closure")
    return depth


def main() -> None:
    ap = argparse.ArgumentParser(description="Depth-norm correlation at initialization")
    ap.add_argument("--npz", required=True, help="transitive closure .npz")
    ap.add_argument("--out", required=True, help="output JSON path")
    args = ap.parse_args()

    depth = node_depths_from_closure(np.load(args.npz))
    max_depth = int(depth.max())

    floors = {
        schedule: initialization_depth_norm_r(depth, max_depth=max_depth, radial_schedule=schedule)
        for schedule in ("log", "linear")
    }

    result = {
        "source_npz": str(Path(args.npz).resolve()),
        "n_nodes": int(len(depth)),
        "max_depth": max_depth,
        "init_depth_norm_r": floors,
        "note": (
            "Computed here: the correlation BEFORE any gradient step, from target_radius() "
            "alone. The released cellular model's TRAINED value on the same closure-derived "
            "depths was measured at 0.957244 during the 2026-08-11 objective review; that "
            "figure is not recomputed by this script. Against the log floor below, training "
            "moves the depth-norm correlation by ~1e-4."
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print(f"nodes                 : {result['n_nodes']:,}")
    print(f"max_depth             : {max_depth}")
    for schedule, r in floors.items():
        print(f"init depth-norm r ({schedule:<6}): {r:.6f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
