"""Dose-response validation of the Task 9 scorer on a TRAINED checkpoint (review item 9).

The unit tests prove the closed forms; they cannot show the scorer separates reading 1 from
reading 2 on a real embedding. These two planted-truth checks can:

 (i)  DIRECTION dose: within each depth, shuffle the directions of a random fraction p of nodes
      (norms untouched). Known truth: learned angular structure is destroyed in proportion to p.
      S_angle must fall monotonically to ~0 at p = 1.
 (ii) RADIUS dose: keep directions, blend each node's norm toward a depth-independent random
      radius until depth-norm r falls from ~0.99 to ~0.67 (Figure 4's prior-collapse level).
      Known truth: NO angular structure changed. S_angle must stay bit-identical; S_poincare and
      the trainer's radius-reading kNN%/Sep are expected to move -- that is the confound.

Usage (one absolute-path invocation):
  <python> scripts/validate_task9_scorer.py --npz <closure> --dir-ckpt <pth> --rad-ckpt <pth> --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))

from score_recipe_checkpoints import load_checkpoint, load_tree  # noqa: E402
from taxembed.eval.angular import TreeIndex, query_bounds, score_embedding, select_queries  # noqa: E402
from taxembed.eval.transplant import trainer_metrics  # noqa: E402
from taxembed.utils.training_pairs import TrainingPairs  # noqa: E402
from train_hierarchical import target_radius  # noqa: E402
from train_small import _build_class_labels  # noqa: E402


def shuffle_directions(x, depth, p, rng):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    unit = x / norms
    out = unit.copy()
    for d in np.unique(depth):
        nodes = np.flatnonzero(depth == d)
        sel = nodes[rng.random(len(nodes)) < p]
        if len(sel) > 1:
            out[sel] = unit[rng.permutation(sel)]
    return out * norms


def blend_radii(x, depth, max_depth, alpha, rng):
    unit = x / np.linalg.norm(x, axis=1, keepdims=True)
    target = target_radius(depth.astype(np.float64), max_depth, "log")
    rand = rng.uniform(0.1, 0.95, size=len(depth))
    return unit * ((1 - alpha) * target + alpha * rand)[:, None]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--dir-ckpt", required=True, help="checkpoint for the direction dose")
    ap.add_argument("--rad-ckpt", required=True, help="checkpoint with planted radii for the radius dose")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-queries", type=int, default=10_000)
    args = ap.parse_args()

    parent, depth = load_tree(args.npz)
    idx = TreeIndex(parent, depth)
    q = select_queries(idx, n=args.n_queries, k=10, seed=0)
    bounds = query_bounds(idx, q, 10)
    pairs = TrainingPairs.load(Path(args.npz))
    class_info = _build_class_labels(pairs, idx.n_nodes)
    out = {"closure": args.npz, "direction_dose": [], "radius_dose": []}

    x_dir, _, _ = load_checkpoint(args.dir_ckpt)
    rng = np.random.default_rng(0)
    for p in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = shuffle_directions(x_dir.astype(np.float64), depth, p, rng)
        s = score_embedding(x, idx, q, 10, "cosine", bounds=bounds)
        out["direction_dose"].append({"p": p, "S_angle": s["S"], "S_cluster_se": s["S_cluster_se"]})
        print(f"[direction dose] p={p:.2f}  S_angle {s['S']:+.4f} (cluster se {s['S_cluster_se']:.4f})",
              flush=True)

    x_rad, _, _ = load_checkpoint(args.rad_ckpt)
    base_s = None
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = blend_radii(x_rad.astype(np.float64), depth, idx.max_depth, alpha, np.random.default_rng(1))
        r = float(np.corrcoef(depth, np.linalg.norm(x, axis=1))[0, 1])
        a = score_embedding(x, idx, q, 10, "cosine", bounds=bounds)
        p_ = score_embedding(x, idx, q, 10, "poincare", bounds=bounds)
        tm = trainer_metrics(x, pairs, class_info)
        if base_s is None:
            base_s = a["s"]
        row = {"alpha": alpha, "depth_norm_r": r, "S_angle": a["S"], "S_poincare": p_["S"],
               "S_angle_identical_to_alpha0": bool(np.array_equal(a["s"], base_s)), **tm}
        out["radius_dose"].append(row)
        print(f"[radius dose] alpha={alpha:.2f}  depth-norm r {r:+.3f} | S_angle {a['S']:+.4f} "
              f"(identical: {row['S_angle_identical_to_alpha0']}) | S_poincare {p_['S']:+.4f} | "
              f"trainer kNN {tm['knn_purity_mean']:.3f}±{tm['knn_purity_sd']:.3f} | "
              f"Sep {tm['class_sep_mean']:.3f}±{tm['class_sep_sd']:.3f}", flush=True)

    d = [r["S_angle"] for r in out["direction_dose"]]
    out["checks"] = {
        "direction_dose_monotone": bool(all(a >= b for a, b in zip(d, d[1:]))),
        "direction_dose_p1_near_zero": bool(abs(d[-1]) < 3 * out["direction_dose"][-1]["S_cluster_se"]),
        "radius_dose_S_angle_invariant": bool(all(r["S_angle_identical_to_alpha0"] for r in out["radius_dose"])),
    }
    print(json.dumps(out["checks"], indent=2))
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
