"""Radius-transplant 2x2 on the Figure 4 runs (Task 9 review item 6).

{prior, canonical} directions x {own, planted, other-arm} radii, each scored on the TRAINER'S OWN
metrics (kNN% / Sep / multiscale kNN -- Figure 4's neighbourhood) and on S_angle / S_poincare /
depth-norm r. Plus the reconstructed init (random directions, planted radii) as the floor.

Reading aid, not a pre-registered test:
  prior-directions-on-planted-radii  ~ canonical  on kNN%/Sep  -> the paper's metrics see a RADIAL gap
  prior-directions-on-planted-radii << canonical                -> the gap is ANGULAR
C7 (MANUSCRIPT_CORRECTIONS_PENDING) shows kNN% RISES when radii are scrambled, so expect the
prior's raw kNN% to flatter it; the transplant rows quantify by how much on the real runs.

Usage: <python> scripts/radius_transplant_2x2.py --npz <closure> --prior-ckpt <pth>
           --canonical-ckpt <pth> --out <json> [--seeds 0 1 2]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))

from score_recipe_checkpoints import load_checkpoint, load_tree  # noqa: E402
from taxembed.eval.angular import TreeIndex, query_bounds, score_embedding, select_queries  # noqa: E402
from taxembed.eval.transplant import planted, trainer_metrics, transplant  # noqa: E402
from taxembed.utils.training_pairs import TrainingPairs  # noqa: E402
from train_small import _build_class_labels  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--prior-ckpt", required=True)
    ap.add_argument("--canonical-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-sample", type=int, default=2000)
    args = ap.parse_args()

    t0 = time.time()
    parent, depth = load_tree(args.npz)
    idx = TreeIndex(parent, depth)
    q = select_queries(idx, n=10_000, k=10, seed=0)
    bounds = query_bounds(idx, q, 10)
    pairs = TrainingPairs.load(Path(args.npz))
    class_info = _build_class_labels(pairs, idx.n_nodes)
    print(f"setup {time.time() - t0:.0f}s", flush=True)

    xp, _, _ = load_checkpoint(args.prior_ckpt)
    xc, _, _ = load_checkpoint(args.canonical_ckpt)
    xp, xc = xp.astype(np.float64), xc.astype(np.float64)
    rng = np.random.default_rng(1)
    init_dirs = rng.standard_normal(xp.shape)

    conditions = {
        "init (random dirs, planted radii)": planted(init_dirs, depth, idx.max_depth),
        "prior raw": xp,
        "prior dirs + planted radii": planted(xp, depth, idx.max_depth),
        "canonical raw": xc,
        "canonical dirs + planted radii": planted(xc, depth, idx.max_depth),
        "canonical dirs + prior radii": transplant(xc, xp),
        "prior dirs + canonical radii": transplant(xp, xc),
    }
    rows = {}
    for name, x in conditions.items():
        t1 = time.time()
        a = score_embedding(x, idx, q, 10, "cosine", bounds=bounds)
        p = score_embedding(x, idx, q, 10, "poincare", bounds=bounds)
        tm = trainer_metrics(x, pairs, class_info, seeds=tuple(args.seeds), n_sample=args.n_sample)
        r = float(np.corrcoef(depth, np.linalg.norm(x, axis=1))[0, 1])
        rows[name] = {"S_angle": a["S"], "S_poincare": p["S"], "depth_norm_r": r, **tm}
        print(f"[{name:<34}] S_angle {a['S']:+.4f} | S_poincare {p['S']:+.4f} | depth-norm {r:+.3f} | "
              f"kNN {tm['knn_purity_mean']:.3f}±{tm['knn_purity_sd']:.3f} | "
              f"Sep {tm['class_sep_mean']:.3f}±{tm['class_sep_sd']:.3f} | "
              f"multiscale {tm['multiscale_knn_mean']} | {time.time() - t1:.0f}s", flush=True)

    out = {"closure": args.npz, "prior_ckpt": args.prior_ckpt, "canonical_ckpt": args.canonical_ckpt,
           "note": "Reading aid, not pre-registered. Trainer kNN%/Sep are radius-confounded (C7).",
           "conditions": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
