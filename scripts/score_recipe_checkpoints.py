"""Score training checkpoints on the radius-free same-depth angular LCA metric (plan v2 Task 9).

Design: docs/specs/2026-09-22-task9-radius-free-scorer-design.md

For every checkpoint: S_angle (primary, cannot read a norm), S_poincare (same pools, reads the
radius), depth-norm r (planted axis), mean radial deviation from target_radius(depth), and the
trainer's own logged fields. Adds a reconstructed INITIALIZATION (planted radius + random
direction) as the null row: S_angle must sit at ~0 there, and depth-norm r at its floor.

One query set (fixed seed) is used for every checkpoint and every arm, so all comparisons are
paired. Per-query scores go to an .npz beside the JSON for paired bootstraps across runs.

Usage (single absolute-path invocation, per CLAUDE.md shell hygiene):
  <python> scripts/score_recipe_checkpoints.py --npz <closure.npz> --out <result.json>
      --run prior=<glob> --run canonical=<glob> [--n-queries 10000] [--k 10] [--seed 0]
Globs are expanded here, not by a shell, and sorted by the checkpoint's own 'epoch' field.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))

from taxembed.eval.angular import (  # noqa: E402
    CLUSTER_LEVEL, TreeIndex, choose_clade_level, paired_S_difference, query_bounds,
    score_embedding, select_queries,
)
from taxembed.eval.subtree import parent_from_closure  # noqa: E402
from train_hierarchical import target_radius  # noqa: E402


def load_tree(npz_path: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(npz_path)
    anc = np.asarray(d["ancestor_idx"], dtype=np.int64)
    des = np.asarray(d["descendant_idx"], dtype=np.int64)
    n = int(max(anc.max(), des.max())) + 1
    depth = np.full(n, -1, dtype=np.int64)
    depth[des] = np.asarray(d["descendant_depth"], dtype=np.int64)
    depth[anc] = np.asarray(d["ancestor_depth"], dtype=np.int64)
    if (depth < 0).any():
        raise ValueError(f"{int((depth < 0).sum())} nodes have no depth in the closure")
    parent = parent_from_closure(anc, des, d["depth_diff"], n)
    return parent, depth


def _epoch_from_name(path: str) -> int:
    m = re.search(r"epoch(\d+)", Path(path).name)
    return int(m.group(1)) if m else -1


def load_checkpoint(path: str) -> tuple[np.ndarray, dict]:
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=False)
    emb = ck.get("embeddings")
    if emb is None:
        emb = ck["state_dict"]["lt.weight"]
    meta = {key: (float(ck[key]) if key in ck and ck[key] is not None else None)
            for key in ("loss", "reg_loss", "depth_norm_corr", "hierarchy_pct",
                        "knn_purity", "class_sep_ratio")}
    meta["epoch"] = int(ck["epoch"]) if "epoch" in ck else None
    z = ck.get("z_embeddings")
    # hyperbolic radius = |z| under the euclidean parametrization (x = tanh(|z|/2) z/|z|)
    radii = np.linalg.norm(z.detach().double().numpy(), axis=1) if z is not None else None
    return emb.detach().float().numpy(), meta, radii


def radial_stats(emb: np.ndarray, depth: np.ndarray, max_depth: int, schedule: str) -> dict:
    norms = np.linalg.norm(emb.astype(np.float64), axis=1)
    target = target_radius(depth.astype(np.float64), max_depth, schedule)
    return {
        "depth_norm_r": float(np.corrcoef(depth, norms)[0, 1]),
        "radial_dev_mean": float(np.abs(norms - target).mean()),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "n_norm_below_0p05": int((norms < 0.05).sum()),
    }


def score_one(label: str, emb: np.ndarray, idx: TreeIndex, queries, k, bounds, seed,
              max_depth, schedule, clade_level, radii=None) -> tuple[dict, dict]:
    if emb.shape[0] != idx.n_nodes:
        raise ValueError(f"{label}: embedding has {emb.shape[0]} rows, closure has {idx.n_nodes} nodes")
    t0 = time.time()
    a = score_embedding(emb, idx, queries, k, "cosine", bounds=bounds, seed=seed,
                        extra_ks=(1, 100), clade_level=clade_level)
    p = score_embedding(emb, idx, queries, k, "poincare", bounds=bounds, seed=seed,
                        radii=radii, clade_level=clade_level)
    row = {
        "S_angle": a["S"], "S_angle_cluster_se": a["S_cluster_se"],
        "S_angle_cluster_ci95": a["S_cluster_ci95"], "n_clusters": a["n_clusters"],
        "S_angle_by_band": a["by_band"], "S_angle_by_clade": a["by_clade"],
        "S_angle_at_k": a["S_at_k"], "hubness_angle": a["hubness"],
        "S_poincare": p["S"], "S_poincare_cluster_se": p["S_cluster_se"],
        "S_poincare_by_band": p["by_band"], "S_poincare_by_clade": p["by_clade"],
        "poincare_radius_source": "z_embeddings" if radii is not None else "2*artanh(|x|)",
        "flag_poincare_above_angle": bool(p["S"] > a["S"]),
        "mean_s_angle": a["mean_s"], "mean_mu0": a["mean_mu0"], "mean_mustar": a["mean_mustar"],
        **radial_stats(emb, idx.depth, max_depth, schedule),
        "seconds": round(time.time() - t0, 1),
    }
    return row, {"s_angle": a["s"], "s_poincare": p["s"]}


def main() -> None:
    ap = argparse.ArgumentParser(description="Task 9 radius-free checkpoint scorer")
    ap.add_argument("--npz", required=True, help="transitive closure .npz the runs trained on")
    ap.add_argument("--out", required=True, help="output JSON (per-query arrays go to <out>.npz)")
    ap.add_argument("--run", action="append", required=True, metavar="ARM=GLOB")
    ap.add_argument("--n-queries", type=int, default=10_000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--radial-schedule", default="log")
    ap.add_argument("--pair", action="append", metavar="ARM_A:ARM_B",
                    help="paired cluster-bootstrap difference of the two arms' mean s")
    ap.add_argument("--max-checkpoints", type=int, default=None,
                    help="score only the first N per arm (smoke runs)")
    args = ap.parse_args()

    t0 = time.time()
    parent, depth = load_tree(args.npz)
    idx = TreeIndex(parent, depth, seed=args.seed)
    max_depth = idx.max_depth
    queries = select_queries(idx, n=args.n_queries, k=args.k, seed=args.seed)
    bounds = query_bounds(idx, queries, args.k)
    clusters = idx.ancestor_at(queries, CLUSTER_LEVEL)
    clade_level = choose_clade_level(idx, queries)       # fixed by tree + queries, pre-embedding
    print(f"tree: {idx.n_nodes:,} nodes, max depth {max_depth}; {len(queries):,} queries; "
          f"{len(np.unique(clusters)):,} bootstrap clusters; clade level {clade_level}; "
          f"setup {time.time() - t0:.1f}s", flush=True)

    result = {
        "design": "docs/specs/2026-09-22-task9-radius-free-scorer-design.md",
        "closure": str(Path(args.npz).resolve()),
        "n_nodes": idx.n_nodes, "max_depth": max_depth,
        "k": args.k, "n_queries": int(len(queries)), "seed": args.seed,
        "radial_schedule": args.radial_schedule,
        "cluster_level": CLUSTER_LEVEL, "clade_level": clade_level,
        "uncertainty_note": ("*_cluster_se / ci95 are CLUSTER bootstraps over depth-3 clades for "
                             "ONE trained run: query-sampling uncertainty only. They say nothing "
                             "about run-to-run spread, which needs independent seeds."),
        "runs": {},
    }
    arrays = {"queries": queries, "mu0": bounds[0], "mustar": bounds[1], "clusters": clusters}

    # Null row: the initialization state, reconstructed (planted radius, random direction).
    rng = np.random.default_rng(args.seed + 1)
    direction = rng.standard_normal((idx.n_nodes, 100))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    init = direction * target_radius(depth.astype(np.float64), max_depth, args.radial_schedule)[:, None]
    row, arr = score_one("init", init, idx, queries, args.k, bounds, args.seed, max_depth,
                         args.radial_schedule, clade_level)
    result["init_null"] = row
    arrays.update({f"init__{key}": val for key, val in arr.items()})
    print(f"[init null] S_angle {row['S_angle']:+.4f} (cluster se {row['S_angle_cluster_se']:.4f}) "
          f"| depth-norm r {row['depth_norm_r']:.6f}", flush=True)

    arm_s: dict[str, list[np.ndarray]] = {}
    for spec in args.run:
        arm, pattern = spec.split("=", 1)
        paths = sorted(glob.glob(pattern), key=_epoch_from_name)
        if not paths:
            raise SystemExit(f"--run {arm}: no files match {pattern}")
        if args.max_checkpoints:
            paths = paths[: args.max_checkpoints]
        rows = []
        for path in paths:                       # one checkpoint in memory at a time
            emb, meta, radii = load_checkpoint(path)
            epoch = meta["epoch"] if meta["epoch"] is not None else _epoch_from_name(path)
            row, arr = score_one(f"{arm}@{epoch}", emb, idx, queries, args.k, bounds, args.seed,
                                 max_depth, args.radial_schedule, clade_level, radii=radii)
            rows.append({"epoch": epoch, "path": path, "trainer": meta, **row})
            arrays.update({f"{arm}__ep{epoch}__{key}": val for key, val in arr.items()})
            arm_s.setdefault(arm, []).append(arr["s_angle"])
            print(f"[{arm} ep{epoch:>3}] S_angle {row['S_angle']:+.4f} "
                  f"(cluster se {row['S_angle_cluster_se']:.4f}) | S_poincare {row['S_poincare']:+.4f} "
                  f"| depth-norm r {row['depth_norm_r']:+.4f} | radial dev {row['radial_dev_mean']:.4f} "
                  f"| loss {meta['loss']} | {row['seconds']}s", flush=True)
        vals = [r["S_angle"] for r in rows]
        result["runs"][arm] = {
            "checkpoints": rows,
            "S_angle_mean_over_checkpoints": float(np.mean(vals)),
            "S_angle_sd_over_checkpoints": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
        }

    # Paired comparisons on each arm's MEAN per-query s over its checkpoints (use rolling-window
    # arms, e.g. epochs 196-200, so annealing noise is averaged rather than sampled once).
    result["paired"] = []
    for pair in args.pair or []:
        a, b = pair.split(":", 1)
        s_a, s_b = np.mean(arm_s[a], axis=0), np.mean(arm_s[b], axis=0)
        diff = paired_S_difference(s_a, s_b, bounds[0], bounds[1], clusters, seed=args.seed)
        result["paired"].append({"a": a, "b": b, "S_angle_a_minus_b": diff})
        print(f"[paired] S_angle {a} - {b} = {diff['diff']:+.4f} "
              f"cluster ci95 [{diff['ci95'][0]:+.4f}, {diff['ci95'][1]:+.4f}]", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    np.savez_compressed(out.with_suffix(".npz"), **arrays)
    print(f"wrote {out} and {out.with_suffix('.npz')}  ({time.time() - t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
