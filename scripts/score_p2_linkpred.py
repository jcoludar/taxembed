"""Score training checkpoints on the P2 held-out link-prediction task (plan v1 Task 6).

Design: docs/plans/2026-09-24-taxembed-p2-heldout-evaluation.md, Task 6.

For every held-out LEAF node whose parent edge was withheld (Task 1/2), rank its true parent
among the candidate pool -- its grandparent's other children (Task 4's `candidate_pool`) -- using
the checkpoint's embedding. PRIMARY METRIC IS COSINE: the candidate pool is depth-homogeneous (all
candidates share the query's grandparent, hence the same target radius), so the planted radius
cannot discriminate between candidates at all and the ranking is driven by angular structure alone.
Poincare distance is computed and stored as a SECONDARY metric on the same pools, so the primary
can be changed in analysis without re-running.

Also reports, once per invocation (these do not depend on the checkpoint):
  - `sibling_chance`  -- the chance floor, 1/|candidates|, that a learned MRR must beat.
  - `vendrov_recall`  -- the mandatory no-learning closure baseline (spec v3 SS3.2), evaluated
    per held-out node against its own true parent. Expected 0.0 BY CONSTRUCTION: the held-out
    parent edge is the one edge every closure derived from the visible graph is missing. That is
    the mirror image of the degeneracy that made us abandon the Ganea split (see the plan's "Why
    this plan departs from the spec" section) and is reported, not hidden.
  - `majority_parent_rate` -- accuracy of always guessing the commonest true parent.

Usage (single absolute-path invocation, per CLAUDE.md shell hygiene):
  <python> scripts/score_p2_linkpred.py --manifest <p2_..._manifest.json> --heldout <..._heldout.npz>
      --checkpoints arm=<glob> [--checkpoints arm2=<glob2> ...] --out <result.json>
      [--metric cosine] [--max-checkpoints N] [--seed 0]
Globs are expanded here, not by a shell, and sorted by the checkpoint's own 'epoch' field.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))

from taxembed.eval.baselines import majority_parent_rate, sibling_chance  # noqa: E402
from taxembed.eval.linkpred import (  # noqa: E402
    _poincare_distance, candidate_pool, linkpred_metrics, rank_of_true_parent, stratify,
)
from taxembed.eval.p2_split import depth_from_closure  # noqa: E402
from taxembed.eval.subtree import parent_from_closure  # noqa: E402
from taxembed.utils.training_pairs import TrainingPairs  # noqa: E402

# spec v3 SS.. bin edges, as given in the Task 6 brief.
POOL_SIZE_BINS = [(2, 2), (3, 5), (6, 20), (21, 10**6)]
DEPTH_BINS = [(11, 15), (16, 21), (22, 28)]

RADIUS_OVERFLOW_BOUND = 100.0  # cosh/sinh overflow ~709.8; a real trained checkpoint measured 3.66


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _epoch_from_name(path: str) -> int:
    m = re.search(r"epoch(\d+)", Path(path).name)
    return int(m.group(1)) if m else -1


def load_checkpoint(path: str) -> tuple[np.ndarray, dict, np.ndarray | None]:
    """Copied from scripts/score_recipe_checkpoints.py:61-75 -- do not diverge, per Task 6 brief."""
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


def load_parent_depth(manifest: dict) -> tuple[np.ndarray, np.ndarray, int]:
    """Manifest supplies `parent`/`depth` directly (the CLI test's simpler fixture form), or
    else they are rebuilt from `source_npz` via TrainingPairs + parent_from_closure +
    depth_from_closure -- the production manifest shape build_p2_split.py writes."""
    if "parent" in manifest and "depth" in manifest:
        parent = np.asarray(manifest["parent"], dtype=np.int64)
        depth = np.asarray(manifest["depth"], dtype=np.int64)
        n_nodes = int(manifest.get("n_nodes", len(parent)))
        return parent, depth, n_nodes
    pairs = TrainingPairs.load(Path(manifest["source_npz"]))
    n_nodes = pairs.n_nodes
    parent = parent_from_closure(pairs.ancestor_idx, pairs.descendant_idx, pairs.depth_diff, n_nodes)
    depth = depth_from_closure(pairs.descendant_idx, pairs.descendant_depth,
                               pairs.ancestor_idx, pairs.ancestor_depth, n_nodes)
    return parent, depth, n_nodes


def _visible_basic_edges(parent: np.ndarray, held_out: np.ndarray, n_nodes: int):
    """The transitive-REDUCTION edges left visible after the P2 holdout: every direct parent
    edge except the held-out leaves' own -- exactly the rows Task 1's `parent_edge_mask` marks
    for removal from the training closure. On a tree this IS the full visible closure for any
    direct-edge membership query: cutting a tree's unique (p, v) edge leaves no alternate path
    from p to v, so basic-edge membership and transitive-closure membership agree for this check.
    """
    idx = np.arange(n_nodes, dtype=np.int64)
    held_mask = np.zeros(n_nodes, dtype=bool)
    held_mask[held_out] = True
    keep = (idx != parent) & ~held_mask
    return parent[keep], idx[keep]


def vendrov_recall(parent: np.ndarray, held_out: np.ndarray, n_nodes: int) -> float:
    """Fraction of held-out nodes whose TRUE parent the Vendrov closure rule marks positive,
    i.e. the DIAGONAL of `baselines.vendrov_closure_rule(visible_anc, visible_desc,
    queries=held_out, candidates=parent[held_out])` -- one (candidate, query) pair per node, not
    a pool. `vendrov_closure_rule` itself builds an O(n_queries * n_candidates) Python set-membership
    matrix, which is unaffordable at P2 production scale (tens of thousands of held-out nodes);
    this computes the identical "is (candidate, query) a visible pair?" check via a vectorized
    pair-encode + np.isin instead. Expected 0.0 on this split by construction (see module docstring).
    """
    held_out = np.asarray(held_out, dtype=np.int64)
    if len(held_out) == 0:
        return float("nan")
    visible_anc, visible_desc = _visible_basic_edges(parent, held_out, n_nodes)
    key = n_nodes + 1
    visible_key = visible_anc.astype(np.int64) * key + visible_desc.astype(np.int64)
    query_key = parent[held_out].astype(np.int64) * key + held_out
    return float(np.isin(query_key, visible_key).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path, help="p2_..._manifest.json")
    ap.add_argument("--heldout", required=True, type=Path, help="p2_..._heldout.npz (test/val)")
    ap.add_argument("--checkpoints", action="append", required=True, metavar="ARM=GLOB")
    ap.add_argument("--out", required=True, type=Path, help="output JSON")
    ap.add_argument("--metric", choices=("cosine", "poincare"), default="cosine",
                    help="which metric populates the un-suffixed 'metrics' key (primary; both "
                         "cosine and poincare are always computed and stored in full)")
    ap.add_argument("--max-checkpoints", type=int, default=None,
                    help="score only the first N per arm (smoke runs)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    manifest = json.loads(args.manifest.read_text())
    parent, depth, n_nodes = load_parent_depth(manifest)

    heldout_data = np.load(args.heldout)
    held = np.asarray(heldout_data["test"], dtype=np.int64)
    depth_held = depth[held]

    # Candidate pools depend only on tree structure, not on any checkpoint -- build once.
    candidates_list = [candidate_pool(parent, depth, node=int(v), strategy="grandparent_children")
                       for v in held]
    n_cand = np.array([len(c) for c in candidates_list], dtype=np.int64)
    true_parents = parent[held]

    print(f"manifest: {n_nodes:,} nodes; {len(held):,} held-out test nodes; "
          f"pool sizes {n_cand.min() if len(n_cand) else 0}-{n_cand.max() if len(n_cand) else 0}; "
          f"setup {time.time() - t0:.1f}s", flush=True)

    result: dict = {
        "design": "docs/plans/2026-09-24-taxembed-p2-heldout-evaluation.md#Task-6",
        "manifest": str(args.manifest.resolve()),
        "heldout": str(args.heldout.resolve()),
        "n_nodes": n_nodes,
        "n_held": int(len(held)),
        "seed": args.seed,
        "primary_metric": args.metric,
        "pool_size_bins": [list(b) for b in POOL_SIZE_BINS],
        "depth_bins": [list(b) for b in DEPTH_BINS],
        "baselines": {
            "sibling_chance_mean": (float(np.mean(sibling_chance(parent, held)))
                                    if len(held) else float("nan")),
            "vendrov_recall": vendrov_recall(parent, held, n_nodes),
            "majority_parent_rate": majority_parent_rate(parent, held),
        },
        "arms": {},
    }
    print(f"[baselines] sibling_chance_mean {result['baselines']['sibling_chance_mean']:.4f} "
          f"| vendrov_recall {result['baselines']['vendrov_recall']:.4f} (expect 0.0 by "
          f"construction) | majority_parent_rate {result['baselines']['majority_parent_rate']:.4f}",
          flush=True)

    for spec in args.checkpoints:
        arm, pattern = spec.split("=", 1)
        paths = sorted(glob.glob(pattern), key=_epoch_from_name)
        if not paths:
            raise SystemExit(f"--checkpoints {arm}: no files match {pattern}")
        if args.max_checkpoints:
            paths = paths[: args.max_checkpoints]

        rows = []
        for path in paths:                        # one checkpoint in memory at a time
            t1 = time.time()
            emb, meta, radii = load_checkpoint(path)
            if emb.shape[0] != n_nodes:
                raise ValueError(f"{path}: embedding has {emb.shape[0]} rows, "
                                 f"manifest/closure has {n_nodes} nodes")
            epoch = meta["epoch"] if meta["epoch"] is not None else _epoch_from_name(path)

            ranks_cos = np.empty(len(held), dtype=np.int64)
            ranks_poin = np.empty(len(held), dtype=np.int64)
            clip_count_total = 0
            for i in range(len(held)):
                v = int(held[i])
                tp = int(true_parents[i])
                cands = candidates_list[i]
                ranks_cos[i] = rank_of_true_parent(emb, v, tp, cands, metric="cosine",
                                                   tie_seed=args.seed)
                r_u = None if radii is None else radii[v]
                r_v = None if radii is None else radii[cands]
                ranks_poin[i] = rank_of_true_parent(emb, v, tp, cands, metric="poincare",
                                                    tie_seed=args.seed, radii=radii)
                _, cc = _poincare_distance(emb[v], emb[cands], r_u=r_u, r_v=r_v)
                clip_count_total += cc

            metrics_cosine = linkpred_metrics(ranks_cos, n_cand)
            metrics_poincare = linkpred_metrics(ranks_poin, n_cand)
            primary_metrics = metrics_cosine if args.metric == "cosine" else metrics_poincare

            by_pool_size = {
                "cosine": stratify(ranks_cos, n_cand, n_cand, bins=POOL_SIZE_BINS),
                "poincare": stratify(ranks_poin, n_cand, n_cand, bins=POOL_SIZE_BINS),
            }
            by_depth = {
                "cosine": stratify(ranks_cos, n_cand, depth_held, bins=DEPTH_BINS),
                "poincare": stratify(ranks_poin, n_cand, depth_held, bins=DEPTH_BINS),
            }

            if radii is not None:
                radius_source = "z_embeddings"
                radii_for_stats = radii
            else:
                # No exact hyperbolic radius available: approximate it from the ball-coordinate
                # point via r = 2*artanh(|x|), the inverse of the euclidean parametrization
                # (x = tanh(|z|/2) z/|z|) -- mirrors scripts/score_recipe_checkpoints.py's
                # fallback label. Needed because |x| itself is bounded in [0, 1) by construction
                # and so can never trip the overflow guard below; the hyperbolic radius can.
                radius_source = "ball_coordinates"
                norms = np.linalg.norm(emb.astype(np.float64), axis=1)
                radii_for_stats = 2.0 * np.arctanh(np.clip(norms, 0.0, 1.0 - 1e-9))

            max_radius = float(np.max(radii_for_stats)) if len(radii_for_stats) else float("nan")
            radius_overflow_risk = bool(max_radius > RADIUS_OVERFLOW_BOUND)
            if radius_overflow_risk:
                print(f"WARNING: {arm}@{epoch} max radius {max_radius:.2f} exceeds "
                      f"{RADIUS_OVERFLOW_BOUND:.0f} -- cosh/sinh overflow risk. This is a "
                      f"genuine discovery about a deeper tree, not a bug to silently tolerate.",
                      flush=True)

            row = {
                "epoch": epoch,
                "path": path,
                "trainer": meta,
                "metrics": primary_metrics,
                "metrics_cosine": metrics_cosine,
                "metrics_poincare": metrics_poincare,
                "by_pool_size": by_pool_size,
                "by_depth": by_depth,
                "poincare_radius_source": radius_source,
                "max_radius": max_radius,
                "radius_overflow_risk": radius_overflow_risk,
                "clip_count_total": int(clip_count_total),
            }
            rows.append(row)
            print(f"[{arm} ep{epoch:>4}] mrr(cos) {metrics_cosine['mrr']:.4f} "
                  f"hits@1(cos) {metrics_cosine['hits_at_1']:.4f} | mrr(poincare) "
                  f"{metrics_poincare['mrr']:.4f} | max_radius {max_radius:.3f} "
                  f"({radius_source}) | clip_count {clip_count_total} | "
                  f"{time.time() - t1:.1f}s", flush=True)

        result["arms"][arm] = {"checkpoints": rows}

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}  md5={md5(out)}", flush=True)
    n_checkpoints = sum(len(a["checkpoints"]) for a in result["arms"].values())
    print(f"scoring complete: {len(result['arms'])} arm(s), {n_checkpoints} checkpoint(s), "
          f"{len(held)} held-out nodes each, {time.time() - t0:.0f}s total", flush=True)


if __name__ == "__main__":
    main()
