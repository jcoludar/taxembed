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

CORRECTION (2026-09-24, fix round 1, IMPORTANT #1). The radius-overflow guard used a single
`> 100` threshold regardless of path, but `load_checkpoint` always returns `emb` as float32
(`.detach().float().numpy()`). On the `ball_coordinates` fallback path (no `z_embeddings` in the
checkpoint), the guard computed `radius = 2*arctanh(|x|)` from THAT float32 array: float32 cannot
represent anything closer to 1.0 than ~1.19e-7, so `|x|` can never get close enough to 1 for the
derived radius to exceed ~17-21, let alone 100 -- the `>100` check was structurally unreachable
on this path. The two paths now use DIFFERENT guards for their genuinely different failure modes:
`z_embeddings` path keeps `>100` on the exact `|z|` (meaningful: under `--euclidean-param`
training, `project_to_ball` is a no-op and `|z|` is genuinely unbounded); `ball_coordinates` path
instead counts nodes with `||x|| >= 1 - 1e-6` (`n_norm_saturated`) -- the reachable failure mode,
where `arctanh` loses precision near the ball boundary. Each checkpoint row records
`radius_guard` (`"z_norm_gt_100"` or `"ball_norm_saturation"`) so the artifact states which test
actually ran.

CORRECTION (2026-09-24, fix round 1, IMPORTANT #2). `vendrov_recall` is tautologically 0.0 for
ANY input on the real P2 split: the same `held_out` array both removes edges from the visible
graph AND supplies the queries, so every query's own edge is guaranteed excluded by construction.
That is a genuine, reportable property of the task (see the docstring below) -- but it also meant
no fixture passed to the OLD single-argument `vendrov_recall` could ever produce a non-zero
result, so a hand-built "does the encode+isin machinery actually work" test was impossible without
decoupling. `vendrov_recall` is now a thin wrapper around `_vendrov_recall_core`, which takes the
"removed" and "queried" node sets separately; production still calls it with the same array for
both (unchanged behaviour), but tests can now hand it different sets and assert a genuine
non-zero recall.

CORRECTION (2026-09-24, fix round 1, IMPORTANT #3). `val` nodes from `--heldout` were read from
the .npz but never counted, scored, or reported -- so `scripts/build_p2_split.py --frac-val 0.05`
(its old default) silently withheld 5% of eligible nodes from every number in this driver's
output, with no trace in the JSON. `n_val` is now recorded at the top level and a warning is
printed when it is non-zero; `build_p2_split.py --frac-val` now defaults to 0.0 so nothing is
withheld without an explicit flag.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))

from taxembed.eval.baselines import (  # noqa: E402
    chance_mrr_mean, degree_prior_metrics, majority_parent_rate, sibling_chance,
)
from taxembed.eval.linkpred import (  # noqa: E402
    candidate_pool, linkpred_metrics, rank_of_true_parent, stratify,
)
from taxembed.eval.p2_split import depth_from_closure  # noqa: E402
from taxembed.eval.subtree import parent_from_closure  # noqa: E402
from taxembed.utils.training_pairs import TrainingPairs  # noqa: E402

# spec v3 SS.. bin edges, as given in the Task 6 brief.
POOL_SIZE_BINS = [(2, 2), (3, 5), (6, 20), (21, 10**6)]
DEPTH_BINS = [(11, 15), (16, 21), (22, 28)]

# z_embeddings path: |z| is genuinely unbounded under --euclidean-param training (the radial
# regulariser's gradient on ||z|| saturates); cosh/sinh overflow ~709.8, a real trained checkpoint
# measured 3.66, so 100 is a real, meaningful bound on that path.
RADIUS_OVERFLOW_BOUND = 100.0
# ball_coordinates path: float32 ||x|| cannot approach 1.0 closer than ~1.19e-7, so >100 can never
# fire there (fix round 1, IMPORTANT #1). The reachable failure mode is norm SATURATION -- ||x||
# within a few ulp of the ball boundary, where arctanh loses precision.
NORM_SATURATION_BOUND = 1.0 - 1e-6


def _json_safe(obj):
    """Recursively replace non-finite floats (NaN, +-inf) with None (fix round 1, MINOR #5).

    With an empty held-out set, `linkpred_metrics` returns `float("nan")` for every metric, and
    `json.dumps`'s default `allow_nan=True` would emit the bare `NaN` token -- not valid JSON
    under a strict (RFC 8259) parser. `null` is the correct JSON representation of "no value".
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


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


def load_parent_depth(manifest: dict,
                      closure_override: Path | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """Manifest supplies `parent`/`depth` directly (the CLI test's simpler fixture form), or
    else they are rebuilt from `source_npz` via TrainingPairs + parent_from_closure +
    depth_from_closure -- the production manifest shape build_p2_split.py writes.

    `closure_override` (C5, 2026-09-24, `p2_amendment_3_20260924`): `build_p2_split.py` records
    `manifest["source_npz"]` as an ABSOLUTE macOS build-time path (e.g.
    `/Users/jcoludar/.../data/taxopy/.../..._transitive.npz`), which does not exist inside the LRZ
    container -- opening it there raises `FileNotFoundError` only after the queue wait (Rule 16).
    When given, `closure_override` is used INSTEAD of `manifest["source_npz"]`, so the caller
    (`scripts/p2_lrz_score.sh`, via `--closure`) can point at the container-mounted path without
    editing the manifest. Never used when the manifest already supplies `parent`/`depth` directly.
    """
    if "parent" in manifest and "depth" in manifest:
        parent = np.asarray(manifest["parent"], dtype=np.int64)
        depth = np.asarray(manifest["depth"], dtype=np.int64)
        n_nodes = int(manifest.get("n_nodes", len(parent)))
        return parent, depth, n_nodes
    npz_path = Path(closure_override) if closure_override is not None else Path(manifest["source_npz"])
    pairs = TrainingPairs.load(npz_path)
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


def _vendrov_recall_core(visible_anc: np.ndarray, visible_desc: np.ndarray,
                         queries: np.ndarray, true_parents: np.ndarray, n_nodes: int) -> float:
    """Fraction of `queries` whose `(true_parents[i], queries[i])` pair is a visible direct edge,
    via a vectorized pair-encode + `np.isin` -- semantically identical to the diagonal of
    `baselines.vendrov_closure_rule(visible_anc, visible_desc, queries, true_parents)` but
    O(n) instead of that function's O(n_queries * n_candidates) Python set matrix (unaffordable
    at P2 scale: tens of thousands of held-out nodes).

    Factored out of `vendrov_recall` (fix round 1, IMPORTANT #2) so a test can hand it a
    visible-edge set that was NOT built by removing the queried nodes' own edges. Production
    (`vendrov_recall` below) always does exactly that removal, which is what makes ITS answer
    0.0 by construction on the real P2 split (see module docstring) -- a fixture passed to
    `vendrov_recall` itself can therefore never exercise a genuine non-zero recall. This
    lower-level function has no such coupling: `queries` and the edges removed to build
    `visible_anc`/`visible_desc` are independent, so a hand-built fixture can give some queried
    pairs that ARE visible, for real non-zero coverage of the encode+isin mechanism.
    """
    queries = np.asarray(queries, dtype=np.int64)
    if len(queries) == 0:
        return float("nan")
    key = n_nodes + 1
    visible_key = np.asarray(visible_anc, dtype=np.int64) * key + np.asarray(visible_desc, dtype=np.int64)
    query_key = np.asarray(true_parents, dtype=np.int64) * key + queries
    return float(np.isin(query_key, visible_key).mean())


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
    return _vendrov_recall_core(visible_anc, visible_desc, held_out, parent[held_out], n_nodes)


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
    ap.add_argument("--closure", type=Path, default=None,
                    help="C5: override manifest['source_npz'], which is an absolute macOS "
                         "build-time path that does not exist inside the LRZ container. Ignored "
                         "when the manifest already supplies parent/depth directly.")
    ap.add_argument("--baselines-key", default="baselines",
                    help="C4: top-level key this invocation's baselines block is written under. "
                         "Default 'baselines' (the real vis00/vis50 pair). Pass "
                         "'baselines_randomdag_vis00' / 'baselines_randomdag_vis50' for the "
                         "matched-control invocations so the merged JSON carries each control's "
                         "OWN floor under the key p2_verdict(..., amendment_2=True) reads, "
                         "instead of every invocation silently writing the same 'baselines' key "
                         "and the merge step handing every control the real tree's floor.")
    args = ap.parse_args()

    t0 = time.time()
    manifest = json.loads(args.manifest.read_text())
    if args.closure is not None and not args.closure.exists():
        raise SystemExit(f"--closure {args.closure} does not exist")
    parent, depth, n_nodes = load_parent_depth(manifest, args.closure)
    if args.closure is not None:
        closure_path_used: str | None = str(args.closure.resolve())
    elif "source_npz" in manifest:
        closure_path_used = str(Path(manifest["source_npz"]))
    else:
        closure_path_used = None

    heldout_data = np.load(args.heldout)
    held = np.asarray(heldout_data["test"], dtype=np.int64)
    val = np.asarray(heldout_data["val"], dtype=np.int64) if "val" in heldout_data else np.array([], dtype=np.int64)
    n_val = int(len(val))
    depth_held = depth[held]
    if n_val:
        print(f"WARNING: --heldout has {n_val:,} 'val' node(s) -- these are held out of training "
              f"but NOT scored by this driver (val is reserved for threshold-tuning outside Task "
              f"6's scope, per the ruling). They are excluded from every number below. If this is "
              f"unintentional, rebuild the split with 'scripts/build_p2_split.py --frac-val 0.0'.",
              flush=True)

    # Candidate pools depend only on tree structure, not on any checkpoint -- build once.
    candidates_list = [candidate_pool(parent, depth, node=int(v), strategy="grandparent_children")
                       for v in held]
    n_cand = np.array([len(c) for c in candidates_list], dtype=np.int64)
    true_parents = parent[held]

    print(f"manifest: {n_nodes:,} nodes; {len(held):,} held-out test nodes; "
          f"pool sizes {n_cand.min() if len(n_cand) else 0}-{n_cand.max() if len(n_cand) else 0}; "
          f"setup {time.time() - t0:.1f}s", flush=True)

    sibling_chance_mean = (float(np.mean(sibling_chance(parent, held)))
                           if len(held) else float("nan"))
    # C1 (degree prior) and C2 (chance_mrr_mean) both depend only on tree structure and the
    # held-out node set, never on any checkpoint -- computed once per invocation, mirroring
    # sibling_chance_mean/vendrov_recall/majority_parent_rate above.
    degree_prior = (degree_prior_metrics(parent, held, candidates_list, true_parents, n_cand)
                    if len(held) else {"n": 0, "n_scored": 0, "n_trivial": 0,
                                        "mean_rank": float("nan"), "mrr": float("nan"),
                                        "hits_at_1": float("nan"), "hits_at_10": float("nan"),
                                        "normalized_rank": float("nan")})
    baselines_block = {
        "sibling_chance_mean": sibling_chance_mean,
        "chance_hits_at_1_mean": sibling_chance_mean,  # C2: same quantity, second explicit name
        "chance_mrr_mean": (chance_mrr_mean(n_cand) if len(held) else float("nan")),
        "vendrov_recall": vendrov_recall(parent, held, n_nodes),
        "majority_parent_rate": majority_parent_rate(parent, held),
        "degree_prior": degree_prior,
    }
    result: dict = {
        "design": "docs/plans/2026-09-24-taxembed-p2-heldout-evaluation.md#Task-6",
        "manifest": str(args.manifest.resolve()),
        "heldout": str(args.heldout.resolve()),
        "closure_path_used": closure_path_used,
        "n_nodes": n_nodes,
        "n_held": int(len(held)),
        "n_val": n_val,
        "seed": args.seed,
        "primary_metric": args.metric,
        "pool_size_bins": [list(b) for b in POOL_SIZE_BINS],
        "depth_bins": [list(b) for b in DEPTH_BINS],
        args.baselines_key: baselines_block,
        "arms": {},
    }
    print(f"[{args.baselines_key}] sibling_chance_mean {sibling_chance_mean:.4f} | "
          f"chance_mrr_mean {baselines_block['chance_mrr_mean']:.4f} | vendrov_recall "
          f"{baselines_block['vendrov_recall']:.4f} (expect 0.0 by construction) | "
          f"majority_parent_rate {baselines_block['majority_parent_rate']:.4f} | "
          f"degree_prior mrr {degree_prior['mrr']:.4f} hits@1 {degree_prior['hits_at_1']:.4f} "
          f"normalized_rank {degree_prior['normalized_rank']:.4f}", flush=True)

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
                # return_clip=True recovers clip_count from THIS call instead of a second,
                # redundant _poincare_distance call on the same pair (fix round 1, MINOR #4).
                ranks_poin[i], cc = rank_of_true_parent(emb, v, tp, cands, metric="poincare",
                                                        tie_seed=args.seed, radii=radii,
                                                        return_clip=True)
                clip_count_total += cc

            metrics_cosine = linkpred_metrics(ranks_cos, n_cand)
            metrics_poincare = linkpred_metrics(ranks_poin, n_cand)
            primary_metrics = metrics_cosine if args.metric == "cosine" else metrics_poincare

            # I2: assert_full_coverage=True -- after C3's k=1 exclusion, every SCORED query
            # (pool size >= 2) must land in exactly one bin; POOL_SIZE_BINS starting at (2, 2) is
            # now a deliberate exclusion of k=1, not a gap, and a scored query landing in no bin
            # would be a real bug in the bin edges, not something to silently under-count.
            # DEPTH_BINS is NOT asserted here: it covers the production depth band [11, 28] only,
            # and small synthetic test trees (this module's own CLI tests) legitimately have
            # held-out nodes at shallower depths outside that band -- a coverage gap there is
            # expected on non-production data, not a bug.
            by_pool_size = {
                "cosine": stratify(ranks_cos, n_cand, n_cand, bins=POOL_SIZE_BINS,
                                   assert_full_coverage=True),
                "poincare": stratify(ranks_poin, n_cand, n_cand, bins=POOL_SIZE_BINS,
                                     assert_full_coverage=True),
            }
            by_depth = {
                "cosine": stratify(ranks_cos, n_cand, depth_held, bins=DEPTH_BINS),
                "poincare": stratify(ranks_poin, n_cand, depth_held, bins=DEPTH_BINS),
            }

            if radii is not None:
                # Exact hyperbolic radius (|z| under the euclidean parametrization): genuinely
                # unbounded, so a >100 threshold is meaningful here (fix round 1, IMPORTANT #1).
                radius_source = "z_embeddings"
                radius_guard = "z_norm_gt_100"
                max_radius = float(np.max(radii)) if len(radii) else float("nan")
                n_norm_saturated = None
                radius_overflow_risk = bool(max_radius > RADIUS_OVERFLOW_BOUND)
            else:
                # No exact hyperbolic radius available: `emb` here is float32 (load_checkpoint
                # casts to float), and float32 cannot represent anything closer to 1.0 than
                # ~1.19e-7 -- so r = 2*artanh(|x|) can never exceed ~17-21 on this path, and a
                # >100 test would be structurally unreachable (fix round 1, IMPORTANT #1). The
                # reachable failure mode instead is norm SATURATION: ||x|| landing at or within
                # a few ulp of 1.0, where arctanh loses all precision. `max_radius` is still
                # reported (informational, via the same formula scripts/score_recipe_checkpoints.py
                # uses), but the GUARD is n_norm_saturated, not max_radius > threshold.
                radius_source = "ball_coordinates"
                radius_guard = "ball_norm_saturation"
                norms = np.linalg.norm(emb.astype(np.float64), axis=1)
                radii_for_stats = 2.0 * np.arctanh(np.clip(norms, 0.0, 1.0 - 1e-9))
                max_radius = float(np.max(radii_for_stats)) if len(radii_for_stats) else float("nan")
                n_norm_saturated = int(np.sum(norms >= NORM_SATURATION_BOUND))
                radius_overflow_risk = n_norm_saturated > 0

            if radius_overflow_risk:
                if radius_guard == "z_norm_gt_100":
                    print(f"WARNING: {arm}@{epoch} max radius {max_radius:.2f} exceeds "
                          f"{RADIUS_OVERFLOW_BOUND:.0f} -- cosh/sinh overflow risk. This is a "
                          f"genuine discovery about a deeper tree, not a bug to silently tolerate.",
                          flush=True)
                else:
                    print(f"WARNING: {arm}@{epoch} {n_norm_saturated} node(s) NORM-SATURATED "
                          f"(||x|| >= {NORM_SATURATION_BOUND}) -- arctanh precision loss risk at "
                          f"the ball boundary (ball_coordinates path has no exact radius to fall "
                          f"back on).", flush=True)

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
                "radius_guard": radius_guard,
                "radius_overflow_risk": radius_overflow_risk,
                "n_norm_saturated": n_norm_saturated,
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
    # allow_nan=False is a second, independent belt: if any non-finite float somehow escapes
    # _json_safe, this raises instead of silently writing invalid JSON.
    out.write_text(json.dumps(_json_safe(result), indent=2, allow_nan=False))
    print(f"wrote {out}  md5={md5(out)}", flush=True)
    n_checkpoints = sum(len(a["checkpoints"]) for a in result["arms"].values())
    print(f"scoring complete: {len(result['arms'])} arm(s), {n_checkpoints} checkpoint(s), "
          f"{len(held)} held-out nodes each, {time.time() - t0:.0f}s total", flush=True)


if __name__ == "__main__":
    main()
