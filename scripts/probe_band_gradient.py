#!/usr/bin/env python
"""Phase 2.0 mechanism-validation probe (E1c): do ANCESTOR-anchored band negatives
actually receive softmax gradient mass FROM THE ANCESTOR?

Read-only. Validates the corrected band design BEFORE building the training sampler.
The loss computes negative distance from the ANCESTOR (d(anc, neg)); a negative gets
gradient ∝ p_j = e^{-d(anc,neg)}/Z. The fan-review BLOCKER: a descendant-anchored band
draws negatives far from the ancestor → p_j≈0 → no gradient. The fix: draw negatives from
SIBLING CLADES OF THE ANCESTOR (subtree(far_anc) ∖ subtree(anchor) at the descendant's depth),
which are near the ancestor → real gradient.

This probe compares, per dataset+checkpoint, the negatives' mean d(anc,neg) and mean p_j:
  - default  : same-depth random (the starved baseline)
  - band-anc : ANCESTOR-anchored sibling-clade band (the corrected design)
  - band-desc: DESCENDANT-anchored band (the buggy design the review flagged)
Stratified by the ANCESTOR depth `da` (the band is only meaningful for mid/deep ancestors).

PASS (corrected design sound) = band-anc gets materially SMALLER d(anc,neg) and HIGHER mean
p_j than default (gradient flows), while band-desc does NOT (confirming the bug + the fix).

Usage:
  .venv/bin/python scripts/probe_band_gradient.py --file <transitive.npz> \
      --checkpoint <pth> --w-far 2 --w-near 0 --batches 30 --seed 0 --tag LABEL
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_hierarchical import TrainingPairs
from _negative_hardness import numpy_poincare_distance, softmax_pj

POOL_CAP = 4000  # cap pool sizes used in membership tests (probe is a statistical estimate)


def build_index(pairs):
    """Ancestor-lineage index from the transitive closure."""
    a_idx = pairs.ancestor_idx; d_idx = pairs.descendant_idx
    a_dep = pairs.ancestor_depth; d_dep = pairs.descendant_depth
    node_anc_by_depth = defaultdict(dict)       # node -> {depth: ancestor}
    anc_depth_lists = defaultdict(list)         # (anc, depth) -> [descendants]
    depth_lists = defaultdict(set)              # depth -> {nodes}
    for i in range(len(a_idx)):
        di = int(d_idx[i]); ai = int(a_idx[i])
        dd = int(d_dep[i]); ad = int(a_dep[i])
        node_anc_by_depth[di][ad] = ai
        anc_depth_lists[(ai, dd)].append(di)
        depth_lists[dd].add(di); depth_lists[ad].add(ai)
    anc_depth_to_nodes = {k: np.fromiter(set(v), dtype=np.int64) for k, v in anc_depth_lists.items()}
    depth_to_nodes = {k: np.fromiter(v, dtype=np.int64) for k, v in depth_lists.items()}
    return node_anc_by_depth, anc_depth_to_nodes, depth_to_nodes


def _capped(arr):
    if arr is None:
        return None
    if len(arr) > POOL_CAP:
        return np.random.choice(arr, POOL_CAP, replace=False)
    return arr


def sample_band(anchor, da, desc, dd, lineage_owner_node, owner_depth, n_neg,
                w_near, w_far, anc_depth_to_nodes, node_anc_by_depth, depth_to_nodes):
    """Draw n_neg band negatives. lineage_owner_node/owner_depth = the node whose lineage
    defines the band (ancestor for band-anc, descendant for band-desc). Returns (negs, is_fallback)."""
    lineage = node_anc_by_depth.get(lineage_owner_node, {})
    far_key = owner_depth - w_far
    if far_key not in lineage:
        return None, True
    far_anc = lineage[far_key]
    far_pool = anc_depth_to_nodes.get((far_anc, dd))
    if far_pool is None or len(far_pool) == 0:
        return None, True
    # exclusion subtree: owner's own subtree (w_near=0) or an ancestor w_near up
    if w_near == 0:
        excl_anc = lineage_owner_node
    else:
        excl_anc = lineage.get(owner_depth - w_near, lineage_owner_node)
    excl_pool = anc_depth_to_nodes.get((excl_anc, dd))
    # rejection sample candidates from far_pool, drop those in the exclusion subtree + self
    cand = np.random.choice(_capped(far_pool), min(len(far_pool), 8 * n_neg + 16), replace=True)
    if excl_pool is not None and len(excl_pool):
        cand = cand[~np.isin(cand, _capped(excl_pool))]
    cand = cand[cand != desc]
    if len(cand) == 0:
        return None, True
    return np.random.choice(cand, n_neg, replace=True), False


def sample_default(desc, dd, n_neg, depth_to_nodes, n_nodes):
    pool = depth_to_nodes.get(dd)
    if pool is not None and len(pool) > 1:
        cands = pool[pool != desc]
        return np.random.choice(cands, n_neg, replace=True)
    return np.random.randint(0, n_nodes, n_neg)


def da_bucket(da):
    if da <= 2:
        return "da<=2(coarse)"
    if da <= 5:
        return "da3-5(mid)"
    return "da>=6(deep)"


def main():
    ap = argparse.ArgumentParser(description="E1c Phase 2.0 band-gradient probe")
    ap.add_argument("--file", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--w-near", type=int, default=0)
    ap.add_argument("--w-far", type=int, default=2)
    ap.add_argument("--n-negatives", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--batches", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    np.random.seed(args.seed)
    pairs = TrainingPairs.load(Path(args.file))
    n_nodes = pairs.n_nodes
    node_anc_by_depth, anc_depth_to_nodes, depth_to_nodes = build_index(pairs)

    from analyze_hierarchy_hyperbolic import load_embeddings
    z = np.asarray(load_embeddings(Path(args.checkpoint)), dtype=np.float64)
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    emb = (np.tanh(norms / 2.0) * z / np.maximum(norms, 1e-8)) if norms.max() >= 1.0 else z

    n_pairs = len(pairs.ancestor_idx)
    # accumulators: arm -> bucket -> dict of running sums
    acc = {arm: defaultdict(lambda: {"d": [], "negmass": [], "dpos": [], "fb": 0, "n": 0})
           for arm in ("default", "band-anc", "band-desc")}

    for _ in range(args.batches):
        rows = np.random.randint(0, n_pairs, args.batch_size)
        A = pairs.ancestor_idx[rows].astype(np.int64); DA = pairs.ancestor_depth[rows].astype(int)
        D = pairs.descendant_idx[rows].astype(np.int64); DD = pairs.descendant_depth[rows].astype(int)
        for i in range(args.batch_size):
            a, da, d, dd = int(A[i]), int(DA[i]), int(D[i]), int(DD[i])
            d_pos = numpy_poincare_distance(emb[a], emb[d])
            b = da_bucket(da)
            # default
            nd = sample_default(d, dd, args.n_negatives, depth_to_nodes, n_nodes)
            # band-anchor (ancestor lineage)
            na, fa = sample_band(a, da, d, dd, a, da, args.n_negatives, args.w_near, args.w_far,
                                 anc_depth_to_nodes, node_anc_by_depth, depth_to_nodes)
            # band-descendant (descendant lineage = the buggy design)
            ndz, fz = sample_band(a, da, d, dd, d, dd, args.n_negatives, args.w_near, args.w_far,
                                  anc_depth_to_nodes, node_anc_by_depth, depth_to_nodes)
            for arm, negs, fb in (("default", nd, False), ("band-anc", na, fa), ("band-desc", ndz, fz)):
                rec = acc[arm][b]; recall = acc[arm]["ALL"]
                rec["n"] += 1; recall["n"] += 1
                if fb or negs is None:
                    rec["fb"] += 1; recall["fb"] += 1
                    continue
                dn = numpy_poincare_distance(emb[a][None, :], emb[negs])  # (n_neg,)
                pj = softmax_pj(np.array([d_pos]), dn[None, :])[0]         # (n_neg,)
                rec["d"].append(dn.mean()); recall["d"].append(dn.mean())
                rec["negmass"].append(pj.sum()); recall["negmass"].append(pj.sum())  # =1-p_pos
                rec["dpos"].append(float(d_pos)); recall["dpos"].append(float(d_pos))

    def fmt(rec):
        n = rec["n"]; fb = rec["fb"]
        dmean = np.mean(rec["d"]) if rec["d"] else float("nan")
        dpos = np.mean(rec["dpos"]) if rec["dpos"] else float("nan")
        nm = np.mean(rec["negmass"]) if rec["negmass"] else float("nan")
        return dmean, dpos, nm, (fb / n if n else float("nan")), n

    print(f"\n=== band-gradient probe :: {args.tag or args.file} "
          f"(N={n_nodes:,}, w_near={args.w_near}, w_far={args.w_far}, n_neg={args.n_negatives}, "
          f"batches={args.batches}) ===")
    print("  (harder negatives => SMALLER d(anc,neg) and LARGER neg-mass=1-p_pos => more gradient)")
    print(f"{'bucket':>14} | {'arm':>10} | {'d(anc,neg)':>11} | {'d(anc,pos)':>11} | "
          f"{'neg-mass':>9} | {'fallback':>9} | {'n':>7}")
    for b in ["ALL", "da<=2(coarse)", "da3-5(mid)", "da>=6(deep)"]:
        for arm in ("default", "band-anc", "band-desc"):
            if acc[arm].get(b) and acc[arm][b]["n"]:
                dmean, dpos, nm, fbrate, n = fmt(acc[arm][b])
                print(f"{b:>14} | {arm:>10} | {dmean:>11.3f} | {dpos:>11.3f} | "
                      f"{nm:>9.4f} | {fbrate:>9.2f} | {n:>7}")
        print(f"{'':>14} | {'-'*10} | {'-'*11} | {'-'*11} | {'-'*9} | {'-'*9} | {'-'*7}")


if __name__ == "__main__":
    main()
