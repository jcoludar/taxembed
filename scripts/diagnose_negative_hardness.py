#!/usr/bin/env python
"""Step-0 diagnostic (E1c): how much within-clade gradient does each sampler give?

Read-only. For one dataset (and optionally a checkpoint), draws negatives via the
production HierarchicalDataLoader exactly as training does, and reports — overall and
stratified by curriculum hop-distance dd:
  - within-class / within-grandparent NEGATIVE FRACTION   (checkpoint-free; sampler-only)
  - softmax p_j MASS on within-clade negatives             (needs --checkpoint)
  - mean negative Poincaré distance                        (needs --checkpoint)

Supports three sampler arms:
  --sampler default  : the production same-depth sampler (thing under test)
  --sampler uniform  : all-node uniform draws (chance-level baseline)
  --sampler tiered   : HierarchicalDataLoader tiered_negatives=True (upper-bound)

Premise under test: at large N the default same-depth sampler draws few within-clade
negatives, so the softmax (which self-weights to near negatives) has nothing within-clade
to sharpen against -> within-clade p_j mass collapses with scale.

Usage:
  .venv/bin/python scripts/diagnose_negative_hardness.py \\
      --file data/taxopy/<ds>/..._transitive.npz \\
      [--checkpoint artifacts/tags/<tag>/<tag>_best.pth] \\
      --n-negatives 50 --batches 40 --seed 0 --sampler default --tag <label>
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))             # train_hierarchical
sys.path.insert(0, str(ROOT / "scripts")) # _negative_hardness, analyze_hierarchy_hyperbolic

from train_hierarchical import TrainingPairs, HierarchicalDataLoader
from _negative_hardness import numpy_poincare_distance, softmax_pj, label_negatives


def _dd_bucket(dd: int) -> str:
    if dd <= 1:
        return "dd<=1"
    if dd <= 9:
        return "dd2-9"
    if dd <= 18:
        return "dd10-18"
    return "dd19+"


def main():
    ap = argparse.ArgumentParser(description="E1c Step-0 negative-hardness diagnostic")
    ap.add_argument("--file", required=True, help="transitive .npz")
    ap.add_argument("--checkpoint", default=None, help="optional .pth for p_j / distance")
    ap.add_argument("--n-negatives", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--batches", type=int, default=40, help="how many batches to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="label for the report line")
    ap.add_argument("--sampler", choices=["default", "uniform", "tiered"], default="default",
                    help="which sampler arm: default (production), uniform (chance baseline), tiered (upper bound)")
    args = ap.parse_args()

    np.random.seed(args.seed)

    pairs = TrainingPairs.load(Path(args.file))
    # n_nodes from pairs — authoritative property; avoids mapping-file off-by-one
    n_nodes = pairs.n_nodes

    loader = HierarchicalDataLoader(
        training_data=pairs, n_nodes=n_nodes,
        batch_size=args.batch_size, n_negatives=args.n_negatives,
        depth_stratify=True,
        tiered_negatives=(args.sampler == "tiered"),
    )

    # --- optional checkpoint: map euclidean-param z -> Poincaré ball before distances ---
    emb = None
    if args.checkpoint:
        from analyze_hierarchy_hyperbolic import load_embeddings
        z = np.asarray(load_embeddings(Path(args.checkpoint)), dtype=np.float64)  # (N,dim); raw z if euclidean-param
        norms = np.linalg.norm(z, axis=1, keepdims=True)
        if norms.max() >= 1.0:   # euclidean-param checkpoint: map tangent z -> Poincaré ball
            emb = np.tanh(norms / 2.0) * z / np.maximum(norms, 1e-8)
        else:
            emb = z

    # --- accumulators, overall + per dd bucket ---
    frac_class = defaultdict(list)
    frac_gp = defaultdict(list)
    pj_class = defaultdict(list)
    neg_dist = defaultdict(list)

    seen = 0
    for ancestors, descendants, negatives, depths in loader:
        anc = ancestors.numpy()
        desc = descendants.numpy()
        negs = negatives.numpy()
        dd = depths.numpy().astype(int)   # depth_diff per pair

        # --- uniform arm: overwrite loader negatives with all-node uniform draws ---
        if args.sampler == "uniform":
            negs = np.random.randint(0, n_nodes, size=negs.shape).astype(np.int64)

        wc, wg = label_negatives(loader._node_class_arr, loader._node_gp_arr, desc, negs)
        # per-anchor fractions
        f_c = wc.mean(axis=1)
        f_g = wg.mean(axis=1)

        pj_in_class = None
        mean_neg_d = None
        if emb is not None:
            d_pos = numpy_poincare_distance(emb[anc], emb[desc])                 # (B,)
            d_neg = numpy_poincare_distance(emb[anc][:, None, :], emb[negs])     # (B,n_neg)
            p_neg = softmax_pj(d_pos, d_neg)                                     # (B,n_neg)
            pj_in_class = (p_neg * wc).sum(axis=1)                               # mass on within-class negs
            mean_neg_d = d_neg.mean(axis=1)

        for i in range(len(desc)):
            b = _dd_bucket(int(dd[i]))
            frac_class[b].append(f_c[i])
            frac_class["ALL"].append(f_c[i])
            frac_gp[b].append(f_g[i])
            frac_gp["ALL"].append(f_g[i])
            if pj_in_class is not None:
                pj_class[b].append(pj_in_class[i])
                pj_class["ALL"].append(pj_in_class[i])
                neg_dist[b].append(mean_neg_d[i])
                neg_dist["ALL"].append(mean_neg_d[i])

        seen += 1
        if seen >= args.batches:
            break

    def m(d, k):
        return float(np.mean(d[k])) if d.get(k) else float("nan")

    def n_count(d, k):
        return len(d[k]) if d.get(k) else 0

    print(f"\n=== negative-hardness :: {args.tag or args.file} "
          f"(N={n_nodes:,}, sampler={args.sampler}, n_neg={args.n_negatives}, "
          f"batches={seen}, ckpt={'yes' if emb is not None else 'no'}) ===")
    print(f"{'bucket':>8} | {'n':>6} | {'within-class frac':>17} | {'within-gp frac':>14} | "
          f"{'pj-mass within-class':>20} | {'mean neg dist':>13}")
    for b in ["ALL", "dd<=1", "dd2-9", "dd10-18", "dd19+"]:
        if not frac_class.get(b):
            continue
        n = n_count(frac_class, b)
        noisy_flag = " *NOISY*" if n < 200 else ""
        print(f"{b:>8} | {n:>6} | {m(frac_class,b):>17.4f} | {m(frac_gp,b):>14.4f} | "
              f"{m(pj_class,b):>20.4f} | {m(neg_dist,b):>13.3f}{noisy_flag}")


if __name__ == "__main__":
    main()
