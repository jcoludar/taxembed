#!/usr/bin/env python3
"""kNN-purity per rank under Poincaré (hyperbolic) distance — anti-Goodhart lock on separation.

The separation-ratio analyzer (`analyze_hierarchy_hyperbolic.py`) compares MEAN inter- vs
intra-group distances. A large ratio *could* in principle be inflated by the radial (depth↔norm)
structure rather than genuine angular clustering. kNN-purity is the complementary, local check:
for each node, look at its k NEAREST neighbours (full hyperbolic distance) and ask what fraction
share its rank-R label. It is a pure neighbour-identity measure — if relatives really are the
nearest points, purity is high; a radial-only artifact (all leaves crammed near the boundary at
random angles) would leave purity at chance. So purity ≫ chance ⇒ the 6–10× separation is real
angular structure, not a Poincaré-metric artifact.

Expectation for a good embedding: purity high at every rank, and purity/chance RISING at finer
ranks (family chance is tiny because families are small, so the over-chance lift is largest there).

Labels come from the SAME machinery the separation analyzer uses (imported, not re-implemented),
so the two metrics are measured on identical group definitions.

Scale note: metazoa has ~498k nodes, so dense pairwise is impossible. We subsample a seeded set of
QUERY nodes per rank but always score them against the FULL labelled pool, via a matmul-expanded
Poincaré distance (memory = one (batch × pool) matrix). Only neighbour ORDER matters for kNN, so
the matmul runs in float32; a float64 cross-check on a small block guards the implementation.

Usage:
    .venv/bin/python scripts/knn_purity_hyperbolic.py \
        --checkpoint artifacts/tags/metazoa_lower_lr_bigger_batch/metazoa_lower_lr_bigger_batch.pth \
        --mapping data/taxopy/metazoa_33208_clean/taxonomy_edges_metazoa_33208_clean.mapping.tsv \
        --ranks phylum class order family \
        --k 10 --queries-per-rank 3000 --repeats 3 --seed 0 \
        -o artifacts/tags/metazoa_lower_lr_bigger_batch/knn_purity/
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# Reuse the analyzer's loaders + label machinery so purity and separation use identical groups.
from analyze_hierarchy_hyperbolic import (  # noqa: E402
    load_embeddings,
    load_mapping,
    load_taxonomy_with_depth,
    get_ancestor_at_rank,
)
from _negative_hardness import numpy_poincare_distance  # noqa: E402  (float64 reference)


# ---------------------------------------------------------------------------
# Batched hyperbolic distance (matmul-expanded; identical formula to numpy_poincare_distance)
# ---------------------------------------------------------------------------

def _prep_sqnorms(emb, eps=1e-5):
    """Return (raw_sqnorm, clipped_sqnorm). Raw feeds ||u-v||^2; clipped feeds the denominator —
    matching numpy_poincare_distance, which clips ONLY the conformal-factor norms, not the diff."""
    raw = np.einsum("ij,ij->i", emb, emb).astype(np.float64)
    clipped = np.clip(raw, 0.0, 1.0 - eps)
    return raw, clipped


def _batch_distances(q_emb, q_raw, q_clip, pool_emb, pool_raw, pool_clip, eps=1e-5):
    """Poincaré distance from each query row to every pool row → (B, P) float32.

    ||u-v||^2 = ||u||^2 + ||v||^2 - 2 u·v ; the matmul (u·v) is the only heavy op.
    """
    dot = q_emb.astype(np.float32) @ pool_emb.astype(np.float32).T          # (B, P)
    sq_diff = (q_raw[:, None] + pool_raw[None, :] - 2.0 * dot.astype(np.float64))
    np.maximum(sq_diff, 0.0, out=sq_diff)                                   # float guard near u≈v
    denom = (1.0 - q_clip)[:, None] * (1.0 - pool_clip)[None, :]
    arg = 1.0 + 2.0 * sq_diff / denom
    np.maximum(arg, 1.0, out=arg)                                          # arccosh domain
    return np.arccosh(arg).astype(np.float32)


def _validate_distance(emb, n=300, seed=0):
    """Cross-check the matmul-expanded distance + neighbour ordering against the float64 helper."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(emb.shape[0], size=min(n, emb.shape[0]), replace=False)
    sub = emb[idx].astype(np.float64)
    raw, clip = _prep_sqnorms(sub)
    fast = _batch_distances(sub, raw, clip, sub, raw, clip)
    # Reference: full float64 pairwise via the broadcasting helper.
    ref = numpy_poincare_distance(sub[:, None, :], sub[None, :, :])
    max_abs = float(np.max(np.abs(fast.astype(np.float64) - ref)))
    # The decisive criterion for kNN is that neighbour ORDER agrees.
    order_ok = True
    for i in range(sub.shape[0]):
        d = fast[i].astype(np.float64).copy()
        d[i] = np.inf
        rd = ref[i].copy()
        rd[i] = np.inf
        if set(np.argpartition(d, 10)[:10]) != set(np.argpartition(rd, 10)[:10]):
            order_ok = False
            break
    print(f"[validate] max|fast-ref| = {max_abs:.2e}  top-10 order agrees = {order_ok}")
    if not order_ok:
        raise SystemExit("Distance validation FAILED — neighbour ordering disagrees with reference.")
    return max_abs


# ---------------------------------------------------------------------------
# kNN-purity
# ---------------------------------------------------------------------------

def build_pool(emb, idx2tax, taxonomy, rank):
    """Embedding indices (and their rank-R labels) for nodes that have an ancestor at `rank`."""
    max_idx = emb.shape[0] - 1
    pool_idx, pool_lab = [], []
    for idx, taxid in idx2tax.items():
        if idx > max_idx:
            continue
        anc = get_ancestor_at_rank(taxid, taxonomy, rank)
        if anc is not None:
            pool_idx.append(idx)
            pool_lab.append(anc)
    return np.asarray(pool_idx, dtype=np.int64), np.asarray(pool_lab, dtype=np.int64)


def chance_purity(pool_lab):
    """Expected purity of a random neighbour = sum_g (n_g / P)^2 (prob two random pool members match)."""
    _, counts = np.unique(pool_lab, return_counts=True)
    p = counts / counts.sum()
    return float(np.sum(p * p))


def knn_purity_for_pool(emb, raw_sqn, clip_sqn, pool_idx, pool_lab,
                        k, n_queries, seed, batch=128):
    """Mean kNN-purity over `n_queries` seeded query nodes drawn from the pool (full pool as candidates).

    Returns (purity_k, purity_1): fraction of the k (and the single) nearest neighbours sharing the
    query's label, averaged over queries. Self is excluded from neighbours.
    """
    P = pool_idx.shape[0]
    pool_emb = emb[pool_idx]
    pool_raw = raw_sqn[pool_idx]
    pool_clip = clip_sqn[pool_idx]

    rng = np.random.default_rng(seed)
    if n_queries >= P:
        qpos = np.arange(P)
    else:
        qpos = rng.choice(P, size=n_queries, replace=False)

    keff = min(k, P - 1)
    hits_k = np.zeros(qpos.shape[0], dtype=np.float64)
    hits_1 = np.zeros(qpos.shape[0], dtype=np.float64)

    for start in range(0, qpos.shape[0], batch):
        bpos = qpos[start:start + batch]
        d = _batch_distances(pool_emb[bpos], pool_raw[bpos], pool_clip[bpos],
                             pool_emb, pool_raw, pool_clip)            # (B, P)
        # Exclude self.
        d[np.arange(bpos.shape[0]), bpos] = np.inf
        # Top-(keff) nearest by partial sort.
        nn = np.argpartition(d, keff, axis=1)[:, :keff]               # (B, keff) unordered
        q_lab = pool_lab[bpos][:, None]                               # (B, 1)
        nn_lab = pool_lab[nn]                                         # (B, keff)
        match = (nn_lab == q_lab)
        hits_k[start:start + bpos.shape[0]] = match.mean(axis=1)
        # k=1: the single closest among the keff.
        nn_d = np.take_along_axis(d, nn, axis=1)                      # (B, keff)
        closest = nn[np.arange(bpos.shape[0]), np.argmin(nn_d, axis=1)]
        hits_1[start:start + bpos.shape[0]] = (pool_lab[closest] == pool_lab[bpos]).astype(float)

    return float(hits_k.mean()), float(hits_1.mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--data-dir", default=None, help="Taxonomy dump dir (default: analyzer's DATA_DIR)")
    ap.add_argument("--ranks", nargs="+", default=["phylum", "class", "order", "family"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--queries-per-rank", type=int, default=3000)
    ap.add_argument("--repeats", type=int, default=3, help="Independent query subsamples → mean±std")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("-o", "--output-dir", default=None)
    args = ap.parse_args()

    emb = load_embeddings(args.checkpoint).astype(np.float32)
    idx2tax = load_mapping(args.mapping)
    valid = set(int(t) for t in idx2tax.values())
    taxonomy = load_taxonomy_with_depth(valid, data_dir=args.data_dir)

    print("\nValidating distance implementation...")
    _validate_distance(emb)

    raw_sqn, clip_sqn = _prep_sqnorms(emb)

    results = {}
    print(f"\n{'='*80}\nkNN-PURITY (k={args.k}, queries/rank={args.queries_per_rank}, "
          f"repeats={args.repeats}, seed={args.seed})\n{'='*80}")
    for rank in args.ranks:
        pool_idx, pool_lab = build_pool(emb, idx2tax, taxonomy, rank)
        n_groups = len(np.unique(pool_lab))
        if pool_idx.shape[0] < 2 or n_groups < 2:
            print(f"\n[{rank}] SKIP — pool={pool_idx.shape[0]} groups={n_groups}")
            continue
        chance = chance_purity(pool_lab)
        pk_runs, p1_runs = [], []
        for r in range(args.repeats):
            pk, p1 = knn_purity_for_pool(
                emb, raw_sqn, clip_sqn, pool_idx, pool_lab,
                k=args.k, n_queries=args.queries_per_rank,
                seed=args.seed + r, batch=args.batch,
            )
            pk_runs.append(pk)
            p1_runs.append(p1)
        pk_m, pk_s = float(np.mean(pk_runs)), float(np.std(pk_runs))
        p1_m, p1_s = float(np.mean(p1_runs)), float(np.std(p1_runs))
        results[rank] = {
            "pool_size": int(pool_idx.shape[0]),
            "n_groups": int(n_groups),
            "chance_purity": chance,
            f"purity_k{args.k}_mean": pk_m,
            f"purity_k{args.k}_std": pk_s,
            "purity_k1_mean": p1_m,
            "purity_k1_std": p1_s,
            f"lift_k{args.k}": pk_m / chance if chance > 0 else float("inf"),
        }
        print(f"\n[{rank}] pool={pool_idx.shape[0]:,}  groups={n_groups:,}  chance={chance:.4f}")
        print(f"    purity@{args.k} = {pk_m:.4f} ± {pk_s:.4f}   "
              f"(lift over chance = {pk_m/chance:6.1f}×)")
        print(f"    purity@1  = {p1_m:.4f} ± {p1_s:.4f}")

    # Summary table.
    print(f"\n{'='*80}\nSUMMARY — kNN-PURITY\n{'='*80}")
    print(f"{'Rank':<10}{'pool':>10}{'groups':>9}{'chance':>9}"
          f"{'purity@'+str(args.k):>14}{'lift':>8}{'purity@1':>11}")
    for rank, r in results.items():
        print(f"{rank:<10}{r['pool_size']:>10,}{r['n_groups']:>9,}{r['chance_purity']:>9.4f}"
              f"{r[f'purity_k{args.k}_mean']:>11.4f}±{r[f'purity_k{args.k}_std']:.3f}"
              f"{r[f'lift_k{args.k}']:>7.1f}×{r['purity_k1_mean']:>10.4f}")

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": str(args.checkpoint),
            "k": args.k, "queries_per_rank": args.queries_per_rank,
            "repeats": args.repeats, "seed": args.seed,
            "results": results,
        }
        (out / "knn_purity.json").write_text(json.dumps(payload, indent=2))
        with (out / "knn_purity.tsv").open("w") as fh:
            fh.write("rank\tpool_size\tn_groups\tchance_purity\t"
                     f"purity_k{args.k}_mean\tpurity_k{args.k}_std\t"
                     f"lift_k{args.k}\tpurity_k1_mean\tpurity_k1_std\n")
            for rank, r in results.items():
                fh.write(f"{rank}\t{r['pool_size']}\t{r['n_groups']}\t{r['chance_purity']:.6f}\t"
                         f"{r[f'purity_k{args.k}_mean']:.6f}\t{r[f'purity_k{args.k}_std']:.6f}\t"
                         f"{r[f'lift_k{args.k}']:.6f}\t{r['purity_k1_mean']:.6f}\t{r['purity_k1_std']:.6f}\n")
        print(f"\nWrote: {out/'knn_purity.json'}\n       {out/'knn_purity.tsv'}")


if __name__ == "__main__":
    main()
