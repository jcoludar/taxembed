"""Application #2 — taxonomy QC / anomaly detection: per-node SIZE-CONDITIONED kNN-impurity score
(z vs a depth+clade-size-matched random-angle null), trivial-baseline comparison, BH-FDR, ranked
output. See docs/specs/2026-06-09-taxembed-paper-design.md §5#2 + §9B.

Local analysis only: pass explicit --checkpoint (the `final`/ep200 .pth) + --mapping. NEVER rely on
run.json (LRZ paths). Re-derive every baseline on the dataset at hand (spec §9C cellular trap).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT / "src"))

from analyze_hierarchy_hyperbolic import (load_embeddings, load_mapping,
                                          load_taxonomy_with_depth)
from knn_purity_hyperbolic import chance_purity, build_pool
from _anomaly_knn import observed_purity, matched_null, pick_device
from taxembed.eval.anomaly import (excess_impurity, matched_null_z, trivial_baselines,
                                   benjamini_hochberg)


def _index_tree(idx2tax, taxonomy=None, parent_col=None):
    """Return (parent_idx, depth, clade_size, degree) int arrays over node indices 0..N-1."""
    n = len(idx2tax)
    tax2idx = {t: i for i, t in idx2tax.items()}
    parent = np.arange(n, dtype=np.int64)
    depth = np.zeros(n, dtype=np.int64)
    for i in range(n):
        t = idx2tax[i]
        p = (parent_col[t] if parent_col is not None else taxonomy.get(t, {}).get("parent", t))
        parent[i] = tax2idx.get(p, i)
        if parent_col is not None:
            d, cur, seen = 0, t, set()
            while cur in parent_col and parent_col[cur] != cur and cur not in seen:
                seen.add(cur); cur = parent_col[cur]; d += 1
            depth[i] = d
        else:
            depth[i] = taxonomy.get(t, {}).get("depth", 0)
    degree = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if parent[i] != i:
            degree[parent[i]] += 1
    clade_size = np.ones(n, dtype=np.int64)
    for i in np.argsort(-depth):
        if parent[i] != i:
            clade_size[parent[i]] += clade_size[i]
    return parent, depth, clade_size, degree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--parent-from-mapping", action="store_true")
    ap.add_argument("--rank", default="family", help="taxdump rank for the label pool")
    ap.add_argument("--rank-from-mapping", default=None,
                    help="Test mode: use this mapping column as the label instead of a taxdump rank")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-null", type=int, default=200, help="matched-null draws per query")
    ap.add_argument("--n-bins", type=int, default=5, help="quantile bins for depth/size matching")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="cuda|mps|cpu|auto (default: auto-detect)")
    ap.add_argument("--knn-batch", type=int, default=1024, help="query rows per kNN batch (GPU memory)")
    ap.add_argument("-o", "--output-dir", required=True)
    args = ap.parse_args()

    emb = load_embeddings(args.checkpoint).astype(np.float32)
    idx2tax = load_mapping(args.mapping)

    taxonomy = None
    if args.parent_from_mapping:
        df = pd.read_csv(args.mapping, sep="\t")
        parent_col = {int(r.taxid): int(r.parent) for r in df.itertuples()}
        parent, depth, clade_size, degree = _index_tree(idx2tax, parent_col=parent_col)
    else:
        taxonomy = load_taxonomy_with_depth(set(idx2tax.values()), args.data_dir)
        parent, depth, clade_size, degree = _index_tree(idx2tax, taxonomy=taxonomy)

    if args.rank_from_mapping:
        df = pd.read_csv(args.mapping, sep="\t")
        lab_of = {int(r.taxid): int(getattr(r, args.rank_from_mapping)) for r in df.itertuples()}
        tax2idx = {t: i for i, t in idx2tax.items()}
        pool_idx, pool_lab = [], []
        for t, lab in lab_of.items():
            if lab >= 0 and t in tax2idx:
                pool_idx.append(tax2idx[t]); pool_lab.append(lab)
        pool_idx = np.asarray(pool_idx, np.int64); pool_lab = np.asarray(pool_lab, np.int64)
        rank_name = args.rank_from_mapping
    else:
        if taxonomy is None:                       # parent came from mapping but rank needs the taxdump
            taxonomy = load_taxonomy_with_depth(set(idx2tax.values()), args.data_dir)
        pool_idx, pool_lab = build_pool(emb, idx2tax, taxonomy, args.rank)
        rank_name = args.rank

    chance = chance_purity(pool_lab)
    device = pick_device(args.device)
    print(f"[device] kNN on {device} (pool={len(pool_idx)}, k={args.k}, batch={args.knn_batch})")

    observed = observed_purity(emb, pool_idx, pool_lab, args.k, device=device, batch=args.knn_batch)
    null_obs = matched_null(observed, pool_idx, depth, clade_size, args.n_null, args.n_bins, args.seed)

    score_z = matched_null_z(observed, null_obs)
    score_excess = excess_impurity(observed, chance)

    pvals = (np.sum(null_obs <= observed[:, None], axis=1) + 1) / (args.n_null + 1)
    qvals = benjamini_hochberg(pvals)

    base = trivial_baselines(emb, parent, depth, clade_size, degree)
    base_pool = {name: vals[pool_idx] for name, vals in base.items()}

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    order = np.argsort(-score_z)
    with (out / "anomaly_ranked.tsv").open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["taxid", "idx", "label", "observed_purity", "score_z", "score_excess",
                    "p_value", "q_value", "clade_size", "depth", "degree", "dist_to_parent_centroid"])
        for j in order:
            idx = int(pool_idx[j])
            w.writerow([idx2tax[idx], idx, int(pool_lab[j]), f"{observed[j]:.6f}",
                        f"{score_z[j]:.6f}", f"{score_excess[j]:.6f}", f"{pvals[j]:.6g}",
                        f"{qvals[j]:.6g}", int(clade_size[idx]), int(depth[idx]), int(degree[idx]),
                        f"{base_pool['dist_to_parent_centroid'][j]:.6f}"])

    np.savez(out / "anomaly_pool.npz",
             pool_idx=pool_idx, pool_lab=pool_lab, observed=observed,
             score_z=score_z, score_excess=score_excess, pvals=pvals, qvals=qvals,
             clade_size=clade_size[pool_idx], depth=depth[pool_idx], degree=degree[pool_idx],
             dist_to_parent_centroid=base_pool["dist_to_parent_centroid"])

    summary = {
        "checkpoint": str(args.checkpoint), "rank": rank_name, "k": args.k,
        "n_null": args.n_null, "n_bins": args.n_bins, "seed": args.seed,
        "device": device, "knn_batch": args.knn_batch,
        "pool_size": int(len(pool_idx)), "chance_purity": chance,
        "n_significant_q05": int(np.sum(qvals <= 0.05)),
        "top10_taxids": [int(idx2tax[int(pool_idx[j])]) for j in order[:10]],
    }
    (out / "anomaly_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
