"""Application #1 — cophenetic fidelity: how faithfully embedded Poincaré distance recovers true tree
distance. Reports LOCAL rank fidelity (kNN-retrieval precision@k, per-query) + global multiplicative
distortion, for the MODEL vs a RADIAL-ONLY null, with taxon-level bootstrap CIs. See
docs/specs/2026-06-09-taxembed-paper-design.md §5#1 + §9B.

Local analysis only: pass explicit --checkpoint (the `final`/ep200 .pth) + --mapping.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))            # for _negative_hardness, analyze_hierarchy_hyperbolic
sys.path.insert(0, str(ROOT / "src"))

from _negative_hardness import numpy_poincare_distance
from analyze_hierarchy_hyperbolic import load_embeddings, load_mapping, load_taxonomy_with_depth
from taxembed.eval.treedist import TreeDistance
from taxembed.eval.nulls import radial_only_null
from taxembed.eval.pairs import sample_pairs_stratified
from taxembed.eval.bootstrap import taxon_bootstrap_ci
from taxembed.eval.fidelity import multiplicative_distortion, knn_retrieval_precision


def build_index_tree(idx2tax, taxonomy=None, parent_col=None):
    """Return (parent_idx, depth) int arrays over node indices 0..N-1.
    parent_col: optional {taxid: parent_taxid} (test mode); else use taxonomy dict from taxdump."""
    n = len(idx2tax)
    tax2idx = {t: i for i, t in idx2tax.items()}
    parent = np.arange(n, dtype=np.int64)
    depth = np.zeros(n, dtype=np.int64)
    for i in range(n):
        t = idx2tax[i]
        p = (parent_col[t] if parent_col is not None else taxonomy.get(t, {}).get("parent", t))
        parent[i] = tax2idx.get(p, i)
        if parent_col is not None:
            # depth via walk (small test trees only)
            d, cur, seen = 0, t, set()
            while cur in parent_col and parent_col[cur] != cur and cur not in seen:
                seen.add(cur); cur = parent_col[cur]; d += 1
            depth[i] = d
        else:
            depth[i] = taxonomy.get(t, {}).get("depth", 0)
    return parent, depth


def _fidelity_block(emb, td, query_idx, cand_idx, k):
    """Per-query kNN-retrieval precision@k over a fixed candidate pool."""
    q = emb[query_idx]                                       # (Q, D)
    c = emb[cand_idx]                                        # (C, D)
    d_emb = numpy_poincare_distance(q[:, None, :], c[None, :, :])   # (Q, C)
    # tree distances query x candidate
    Q, C = len(query_idx), len(cand_idx)
    aa = np.repeat(query_idx, C); bb = np.tile(cand_idx, Q)
    d_tree = td.path_length(aa, bb).reshape(Q, C).astype(float)
    # mask self
    self_mask = query_idx[:, None] == cand_idx[None, :]
    d_emb = d_emb.copy(); d_tree = d_tree.copy()
    d_emb[self_mask] = np.inf; d_tree[self_mask] = np.inf
    return knn_retrieval_precision(d_emb, d_tree, k=k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--parent-from-mapping", action="store_true",
                    help="Test mode: read a 'parent' column from the mapping instead of the taxdump")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-queries", type=int, default=3000)
    ap.add_argument("--n-cands", type=int, default=2000)
    ap.add_argument("--n-pairs", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--output-dir", required=True)
    args = ap.parse_args()

    emb = load_embeddings(args.checkpoint)
    idx2tax = load_mapping(args.mapping)
    if args.parent_from_mapping:
        df = pd.read_csv(args.mapping, sep="\t")
        parent_col = {int(r.taxid): int(r.parent) for r in df.itertuples()}
        parent, depth = build_index_tree(idx2tax, parent_col=parent_col)
    else:
        taxonomy = load_taxonomy_with_depth(set(idx2tax.values()), args.data_dir)
        parent, depth = build_index_tree(idx2tax, taxonomy=taxonomy)

    td = TreeDistance(parent, depth)
    rng = np.random.default_rng(args.seed)
    n = len(emb)
    query_idx = rng.choice(n, size=min(args.n_queries, n), replace=False)
    cand_idx = rng.choice(n, size=min(args.n_cands, n), replace=False)

    null = radial_only_null(emb, seed=args.seed)

    result = {}
    for name, E in [("model", emb), ("radial_only_null", null)]:
        per_q = _fidelity_block(E, td, query_idx, cand_idx, args.k)
        mean, lo, hi = taxon_bootstrap_ci(per_q, seed=args.seed)
        # global distortion on a stratified pair sample
        a = rng.integers(0, n, args.n_pairs); b = rng.integers(0, n, args.n_pairs)
        dt = td.path_length(a, b)
        keep = sample_pairs_stratified(dt, bin_edges=[0, 2, 4, 6, 8, 100], per_bin=args.n_pairs // 10, seed=args.seed)
        de = numpy_poincare_distance(E[a[keep]], E[b[keep]])
        dist = multiplicative_distortion(de, dt[keep].astype(float))
        result[name] = {"knn_precision": {"mean": mean, "lo": lo, "hi": hi, "k": args.k},
                        "distortion": dist}

    result["delta_knn_precision"] = result["model"]["knn_precision"]["mean"] - \
        result["radial_only_null"]["knn_precision"]["mean"]

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "cophenetic_fidelity.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
