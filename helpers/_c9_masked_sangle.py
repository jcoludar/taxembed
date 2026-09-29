#!/usr/bin/env python3
"""C9 — is the reported S_angle carried by the placeholder taxa? Answers it WITHOUT a retrain.

THE POINT. C9's decision (MANUSCRIPT_CORRECTIONS_PENDING.md) is DO NOT RETRAIN, and reason 2 is
that the question a retrain would answer is answerable by masking at SCORING time. A recommendation
that is only argued is not evidence; this runs it.

WHAT IS MASKED. 91,317 embedded taxa (43.9 % of Bacteria, 48.1 % of Archaea, 8.91 % of the whole
embedding) match r"\\b(?:bacterium|archaeon)\\b" -- the "<clade> bacterium|archaeon <id>" form whose
NCBI placement is just the clade the submitter typed. 91,305 of 91,317 (99.99 %) are LEAVES
(helpers/_c9_mask_feasibility.py), so pruning them cannot orphan a subtree.

🧨 THE NUMBER THAT MOTIVATES THIS: the depth-4 same-depth pool -- the pool S_angle ranks a depth-4
query against -- is **98.8 % placeholder** (48,987 of 49,560). Depth 6 is 61.7 %, depth 8 49.1 %,
depth 9 34.3 %. For a shallow query, nearly every "cousin" it is scored against is a
tautologically-placed node.

TWO CONDITIONS, one scorer (src/taxembed/eval/angular.py), production settings n=10,000 k=10 seed=0:

  A  SPLIT BY QUERY (primary, cheap). One tree, one pool, one query draw; partition the queries into
     placeholder and non-placeholder and score each. Only the query set differs, so a gap is a
     statement about the queries and nothing else. Asks: DO PLACEHOLDER TAXA SCORE BETTER THAN REAL
     ONES?

  B  POOL-MASKED (secondary). Prune the 91,305 placeholder leaves, rebuild the TreeIndex, re-score
     the surviving non-placeholder queries. Asks: DOES A REAL TAXON'S SCORE SURVIVE WHEN ITS
     TAUTOLOGICAL COUSINS LEAVE THE POOL? This is the closest no-GPU proxy for what a retrain does
     to the evaluation -- it is NOT a retrain, because the geometry was still fit with those nodes
     present, and that limit is stated in the output.

------------------------------------------------------------------------------------------------
PREDICTIONS, WRITTEN BEFORE THE RUN
------------------------------------------------------------------------------------------------
G1 (GATE) The init null -- random directions at the planted radii -- must score S ~ 0 in BOTH
   conditions. It is 0 by construction for a correct harness, so a nonzero value in condition B
   means the pruning broke the tree, the depths, or the per-query bounds. This gate CAN fail and is
   the whole reason condition B is trustworthy if it passes.
G2 S_angle is oracle-normalised PER QUERY (mu0 and mu* are recomputed from each query's actual
   pool), so it is DESIGNED to be robust to pool composition. Naive expectation: the overall S
   moves little.
G3 THE QUESTION. Prediction, stated so it can be wrong: the shallow band moves most, because
   pruning takes the depth-4 pool from 49,560 to ~573 members.
     |ΔS| < 0.02  => the headline is not carried by placeholder taxa; C9 is one Methods sentence
                     and the DO-NOT-RETRAIN call is confirmed on its own terms.
     |ΔS| >= 0.05 => material; the C9 retrain triggers get re-examined, and the masked number is
                     reported beside the full one in Methods either way.
   I do not know which. Both are recorded here before the run.

Cost: ~1 core, no GPU, no training (Rule 18). Reads the shipped artifact read-only.
Written 2026-09-29.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings")
sys.path.insert(0, str(ROOT / "helpers"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from _p3_placement_score import EMB, OLD_DATE, load_safetensors  # noqa: E402
from score_recipe_checkpoints import load_tree  # noqa: E402
from taxembed.eval.angular import (  # noqa: E402
    CLUSTER_LEVEL, TreeIndex, query_bounds, score_embedding, select_queries,
)
from taxembed.eval.radial import target_radius  # noqa: E402

CLOSURE = (ROOT / "data" / "taxopy" / "cellular_organisms_131567_clean"
           / "taxonomy_edges_cellular_organisms_131567_clean_transitive.npz")
MAPPING = (ROOT / "data" / "taxopy" / "cellular_organisms_131567_clean"
           / "taxonomy_edges_cellular_organisms_131567_clean.mapping.tsv")
NAMES = ROOT / "data" / f"taxdump_archive_{OLD_DATE}" / "names.dmp"
PLACEHOLDER = re.compile(r"\b(?:bacterium|archaeon)\b", re.IGNORECASE)
N_QUERIES, K, SEED, SCHEDULE = 10_000, 10, 0, "log"


def load_index_to_taxid(path: Path) -> dict[int, int]:
    out = {}
    with path.open() as fh:
        fh.readline()
        for line in fh:
            a, _, b = line.partition("\t")
            if a.strip().isdigit() and b.strip().isdigit():
                out[int(b)] = int(a)
    return out


def scientific_names(path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "scientific name" not in line:
                continue
            p = line.split("\t|\t")
            if len(p) < 2:
                continue
            try:
                out[int(p[0].strip())] = p[1].strip()
            except ValueError:
                continue
    return out


def init_null(n_nodes, depth, max_depth, seed):
    rng = np.random.default_rng(seed + 1)
    d = rng.standard_normal((n_nodes, 100))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return d * target_radius(depth.astype(np.float64), max_depth, SCHEDULE)[:, None]


def main() -> None:
    print("=" * 96)
    print("C9 — masked S_angle: is the headline carried by placeholder taxa? (no retrain)")
    print("=" * 96)
    t0 = time.time()

    parent, depth = load_tree(str(CLOSURE))
    n = len(parent)
    emb = load_safetensors(EMB)
    print(f"  closure {n:,} nodes; embedding {emb.shape}")
    if emb.shape[0] != n:
        print(f"  ⚠ embedding rows {emb.shape[0]:,} != closure nodes {n:,}")

    i2t = load_index_to_taxid(MAPPING)
    names = scientific_names(NAMES)
    is_ph = np.zeros(n, dtype=bool)
    for i in range(n):
        t = i2t.get(i)
        if t is not None and PLACEHOLDER.search(names.get(t, "")):
            is_ph[i] = True
    print(f"  placeholder nodes in the closure: {int(is_ph.sum()):,} "
          f"({100*is_ph.mean():.2f} %)   setup {time.time()-t0:.0f}s")

    out: dict = {"purpose": "C9 masked S_angle; answers the retrain question without retraining",
                 "n_nodes": n, "n_placeholder": int(is_ph.sum()),
                 "n_queries": N_QUERIES, "k": K, "seed": SEED}

    # ---------------- condition A: one tree, split the queries
    print("\n--- A: SPLIT BY QUERY (same tree, same pools) " + "-" * 40)
    idx = TreeIndex(parent, depth, seed=SEED)
    queries = select_queries(idx, n=N_QUERIES, k=K, seed=SEED)
    bounds = query_bounds(idx, queries, K)
    a_all = score_embedding(emb, idx, queries, K, bounds=bounds, seed=SEED)
    qph = is_ph[queries]
    print(f"  queries {len(queries):,}: {int(qph.sum()):,} placeholder / "
          f"{int((~qph).sum()):,} real")
    print(f"  S_angle ALL queries        {a_all['S']:+.4f}  (cluster se {a_all['S_cluster_se']:.4f})")
    res_a = {"all": {"n": int(len(queries)), "S": a_all["S"], "se": a_all["S_cluster_se"],
                     "by_band": a_all["by_band"]}}
    for label, m in (("placeholder", qph), ("real", ~qph)):
        if m.sum() < 50:
            print(f"  S_angle {label:<18} n={int(m.sum())} — too few to score")
            continue
        sub = score_embedding(emb, idx, queries[m], K, seed=SEED)
        res_a[label] = {"n": int(m.sum()), "S": sub["S"], "se": sub["S_cluster_se"],
                        "by_band": sub["by_band"]}
        print(f"  S_angle {label:<18} {sub['S']:+.4f}  (cluster se {sub['S_cluster_se']:.4f})"
              f"  n={int(m.sum()):,}")
    if "placeholder" in res_a and "real" in res_a:
        gap = res_a["placeholder"]["S"] - res_a["real"]["S"]
        res_a["gap_placeholder_minus_real"] = gap
        print(f"  GAP (placeholder − real): {gap:+.4f}")
    out["condition_A_split_by_query"] = res_a

    # ---------------- condition B: prune placeholder leaves, rescore the real queries
    print("\n--- B: POOL-MASKED (placeholder leaves pruned) " + "-" * 39)
    n_children = np.bincount(parent, minlength=n)
    n_children[parent == np.arange(n)] -= 1
    prunable = is_ph & (n_children == 0)
    keep = ~prunable
    print(f"  pruning {int(prunable.sum()):,} placeholder LEAVES; "
          f"{int((is_ph & ~prunable).sum()):,} placeholder non-leaves kept (cannot orphan)")

    old2new = np.full(n, -1, dtype=np.int64)
    old2new[keep] = np.arange(int(keep.sum()))
    p_new = old2new[parent[keep]]
    if (p_new < 0).any():
        raise SystemExit("a kept node's parent was pruned — pruning was not leaf-only")
    idx_b = TreeIndex(p_new, depth[keep], seed=SEED)
    emb_b = emb[keep]
    print(f"  pruned tree: {idx_b.n_nodes:,} nodes, max depth {idx_b.max_depth}")

    q_real_old = queries[~qph]
    q_b = old2new[q_real_old]
    q_b = q_b[q_b >= 0]
    bounds_b = query_bounds(idx_b, q_b, K)
    b_real = score_embedding(emb_b, idx_b, q_b, K, bounds=bounds_b, seed=SEED)
    print(f"  S_angle real queries, MASKED pool  {b_real['S']:+.4f} "
          f"(cluster se {b_real['S_cluster_se']:.4f})  n={len(q_b):,}")
    out["condition_B_pool_masked"] = {"n": int(len(q_b)), "S": b_real["S"],
                                      "se": b_real["S_cluster_se"], "by_band": b_real["by_band"],
                                      "n_nodes_after_prune": int(idx_b.n_nodes)}

    # ---------------- G1: the init null must be ~0 in BOTH trees
    print("\n--- G1 GATE: init null must be ~0 in both " + "-" * 44)
    z_a = score_embedding(init_null(n, depth, idx.max_depth, SEED), idx, queries[~qph], K, seed=SEED)
    z_b = score_embedding(init_null(idx_b.n_nodes, depth[keep], idx_b.max_depth, SEED),
                          idx_b, q_b, K, bounds=bounds_b, seed=SEED)
    g1 = abs(z_a["S"]) < 0.02 and abs(z_b["S"]) < 0.02
    print(f"  init null, FULL tree   {z_a['S']:+.4f}")
    print(f"  init null, PRUNED tree {z_b['S']:+.4f}")
    print(f"  G1 {'PASS' if g1 else 'FAIL — condition B is VOID'}")
    out["G1_init_null"] = {"full": z_a["S"], "pruned": z_b["S"], "pass": bool(g1)}

    # ---------------- verdict against the pre-declared thresholds
    d_s = out["condition_B_pool_masked"]["S"] - res_a.get("real", {}).get("S", float("nan"))
    out["delta_masked_minus_real_same_queries"] = d_s
    print("\n" + "=" * 96)
    print(f"  real queries, FULL pool   {res_a.get('real', {}).get('S', float('nan')):+.4f}")
    print(f"  real queries, MASKED pool {out['condition_B_pool_masked']['S']:+.4f}")
    print(f"  ΔS = {d_s:+.4f}")
    verdict = ("NOT CARRIED BY PLACEHOLDERS (|ΔS| < 0.02)" if abs(d_s) < 0.02 else
               "MATERIAL (|ΔS| >= 0.05) — re-examine the C9 retrain triggers" if abs(d_s) >= 0.05
               else "INTERMEDIATE (0.02 <= |ΔS| < 0.05) — report both numbers")
    out["verdict"] = verdict if g1 else "VOID (G1 failed)"
    print(f"  VERDICT: {out['verdict']}")
    print("=" * 96)

    outp = ROOT / "results" / "c9_masked_sangle_20260929.json"
    out["limit"] = ("Condition B masks at SCORING time only. The geometry was still FIT with the "
                    "placeholder nodes present, so this bounds what a retrain could change at "
                    "evaluation; it does not simulate a retrain.")
    json.dump(out, outp.open("w"), indent=2, default=float)
    print(f"\nwritten: {outp}   total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
