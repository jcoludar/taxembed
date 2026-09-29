#!/usr/bin/env python3
"""P3 — non-embedding baselines on the IDENTICAL pools. READ-ONLY, NO VERDICT.

WHY. `helpers/_p3_confound_diagnostics.py` (2026-09-29) established two things:

  P1  Tree distance from p_old is CONSTANT across the amendment-2 pool, 1,241 / 1,241 pools.
      The prescribed "tree-distance-matched control" is VACUOUS — it cannot move a candidate.
  P2  Substituting p_old for v as the query point reproduces the effect: 0.4444 vs the primary's
      0.4400, paired delta -0.0044 CI95 [-0.0145, +0.0062], containing zero.
      ⇒ the moved taxon's OWN coordinate contributes nothing measurable. The P3 statistic is a
        statement about the PAIR (p_old, p_new), not about v.

So the claim under test is no longer "the embedding anticipates where a taxon will move". It is:

    In the embedding of the OLD tree, the future parent p_new sits nearer p_old than
    topologically equivalent alternatives do.

"Topologically equivalent" is now literal, by P1: every candidate is at the same depth AND the same
tree distance from p_old. The discrete old tree therefore cannot tell them apart *by those two
features*. The open question is whether it can tell them apart by ANY cheap combinatorial feature —
because if it can, the embedding is not contributing the information, and the P2 diagnosis's own
lesson applies: the `degree_prior` baseline is what proved P2's pool and harness sound, and it beat
every embedding arm.

------------------------------------------------------------------------------------------------
PREDICTIONS, WRITTEN BEFORE THE RUN
------------------------------------------------------------------------------------------------
B0 random                   -> 0.500 exactly. Harness check; anything else voids the run.
B1 subtree size, LARGER 1st -> genuinely unsure. NCBI plausibly moves taxa INTO large, actively
                               curated groups, which would put this below 0.5.
B2 |taxid(q) - taxid(p_old)|, NEARER 1st
                            -> predicted ~0.5 (NO effect). The hyperbolic layout's ordering of
                               sibling branches around L is set by random initialization, not by
                               accession order, so taxid adjacency should carry nothing. If this
                               comes in well below 0.5 it is a SERIOUS artifact and the embedding
                               result needs re-examining against it, not alongside it.
B3 n_children, MORE 1st     -> unsure; a coarser proxy for the same thing as B1.

Each baseline's mirror direction is 1 - value, so one number covers both; no direction is chosen
after seeing the result.

MULTIPLE COMPARISONS. Four baselines are tried and the STRONGEST is compared against the
embedding. That is a garden of forking paths biased AGAINST the embedding claim — it can only make
the baseline look better than it is. Conservative for our purpose, and stated rather than hidden.

Cost: combinatorial features on pools already built; one pass, ~1 core, no GPU (Rule 18).
Written 2026-09-29.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings")
sys.path.insert(0, str(ROOT / "helpers"))
sys.path.insert(0, str(ROOT / "src"))

from _p3_placement_score import (  # noqa: E402
    EMB, MAPPING, NEW_DATE, OLD_DATE, PoolIndex, SEED, boot_ci, build_tree, child_toward,
    lca, load_mapping, load_safetensors, poincare_dist,
)
from taxembed.eval.release_diff import (  # noqa: E402
    canonicalize_taxid, parse_delnodes, parse_merged, parse_parents, reclassified_taxa,
)


def rank_by(score: np.ndarray, cand: np.ndarray, target: int, jitter: np.ndarray) -> float:
    """normalized_rank of `target` when candidates are sorted by `score` ascending."""
    order = np.argsort(score + jitter, kind="stable")
    pos = int(np.flatnonzero(cand[order] == target)[0])
    return pos / (len(cand) - 1)


def main() -> None:
    print("=" * 92)
    print("P3 COMBINATORIAL BASELINES — identical pools, non-embedding scores. No verdict.")
    print("=" * 92)

    emb = load_safetensors(EMB)
    taxid2row = load_mapping(MAPPING)

    old_dir = ROOT / "data" / f"taxdump_archive_{OLD_DATE}"
    new_dir = ROOT / "data" / f"taxdump_archive_{NEW_DATE}"
    old_parent_map = parse_parents(old_dir / "nodes.dmp")
    new_parent_map = parse_parents(new_dir / "nodes.dmp")
    new_merged = parse_merged(new_dir / "merged.dmp")
    new_del = parse_delnodes(new_dir / "delnodes.dmp")

    taxids, idx, parent, depth, tin, tout = build_tree(old_parent_map)
    n = len(taxids)
    emb_rows = np.full(n, -1, dtype=np.int64)
    for t, r in taxid2row.items():
        i = idx.get(t)
        if i is not None:
            emb_rows[i] = r
    pool_index = PoolIndex(depth, tin, emb_rows >= 0)

    subtree_size = (tout - tin).astype(np.int64)          # euler convention, see eval/subtree.py
    n_children = np.bincount(parent, minlength=n).astype(np.int64)
    n_children[parent == np.arange(n)] -= 1               # a root parents itself; do not count it
    print(f"old tree {n:,} nodes; subtree_size and n_children built")

    moved = reclassified_taxa(set(taxid2row), old_parent_map, new_parent_map, new_merged, new_del)
    primary_q = []
    for t in moved:
        v = idx.get(int(t))
        po = canonicalize_taxid(old_parent_map[int(t)], new_merged, new_del)
        pn = new_parent_map[int(t)]
        if v is None or po is None:
            continue
        vo, vn = idx.get(int(po)), idx.get(int(pn))
        if vo is None or vn is None:
            continue
        if emb_rows[v] < 0 or emb_rows[vo] < 0 or emb_rows[vn] < 0:
            continue
        primary_q.append((v, vn, vo))

    rng = np.random.default_rng(SEED)
    out: dict[str, list[float]] = {k: [] for k in
                                   ("emb_v", "emb_pold", "b0_random", "b1_subtree",
                                    "b2_taxid", "b3_children")}
    struct = {"p_new_is_sibling_of_p_old": 0, "total": 0}

    for (v, vn, vo) in primary_q:
        L = lca(int(vo), int(vn), parent, depth)
        cand = pool_index.candidates(depth[vn], L, tin, tout)
        if len(cand) == 0:
            continue
        c_old = child_toward(L, vo, parent)
        if c_old is not None and len(cand):
            in_old_branch = (tin[cand] >= tin[c_old]) & (tin[cand] < tout[c_old])
            cand = cand[~in_old_branch]
        cand = cand[(cand != vo) & (cand != v)]
        if len(cand) < 2 or not np.any(cand == vn):
            continue
        rows = emb_rows[cand]
        ok = rows >= 0
        cand, rows = cand[ok], rows[ok]
        if len(cand) < 2 or emb_rows[vn] < 0:
            continue

        jitter = rng.random(len(cand)) * 1e-12
        struct["total"] += 1
        if int(parent[vo]) == L and int(parent[vn]) == L:
            struct["p_new_is_sibling_of_p_old"] += 1

        out["emb_v"].append(rank_by(
            poincare_dist(emb[emb_rows[v]], emb[rows]), cand, vn, jitter))
        out["emb_pold"].append(rank_by(
            poincare_dist(emb[emb_rows[vo]], emb[rows]), cand, vn, jitter))
        out["b0_random"].append(rank_by(
            rng.random(len(cand)), cand, vn, jitter * 0))
        out["b1_subtree"].append(rank_by(
            -subtree_size[cand].astype(np.float64), cand, vn, jitter))     # larger first
        out["b2_taxid"].append(rank_by(
            np.abs(taxids[cand] - taxids[vo]).astype(np.float64), cand, vn, jitter))  # nearer 1st
        out["b3_children"].append(rank_by(
            -n_children[cand].astype(np.float64), cand, vn, jitter))       # more children first

    print(f"\nscored {len(out['emb_v']):,} taxa "
          f"(p_new is a direct sibling of p_old in "
          f"{100 * struct['p_new_is_sibling_of_p_old'] / max(struct['total'], 1):.1f} %)\n")
    print(f"  {'score':<34} {'mean':>8}  {'CI95':>20}")
    print("  " + "-" * 66)
    res = {}
    for k in ("emb_v", "emb_pold", "b0_random", "b1_subtree", "b2_taxid", "b3_children"):
        a = np.asarray(out[k])
        lo, hi = boot_ci(a)
        res[k] = {"mean": float(a.mean()), "ci95": [lo, hi]}
        label = {"emb_v": "EMBEDDING from v (primary)",
                 "emb_pold": "EMBEDDING from p_old",
                 "b0_random": "B0 random (must be 0.500)",
                 "b1_subtree": "B1 subtree size, larger first",
                 "b2_taxid": "B2 |taxid - taxid(p_old)|, nearer",
                 "b3_children": "B3 n_children, more first"}[k]
        print(f"  {label:<34} {a.mean():>8.4f}  [{lo:.4f}, {hi:.4f}]")

    best_b = min(("b1_subtree", "b2_taxid", "b3_children"),
                 key=lambda k: min(res[k]["mean"], 1 - res[k]["mean"]))
    bm = res[best_b]["mean"]
    bm_eff = min(bm, 1 - bm)
    print(f"\n  strongest combinatorial baseline (either direction): {best_b} "
          f"-> effective {bm_eff:.4f}")
    print(f"  embedding from p_old: {res['emb_pold']['mean']:.4f}")

    # paired: does the embedding beat the strongest baseline, taxon by taxon?
    b_arr = np.asarray(out[best_b])
    if bm > 0.5:
        b_arr = 1.0 - b_arr                       # read the baseline in its stronger direction
    d = np.asarray(out["emb_pold"]) - b_arr
    lo_d, hi_d = boot_ci(d)
    print(f"  PAIRED (embedding_pold - {best_b}), negative = embedding better: "
          f"{d.mean():+.4f} CI95 [{lo_d:+.4f}, {hi_d:+.4f}]")

    outp = ROOT / "results" / "p3_combinatorial_baselines_20260929.json"
    json.dump({
        "purpose": "non-embedding baselines on the P3 amendment-2 pools; read-only, no verdict",
        "old_date": OLD_DATE, "new_date": NEW_DATE,
        "n_scored": len(out["emb_v"]),
        "p_new_is_direct_sibling_of_p_old_frac":
            struct["p_new_is_sibling_of_p_old"] / max(struct["total"], 1),
        "scores": res,
        "strongest_baseline": best_b,
        "paired_embedding_minus_best_baseline": {"mean": float(d.mean()), "ci95": [lo_d, hi_d]},
        "multiple_comparisons_note":
            "3 baselines tried, strongest selected; biased AGAINST the embedding claim",
    }, outp.open("w"), indent=2)
    print(f"\nwritten: {outp}")


if __name__ == "__main__":
    main()
