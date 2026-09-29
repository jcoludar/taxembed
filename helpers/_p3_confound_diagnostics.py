#!/usr/bin/env python3
"""P3 — diagnose the residual confound qualification 3 names. READ-ONLY, NO VERDICT.

WHY THIS EXISTS. The 2026-09-29 P3 result (ANTICIPATES, primary 0.4400 vs chance 0.5, four gates
passing) closed with one uncontrolled residual, owed before any manuscript sentence:

    "Revisions typically move a taxon to a NEARBY sister group. The pool excludes p_old's branch
     but NOT branches ADJACENT to it. If p_new is systematically tree-nearer to p_old than a
     uniform pool draw, the embedding ranks it well for encoding the OLD tree faithfully, not for
     anticipating anything. Gate (b) draws uniformly, so it does NOT catch this. The fix: match
     each pseudo-parent on TREE DISTANCE from p_old, not only depth and subtree."

This helper does NOT implement that fix. It first MEASURES whether the thing to be matched varies
at all, because the prescribed fix may be vacuous, and a control that could not have changed
anything is not evidence ([[feedback_a_check_that_could_not_have_failed_is_not_evidence]]).

------------------------------------------------------------------------------------------------
PREDICTIONS, WRITTEN BEFORE THE RUN (falsifiable; the confirmation/refutation is the output)
------------------------------------------------------------------------------------------------

P1 — TREE DISTANCE IS ALREADY CONSTANT ACROSS THE AMENDMENT-2 POOL. Predicted: for >=99% of
     primary queries, every candidate q in the pool has the SAME path distance to p_old, and it
     equals path(p_old, p_new).

     The argument: the pool is (i) depth-homogeneous -- every q sits at depth(p_new) -- and
     (ii) after amendment 2, no q lies in p_old's child-branch of L, so LCA(p_old, q) == L for
     EVERY q. Hence
         path(p_old, q) = (depth(p_old) - depth(L)) + (depth(p_new) - depth(L))
     which contains no term that varies with q.

     If P1 holds, the prescribed tree-distance matching cannot move a single candidate, the owed
     control is VACUOUS AS SPECIFIED, and the real residual must be restated. If P1 is REFUTED,
     the prescribed fix is live and I build it.

P2 — THE REAL RESIDUAL, RESTATED GEOMETRICALLY. Unweighted path distance is not what drives the
     statistic; hyperbolic position does. Sibling branches of L are equidistant in path length but
     NOT in the embedding: the layout gives each child of L an angular sector, so some branches sit
     angularly nearer p_old's branch than others. If p_new tends to land in a branch that is
     EMBEDDING-near p_old, then v -- which was trained adjacent to p_old -- ranks p_new well for
     encoding the OLD tree, which is exactly the confound, surviving P1.

     The decisive test is a SUBSTITUTION control, not a matching one: recompute the identical
     statistic on the identical pool with the query point p_old INSTEAD OF v.
       - if the effect is "v anticipates its new parent", p_old must be WEAKER than v: v's offset
         from p_old is the only place anticipation could live;
       - if the effect is "p_new sits near p_old and v sits near p_old", p_old is EQUAL OR
         STRONGER, and the ANTICIPATES verdict is explained away without any anticipation.
     Reported as the PAIRED delta (rank_v - rank_pold) per taxon, with a bootstrap CI.

     I do not know this one's answer. That is the point of running it.

------------------------------------------------------------------------------------------------
Pool construction is IMPORTED from helpers/_p3_placement_score.py rather than re-implemented, so
the pool measured here is the pool that was scored. The reproduction of primary = 0.4400 on the
same queries is the check that the import did its job; if that number does not come back, nothing
below is about the scored pool and the run is void.

Cost: single pass, ~1 core, no GPU, no training (Rule 18 respected).
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


def path_distance(a: int, b: int, parent, depth) -> int:
    """Unweighted tree path length between two nodes."""
    anc = lca(int(a), int(b), parent, depth)
    return int(depth[a]) + int(depth[b]) - 2 * int(depth[anc])


def main() -> None:
    print("=" * 92)
    print("P3 CONFOUND DIAGNOSTICS — read-only, no verdict")
    print("  P1: is tree distance from p_old already constant across the amendment-2 pool?")
    print("  P2: does p_old, substituted for v, reproduce the effect? (paired)")
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
    embedded_mask = emb_rows >= 0
    pool_index = PoolIndex(depth, tin, embedded_mask)
    print(f"old tree {n:,} nodes, {int(embedded_mask.sum()):,} embedded")

    moved = reclassified_taxa(set(taxid2row), old_parent_map, new_parent_map, new_merged, new_del)

    # ---- rebuild the PRIMARY query list exactly as the scorer does
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
    print(f"primary queries: {len(primary_q):,}\n")

    rng = np.random.default_rng(SEED)
    rank_v, rank_old, pool_sz = [], [], []
    pd_constant, pd_varies, pd_target_equals = 0, 0, 0
    pd_spread_examples = []

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
        pool_sz.append(len(cand))
        if len(cand) < 2:
            continue
        if not np.any(cand == vn):
            continue
        rows = emb_rows[cand]
        ok = rows >= 0
        cand, rows = cand[ok], rows[ok]
        if len(cand) < 2 or emb_rows[vn] < 0:
            continue

        # ---- P1: path distance from p_old to every candidate in THIS pool
        pds = {path_distance(vo, int(q), parent, depth) for q in cand}
        pd_t = path_distance(vo, int(vn), parent, depth)
        if len(pds) == 1:
            pd_constant += 1
            if next(iter(pds)) == pd_t:
                pd_target_equals += 1
        else:
            pd_varies += 1
            if len(pd_spread_examples) < 5:
                pd_spread_examples.append(
                    (int(taxids[v]), sorted(pds), pd_t, len(cand)))

        # ---- P2: the same pool, ranked from v and from p_old
        jitter = rng.random(len(cand)) * 1e-12
        for src, store in ((v, rank_v), (vo, rank_old)):
            d = poincare_dist(emb[emb_rows[src]], emb[rows]) + jitter
            order = np.argsort(d, kind="stable")
            pos = int(np.flatnonzero(cand[order] == vn)[0])
            store.append(pos / (len(cand) - 1))

    rank_v = np.asarray(rank_v)
    rank_old = np.asarray(rank_old)
    n_scored = len(rank_v)

    print("-" * 92)
    print("P1 — TREE DISTANCE FROM p_old ACROSS THE POOL")
    tot = pd_constant + pd_varies
    print(f"  pools where path(p_old, q) is CONSTANT over all q : {pd_constant:,} / {tot:,} "
          f"({100 * pd_constant / max(tot, 1):.2f} %)")
    print(f"  ... and that constant equals path(p_old, p_new)   : {pd_target_equals:,} / {tot:,} "
          f"({100 * pd_target_equals / max(tot, 1):.2f} %)")
    print(f"  pools where it VARIES                             : {pd_varies:,}")
    for ex in pd_spread_examples:
        print(f"      e.g. taxid {ex[0]}: distances {ex[1]}, target at {ex[2]}, pool {ex[3]}")
    print(f"  PREDICTION P1 (>=99 % constant): "
          f"{'CONFIRMED' if pd_constant >= 0.99 * max(tot, 1) else 'REFUTED'}")

    print("\n" + "-" * 92)
    print("P2 — SUBSTITUTION CONTROL: the same pool, ranked from p_old instead of v")
    lo_v, hi_v = boot_ci(rank_v)
    lo_o, hi_o = boot_ci(rank_old)
    print(f"  rank from v      (PRIMARY, must reproduce 0.4400)  "
          f"mean={rank_v.mean():.4f}  CI95=[{lo_v:.4f}, {hi_v:.4f}]  n={n_scored:,}")
    print(f"  rank from p_old  (substitution control)            "
          f"mean={rank_old.mean():.4f}  CI95=[{lo_o:.4f}, {hi_o:.4f}]")
    delta = rank_v - rank_old
    lo_d, hi_d = boot_ci(delta)
    print(f"  PAIRED delta (v - p_old), negative = v knows more  "
          f"mean={delta.mean():+.4f}  CI95=[{lo_d:+.4f}, {hi_d:+.4f}]")
    print(f"  median pool {int(np.median(pool_sz)) if pool_sz else 0}")

    repro_ok = abs(rank_v.mean() - 0.4400) < 0.005
    print(f"\n  reproduction check (primary within 0.005 of 0.4400): "
          f"{'OK' if repro_ok else 'MISMATCH — everything above is void'}")

    out = ROOT / "results" / "p3_confound_diagnostics_20260929.json"
    json.dump({
        "purpose": "read-only diagnosis of P3 qualification 3; no verdict, no amendment applied",
        "old_date": OLD_DATE, "new_date": NEW_DATE,
        "n_primary_queries": len(primary_q), "n_scored": n_scored,
        "p1_tree_distance": {
            "pools_constant": pd_constant, "pools_varying": pd_varies,
            "constant_equals_target_distance": pd_target_equals,
            "prediction_confirmed": bool(pd_constant >= 0.99 * max(tot, 1)),
        },
        "p2_substitution": {
            "rank_from_v": {"mean": float(rank_v.mean()), "ci95": [lo_v, hi_v]},
            "rank_from_p_old": {"mean": float(rank_old.mean()), "ci95": [lo_o, hi_o]},
            "paired_delta_v_minus_pold": {"mean": float(delta.mean()), "ci95": [lo_d, hi_d]},
        },
        "reproduction_of_primary_0p4400": bool(repro_ok),
    }, out.open("w"), indent=2)
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
