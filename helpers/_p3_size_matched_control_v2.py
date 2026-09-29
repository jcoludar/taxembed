#!/usr/bin/env python3
"""P3 — does the placement signal survive size-matching? v2, with the RIGHT null.

🛑 DEPRECATED 2026-09-29, same day, by helpers/_p3_size_residual_control_v3.py — and its
"BEYOND SIZE" readings at k=4 (-0.0792) and k=8 (-0.0356) are WITHDRAWN, not carried forward.
H1 (random, R=25) PASSED at 0.4998 / 0.5012 / 0.4982, so the harness is sound. But H2 FAILED, and
worse than v1 rather than better: size inside the k-NN pool read 0.2445 / 0.2575 / 0.2907 against a
predicted ~0.5. The cause is not a bug — subtree size is heavy-tailed and p_new sits in the UPPER
TAIL (B1: NCBI moves taxa into big groups), so the k nodes nearest in log-size to a large target are
drawn ASYMMETRICALLY FROM BELOW and the target stays among the largest in its own "matched" pool.
That also breaks gate (e): its pseudo-target is drawn UNIFORMLY, so it is typically SMALL and never
feels the tail asymmetry the real target feels. Gate (e) measures the floor for a TYPICAL pool
member, not for a LARGE one — so it is confounded by the very quantity it exists to control.
v3 conditions on size instead of matching on it. Kept for the record and for H1/H2.

DEPRECATES helpers/_p3_size_matched_control.py (2026-09-29, same day). That version was not wrong
in its pool construction -- v2 reuses it verbatim -- but its control was the wrong instrument
twice over, and one of the two showed up as a visible defect:

  1. Its random control used ONE draw per query, SE ~0.010, and came back 0.5240 / 0.5206 at bands
     +-1 / +-2 (CIs excluding 0.5) while +-0.5 and inf passed. A synthetic check of `rank_by` over
     the real pool-size distribution (20,000 reps per size, sizes 2..120) returns 0.4941-0.5036 --
     UNBIASED. So those two were single-draw noise, but a control that noisy cannot gate anything.
     v2 averages R=25 replicates per query, dropping its SE ~5x.

  2. More important: A RANDOM SCORE IS THE WRONG NULL FOR A SIZE-MATCHED POOL. The matched pool is
     built around the TARGET's size, so the target is size-central in it by construction. Embedding
     distance correlates with subtree size, so size-centrality pushes the EMBEDDING's rank toward
     0.5 in a way a size-independent random score never feels. The random control therefore cannot
     see the bias that actually applies to the quantity being reported.

THE RIGHT NULL (gate e) is the one the original pre-registration already used in gate (b): apply
the IDENTICAL procedure to a pseudo-target. Draw t_null uniformly from the unmatched amendment-2
pool, build the matched pool around t_null, and rank t_null with the REAL embedding. That absorbs
the size-centrality effect, the pool-size distribution and the metric all at once. The primary is
then read AGAINST that empirical floor, not against 0.5.

Direction of the size-centrality effect, stated before the run: it pushes the primary TOWARD 0.5,
i.e. it is CONSERVATIVE for the embedding claim. Gate (e) measures how much.

------------------------------------------------------------------------------------------------
WHAT v1 ESTABLISHED AND v2 CARRIES FORWARD (results/p3_size_matched_control_20260929.json)
------------------------------------------------------------------------------------------------
  band     n_scored  med pool   size(B1)              emb from p_old
  +-0.5         961         5   0.4746 [.455,.495]    0.4613 [.437,.486]
  +-1         1,062        11   0.4129 [.396,.430]    0.4883 [.468,.509]
  +-2         1,139        18   0.3780 [.363,.394]    0.4668 [.448,.485]
  inf         1,241        30   0.3155 [.299,.333]    0.4444 [.426,.463]

Size is NEVER fully neutralized -- even at +-0.5 (within ~1.4x) B1 is 0.4746 with an upper bound of
0.495. By v1's own pre-declared gate, no band is clean. v2 therefore adds K-NEAREST-IN-SIZE
matching, which fixes the pool size and squeezes the residual size spread much harder than a fixed
band can.

------------------------------------------------------------------------------------------------
PREDICTIONS, WRITTEN BEFORE THE RUN
------------------------------------------------------------------------------------------------
H1  Random control, R=25 replicates -> 0.500 +- 0.005 at EVERY band and every k. Anything else is
    a harness bug and voids the run.
H2  Size (B1) inside a k-nearest-in-size pool -> much closer to 0.5 than v1's bands achieved;
    k=4 should be closest. This is the matching-worked check.
H3  Gate (e), the matched-procedure null -> somewhere in 0.47..0.53; whatever it is, it is THE
    FLOOR, and the verdict below is read against it, not against 0.5.
H4  THE QUESTION: primary minus gate (e).
      clearly negative, CI excluding 0  => the embedding carries placement information BEYOND
                                           subtree size; P3 survives, restated at pair level and
                                           with the "weaker than a one-line baseline" caveat.
      contains 0                        => the P3 effect was a subtree-size proxy; ANTICIPATES does
                                           not survive its own control and P3 joins P2.
    I do not know which. Both are written here so neither can be adopted after the fact.

Cost: distances over each unmatched pool computed ONCE and reused for every band, k and null draw;
~1 core, no GPU (Rule 18).
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

KS = [4, 8, 16]          # k-nearest-in-size pool sizes (including the target)
R_RANDOM = 25            # replicates for the random control
R_NULL = 5               # pseudo-targets per query for gate (e)


def nrank(score: np.ndarray, target_pos: int) -> float:
    """normalized_rank of the element at target_pos when `score` is sorted ascending."""
    order = np.argsort(score, kind="stable")
    return int(np.flatnonzero(order == target_pos)[0]) / (len(score) - 1)


def knn_size_pool(log_s: np.ndarray, t_pos: int, k: int) -> np.ndarray:
    """Positions of the k entries closest in log-size to entry t_pos (t_pos always included)."""
    d = np.abs(log_s - log_s[t_pos])
    d[t_pos] = -1.0                                  # force the target in first
    return np.argsort(d, kind="stable")[:k]


def main() -> None:
    print("=" * 100)
    print("P3 SIZE-MATCHED CONTROL v2 — k-nearest-in-size pools, matched-procedure null (gate e)")
    print("=" * 100)

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
    log_size = np.log2(np.clip((tout - tin).astype(np.float64), 1.0, None))

    moved = reclassified_taxa(set(taxid2row), old_parent_map, new_parent_map, new_merged, new_del)
    queries = []
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
        queries.append((v, vn, vo))

    rng = np.random.default_rng(SEED)
    acc = {k: {"emb_pold": [], "emb_v": [], "size": [], "rand": [], "null": []} for k in KS}

    for (v, vn, vo) in queries:
        L = lca(int(vo), int(vn), parent, depth)
        cand = pool_index.candidates(depth[vn], L, tin, tout)
        if len(cand) == 0:
            continue
        c_old = child_toward(L, vo, parent)
        if c_old is not None:
            cand = cand[~((tin[cand] >= tin[c_old]) & (tin[cand] < tout[c_old]))]
        cand = cand[(cand != vo) & (cand != v)]
        if len(cand) < 2 or not np.any(cand == vn):
            continue
        rows = emb_rows[cand]
        cand, rows = cand[rows >= 0], rows[rows >= 0]
        if len(cand) < 2 or not np.any(cand == vn):
            continue

        # distances over the WHOLE unmatched pool, once; every band/k/null reuses them
        d_pold = poincare_dist(emb[emb_rows[vo]], emb[rows])
        d_v = poincare_dist(emb[emb_rows[v]], emb[rows])
        ls = log_size[cand]
        neg_size = -np.exp2(ls)
        t_pos = int(np.flatnonzero(cand == vn)[0])

        for k in KS:
            if len(cand) < k:
                continue
            sel = knn_size_pool(ls, t_pos, k)
            tp = int(np.flatnonzero(sel == t_pos)[0])
            a = acc[k]
            a["emb_pold"].append(nrank(d_pold[sel], tp))
            a["emb_v"].append(nrank(d_v[sel], tp))
            a["size"].append(nrank(neg_size[sel], tp))
            a["rand"].append(float(np.mean(
                [nrank(rng.random(k), tp) for _ in range(R_RANDOM)])))
            # GATE E: identical procedure, pseudo-target drawn uniformly from the unmatched pool
            others = np.flatnonzero(np.arange(len(cand)) != t_pos)
            picks = rng.choice(others, size=min(R_NULL, len(others)), replace=False)
            vals = []
            for p in picks:
                sel_n = knn_size_pool(ls, int(p), k)
                tp_n = int(np.flatnonzero(sel_n == int(p))[0])
                vals.append(nrank(d_pold[sel_n], tp_n))
            a["null"].append(float(np.mean(vals)))

    print(f"\nqueries built: {len(queries):,}\n")
    hdr = (f"{'k':>4}{'n':>8}{'H1 random':>20}{'H2 size':>20}"
           f"{'GATE e null':>20}{'PRIMARY emb_pold':>22}")
    print(hdr)
    print("-" * len(hdr))
    out = {}
    for k in KS:
        a = acc[k]
        if not a["emb_pold"]:
            continue
        row = {"n_scored": len(a["emb_pold"])}
        for key in ("rand", "size", "null", "emb_pold", "emb_v"):
            arr = np.asarray(a[key])
            lo, hi = boot_ci(arr)
            row[key] = {"mean": float(arr.mean()), "ci95": [lo, hi]}
        # the question: primary vs the matched-procedure floor, paired per query
        d = np.asarray(a["emb_pold"]) - np.asarray(a["null"])
        lo_d, hi_d = boot_ci(d)
        row["primary_minus_gate_e"] = {"mean": float(d.mean()), "ci95": [lo_d, hi_d]}
        row["h1_random_ok"] = bool(abs(row["rand"]["mean"] - 0.5) <= 0.005)
        out[str(k)] = row
        print(f"{k:>4}{row['n_scored']:>8,}"
              f"{row['rand']['mean']:>12.4f} {'OK ' if row['h1_random_ok'] else 'BAD'}"
              f"{row['size']['mean']:>14.4f} [{row['size']['ci95'][0]:.3f}]"
              f"{row['null']['mean']:>14.4f} [{row['null']['ci95'][0]:.3f},{row['null']['ci95'][1]:.3f}]"
              f"{row['emb_pold']['mean']:>14.4f} [{row['emb_pold']['ci95'][0]:.3f},{row['emb_pold']['ci95'][1]:.3f}]")

    print(f"\n{'k':>4}   PRIMARY − GATE E  (negative = embedding beats its own matched floor)")
    print("-" * 70)
    for k in KS:
        if str(k) not in out:
            continue
        d = out[str(k)]["primary_minus_gate_e"]
        verdict = ("BEYOND SIZE" if d["ci95"][1] < 0 else
                   "WORSE THAN FLOOR" if d["ci95"][0] > 0 else "NO EVIDENCE BEYOND SIZE")
        print(f"{k:>4}   {d['mean']:+.4f}  CI95 [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]   {verdict}")

    outp = ROOT / "results" / "p3_size_matched_control_v2_20260929.json"
    json.dump({
        "purpose": "k-nearest-in-size matched pools with a matched-procedure null (gate e)",
        "deprecates": "helpers/_p3_size_matched_control.py (random-score null, 1 draw)",
        "old_date": OLD_DATE, "new_date": NEW_DATE,
        "k_values": KS, "r_random": R_RANDOM, "r_null": R_NULL,
        "n_queries": len(queries), "results": out,
    }, outp.open("w"), indent=2)
    print(f"\nwritten: {outp}")


if __name__ == "__main__":
    main()
