#!/usr/bin/env python3
"""P3 — is the embedding's placement signal anything more than a SUBTREE-SIZE proxy?

🛑 DEPRECATED 2026-09-29, same day, by helpers/_p3_size_matched_control_v2.py. Its POOL
CONSTRUCTION is sound and v2 reuses it verbatim; its CONTROL was the wrong instrument twice:
  (1) the random control used ONE draw per query (SE ~0.010) and read 0.5240 / 0.5206 at bands
      +-1 / +-2, CIs excluding 0.5, while +-0.5 and inf passed. `rank_by` was then shown UNBIASED
      by synthetic test (20,000 reps per pool size, 2..120, all 0.494-0.504), so those were noise
      -- but a control that noisy cannot gate anything.
  (2) a RANDOM SCORE is the wrong null for a size-matched pool at all: the pool is built around the
      target's size, so the target is size-central by construction, and embedding distance
      correlates with size. The bias that applies to the reported quantity is invisible to a
      size-independent score. v2 replaces it with a matched-PROCEDURE null (gate e).
Its numbers are kept as the fixed-band robustness surface and are quoted in v2's docstring.

WHY. `helpers/_p3_combinatorial_baselines.py` (2026-09-29) on the identical amendment-2 pools:

    EMBEDDING from v (primary)           0.4400  [0.4214, 0.4588]
    EMBEDDING from p_old                 0.4444  [0.4257, 0.4628]
    B0 random (harness check)            0.4937  [0.4761, 0.5110]   <- contains 0.5, PASS
    B1 subtree size, larger first        0.3146  [0.2982, 0.3313]   <- CRUSHES the embedding
    B2 |taxid - taxid(p_old)|, nearer    0.5571  [0.5392, 0.5747]
    B3 n_children, more first            0.3188  [0.3020, 0.3359]
    PAIRED (emb_pold - B1) = +0.1298 CI95 [+0.1107, +0.1488]

NCBI moves taxa into BIG groups, and "pick the largest candidate branch" predicts the destination
far better than the embedding does. A hyperbolic embedding partially encodes subtree size (angular
sector width scales with it), so 0.4444 may be nothing but a weak, lossy proxy for 0.3146.

THE TEST. Match the candidate pool on subtree size as well as depth and subtree membership, then
re-rank. Matching on a property of p_new is the same move amendment 2 made with depth and the
L-branch, and is legitimate for the same reason: it holds constant the thing that would otherwise
do the explaining.

Reported as a ROBUSTNESS SURFACE over the band, not one defended cutoff
([[reference_borrowed_from_exondomaincompare]]): bands of 0.5 / 1.0 / 2.0 in log2 size
(within ~1.4x / 2x / 4x of p_new's subtree size), plus the unmatched pool as the 'inf' row.

------------------------------------------------------------------------------------------------
PREDICTIONS, WRITTEN BEFORE THE RUN
------------------------------------------------------------------------------------------------
G  (the GATE, must pass or the band is void) B1 re-scored inside each size-matched pool must
   collapse to ~0.5. If size still predicts inside the matched pool, the match is too loose and
   that band says nothing. This gate CAN fail, and the tightest bands are where it should pass
   most cleanly.

E  (the question) The embedding, re-scored inside the matched pool:
     - stays near 0.44 with a CI excluding 0.5 => it carries placement information BEYOND subtree
       size, and P3 survives as a (restated, pair-level) result.
     - rises to ~0.5 => the entire P3 effect was a size proxy, `ANTICIPATES` does not survive its
       own control, and the honest verdict is that P3 joins P2.
   I do not know which. Both outcomes are written here before the run so neither can be adopted
   after the fact.

N  n_scored falls as the band tightens (queries whose matched pool drops below 2 are lost) and the
   surviving population is no longer the same one. n_scored is reported per band for that reason;
   a band below the pre-registered power floor of 1,000 is DESCRIPTIVE, not a verdict.

Cost: 4 bands x one pass, ~1 core, no GPU (Rule 18).
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

BANDS = [0.5, 1.0, 2.0, float("inf")]        # log2 units; inf = the unmatched amendment-2 pool


def rank_by(score, cand, target, jitter) -> float:
    order = np.argsort(score + jitter, kind="stable")
    return int(np.flatnonzero(cand[order] == target)[0]) / (len(cand) - 1)


def main() -> None:
    print("=" * 96)
    print("P3 SIZE-MATCHED CONTROL — is the placement signal more than a subtree-size proxy?")
    print("=" * 96)

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
    size = (tout - tin).astype(np.float64)
    log_size = np.log2(np.clip(size, 1.0, None))

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

    acc = {b: {"emb_v": [], "emb_pold": [], "size": [], "rand": [], "pool": []} for b in BANDS}
    rng = np.random.default_rng(SEED)

    for (v, vn, vo) in primary_q:
        L = lca(int(vo), int(vn), parent, depth)
        cand0 = pool_index.candidates(depth[vn], L, tin, tout)
        if len(cand0) == 0:
            continue
        c_old = child_toward(L, vo, parent)
        if c_old is not None:
            cand0 = cand0[~((tin[cand0] >= tin[c_old]) & (tin[cand0] < tout[c_old]))]
        cand0 = cand0[(cand0 != vo) & (cand0 != v)]
        if len(cand0) < 2 or not np.any(cand0 == vn):
            continue
        rows0 = emb_rows[cand0]
        ok = rows0 >= 0
        cand0 = cand0[ok]
        if len(cand0) < 2 or emb_rows[vn] < 0:
            continue

        for b in BANDS:
            if np.isinf(b):
                cand = cand0
            else:
                keep = np.abs(log_size[cand0] - log_size[vn]) <= b
                cand = cand0[keep]
            if len(cand) < 2 or not np.any(cand == vn):
                continue
            rows = emb_rows[cand]
            jit = rng.random(len(cand)) * 1e-12
            a = acc[b]
            a["pool"].append(len(cand))
            a["emb_v"].append(rank_by(poincare_dist(emb[emb_rows[v]], emb[rows]), cand, vn, jit))
            a["emb_pold"].append(rank_by(poincare_dist(emb[emb_rows[vo]], emb[rows]), cand, vn, jit))
            a["size"].append(rank_by(-size[cand], cand, vn, jit))
            a["rand"].append(rank_by(rng.random(len(cand)), cand, vn, jit * 0))

    print(f"\n{'band (log2)':<13}{'n_scored':>9}{'med pool':>9}"
          f"{'GATE size':>22}{'EMB from p_old':>24}{'EMB from v':>24}")
    print("-" * 101)
    out = {}
    for b in BANDS:
        a = acc[b]
        if not a["emb_v"]:
            print(f"{str(b):<13}{'0':>9}   (nothing scored)")
            continue
        row = {}
        for k in ("size", "emb_pold", "emb_v", "rand"):
            arr = np.asarray(a[k])
            lo, hi = boot_ci(arr)
            row[k] = {"mean": float(arr.mean()), "ci95": [lo, hi]}
        gate_ok = row["size"]["ci95"][0] <= 0.5 <= row["size"]["ci95"][1]
        row["gate_size_neutralized"] = bool(gate_ok)
        row["n_scored"] = len(a["emb_v"])
        row["median_pool"] = int(np.median(a["pool"]))
        out[("inf" if np.isinf(b) else f"{b:g}")] = row
        tag = "PASS" if gate_ok else "FAIL"
        lbl = "inf (unmatched)" if np.isinf(b) else f"±{b:g}"
        print(f"{lbl:<13}{len(a['emb_v']):>9,}{int(np.median(a['pool'])):>9}"
              f"   {row['size']['mean']:.4f} [{row['size']['ci95'][0]:.3f},"
              f"{row['size']['ci95'][1]:.3f}] {tag:<4}"
              f"   {row['emb_pold']['mean']:.4f} [{row['emb_pold']['ci95'][0]:.3f},"
              f"{row['emb_pold']['ci95'][1]:.3f}]"
              f"   {row['emb_v']['mean']:.4f} [{row['emb_v']['ci95'][0]:.3f},"
              f"{row['emb_v']['ci95'][1]:.3f}]")

    print("\n(random control per band, must contain 0.500)")
    for k, row in out.items():
        r = row["rand"]
        ok = r["ci95"][0] <= 0.5 <= r["ci95"][1]
        print(f"   band {k:<5} random {r['mean']:.4f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] "
              f"{'OK' if ok else 'BROKEN'}")

    outp = ROOT / "results" / "p3_size_matched_control_20260929.json"
    json.dump({
        "purpose": "does the P3 placement signal survive matching candidates on subtree size?",
        "old_date": OLD_DATE, "new_date": NEW_DATE,
        "bands_log2": ["0.5", "1", "2", "inf"],
        "prereg_power_floor": 1000,
        "results": out,
    }, outp.open("w"), indent=2)
    print(f"\nwritten: {outp}")


if __name__ == "__main__":
    main()
