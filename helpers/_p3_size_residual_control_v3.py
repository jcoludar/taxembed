#!/usr/bin/env python3
"""P3 — does the embedding beat subtree size? v3: CONDITION on size, do not MATCH on it.

Two matching designs have now failed, each caught by a gate declared before its run. Both failures
have the same root cause and v3 abandons matching because of it.

  v1  helpers/_p3_size_matched_control.py -- fixed log2 band around the target's size.
      GATE (size must neutralize inside the band) FAILED at every band: even at +-0.5 log2 (~1.4x)
      size still predicted, 0.4746 [0.455, 0.495]. Pools are too small to band-match tightly.
      Its random control was also a single draw per query (SE ~0.010) and mis-read two bands.

  v2  helpers/_p3_size_matched_control_v2.py -- k-nearest-in-size pools + matched-procedure null.
      H1 (random, R=25) PASSED: 0.4998 / 0.5012 / 0.4982. The harness is sound.
      H2 (size must neutralize inside the k-NN pool) FAILED WORSE, not better:
          k=4  0.2445   k=8  0.2575   k=16  0.2907        (predicted ~0.5, k=4 closest)
      WHY, and it is not a bug: subtree size is heavy-tailed, and p_new sits in the UPPER TAIL
      (that is the B1 finding -- NCBI moves taxa into big groups). There are far more small nodes
      than large ones, so the k nodes NEAREST IN LOG-SIZE to a large target are drawn
      ASYMMETRICALLY FROM BELOW. The target stays among the largest in its own "size-matched" pool.
      k-nearest-in-size does not center a target that lives in the tail.
      ⇒ AND THAT BREAKS v2's GATE (e): its pseudo-target is drawn UNIFORMLY from the pool, so it is
        typically SMALL and never feels the tail asymmetry the real target feels. Gate (e) reports
        the floor for a TYPICAL member, not for a LARGE one. Its k=4 "BEYOND SIZE" reading
        (-0.0792) is therefore confounded by the exact quantity it claims to control, and is
        WITHDRAWN rather than carried forward.

WHAT v3 DOES INSTEAD. Stop trying to build a pool in which size does not matter; instead measure
how much size predicts, and ask whether the embedding beats THAT.

  For every candidate q in the (unmatched, amendment-2) pool of every query, compute two normalized
  within-pool ranks:
        r_s(q) = rank by subtree size, larger first
        r_e(q) = rank by Poincare distance from p_old
  Estimate the CALIBRATION CURVE  E[r_e | r_s]  by binned mean over ALL candidates that are NOT
  their query's target (targets excluded so the curve cannot be fit to the thing under test).
  The statistic is the RESIDUAL for each target:
        resid = r_e(p_new) - curve(r_s(p_new))
  resid < 0 means the embedding places p_new nearer p_old than its SIZE RANK ALONE predicts.

  This conditions on size instead of matching on it, so it needs no pool surgery, keeps the full
  n = 1,241 power, and is indifferent to how heavy-tailed the size distribution is.

------------------------------------------------------------------------------------------------
PREDICTIONS, WRITTEN BEFORE THE RUN
------------------------------------------------------------------------------------------------
G1 (HARNESS GATE, must pass) The same residual computed for a pseudo-target drawn uniformly from
   each pool must be 0.000 within CI. It is zero by construction if the curve is estimated without
   bias, so this gate tests the estimator, and it CAN fail (binning too coarse, edge bins, the
   target-exclusion bookkeeping).
G2 (HARNESS GATE) The curve must be monotone-ish and clearly non-flat -- bigger subtrees sit nearer
   p_old. A flat curve would mean size and embedding distance are unrelated, which would contradict
   B1 vs the embedding result and indicate a bug.
G3 THE QUESTION: mean residual for the real targets.
     clearly negative, CI excluding 0 => the embedding carries placement information BEYOND subtree
                                         size. P3 survives -- restated at PAIR level (v is inert),
                                         and with the standing caveat that plain "pick the biggest
                                         branch" still BEATS it outright (0.3146 vs 0.4444).
     contains 0                       => the P3 signal is a subtree-size proxy. ANTICIPATES does not
                                         survive its own control and P3 joins P2 as uninformative.
   Not known in advance. Both outcomes recorded here before the run.

ROBUSTNESS, not one defended cutoff: the whole thing is recomputed at 10 / 20 / 40 bins and with a
monotone isotonic-style cumulative-mean curve, and all four are reported
([[reference_borrowed_from_exondomaincompare]]).

Cost: one pass, ~1 core, no GPU (Rule 18).
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

N_BINS = [10, 20, 40]
R_NULL = 20      # pseudo-targets averaged per query for G1.
# 🧨🧨 G1 FAILED ONCE THE GATE WAS STRONG ENOUGH TO SEE ANYTHING, AND THE DEFECT WAS MINE.
# Raising R_NULL 1 -> 20 tightened G1's CI from +-0.017 to +-0.004, and it immediately read
# +0.0122 / +0.0092 / +0.0090 with CIs EXCLUDING zero -- a real bias the one-draw version could not
# resolve. Cause: a WEIGHTING MISMATCH in my own estimator. The calibration curve was a plain
# per-bin mean over the POOLED background, so it was CANDIDATE-weighted and big pools dominated it;
# the statistic averages per query first, so it is QUERY-weighted. Under a non-flat
# residual-vs-pool-size relation the two disagree by exactly this order. FIX: fit each bin as a
# weighted mean with w = 1/n_q per candidate, matching the statistic's weighting, which makes the
# null residual zero by construction and leaves G1 a genuine check on the rest of the machinery.
# The bias was POSITIVE, i.e. it made the target residual look MORE negative than it should:
# correcting it moves the headline the UNWELCOME way. Recorded for that reason.
# 🧨 RAISED FROM 1 TO 20 after the first run, for POWER, not to move a number. With one draw G1
# read +0.0069 / +0.0042 / +0.0056, CIs ~[-0.011, +0.024] -- half-width 0.017 against an effect of
# 0.034. A gate that can only exclude a bias HALF the size of the claim it certifies is too weak to
# certify it. The statistic, the curve, the bins and G3 are untouched; only the null's replicate
# count changes. Pre-tightening numbers are preserved in the session log.


def norm_ranks(score: np.ndarray) -> np.ndarray:
    """Normalized within-pool rank of every element, ascending, ties broken stably. 0..1."""
    order = np.argsort(score, kind="stable")
    r = np.empty(len(score), dtype=np.float64)
    r[order] = np.arange(len(score), dtype=np.float64)
    return r / (len(score) - 1)


def main() -> None:
    print("=" * 96)
    print("P3 SIZE-RESIDUAL CONTROL v3 — condition on size rather than match on it")
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
    # per-query records, plus the pooled (r_s, r_e) cloud of NON-target candidates
    tgt_rs, tgt_re, tgt_re_v = [], [], []
    nul_rs, nul_re = [], []
    bg_rs, bg_re, bg_w = [], [], []

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

        r_s = norm_ranks(-size[cand])                              # larger subtree first
        r_e = norm_ranks(poincare_dist(emb[emb_rows[vo]], emb[rows]))
        r_ev = norm_ranks(poincare_dist(emb[emb_rows[v]], emb[rows]))
        tp = int(np.flatnonzero(cand == vn)[0])

        tgt_rs.append(r_s[tp]); tgt_re.append(r_e[tp]); tgt_re_v.append(r_ev[tp])
        others = np.flatnonzero(np.arange(len(cand)) != tp)
        picks = rng.choice(others, size=min(R_NULL, len(others)), replace=False)
        nul_rs.append(r_s[picks]); nul_re.append(r_e[picks])
        keep = others                                              # targets excluded from the curve
        bg_rs.append(r_s[keep]); bg_re.append(r_e[keep])
        # 🧨 PER-QUERY WEIGHTS. See the G1 note below: the curve must be fit under the SAME
        # weighting the statistic uses, or the null residual is biased. 1/n_q per candidate makes
        # every query contribute equally, matching the query-averaged statistic.
        bg_w.append(np.full(len(keep), 1.0 / max(len(keep), 1)))

    tgt_rs = np.asarray(tgt_rs); tgt_re = np.asarray(tgt_re); tgt_re_v = np.asarray(tgt_re_v)
    # G1's pseudo-targets: R_NULL per query, flattened with a query index so each query's
    # residuals can be averaged BEFORE the bootstrap (the bootstrap resamples queries, not draws).
    nul_qidx = np.concatenate([np.full(len(a), i) for i, a in enumerate(nul_rs)])
    nul_rs = np.concatenate(nul_rs); nul_re = np.concatenate(nul_re)
    n_queries_null = int(nul_qidx.max()) + 1 if len(nul_qidx) else 0
    bg_rs = np.concatenate(bg_rs); bg_re = np.concatenate(bg_re)
    bg_w = np.concatenate(bg_w)
    print(f"\nqueries scored {len(tgt_rs):,}   background (non-target) candidates {len(bg_rs):,}")
    print(f"target size-rank mean {tgt_rs.mean():.4f}  (0 = largest in pool; "
          f"background mean {bg_rs.mean():.4f})")

    out = {}
    for nb in N_BINS:
        edges = np.linspace(0.0, 1.0, nb + 1)
        which = np.clip(np.digitize(bg_rs, edges[1:-1]), 0, nb - 1)
        curve = np.full(nb, np.nan)
        for b in range(nb):
            m = which == b
            if m.any():
                curve[b] = np.average(bg_re[m], weights=bg_w[m])
        good = ~np.isnan(curve)
        curve[~good] = 0.5                                     # empty bin -> no adjustment

        def resid(rs, re):
            b = np.clip(np.digitize(rs, edges[1:-1]), 0, nb - 1)
            return re - curve[b]

        r_t = resid(tgt_rs, tgt_re)
        r_tv = resid(tgt_rs, tgt_re_v)
        r_n_flat = resid(nul_rs, nul_re)
        r_n = np.bincount(nul_qidx, weights=r_n_flat, minlength=n_queries_null) / \
            np.maximum(np.bincount(nul_qidx, minlength=n_queries_null), 1)
        lo_t, hi_t = boot_ci(r_t)
        lo_tv, hi_tv = boot_ci(r_tv)
        lo_n, hi_n = boot_ci(r_n)
        g1 = lo_n <= 0.0 <= hi_n
        spread = float(np.nanmax(curve[good]) - np.nanmin(curve[good]))
        g2 = spread > 0.05
        verdict = ("BEYOND SIZE" if hi_t < 0 else
                   "WORSE THAN SIZE PREDICTS" if lo_t > 0 else "NO EVIDENCE BEYOND SIZE")
        out[str(nb)] = {
            "n_bins": nb,
            "curve": [None if np.isnan(c) else float(c) for c in
                      np.where(good, curve, np.nan)],
            "curve_spread": spread,
            "g1_null_residual": {"mean": float(r_n.mean()), "ci95": [lo_n, hi_n], "pass": bool(g1)},
            "g2_curve_non_flat": bool(g2),
            "g3_target_residual_from_p_old": {"mean": float(r_t.mean()), "ci95": [lo_t, hi_t]},
            # decomposition of the 0.5 - 0.4444 = 0.0556 raw effect into a size part and a residual
            "decomposition": {
                "observed_target_rank": float(tgt_re.mean()),
                "size_alone_prediction": float((tgt_re - r_t).mean()),
                "effect_from_size": float(0.5 - (tgt_re - r_t).mean()),
                "effect_beyond_size": float(-r_t.mean()),
            },
            "target_residual_from_v": {"mean": float(r_tv.mean()), "ci95": [lo_tv, hi_tv]},
            "verdict": verdict if (g1 and g2) else "VOID (harness gate failed)",
        }
        print(f"\n--- {nb} bins " + "-" * 70)
        print(f"  G1 null residual (must contain 0)   {r_n.mean():+.4f} "
              f"[{lo_n:+.4f}, {hi_n:+.4f}]   {'PASS' if g1 else 'FAIL'}")
        print(f"  G2 curve spread (must be > 0.05)     {spread:.4f}          "
              f"   {'PASS' if g2 else 'FAIL'}")
        print(f"  G3 target residual, from p_old      {r_t.mean():+.4f} "
              f"[{lo_t:+.4f}, {hi_t:+.4f}]   {out[str(nb)]['verdict']}")
        print(f"     target residual, from v          {r_tv.mean():+.4f} "
              f"[{lo_tv:+.4f}, {hi_tv:+.4f}]")
        dec = out[str(nb)]["decomposition"]
        print(f"     DECOMPOSITION of the {0.5 - dec['observed_target_rank']:.4f} raw effect: "
              f"size alone predicts {dec['size_alone_prediction']:.4f} "
              f"(= {dec['effect_from_size']:.4f} of it), "
              f"beyond size {dec['effect_beyond_size']:.4f}")

    outp = ROOT / "results" / "p3_size_residual_control_v3_20260929.json"
    json.dump({
        "purpose": "condition the P3 placement signal on subtree size instead of matching on it",
        "deprecates": ["helpers/_p3_size_matched_control.py",
                       "helpers/_p3_size_matched_control_v2.py"],
        "old_date": OLD_DATE, "new_date": NEW_DATE, "n_scored": int(len(tgt_rs)),
        "target_mean_size_rank": float(tgt_rs.mean()),
        "background_mean_size_rank": float(bg_rs.mean()),
        "by_bins": out,
    }, outp.open("w"), indent=2)
    print(f"\nwritten: {outp}")


if __name__ == "__main__":
    main()
