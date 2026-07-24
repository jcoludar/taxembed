"""Task 17 (exploratory): Radius / graded-confidence calibration for the Taxonomy Bridge — PLA2.

Hypothesis: proteins from *deeper* taxa (farther from root in the embedding's depth metric)
should be placed *nearer the Poincaré ball boundary* (larger norm). If so, the placed norm can
serve as a graded confidence signal (higher-norm = more reliable coarse-rank classification).

Steps
-----
1. LOSO placement — honest out-of-fold placed positions, identical CV to read_eval.py.
2. Spearman(radius, true_depth) over all proteins.
3. Reliability curve: bin proteins by radius quintile; per bin, compute coarse-rank (class or
   order for the 8 Sauropsida species without a class node) accuracy from the LOSO retrieval.
4. Emit results/radius_pla2.json with rho, p-value, per-bin table, n, and a verdict field.
5. Print a readable summary.

Exploratory: a null result (verdict="cut") is equally valid — reported honestly, not massaged.

Run:
  python -m taxembed.bridge.radius_calib
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_HERE = Path(__file__).resolve().parent

from . import config  # noqa: E402
from .h5_io import load_embeddings  # noqa: E402
from .core import Bridge, TaxonomyEmbedding, retrieve_nearest  # noqa: E402
from .splits import leave_species_out  # noqa: E402
from .taxdump import TaxonResolver  # noqa: E402

from .eval_utils import RankLookup  # noqa: E402

SEED = 0
ALPHA = 0.01          # from read_eval alpha selection (frozen; reuse)
RETRIEVE_CHUNK = 8192
N_BINS = 5            # quintiles for reliability curve
# Coarse rank for scoring. We use "class" where present, fall back to "order" for the 8
# Sauropsida species (4 Crocodiles + 4 Turtles) that lack a class node — same handling as
# read_eval.py's class_rank_note. Documented, not hidden.
COARSE_RANK = "class"
FALLBACK_RANK = "order"


# ---------------------------------------------------------------------------- data loading
def load_pla2():
    """Return (ann DataFrame, reps ndarray). Identical to read_eval.load_pla2 minus clustering."""
    emb = load_embeddings(config.PLA2_H5)
    ann = pd.read_csv(config.PLA2_CSV)
    res = pd.read_csv(config.PLA2_RESOLUTION, sep="\t")
    res = res.rename(columns={"species": "Species"})
    ann = ann.merge(res[["Species", "taxid", "idx"]], on="Species", how="left", validate="m:1")
    missing = ann[ann["taxid"].isna()]
    if len(missing):
        raise SystemExit(f"{len(missing)} proteins have unresolved species")
    ann["taxid"] = ann["taxid"].astype(int)
    ann["idx"] = ann["idx"].astype(int)
    ids = ann["identifier"].tolist()
    if set(ids) - set(emb):
        raise SystemExit(f"identifiers missing from h5: {sorted(set(ids) - set(emb))[:5]}")
    ann = ann.reset_index(drop=True)
    reps = np.stack([emb[i].astype(np.float64) for i in ann["identifier"]])
    return ann, reps


# ---------------------------------------------------------------------------- taxonomy helpers
def coarse_rank_hit(true_taxid: int, pred_taxid: int, ranklut: RankLookup) -> tuple[bool, bool, str]:
    """Return (scored, hit, rank_used).

    For taxa that have a 'class' node: scored=True, rank_used='class'.
    For the 8 Sauropsida taxa without 'class': fall back to 'order'.
    If neither rank exists in true lineage: scored=False.
    """
    trm = ranklut.rank_map(int(true_taxid))
    prm = ranklut.rank_map(int(pred_taxid))
    if COARSE_RANK in trm:
        rank_used = COARSE_RANK
    elif FALLBACK_RANK in trm:
        rank_used = FALLBACK_RANK
    else:
        return False, False, ""
    hit = prm.get(rank_used) == trm.get(rank_used)
    return True, bool(hit), rank_used


# ---------------------------------------------------------------------------- LOSO placement
def run_loso_placement(reps, positions, ann, te_emb):
    """LOSO placement. For each held species: align bridge on train, place held, retrieve nearest.
    Returns parallel arrays: placed_positions (N, d), radii (N,), pred_taxids (N,)."""
    species = ann["Species"].to_numpy()
    placed_all = np.empty_like(te_emb.positions[:1])  # placeholder shape
    placed_all = np.zeros((len(ann), te_emb.dim), dtype=np.float64)
    pred_taxids = np.empty(len(ann), dtype=np.int64)

    t0 = time.time()
    for tr, te, held in leave_species_out(species):
        br = Bridge.align(reps[tr], positions[tr], pca_dim=config.PCA_DIM, alpha=ALPHA)
        placed = br.place(reps[te])                    # (n_held, d) in the ball
        placed_all[te] = placed
        nn = retrieve_nearest(placed, te_emb.positions, k=1, chunk=RETRIEVE_CHUNK).ravel()
        pred_taxids[te] = te_emb.idx2taxid[nn]

    print(f"[LOSO] placement done in {time.time() - t0:.1f}s")
    radii = np.linalg.norm(placed_all, axis=1)
    return placed_all, radii, pred_taxids


# ---------------------------------------------------------------------------- reliability curve
def compute_reliability_curve(radii, true_taxids, pred_taxids, ranklut, n_bins=N_BINS):
    """Bin proteins by radius quintile; per bin compute coarse-rank accuracy and average true_depth."""
    bins = pd.qcut(radii, q=n_bins, labels=False, duplicates="drop")
    n_actual_bins = int(bins.max()) + 1

    rows = []
    for b in range(n_actual_bins):
        mask = bins == b
        if not mask.any():
            continue
        r_min = float(radii[mask].min())
        r_max = float(radii[mask].max())
        r_mean = float(radii[mask].mean())
        n_prot = int(mask.sum())
        hits, scored_count = [], 0
        for tt, pt in zip(true_taxids[mask], pred_taxids[mask]):
            scored, hit, rank_used = coarse_rank_hit(int(tt), int(pt), ranklut)
            if scored:
                hits.append(int(hit))
                scored_count += 1
        acc = float(np.mean(hits)) if hits else float("nan")
        rows.append({
            "bin": b,
            "radius_min": r_min,
            "radius_max": r_max,
            "radius_mean": r_mean,
            "n_proteins": n_prot,
            "n_scored": scored_count,
            "coarse_rank_acc": acc,
        })
    return rows


def check_monotone(rows):
    """Check if coarse-rank accuracy increases monotonically with radius bin.
    Returns (is_monotone_ish, direction). 'ish' = at most one dip/rise."""
    accs = [r["coarse_rank_acc"] for r in rows if not np.isnan(r["coarse_rank_acc"])]
    if len(accs) < 2:
        return False, "insufficient_data"
    diffs = np.diff(accs)
    n_pos = int((diffs > 0).sum())
    n_neg = int((diffs < 0).sum())
    if n_pos > n_neg:
        direction = "increasing"
        monotone_ish = n_neg <= 1          # at most one dip
    elif n_neg > n_pos:
        direction = "decreasing"
        monotone_ish = n_pos <= 1
    else:
        direction = "flat"
        monotone_ish = False
    return bool(monotone_ish), direction


# ---------------------------------------------------------------------------- main
def main():
    t_start = time.time()
    print("=== Task 17 (exploratory): Radius calibration — PLA2 ===")

    ann, reps = load_pla2()
    print(f"loaded {len(ann)} proteins, {ann['Species'].nunique()} species")

    te_emb = TaxonomyEmbedding(config.CKPT, config.TAXMAP, config.EDGELIST)
    resolver = TaxonResolver(config.TAXDUMP_DIR)
    ranklut = RankLookup(resolver, config.RANKS)
    positions = te_emb.positions[ann["idx"].to_numpy()]   # (N, 100) true node positions

    # Step 1: LOSO placement → radii
    placed_all, radii, pred_taxids = run_loso_placement(reps, positions, ann, te_emb)
    true_taxids = ann["taxid"].to_numpy()

    # Step 2: true_depth and Spearman(radius, true_depth)
    true_depth = te_emb.depth[ann["idx"].to_numpy()]
    rho, p_spearman = spearmanr(radii, true_depth)
    rho = float(rho)
    p_spearman = float(p_spearman)
    print(f"[Step 2] Spearman(radius, true_depth): rho={rho:.4f}, p={p_spearman:.4e}  "
          f"(n={len(radii)}; depth range {true_depth.min()}–{true_depth.max()})")

    # Step 3: reliability curve
    rel_rows = compute_reliability_curve(radii, true_taxids, pred_taxids, ranklut)
    monotone_ish, direction = check_monotone(rel_rows)
    print(f"[Step 3] Reliability curve ({N_BINS} bins by radius):")
    print(f"  {'bin':>3}  {'r_min':>6}  {'r_max':>6}  {'n':>4}  {'coarse_acc':>10}")
    for row in rel_rows:
        print(f"  {row['bin']:>3}  {row['radius_min']:>6.4f}  {row['radius_max']:>6.4f}  "
              f"{row['n_proteins']:>4}  {row['coarse_rank_acc']:>10.4f}")
    print(f"  monotone direction: {direction}, monotone-ish: {monotone_ish}")

    # Step 4: verdict
    rho_sig = bool(rho > 0 and p_spearman < 0.05)
    verdict = "calibrated" if (rho_sig and monotone_ish and direction == "increasing") else "cut"

    results = {
        "exploratory": True,
        "meta": {
            "n_proteins": int(len(ann)),
            "n_species": int(ann["Species"].nunique()),
            "alpha": float(ALPHA),
            "pca_dim": config.PCA_DIM,
            "seed": SEED,
            "n_bins": N_BINS,
            "coarse_rank": COARSE_RANK,
            "fallback_rank": FALLBACK_RANK,
            "coarse_rank_note": (
                "Class rank used for all species that have an NCBI 'class' node. "
                "8 species (4 Crocodiles + 4 Turtles) lack a class node (Sauropsida clade); "
                "these fall back to 'order' rank. Identical handling to read_eval.py."
            ),
            "depth_range": [int(true_depth.min()), int(true_depth.max())],
            "radius_range": [float(radii.min()), float(radii.max())],
        },
        "spearman": {
            "rho": rho,
            "p_value": p_spearman,
            "n": int(len(radii)),
            "significant_p05": bool(rho_sig),
            "interpretation": (
                f"rho={rho:.4f} (p={p_spearman:.4e}): "
                + ("positive and significant — deeper taxa produce larger placed norm"
                   if rho_sig else
                   "not significant or non-positive — no reliable depth→radius gradient")
            ),
        },
        "reliability_curve": {
            "bins": rel_rows,
            "monotone_ish": monotone_ish,
            "direction": direction,
        },
        "verdict": verdict,
        "verdict_rationale": (
            "calibrated: Spearman rho>0 significant (p<0.05) AND reliability monotone-ish increasing"
            if verdict == "calibrated" else
            "cut: condition not met — rho not positive/significant OR reliability not monotone-ish increasing. "
            "The radius-as-confidence claim is dropped; this is a clean null result."
        ),
        "runtime_sec": None,
    }
    results["runtime_sec"] = round(time.time() - t_start, 1)

    out_path = _HERE / "results" / "radius_pla2.json"
    out_path.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 64)
    print(f"VERDICT: {verdict.upper()}")
    print(f"  Spearman rho = {rho:.4f}  p = {p_spearman:.4e}  (sig@0.05: {rho_sig})")
    print(f"  Reliability monotone-ish: {monotone_ish} ({direction})")
    print(f"  {results['verdict_rationale']}")
    print(f"\nwrote {out_path}  (runtime {results['runtime_sec']}s)")
    return results


if __name__ == "__main__":
    main()
