#!/usr/bin/env python3
"""§4.5 TimeTree — the scorer. Executes results/timetree_preregistration.json + amendment 1.

PRE-REGISTRATION frozen 2026-09-29 at SHA256
  a3e733c781914a80046a7e116f491951269422d46ff31bf88bd2929521eb7fba
before any divergence time, embedded distance or correlation was computed;
timetree_amendment_1_20260929 added after the LABELS were fetched but still before any
correlation. This file implements that document and prints every deviation as an amendment.

QUESTION. Does the shipped embedding's geometry track evolutionary RELATEDNESS (divergence time),
or NCBI's classification CONVENTION?

PRIMARY. Spearman rho( ANGULAR distance, divergence time ) per stratum. Angular (1 - cosine of
direction) reads no norm. 🛑 Poincare distance is NOT the primary: the radial prior is PLANTED
(norm = target_radius(depth), held by the radial regularizer) and depth correlates with divergence
time, so Poincare could deliver the answer from GIVEN structure and have it read as learned.

CO-REPORTED PRIMARY COMPARATOR. Spearman rho( NCBI path length, divergence time ) on the SAME
pairs. Finding 4 §4.7: name the cheapest statistic of the training data that could pass, and
instrument it so it cannot pass unnoticed. If the comparator wins, steelman (ii) is CONFIRMED.

GATES, each able to fail:
  b  init null   random directions at the TRAINED radii; |rho_angular| <= 0.05 required, else the
                 angular metric is not radius-free in practice and the primary is VOID.
  d  shuffle     divergence times permuted within stratum; rho ~ 0 required.
  a  power       >= 1000 pairs and >= 300 distinct taxa per stratum.
AMENDMENT 1 replaces gate (c)'s single all_total >= 1 threshold (which every pair cleared, so it
could not have failed) with a REPORTED SURFACE at >= 1 / >= 10 / >= 25, and adds an LCA-depth-binned
SECONDARY surface because uniform within-stratum pairs turned out to be overwhelmingly ancient
(median ~330 Mya) -- which biases the comparison toward NO_EVIDENCE, i.e. hides a difference rather
than inventing one.

UNCERTAINTY. 🛑 CLUSTER BOOTSTRAP OVER TAXA, never over pairs: 6,000 pairs arise from a few
thousand taxa and are not independent. Resample taxa with replacement, keep the pairs whose BOTH
endpoints survive, recompute rho. n_boot = 1000, seed 0.

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
    EMB, MAPPING, OLD_DATE, build_tree, lca, load_mapping, load_safetensors, poincare_dist,
)
from taxembed.eval.release_diff import parse_parents  # noqa: E402

PAIRS = ROOT / "results" / "timetree_pairs_20260929.json"
NODES = ROOT / "data" / f"taxdump_archive_{OLD_DATE}" / "nodes.dmp"
OUT = ROOT / "results" / "timetree_result_20260929.json"
N_BOOT, SEED = 1000, 0
AT_CUTS = [1, 10, 25]
PREREG_SHA = "a3e733c781914a80046a7e116f491951269422d46ff31bf88bd2929521eb7fba"


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho via Pearson on ranks, average ranks for ties."""
    if len(x) < 3:
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def _rank(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="stable")
    r = np.empty(len(a), dtype=np.float64)
    r[order] = np.arange(len(a), dtype=np.float64)
    # average ties
    a_sorted = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a_sorted[j + 1] == a_sorted[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def taxon_cluster_boot(a_idx, b_idx, dist, age, n_boot=N_BOOT, seed=SEED):
    """Resample TAXA with replacement; keep pairs whose both endpoints survive."""
    taxa = np.unique(np.concatenate([a_idx, b_idx]))
    pos = {int(t): i for i, t in enumerate(taxa)}
    ai = np.array([pos[int(t)] for t in a_idx])
    bi = np.array([pos[int(t)] for t in b_idx])
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        mult = np.bincount(rng.integers(0, len(taxa), size=len(taxa)), minlength=len(taxa))
        keep = (mult[ai] > 0) & (mult[bi] > 0)
        if keep.sum() >= 50:
            out.append(spearman(dist[keep], age[keep]))
    out = np.asarray([v for v in out if np.isfinite(v)])
    if len(out) == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def main() -> None:
    print("=" * 100)
    print("§4.5 TIMETREE — executing results/timetree_preregistration.json (+ amendment 1)")
    print(f"   frozen SHA256 {PREREG_SHA}")
    print("=" * 100)

    d = json.loads(PAIRS.read_text())
    rows = [r for r in d["pairs"] if r.get("age") is not None and "error" not in r]
    print(f"  pairs with an age: {len(rows):,}")

    emb = load_safetensors(EMB)
    taxid2row = load_mapping(MAPPING)
    parent_map = parse_parents(NODES)
    taxids, idx, parent, depth, tin, tout = build_tree(parent_map)

    # direction-only (radius-free) embedding
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    unit = emb / np.clip(norm, 1e-12, None)

    rng_null = np.random.default_rng(SEED + 7)
    dirs = rng_null.normal(size=emb.shape).astype(np.float32)
    dirs /= np.clip(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12, None)
    emb_null = (dirs * norm).astype(np.float32)               # planted radius, random direction
    unit_null = dirs

    result = {"preregistration": "results/timetree_preregistration.json",
              "preregistration_sha256": PREREG_SHA,
              "amendment": "timetree_amendment_1_20260929",
              "embedding": str(EMB), "clade": "cellular_organisms_131567_clean",
              "n_boot": N_BOOT, "seed": SEED, "strata": {}}

    for stratum in ("Vertebrata", "Insecta"):
        sr = [r for r in rows if r["stratum"] == stratum]
        a_t = np.array([r["a"] for r in sr], dtype=np.int64)
        b_t = np.array([r["b"] for r in sr], dtype=np.int64)
        age = np.array([r["age"] for r in sr], dtype=np.float64)
        at = np.array([r.get("all_total", 0) for r in sr], dtype=np.int64)

        ok = np.array([(int(x) in idx and int(y) in idx and int(x) in taxid2row
                        and int(y) in taxid2row) for x, y in zip(a_t, b_t)])
        a_t, b_t, age, at = a_t[ok], b_t[ok], age[ok], at[ok]
        ai = np.array([idx[int(t)] for t in a_t])
        bi = np.array([idx[int(t)] for t in b_t])
        ra = np.array([taxid2row[int(t)] for t in a_t])
        rb = np.array([taxid2row[int(t)] for t in b_t])

        # distances
        ang = 1.0 - np.einsum("ij,ij->i", unit[ra], unit[rb])
        ang_null = 1.0 - np.einsum("ij,ij->i", unit_null[ra], unit_null[rb])
        poin = np.array([poincare_dist(emb[x], emb[y][None, :])[0] for x, y in zip(ra, rb)])
        poin_null = np.array([poincare_dist(emb_null[x], emb_null[y][None, :])[0]
                              for x, y in zip(ra, rb)])
        lcad = np.array([depth[lca(int(x), int(y), parent, depth)] for x, y in zip(ai, bi)])
        pathlen = depth[ai] + depth[bi] - 2 * lcad

        n_taxa = len(np.unique(np.concatenate([a_t, b_t])))
        g_a = len(age) >= 1000 and n_taxa >= 300
        print(f"\n{'='*100}\n  {stratum}: {len(age):,} pairs, {n_taxa:,} distinct taxa   "
              f"GATE a {'PASS' if g_a else 'FAIL'}")

        rho_ang = spearman(ang, age)
        ci_ang = taxon_cluster_boot(a_t, b_t, ang, age)
        rho_path = spearman(pathlen.astype(float), age)
        ci_path = taxon_cluster_boot(a_t, b_t, pathlen.astype(float), age)
        rho_poin = spearman(poin, age)
        rho_ang_null = spearman(ang_null, age)
        rho_poin_null = spearman(poin_null, age)

        rng_s = np.random.default_rng(SEED + 3)
        rho_shuf = spearman(ang, rng_s.permutation(age))

        g_b = abs(rho_ang_null) <= 0.05
        g_d = abs(rho_shuf) <= 0.05
        print(f"    PRIMARY   angular      rho={rho_ang:+.4f}  CI95[{ci_ang[0]:+.4f},{ci_ang[1]:+.4f}]")
        print(f"    COMPARATOR NCBI path   rho={rho_path:+.4f}  CI95[{ci_path[0]:+.4f},{ci_path[1]:+.4f}]")
        print(f"    secondary Poincare     rho={rho_poin:+.4f}   (DESCRIPTIVE ONLY)")
        print(f"    GATE b  init null angular   {rho_ang_null:+.4f}  "
              f"{'PASS' if g_b else 'FAIL — primary VOID'}   "
              f"(Poincare init null {rho_poin_null:+.4f}, expected NONZERO)")
        print(f"    GATE d  shuffle control     {rho_shuf:+.4f}  {'PASS' if g_d else 'FAIL'}")

        # amendment 1: all_total surface
        surface = {}
        print("    all_total surface (amendment 1):")
        for cut in AT_CUTS:
            m = at >= cut
            if m.sum() < 300:
                surface[str(cut)] = {"n": int(m.sum()), "underpowered": True}
                print(f"      >= {cut:>2}: n={int(m.sum()):>5,}  UNDERPOWERED")
                continue
            ra_ = spearman(ang[m], age[m])
            rp_ = spearman(pathlen[m].astype(float), age[m])
            surface[str(cut)] = {"n": int(m.sum()), "rho_angular": ra_, "rho_path": rp_}
            print(f"      >= {cut:>2}: n={int(m.sum()):>5,}  angular {ra_:+.4f}   path {rp_:+.4f}")

        # amendment 2: LCA-depth secondary, TERCILE bins (amendment 1's fixed bins were vacuous —
        # 2,945/2,993 and 2,991/2,991 pairs landed in one bin. A quantile bin cannot be empty.)
        bins = {}
        q = np.percentile(lcad, [0, 33.333, 66.667, 100]).astype(int)
        edges = sorted(set(int(v) for v in q))
        print(f"    LCA depth distribution: min {lcad.min()} q33 {q[1]} q67 {q[2]} "
              f"max {lcad.max()}  -> tercile edges {edges}")
        if len(edges) < 3:
            print("      ⚠ SECONDARY HAS NO RESOLUTION ON THIS TREE (LCA depth too concentrated "
                  "for terciles to separate) — reported, not papered over")
            bins["no_resolution"] = {"lca_depth_min": int(lcad.min()),
                                     "lca_depth_max": int(lcad.max()),
                                     "distinct_values": int(len(np.unique(lcad)))}
        print("    LCA-depth secondary surface (amendment 2, cannot overturn the primary):")
        pairs_lohi = [(edges[i], edges[i + 1] - (1 if i + 2 < len(edges) else 0))
                      for i in range(len(edges) - 1)] if len(edges) >= 3 else []
        for lo, hi in pairs_lohi:
            m = (lcad >= lo) & (lcad <= hi)
            if m.sum() < 300:
                bins[f"{lo}-{hi}"] = {"n": int(m.sum()), "underpowered": True}
                print(f"      LCA depth {lo}-{hi}: n={int(m.sum()):>5,}  UNDERPOWERED")
                continue
            ra_ = spearman(ang[m], age[m])
            rp_ = spearman(pathlen[m].astype(float), age[m])
            bins[f"{lo}-{hi}"] = {"n": int(m.sum()), "rho_angular": ra_, "rho_path": rp_,
                                  "age_median": float(np.median(age[m]))}
            print(f"      LCA depth {lo}-{hi}: n={int(m.sum()):>5,}  angular {ra_:+.4f}   "
                  f"path {rp_:+.4f}   age median {np.median(age[m]):.0f} Mya")

        gates_ok = g_a and g_b and g_d
        if not gates_ok:
            verdict = "UNINFORMATIVE"
        elif ci_ang[0] > ci_path[1] or (rho_ang > rho_path and ci_ang[0] > ci_path[1]):
            verdict = "TRACKS_RELATEDNESS"
        elif ci_path[0] > ci_ang[1]:
            verdict = "TRACKS_CONVENTION"
        else:
            verdict = "NO_EVIDENCE"
        print(f"    VERDICT ({stratum}): {verdict}")

        result["strata"][stratum] = {
            "n_pairs": int(len(age)), "n_taxa": int(n_taxa),
            "primary_angular": {"rho": rho_ang, "ci95": list(ci_ang)},
            "comparator_ncbi_path": {"rho": rho_path, "ci95": list(ci_path)},
            "secondary_poincare_descriptive": rho_poin,
            "gate_b_init_null_angular": {"rho": rho_ang_null, "pass": bool(g_b)},
            "init_null_poincare_expected_nonzero": rho_poin_null,
            "gate_d_shuffle": {"rho": rho_shuf, "pass": bool(g_d)},
            "gate_a_power": bool(g_a),
            "all_total_surface": surface, "lca_depth_secondary": bins,
            "verdict": verdict,
        }

    json.dump(result, OUT.open("w"), indent=2, default=float)
    print(f"\n{'='*100}\nwritten: {OUT}")


if __name__ == "__main__":
    main()
