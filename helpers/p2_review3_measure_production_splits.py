#!/usr/bin/env python3
"""REVIEW WAVE 3 (read-only): measure the 12 PRODUCTION metazoa splits, not mollusca.

Every C1/amendment_4 number in the ledger and in `results/p2_heldout_preregistration.json`'s
`p2_amendment_4_20260924` block was measured on **mollusca_6447_clean** (32,017 nodes, 740 held-out
nodes per seed). The array about to be submitted trains on **metazoa_33208_clean** (498,246 nodes,
28,418 held-out nodes per seed). A finding travels; its scope does not. This script re-measures the
load-bearing quantities on the splits that will actually be scored.

Writes nothing except stdout. Touches no production file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))

from taxembed.eval.baselines import (  # noqa: E402
    chance_mrr_mean, degree_prior_metrics, sibling_chance,
)
from taxembed.eval.linkpred import candidate_pool  # noqa: E402
from taxembed.eval.p2_split import depth_from_closure  # noqa: E402
from taxembed.eval.subtree import parent_from_closure  # noqa: E402
from taxembed.utils.training_pairs import TrainingPairs  # noqa: E402

SPLITS = _REPO / "data" / "p2_splits"
REAL_CLOSURE = (_REPO / "data" / "taxopy" / "metazoa_33208_clean"
                / "taxonomy_edges_metazoa_33208_clean_transitive.npz")


def harmonic_mean_over(n_cand: np.ndarray) -> float:
    """mean(H_k/k) over the given pool sizes -- the same formula chance_mrr_mean uses."""
    n_cand = np.asarray(n_cand, dtype=np.int64)
    if len(n_cand) == 0:
        return float("nan")
    max_k = int(n_cand.max())
    h = np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, max_k + 1, dtype=np.float64))])
    return float(np.mean(h[n_cand] / n_cand.astype(np.float64)))


def load_tree(npz: Path):
    pairs = TrainingPairs.load(npz)
    n = pairs.n_nodes
    parent = parent_from_closure(pairs.ancestor_idx, pairs.descendant_idx, pairs.depth_diff, n)
    depth = depth_from_closure(pairs.descendant_idx, pairs.descendant_depth,
                               pairs.ancestor_idx, pairs.ancestor_depth, n)
    return pairs, parent, depth, n


def measure(label: str, closure: Path, heldout: Path) -> dict:
    pairs, parent, depth, n = load_tree(closure)
    held = np.asarray(np.load(heldout)["test"], dtype=np.int64)
    val = np.asarray(np.load(heldout)["val"], dtype=np.int64)
    cands = [candidate_pool(parent, depth, node=int(v)) for v in held]
    n_cand = np.array([len(c) for c in cands], dtype=np.int64)
    true_parents = parent[held]

    sc = sibling_chance(parent, held)
    # independent cross-check: sibling_chance's 1/fanout[grandparent] must equal 1/|pool|
    pool_mismatch = int(np.sum(np.abs(sc - 1.0 / np.maximum(n_cand, 1)) > 1e-12))

    scored = n_cand >= 2
    out = {
        "label": label,
        "n_nodes": int(n), "n_pairs": int(len(pairs)),
        "n_held": int(len(held)), "n_val": int(len(val)),
        "n_trivial_k1": int((~scored).sum()),
        "k1_share": float((~scored).mean()),
        "pool_min": int(n_cand.min()), "pool_max": int(n_cand.max()),
        "sibling_chance_mean_AS_CODED_all_k": float(np.mean(sc)),
        "sibling_chance_mean_SCORED_k_ge_2": float(np.mean(sc[scored])),
        "chance_mrr_mean_AS_CODED_all_k": float(chance_mrr_mean(n_cand)),
        "chance_mrr_mean_SCORED_k_ge_2": harmonic_mean_over(n_cand[scored]),
        "pool_vs_sibling_chance_mismatches": pool_mismatch,
        "held_sorted_key": int(held.sum()),
    }
    dp = degree_prior_metrics(parent, held, cands, true_parents, n_cand)
    out["degree_prior"] = dp
    return out, held, parent, depth, pairs


def heldout_ancestry_check(train_npz: Path, held: np.ndarray, depth: np.ndarray) -> dict:
    """Task-2 plan-defect invariant at METAZOA scale: every held-out node must appear in exactly
    depth-1 training rows (its full retained ancestry, exempt from visibility thinning)."""
    tp = TrainingPairs.load(train_npz)
    counts = np.bincount(np.asarray(tp.descendant_idx, dtype=np.int64), minlength=len(depth))
    want = depth[held] - 1
    got = counts[held]
    return {"train_npz": train_npz.name,
            "n_held_with_zero_rows": int(np.sum(got == 0)),
            "n_mismatch_vs_depth_minus_1": int(np.sum(got != want)),
            "min_rows": int(got.min()), "max_rows": int(got.max())}


def main() -> int:
    rows = []
    held_by_key = {}
    for seed in (0, 1, 2):
        real_held = SPLITS / f"p2_metazoa_33208_clean_seed{seed}_heldout.npz"
        r, held_r, parent_r, depth_r, _ = measure(f"real_s{seed}", REAL_CLOSURE, real_held)
        rows.append(r)
        held_by_key[f"real_s{seed}"] = held_r

        dm_closure = SPLITS / f"taxonomy_edges_metazoa_33208_clean_degmatch_seed{seed}_transitive.npz"
        dm_held = SPLITS / f"p2_metazoa_33208_clean_degmatch_seed{seed}_heldout.npz"
        d, held_d, parent_d, depth_d, _ = measure(f"degmatch_s{seed}", dm_closure, dm_held)
        rows.append(d)
        held_by_key[f"degmatch_s{seed}"] = held_d

        print(f"\n--- seed {seed}: held-out node sets identical real vs degmatch? "
              f"{np.array_equal(held_r, held_d)}")
        print(f"    depth arrays identical? {np.array_equal(depth_r, depth_d)}")
        print(f"    parent arrays identical? {np.array_equal(parent_r, parent_d)}  "
              f"(must be False -- the control must differ)")
        fan_r = np.bincount(parent_r[np.arange(len(parent_r)) != parent_r], minlength=len(parent_r))
        fan_d = np.bincount(parent_d[np.arange(len(parent_d)) != parent_d], minlength=len(parent_d))
        print(f"    per-node fan-out identical elementwise? {np.array_equal(fan_r, fan_d)}  "
              f"(amendment_4's core claim, at METAZOA scale)")

        for vis in ("vis00", "vis50"):
            print("    " + json.dumps(heldout_ancestry_check(
                SPLITS / f"p2_metazoa_33208_clean_{vis}_seed{seed}_train.npz", held_r, depth_r)))
            print("    " + json.dumps(heldout_ancestry_check(
                SPLITS / f"p2_metazoa_33208_clean_degmatch_{vis}_seed{seed}_train.npz",
                held_d, depth_d)))

    print("\n================ PER-SPLIT MEASUREMENTS ================")
    for r in rows:
        print(json.dumps(r, indent=2))

    print("\n================ THE LOAD-BEARING COMPARISON ================")
    for seed in (0, 1, 2):
        rr = next(x for x in rows if x["label"] == f"real_s{seed}")
        dd = next(x for x in rows if x["label"] == f"degmatch_s{seed}")
        print(f"seed {seed}:")
        print(f"  degree_prior normalized_rank  real {rr['degree_prior']['normalized_rank']:.5f}  "
              f"degmatch {dd['degree_prior']['normalized_rank']:.5f}  "
              f"-> real BELOW control? {rr['degree_prior']['normalized_rank'] < dd['degree_prior']['normalized_rank']}"
              f"   (True == C1's exploit is OPEN: a fan-out-only model clears below_control_all_seeds)")
        print(f"  degree_prior mrr              real {rr['degree_prior']['mrr']:.5f}  "
              f"degmatch {dd['degree_prior']['mrr']:.5f}")
        print(f"  sibling_chance_mean ratio degmatch/real "
              f"{dd['sibling_chance_mean_AS_CODED_all_k'] / rr['sibling_chance_mean_AS_CODED_all_k']:.4f}")
        print(f"  k=1 share                     real {rr['k1_share']:.5f}  "
              f"degmatch {dd['k1_share']:.5f}")
        print(f"  chance_mrr_mean  as-coded(all k) real {rr['chance_mrr_mean_AS_CODED_all_k']:.5f} "
              f"vs scored(k>=2) {rr['chance_mrr_mean_SCORED_k_ge_2']:.5f}  "
              f"| degmatch as-coded {dd['chance_mrr_mean_AS_CODED_all_k']:.5f} "
              f"vs scored {dd['chance_mrr_mean_SCORED_k_ge_2']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
