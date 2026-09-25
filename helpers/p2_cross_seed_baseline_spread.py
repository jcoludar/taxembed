"""How much do sibling_chance_mean / chance_mrr_mean / degree_prior actually vary across seeds?

WHY (2026-09-24, Part 2 of the p2_amendment_4_20260924 task -- results/
p2_heldout_preregistration.json). `scripts/p2_lrz_score.sh` used to write every seed's own
baselines block under the SAME top-level key ("baselines" for the real-tree pair, "baselines_
<control>" for a control) -- each seed draws its OWN held-out set (Task 2's build_p2_split.py
writes one manifest/heldout PER SEED), so these are themselves per-seed quantities, and
`merge_p2_scorer_outputs` silently kept whichever file happened to merge first ("seed 0 wins").
The fix (taxembed.eval.preregistration._p2_baselines_for_seed,
assert_baselines_agree_across_seeds) keeps each seed's own value and gates the ACTUAL run against
it, only asserting cross-seed agreement rather than assuming it. This script measures how large
that cross-seed spread actually is, on the real mollusca_6447_clean closure, at visibility 0.0,
seeds 0/1/2 -- read-only, no training, no embeddings needed (these baselines depend only on tree
structure + the held-out node set).

Read-only. Usage: <venv-python> helpers/p2_cross_seed_baseline_spread.py [clade]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/src")

from taxembed.eval.baselines import chance_mrr_mean, degree_prior_metrics, sibling_chance  # noqa: E402
from taxembed.eval.linkpred import candidate_pool                                          # noqa: E402
from taxembed.eval.p2_split import depth_from_closure, eligible_nodes, select_holdout       # noqa: E402
from taxembed.eval.subtree import parent_from_closure                                       # noqa: E402
from taxembed.utils.training_pairs import TrainingPairs                                     # noqa: E402

DATA = Path("/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/data/taxopy")
SEEDS = (0, 1, 2)


def measure_seed(parent, depth, seed: int) -> dict:
    elig = eligible_nodes(parent, depth, leaves_only=True)
    split = select_holdout(elig, frac_test=0.10, frac_val=0.0, seed=seed)
    held = split["test"]
    sc = sibling_chance(parent, held)
    cands = [candidate_pool(parent, depth, node=int(v), strategy="grandparent_children") for v in held]
    n_cand = np.array([len(c) for c in cands], dtype=np.int64)
    true_parents = parent[held]
    dp = degree_prior_metrics(parent, held, cands, true_parents, n_cand)
    cmm = chance_mrr_mean(n_cand)
    return {
        "seed": seed, "n_held": len(held),
        "sibling_chance_mean": float(sc.mean()),
        "chance_mrr_mean": float(cmm),
        "degree_prior_mrr": float(dp["mrr"]),
        "degree_prior_normalized_rank": float(dp["normalized_rank"]),
    }


def spread_report(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    spread_abs = float(arr.max() - arr.min())
    spread_rel = spread_abs / mean if mean != 0 else float("nan")
    return {"values": values, "mean": mean, "max_minus_min": spread_abs,
           "relative_spread_pct": spread_rel * 100.0}


def main() -> int:
    clade = sys.argv[1] if len(sys.argv) > 1 else "mollusca_6447_clean"
    pairs = TrainingPairs.load(DATA / clade / f"taxonomy_edges_{clade}_transitive.npz")
    n = pairs.n_nodes
    parent = parent_from_closure(pairs.ancestor_idx, pairs.descendant_idx, pairs.depth_diff, n)
    depth = depth_from_closure(pairs.descendant_idx, pairs.descendant_depth,
                               pairs.ancestor_idx, pairs.ancestor_depth, n)

    print(f"clade = {clade}   n_nodes = {n:,}\n")
    rows = [measure_seed(parent, depth, s) for s in SEEDS]
    for r in rows:
        print(f"  seed {r['seed']}: n_held {r['n_held']:,}  sibling_chance_mean "
              f"{r['sibling_chance_mean']:.5f}  chance_mrr_mean {r['chance_mrr_mean']:.5f}  "
              f"degree_prior.mrr {r['degree_prior_mrr']:.5f}  degree_prior.normalized_rank "
              f"{r['degree_prior_normalized_rank']:.5f}")

    print("\n=== CROSS-SEED SPREAD (3 seeds, real mollusca_6447_clean, vis-independent) ===")
    for field in ("sibling_chance_mean", "chance_mrr_mean", "degree_prior_mrr",
                 "degree_prior_normalized_rank"):
        rep = spread_report([r[field] for r in rows])
        print(f"  {field:<28} values {['%.5f' % v for v in rep['values']]}  "
              f"max-min {rep['max_minus_min']:.5f}  relative spread "
              f"{rep['relative_spread_pct']:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
