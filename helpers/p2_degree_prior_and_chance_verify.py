"""Re-derive, read-only, the C1/C2/C3 review-finding numbers on the REAL metazoa P2 split (seed 0)
and its RandomDAG counterpart, using the ACTUAL production functions this fix ships
(`taxembed.eval.baselines.degree_prior_metrics`, `.chance_mrr_mean`, and
`taxembed.eval.linkpred.linkpred_metrics`'s C3 pool-size-1 exclusion) -- not a hand-rolled
reimplementation, so these numbers are exactly what `scripts/score_p2_linkpred.py` will write to
`baselines.degree_prior` / `baselines.chance_mrr_mean` on a real run.

WHY THIS SUPERSEDES AN EARLIER VERSION OF THIS FILE. A first pass at this script duplicated the
ranking logic by hand and measured degree prior MRR 0.5750 / hits@1 0.4243 / normalized_rank
0.1485 on the real tree BEFORE `linkpred.linkpred_metrics` excluded pool-size-1 queries (C3). Once
C3 landed, re-running against the FIXED `linkpred_metrics` (which the production
`degree_prior_metrics` calls) changed the degree prior's OWN numbers too -- k=1 is a free win for
ANY ranker, including the degree prior, so excluding it moves the degree prior's own MRR down, not
just a learned model's. The numbers below are POST-C3 (k=1 excluded from the degree prior's own
score, exactly as production will compute it) and are the ones written into
`p2_amendment_3_20260924`, not the earlier pre-C3 measurement.

Usage: <venv-python> helpers/p2_degree_prior_and_chance_verify.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from taxembed.eval.baselines import chance_mrr_mean, degree_prior_metrics  # noqa: E402
from taxembed.eval.linkpred import candidate_pool  # noqa: E402
from taxembed.eval.p2_split import (  # noqa: E402
    depth_from_closure, eligible_nodes, select_holdout,
)
from taxembed.eval.randomdag import randomize_parents  # noqa: E402
from taxembed.eval.subtree import parent_from_closure  # noqa: E402
from taxembed.utils.training_pairs import TrainingPairs  # noqa: E402

MANIFEST = _REPO / "data" / "p2_splits" / "p2_metazoa_33208_clean_vis00_seed0_manifest.json"


def describe_tree(label: str, parent: np.ndarray, depth: np.ndarray, held: np.ndarray,
                  true_parents: np.ndarray) -> dict:
    t0 = time.time()
    candidates_list = [candidate_pool(parent, depth, node=int(v), strategy="grandparent_children")
                       for v in held]
    n_cand = np.array([len(c) for c in candidates_list], dtype=np.int64)

    dp_metrics = degree_prior_metrics(parent, held, candidates_list, true_parents, n_cand)
    chance_mrr = chance_mrr_mean(n_cand)
    chance_hits1 = float(np.mean(1.0 / n_cand.astype(np.float64)))
    k1_share = float(np.mean(n_cand == 1))

    print(f"\n=== {label} ===  ({time.time() - t0:.1f}s)")
    print(f"  n_held = {len(held):,}   pool sizes {n_cand.min()}-{n_cand.max()}")
    print(f"  degree prior (POST-C3, k=1 excluded): MRR {dp_metrics['mrr']:.4f}  "
          f"hits@1 {dp_metrics['hits_at_1']:.4f}  normalized_rank {dp_metrics['normalized_rank']:.4f}"
          f"  (n_scored {dp_metrics['n_scored']}, n_trivial {dp_metrics['n_trivial']})")
    print(f"  analytic chance: MRR {chance_mrr:.4f}  hits@1 {chance_hits1:.4f}")
    print(f"  k=1 pool-size share: {k1_share * 100:.2f}%")
    return {"degree_prior": dp_metrics, "chance_mrr_mean": chance_mrr,
           "chance_hits_at_1_mean": chance_hits1, "k1_share": k1_share,
           "n_cand_min": int(n_cand.min()), "n_cand_max": int(n_cand.max())}


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    source_npz = Path(manifest["source_npz"])
    print(f"source_npz: {source_npz}  (exists: {source_npz.exists()})")
    pairs = TrainingPairs.load(source_npz)
    n = pairs.n_nodes
    parent = parent_from_closure(pairs.ancestor_idx, pairs.descendant_idx, pairs.depth_diff, n)
    depth = depth_from_closure(pairs.descendant_idx, pairs.descendant_depth,
                               pairs.ancestor_idx, pairs.ancestor_depth, n)

    heldout_path = _REPO / "data" / "p2_splits" / manifest["heldout_npz"]
    heldout = np.load(heldout_path)
    held_real = np.asarray(heldout["test"], dtype=np.int64)
    true_parents_real = parent[held_real]

    real = describe_tree("REAL metazoa_33208_clean seed0 (band [11,28], test split)",
                         parent, depth, held_real, true_parents_real)

    # ---- RandomDAG control: same seed, same construction as build_p2_randomdag_split.py ----
    rand_parent = randomize_parents(parent, depth, seed=0)
    elig_rand = eligible_nodes(rand_parent, depth, leaves_only=True)
    split_rand = select_holdout(elig_rand, frac_test=0.10, frac_val=0.0, seed=0)
    held_rand = split_rand["test"]
    true_parents_rand = rand_parent[held_rand]

    rand = describe_tree("RANDOMISED (RandomDAG) metazoa seed0", rand_parent, depth, held_rand,
                         true_parents_rand)

    print("\n=== Final numbers for p2_amendment_3_20260924 (production functions, POST-C3) ===")
    print(f"  real tree degree prior:      MRR {real['degree_prior']['mrr']:.4f}  "
          f"hits@1 {real['degree_prior']['hits_at_1']:.4f}  "
          f"normalized_rank {real['degree_prior']['normalized_rank']:.4f}")
    print(f"  real tree chance:            MRR {real['chance_mrr_mean']:.4f}  "
          f"hits@1 {real['chance_hits_at_1_mean']:.4f}")
    print(f"  randomised degree prior nr:  {rand['degree_prior']['normalized_rank']:.4f}")
    print(f"  real tree k=1 share:         {real['k1_share']*100:.2f}%")
    print(f"  randomised tree k=1 share:   {rand['k1_share']*100:.2f}%")


if __name__ == "__main__":
    main()
