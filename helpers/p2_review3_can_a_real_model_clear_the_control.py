#!/usr/bin/env python3
"""REVIEW WAVE 3 (read-only): is `below_control_all_seeds` REACHABLE by a real model?

The question. amendment_4 replaced RandomDAG with `degree_matched_shuffle` because the degree prior
scored BETTER on the RandomDAG control than on the real tree, which let a learned-nothing model
clear `below_control_all_seeds` (C1's exploit, a false GENERALISES). Measured at metazoa scale
(helpers/p2_review3_measure_production_splits.py) the new control inverts that sign -- but by a
LARGER factor than RandomDAG's, in the opposite direction:

    degree-prior normalized_rank   real 0.1538   RandomDAG 0.2487 (control 1.6x WORSE)
    degree-prior normalized_rank   real 0.1538   degmatch  0.0404 (control 3.8x BETTER)

So the GENERALISES bar `real_nr < control_nr` now demands the real arm beat a control whose task is
measurably much easier. This script asks whether that bar is reachable by a model that HAS learned:
it scores an EXISTING trained mollusca checkpoint through the PRODUCTION scorer's own code path
(`taxembed.eval.linkpred.rank_of_true_parent` + `linkpred_metrics`, cosine, tie_seed 0) on the real
P2 mollusca split, and compares its normalized_rank to the degree prior's on the same tree and on
that tree's degree-matched shuffle.

⚠ SCOPE. The checkpoint was trained on the FULL mollusca closure, not on a P2 split -- it has seen
every held-out parent edge. Its score is therefore an UPPER BOUND on what a P2 arm can reach, not an
estimate of one. That is exactly what makes it decisive in one direction: if even a model that saw
the answers cannot get below the control's training-free floor, no P2 arm will.

No epoch loop, no training, one checkpoint, single core (Rule 18: a closed-form audit over an npz).
Writes nothing except stdout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))

from taxembed.eval.baselines import chance_mrr_mean, degree_prior_metrics, sibling_chance  # noqa: E402
from taxembed.eval.linkpred import candidate_pool, linkpred_metrics, rank_of_true_parent  # noqa: E402
from taxembed.eval.p2_split import depth_from_closure  # noqa: E402
from taxembed.eval.randomdag import closure_from_parent, degree_matched_shuffle  # noqa: E402
from taxembed.eval.subtree import parent_from_closure  # noqa: E402
from taxembed.utils.training_pairs import TrainingPairs  # noqa: E402

SPLITS = _REPO / "data" / "p2_splits"
CKPT = _REPO / "results" / "task9_runs_mollusca" / "echino_canonical_s0_epoch200.pth"
MANIFEST = SPLITS / "p2_mollusca_6447_clean_vis00_seed0_manifest.json"
HELDOUT = SPLITS / "p2_mollusca_6447_clean_seed0_heldout.npz"


def tree_from(pairs: TrainingPairs):
    n = pairs.n_nodes
    parent = parent_from_closure(pairs.ancestor_idx, pairs.descendant_idx, pairs.depth_diff, n)
    depth = depth_from_closure(pairs.descendant_idx, pairs.descendant_depth,
                               pairs.ancestor_idx, pairs.ancestor_depth, n)
    return parent, depth, n


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    closure = Path(manifest["source_npz"])
    if not closure.exists():
        raise SystemExit(f"closure missing: {closure}")
    pairs = TrainingPairs.load(closure)
    parent, depth, n = tree_from(pairs)
    held = np.asarray(np.load(HELDOUT)["test"], dtype=np.int64)
    cands = [candidate_pool(parent, depth, node=int(v)) for v in held]
    n_cand = np.array([len(c) for c in cands], dtype=np.int64)
    true_parents = parent[held]

    import torch
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    emb = ck.get("embeddings")
    if emb is None:
        emb = ck["state_dict"]["lt.weight"]
    emb = emb.detach().float().numpy()
    if emb.shape[0] != n:
        raise SystemExit(f"checkpoint has {emb.shape[0]} rows, closure has {n} nodes")
    print(f"checkpoint {CKPT.name}: epoch {ck.get('epoch')}, {emb.shape} on {n:,}-node tree, "
          f"{len(held):,} held-out nodes")

    ranks = np.array([rank_of_true_parent(emb, int(v), int(tp), c, metric="cosine", tie_seed=0)
                      for v, tp, c in zip(held, true_parents, cands)], dtype=np.int64)
    learned = linkpred_metrics(ranks, n_cand)

    dp_real = degree_prior_metrics(parent, held, cands, true_parents, n_cand)

    # the degree-matched control's OWN degree prior, on the same clade/seed the control would use
    shuffled = degree_matched_shuffle(parent, depth, seed=0)
    dm_pairs = closure_from_parent(shuffled, depth)
    dm_parent, dm_depth, dm_n = tree_from(dm_pairs)
    dm_cands = [candidate_pool(dm_parent, dm_depth, node=int(v)) for v in held]
    dm_n_cand = np.array([len(c) for c in dm_cands], dtype=np.int64)
    dp_dm = degree_prior_metrics(dm_parent, held, dm_cands, dm_parent[held], dm_n_cand)

    print("\n--- mollusca_6447_clean seed 0, cosine, production code path ---")
    print(f"LEARNED (full-closure-trained checkpoint, an UPPER BOUND):")
    print(json.dumps(learned, indent=2))
    print(f"chance_mrr_mean as-coded(all k) {chance_mrr_mean(n_cand):.5f}  | "
          f"sibling_chance_mean as-coded {float(np.mean(sibling_chance(parent, held))):.5f}")
    print(f"\nDEGREE PRIOR, real tree     : mrr {dp_real['mrr']:.5f}  "
          f"normalized_rank {dp_real['normalized_rank']:.5f}")
    print(f"DEGREE PRIOR, degmatch tree : mrr {dp_dm['mrr']:.5f}  "
          f"normalized_rank {dp_dm['normalized_rank']:.5f}   "
          f"(k=1 share {float((dm_n_cand <= 1).mean()):.5f})")

    print("\n================ THE BAR ================")
    print(f"real arm's best possible normalized_rank (this leaky upper bound) : "
          f"{learned['normalized_rank']:.5f}")
    print(f"the control's TRAINING-FREE normalized_rank floor                 : "
          f"{dp_dm['normalized_rank']:.5f}")
    clears = learned["normalized_rank"] < dp_dm["normalized_rank"]
    print(f"could a real arm clear `below_control_all_seeds` if the control merely matched its own")
    print(f"degree prior?  {clears}")
    if not clears:
        print("  => 🛑 NOT REACHABLE. Even a model that trained on every held-out edge scores WORSE")
        print("     (higher) normalized_rank than the control's TRAINING-FREE prior. The control arm")
        print("     will be trained too, so its own score can only be <= that prior. GENERALISES is")
        print("     therefore structurally unreachable and the 12-run array reads MEMORISES for a")
        print("     reason that is a property of the CONTROL's construction, not of the geometry.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
