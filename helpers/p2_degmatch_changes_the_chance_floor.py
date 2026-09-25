"""Does the degree-matched shuffle actually close the chance-floor gap RandomDAG opened?

WHY (2026-09-24, p2_amendment_4_20260924). helpers/p2_randomdag_changes_the_chance_floor.py
measured that `randomdag.randomize_parents` (i.i.d. uniform resample of each node's parent)
collapses the real taxonomy's heavy-tailed fan-out toward the mean -- mean fan-out 4.61 -> 1.96 on
mollusca_6447_clean seed 0, raising P2's chance floor 5.4x. `degree_matched_shuffle` (same module)
instead permutes the REAL parent-label multiset per depth level, which should preserve fan-out --
and therefore the chance floor and the training-free degree prior -- EXACTLY. This script measures
that claim directly, on the same clade and seed the RandomDAG measurement used, with the SAME
production functions (`taxembed.eval.baselines.sibling_chance`, `.degree_prior_metrics`,
`.chance_mrr_mean`) so the numbers are exactly what `scripts/score_p2_linkpred.py` would write.

Read-only. Usage: <venv-python> helpers/p2_degmatch_changes_the_chance_floor.py [clade]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/src")

from taxembed.eval.baselines import chance_mrr_mean, degree_prior_metrics, sibling_chance  # noqa: E402
from taxembed.eval.linkpred import candidate_pool                                          # noqa: E402
from taxembed.eval.p2_split import depth_from_closure, eligible_nodes, select_holdout       # noqa: E402
from taxembed.eval.randomdag import degree_matched_shuffle                                  # noqa: E402
from taxembed.eval.subtree import parent_from_closure                                       # noqa: E402
from taxembed.utils.training_pairs import TrainingPairs                                     # noqa: E402

DATA = Path("/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/data/taxopy")


def fanout_of(parent: np.ndarray) -> np.ndarray:
    real = np.arange(len(parent)) != parent
    return np.bincount(parent[real], minlength=len(parent))


def describe(name: str, v: np.ndarray) -> None:
    ps = np.percentile(v, [50, 90, 99, 100])
    print(f"  {name:<34} mean {v.mean():8.5f} | p50 {ps[0]:8.4f} | p90 {ps[1]:8.4f} | "
          f"p99 {ps[2]:8.4f} | max {ps[3]:9.2f}")


def main() -> int:
    clade = sys.argv[1] if len(sys.argv) > 1 else "mollusca_6447_clean"
    seed = 0
    pairs = TrainingPairs.load(DATA / clade / f"taxonomy_edges_{clade}_transitive.npz")
    n = pairs.n_nodes
    parent = parent_from_closure(pairs.ancestor_idx, pairs.descendant_idx, pairs.depth_diff, n)
    depth = depth_from_closure(pairs.descendant_idx, pairs.descendant_depth,
                               pairs.ancestor_idx, pairs.ancestor_depth, n)

    degmatch = degree_matched_shuffle(parent, depth, seed=seed)

    print(f"clade = {clade}   n_nodes = {n:,}\n")

    # ---- 1. fan-out multiset: mean, max, and exact equality of the sorted multisets ----
    fo_real = fanout_of(parent)
    fo_degmatch = fanout_of(degmatch)
    internal_real = fo_real[fo_real > 0]
    internal_degmatch = fo_degmatch[fo_degmatch > 0]
    multiset_identical = bool(np.array_equal(np.sort(fo_real), np.sort(fo_degmatch)))
    print("FAN-OUT over nodes that have at least one child:")
    describe("real fanout", internal_real.astype(float))
    describe("degree-matched fanout", internal_degmatch.astype(float))
    print(f"  mean fanout:  real {internal_real.mean():.5f}  vs  degree-matched "
          f"{internal_degmatch.mean():.5f}")
    print(f"  max fanout:   real {int(internal_real.max())}  vs  degree-matched "
          f"{int(internal_degmatch.max())}")
    print(f"  FULL SORTED FAN-OUT MULTISET IDENTICAL (every node, incl. zero-child leaves): "
          f"{multiset_identical}")
    if not multiset_identical:
        print("  🛑 THE CONTROL DOES NOT PRESERVE FAN-OUT -- the construction has a bug.")

    # ---- 2. sibling_chance (the P2 chance floor) on both trees' own band-eligible leaves ----
    elig_real = eligible_nodes(parent, depth, leaves_only=True)
    elig_degmatch = eligible_nodes(degmatch, depth, leaves_only=True)
    sc_real = sibling_chance(parent, elig_real)
    sc_degmatch = sibling_chance(degmatch, elig_degmatch)
    print(f"\nBAND-ELIGIBLE LEAVES (depth in [11,28]): real {len(elig_real):,} vs "
          f"degree-matched {len(elig_degmatch):,}")
    print("\nSIBLING CHANCE (1 / grandparent fan-out) -- P2's chance floor:")
    describe("real tree", sc_real)
    describe("degree-matched tree", sc_degmatch)
    ratio = sc_degmatch.mean() / max(sc_real.mean(), 1e-12)
    print(f"\n  mean chance floor ratio degree-matched/real = {ratio:.4f}  "
          f"(RandomDAG's own ratio, for comparison, measured 5.403 on this clade/seed)")

    # ---- 3. training-free degree prior, on the SAME held-out split each tree would score ----
    split_real = select_holdout(elig_real, frac_test=0.10, frac_val=0.0, seed=seed)
    split_degmatch = select_holdout(elig_degmatch, frac_test=0.10, frac_val=0.0, seed=seed)
    held_real, held_degmatch = split_real["test"], split_degmatch["test"]

    def degree_prior_on(p, held):
        cands = [candidate_pool(p, depth, node=int(v), strategy="grandparent_children") for v in held]
        n_cand = np.array([len(c) for c in cands], dtype=np.int64)
        true_parents = p[held]
        return degree_prior_metrics(p, held, cands, true_parents, n_cand), n_cand

    dp_real, nc_real = degree_prior_on(parent, held_real)
    dp_degmatch, nc_degmatch = degree_prior_on(degmatch, held_degmatch)
    print(f"\nDEGREE PRIOR (training-free, ranks candidates by descending fan-out), on this "
          f"seed's OWN held-out test set:")
    print(f"  real tree:           MRR {dp_real['mrr']:.4f}  hits@1 {dp_real['hits_at_1']:.4f}  "
          f"normalized_rank {dp_real['normalized_rank']:.4f}  (n_scored {dp_real['n_scored']}, "
          f"n_trivial {dp_real['n_trivial']})")
    print(f"  degree-matched tree: MRR {dp_degmatch['mrr']:.4f}  "
          f"hits@1 {dp_degmatch['hits_at_1']:.4f}  "
          f"normalized_rank {dp_degmatch['normalized_rank']:.4f}  "
          f"(n_scored {dp_degmatch['n_scored']}, n_trivial {dp_degmatch['n_trivial']})")

    chance_real = chance_mrr_mean(nc_real)
    chance_degmatch = chance_mrr_mean(nc_degmatch)
    print(f"\nANALYTIC CHANCE MRR (mean H_k/k): real {chance_real:.4f}  vs  "
          f"degree-matched {chance_degmatch:.4f}")

    print("\n=== SUMMARY (p2_amendment_4_20260924) ===")
    print(f"  fan-out multiset identical:           {multiset_identical}")
    print(f"  mean fan-out   real / degree-matched: {internal_real.mean():.4f} / "
          f"{internal_degmatch.mean():.4f}")
    print(f"  max fan-out    real / degree-matched: {int(internal_real.max())} / "
          f"{int(internal_degmatch.max())}")
    print(f"  sibling_chance mean real / degree-matched: {sc_real.mean():.5f} / "
          f"{sc_degmatch.mean():.5f}  (ratio {ratio:.4f})")
    print(f"  degree_prior normalized_rank real / degree-matched: "
          f"{dp_real['normalized_rank']:.4f} / {dp_degmatch['normalized_rank']:.4f}")
    if multiset_identical and abs(ratio - 1.0) < 0.05:
        print("  ✅ the chance floor and degree prior MATCH between the two trees -- the control "
              "closes the RandomDAG confound by construction.")
    else:
        print("  🛑 the chance floor still differs materially -- investigate before using this "
              "as P2's control.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
