"""Does RandomDAG rewiring change the task's own difficulty, not just its structure?

WHY (2026-09-24, P2 Task 7 review). `randomdag.randomize_parents` preserves every node's DEPTH, and
therefore the closure pair count exactly (sum of depths). It does NOT obviously preserve the
FAN-OUT distribution: it draws each node's new parent uniformly with replacement from the level
above, whereas a real taxonomy's fan-out is heavy-tailed (a few genera with hundreds of children,
many with one).

That matters for how P2's RandomDAG control may be READ. P2's candidate pool is the grandparent's
children, so the chance rate of guessing the true parent is 1/fanout(grandparent)
(`baselines.sibling_chance`). If rewiring shifts that distribution, the RandomDAG arm is not merely
a scrambled taxonomy -- it is a task of DIFFERENT DIFFICULTY, and comparing raw MRR across the two
would confound "did the model memorise shape" with "was the control easier".

So measure it: real vs randomised sibling-chance on the same nodes, same seed.

Read-only. Usage: <venv-python> helpers/p2_randomdag_changes_the_chance_floor.py [clade]
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/src")

from taxembed.eval.baselines import sibling_chance                      # noqa: E402
from taxembed.eval.p2_split import depth_from_closure, eligible_nodes   # noqa: E402
from taxembed.eval.randomdag import randomize_parents                   # noqa: E402
from taxembed.eval.subtree import parent_from_closure                   # noqa: E402
from taxembed.utils.training_pairs import TrainingPairs                 # noqa: E402

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
    pairs = TrainingPairs.load(DATA / clade / f"taxonomy_edges_{clade}_transitive.npz")
    n = pairs.n_nodes
    parent = parent_from_closure(pairs.ancestor_idx, pairs.descendant_idx, pairs.depth_diff, n)
    depth = depth_from_closure(pairs.descendant_idx, pairs.descendant_depth,
                               pairs.ancestor_idx, pairs.ancestor_depth, n)

    rand = randomize_parents(parent, depth, seed=0)

    print(f"clade = {clade}   n_nodes = {n:,}\n")

    # invariant the control is supposed to preserve
    real_pairs = int(depth[depth > 0].sum())
    print(f"closure pair count, both trees = sum(depth) = {real_pairs:,}  (preserved by construction)")
    print(f"depths identical after rewiring: {np.array_equal(depth, depth)}  "
          f"(parents all drawn from depth-1: "
          f"{bool((depth[rand[depth > 0]] == depth[depth > 0] - 1).all())})\n")

    fo_real = fanout_of(parent)
    fo_rand = fanout_of(rand)
    internal_real = fo_real[fo_real > 0]
    internal_rand = fo_rand[fo_rand > 0]
    print("FAN-OUT over nodes that have at least one child:")
    describe("real fanout", internal_real.astype(float))
    describe("randomised fanout", internal_rand.astype(float))
    print(f"  internal nodes: real {len(internal_real):,} vs randomised {len(internal_rand):,}")
    print(f"  leaves:         real {int((fo_real == 0).sum()):,} vs "
          f"randomised {int((fo_rand == 0).sum()):,}")

    # the quantity P2 actually compares against
    elig_real = eligible_nodes(parent, depth, leaves_only=True)
    elig_rand = eligible_nodes(rand, depth, leaves_only=True)
    print(f"\nBAND-ELIGIBLE LEAVES (depth in [11,28]): real {len(elig_real):,} vs "
          f"randomised {len(elig_rand):,}")

    if len(elig_real) and len(elig_rand):
        sc_real = sibling_chance(parent, elig_real)
        sc_rand = sibling_chance(rand, elig_rand)
        print("\nSIBLING CHANCE (1 / grandparent fan-out) -- P2's chance floor:")
        describe("real tree", sc_real)
        describe("randomised tree", sc_rand)
        ratio = sc_rand.mean() / max(sc_real.mean(), 1e-12)
        print(f"\n  mean chance floor ratio randomised/real = {ratio:.3f}")
        if abs(ratio - 1.0) > 0.05:
            print("  🛑 THE CONTROL IS A DIFFERENT TASK: raw MRR is NOT comparable across arms.")
            print("     Compare a chance-ADJUSTED quantity (MRR - sibling_chance, or")
            print("     normalized_rank), never raw MRR.")
        else:
            print("  ✅ chance floors match within 5%: raw MRR is comparable across arms.")

    print("\nfan-out histogram, real (top 8):",
          dict(sorted(Counter(internal_real.tolist()).items())[:8]))
    print("fan-out histogram, rand (top 8):",
          dict(sorted(Counter(internal_rand.tolist()).items())[:8]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
