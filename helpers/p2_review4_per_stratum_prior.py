"""READ-ONLY PROBE (review 4, 2026-09-26).

amendment_6 divides each side's PER-DEPTH-STRATUM normalized_rank by that tree's AGGREGATE
training-free degree prior (preregistration.py:830-832). The pre-registration's
`stated_limitations.strata_use_the_aggregate_prior` defends this as "a uniform positive rescaling
of each side: it cannot reorder strata within an arm".

That is true and beside the point: the sign test is a CROSS-ARM comparison, so what matters is
whether the AGGREGATE difficulty ratio (real_prior / control_prior) equals the PER-STRATUM one.
This measures both on the real production metazoa splits, seed 0 -- the same splits the array
will train on -- and reports the per-stratum allowance error the amendment grants.

Writes nothing. Touches no file under src/, scripts/, tests/ or results/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))

from taxembed.eval.baselines import _fanout_of, degree_prior_rank  # noqa: E402
from taxembed.eval.linkpred import candidate_pool, linkpred_metrics  # noqa: E402
from taxembed.eval.p2_split import depth_from_closure  # noqa: E402
from taxembed.eval.subtree import parent_from_closure  # noqa: E402
from taxembed.utils.training_pairs import TrainingPairs  # noqa: E402

SPLITS = _REPO / "data" / "p2_splits"
REAL_CLOSURE = (_REPO / "data" / "taxopy" / "metazoa_33208_clean"
                / "taxonomy_edges_metazoa_33208_clean_transitive.npz")
DEPTH_BINS = [(11, 15), (16, 21), (22, 28)]      # scripts/score_p2_linkpred.py:90


def measure(closure: Path, heldout: Path):
    pairs = TrainingPairs.load(closure)
    n = pairs.n_nodes
    parent = parent_from_closure(pairs.ancestor_idx, pairs.descendant_idx, pairs.depth_diff, n)
    depth = depth_from_closure(pairs.descendant_idx, pairs.descendant_depth,
                               pairs.ancestor_idx, pairs.ancestor_depth, n)
    held = np.asarray(np.load(heldout)["test"], dtype=np.int64)
    cands = [candidate_pool(parent, depth, node=int(v)) for v in held]
    n_cand = np.array([len(c) for c in cands], dtype=np.int64)
    true_parents = parent[held]
    fanout = _fanout_of(parent)
    ranks = np.array([degree_prior_rank(parent, int(v), int(tp), c, tie_seed=0, fanout=fanout)
                      for v, tp, c in zip(held, true_parents, cands)], dtype=np.int64)
    agg = linkpred_metrics(ranks, n_cand)["normalized_rank"]
    per = {}
    dh = depth[held]
    for lo, hi in DEPTH_BINS:
        sel = (dh >= lo) & (dh <= hi)
        m = linkpred_metrics(ranks[sel], n_cand[sel])
        per[f"{lo}-{hi}"] = (m["normalized_rank"], m["n_scored"])
    return agg, per


def main() -> int:
    seed = 0
    print("measuring the TRAINING-FREE degree prior per depth stratum, metazoa seed 0 ...")
    real_agg, real_per = measure(
        REAL_CLOSURE, SPLITS / f"p2_metazoa_33208_clean_seed{seed}_heldout.npz")
    ctrl_agg, ctrl_per = measure(
        SPLITS / f"taxonomy_edges_metazoa_33208_clean_degmatch_seed{seed}_transitive.npz",
        SPLITS / f"p2_metazoa_33208_clean_degmatch_seed{seed}_heldout.npz")

    agg_ratio = real_agg / ctrl_agg
    print()
    print(f"AGGREGATE degree_prior normalized_rank: real {real_agg:.5f}  "
          f"control {ctrl_agg:.5f}  ratio {agg_ratio:.4f}")
    print("   (this single ratio is the allowance amendment_6 grants IN EVERY STRATUM)")
    print()
    print(f"{'stratum':>10} {'real prior':>11} {'ctrl prior':>11} {'true ratio':>11} "
          f"{'aggregate':>10} {'over-allow':>11} {'n_scored real':>14}")
    worst = 0.0
    for k in real_per:
        r, nr_ = real_per[k]
        c, nc_ = ctrl_per[k]
        true_ratio = r / c
        over = agg_ratio / true_ratio
        worst = max(worst, abs(np.log(over)))
        print(f"{k:>10} {r:11.5f} {c:11.5f} {true_ratio:11.4f} {agg_ratio:10.4f} "
              f"{over:10.3f}x {nr_:14d}")
    print()
    print(f"worst |log| discrepancy between the aggregate allowance and the stratum's own: "
          f"{worst:.4f}")
    print()
    print("READING: in a stratum where the aggregate over-allows (over-allow > 1), a real arm")
    print("whose per-stratum normalized_rank is WORSE than its control's own-prior-scaled value")
    print("still scores a NEGATIVE diff and passes `sign_consistent` -- i.e. GENERALISES can be")
    print("reached through a stratum the per-stratum prior would have failed.")

    # Make the consequence concrete for the stratum with the largest over-allowance.
    k = max(real_per, key=lambda k: agg_ratio / (real_per[k][0] / ctrl_per[k][0]))
    r, _ = real_per[k]
    c, _ = ctrl_per[k]
    true_ratio = r / c
    # An arm and a control that are EXACTLY at parity on the per-stratum prior scale:
    ctrl_nr = c * 0.5                 # control halves its own per-stratum prior
    arm_nr = r * 0.5                  # arm halves its own per-stratum prior -> genuine TIE
    d_agg = arm_nr / real_agg - ctrl_nr / ctrl_agg
    d_str = arm_nr / r - ctrl_nr / c
    print()
    print(f"worked example, stratum {k} (true ratio {true_ratio:.3f}, aggregate {agg_ratio:.3f}):")
    print(f"  arm and control each halve THEIR OWN per-stratum prior -> a genuine TIE")
    print(f"  diff as coded (aggregate prior) : {d_agg:+.6f}   sign_consistent needs < 0 -> "
          f"{d_agg < 0}")
    print(f"  diff with the per-stratum prior : {d_str:+.6f}   -> {d_str < 0}")
    print(f"  the two DISAGREE: {(d_agg < 0) != (d_str < 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
