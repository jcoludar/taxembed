#!/usr/bin/env python3
"""P2 below-chance diagnosis: do the curriculum phase boundaries explain the collapse?

Two questions, both answered from local files only (no LRZ, no retrain):

  Q1  What are the REAL auto-curriculum phase boundaries for these splits (max_depth from the
      actual .npz, n_epochs=200)? Do they coincide with the observed inflection epochs
      (flat->learning at ~40, peak at ~80, decay through 120-200)?

  Q2  Is a held-out node PRESENT in the training closure at all? That decides whether the
      negative-sampler leak (a held-out node drawn as a negative for its own true parent) is
      even reachable in each arm -- it needs the node to be in `_depth_to_nodes`, which is
      built only from nodes appearing in the training pairs.

Written 2026-09-29 for the P2 below-chance-MRR diagnosis.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings")
SPLITDIR = ROOT / "data" / "p2_splits"
sys.path.insert(0, str(ROOT))

N_EPOCHS = 200


def auto_curriculum_phases(max_depth: int, n_epochs: int):
    """Verbatim copy of train_small.py::auto_curriculum_phases (pinned below by import check)."""
    e1 = 1
    e2 = max(2, int(n_epochs * 0.2))
    e3 = max(e2 + 1, int(n_epochs * 0.4))
    e4 = max(e3 + 1, int(n_epochs * 0.6))
    return [(e1, 1), (e2, max(1, max_depth // 4)), (e3, max(2, max_depth // 2)), (e4, None)]


def check_copy_matches_production():
    """A verifier I wrote shares my blind spot unless it is the SAME function."""
    from train_small import auto_curriculum_phases as prod
    for md in (10, 20, 28, 33, 40):
        assert prod(md, N_EPOCHS) == auto_curriculum_phases(md, N_EPOCHS), md
    return True


def main():
    print("pinned against production auto_curriculum_phases:", check_copy_matches_production())
    print()

    for arm in ("vis00", "vis50"):
        train = SPLITDIR / f"p2_metazoa_33208_clean_{arm}_seed0_train.npz"
        if not train.exists():
            print(f"MISSING {train}")
            continue
        d = np.load(train, allow_pickle=False)
        keys = list(d.keys())
        anc, des = d["ancestor_idx"], d["descendant_idx"]
        dd = d["depth_diff"]
        desc_depth = d["descendant_depth"] if "descendant_depth" in keys else None

        max_depth = int(desc_depth.max()) if desc_depth is not None else int(dd.max())
        phases = auto_curriculum_phases(max_depth, N_EPOCHS)

        print("=" * 78)
        print(f"{arm}  ({train.name})")
        print("=" * 78)
        print(f"  keys                 : {keys}")
        print(f"  n_pairs              : {len(anc):,}")
        print(f"  max descendant_depth : {max_depth}")
        print(f"  depth_diff min/max   : {int(dd.min())} / {int(dd.max())}")
        print(f"  n pairs at dd==1     : {int((dd == 1).sum()):,}")
        print()
        print("  AUTO CURRICULUM PHASES (n_epochs=200):")
        names = ["phase1", "phase2", "phase3", "phase4"]
        for i, (ep, mdd) in enumerate(phases):
            end = phases[i + 1][0] - 1 if i + 1 < len(phases) else N_EPOCHS
            cap = "ALL PAIRS" if mdd is None else f"depth_diff <= {mdd}"
            n_rows = len(dd) if mdd is None else int((dd <= mdd).sum())
            print(f"    {names[i]}: epochs {ep:>3}-{end:<3}  {cap:<22} "
                  f"rows={n_rows:>12,}  ({100.0 * n_rows / len(dd):5.1f}% of closure)")

        # Q2 -- are held-out nodes present in the training closure?
        held = SPLITDIR / "p2_metazoa_33208_clean_seed0_heldout.npz"
        if held.exists():
            h = np.load(held, allow_pickle=False)
            hkeys = list(h.keys())
            print()
            print(f"  heldout npz keys     : {hkeys}")
            test = h["test"]
            print(f"  test array shape     : {test.shape}  dtype={test.dtype}")
            # Expect columns (node, true_parent) or a structured array.
            if test.ndim == 2 and test.shape[1] >= 2:
                v = test[:, 0].astype(np.int64)
                p = test[:, 1].astype(np.int64)
            else:
                v, p = test.astype(np.int64), None
            if True:
                present = np.isin(v, np.union1d(np.unique(anc), np.unique(des)))
                as_desc = np.isin(v, np.unique(des))
                print(f"  n held-out nodes     : {len(v):,}")
                print(f"  present in closure   : {int(present.sum()):,} "
                      f"({100.0 * present.mean():.2f}%)")
                print(f"  present AS DESCENDANT: {int(as_desc.sum()):,} "
                      f"({100.0 * as_desc.mean():.2f}%)   <- gates _depth_to_nodes membership")
                print("  => negative-sampler leak (held-out node drawn as a negative for its own "
                      "true parent) is " + ("REACHABLE" if as_desc.any() else "UNREACHABLE")
                      + " in this arm")
                if p is not None:
                    print(f"  n held-out parents   : {len(np.unique(p)):,} distinct")
        print()


if __name__ == "__main__":
    main()
