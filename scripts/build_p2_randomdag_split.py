#!/usr/bin/env python
"""Build the P2 RandomDAG-control split: a depth-preserving randomised closure, then the SAME
held-out-leaf split logic used for the real-tree arms (Task 8 brief, "WHAT CHANGED SINCE THE
BRIEF WAS WRITTEN").

Two steps, in order:
  1. Load the real transitive closure, recover its `parent`/`depth` arrays, and rewire every
     non-root node to a uniformly random parent at depth-1 via
     `taxembed.eval.randomdag.randomize_parents` -- depth is preserved EXACTLY (spec v3 SS.P2.4),
     so the RandomDAG arm's own closure pair count equals the real tree's. The randomised closure
     is expanded via `closure_from_parent` and WRITTEN TO DISK as its own transitive-closure
     `.npz`, in the identical `TrainingPairs` schema as any other closure file.
  2. Hand that randomised closure to `scripts/build_p2_split.py::build_split` -- imported, not
     duplicated -- so the RandomDAG arm holds out parent edges from the RANDOMISED tree and is
     later scored (`scripts/score_p2_linkpred.py`) against RANDOMISED parents. Node identities
     (indices 0..n-1) are untouched by rewiring, so they line up 1:1 with the real-tree arms and
     the SAME `--mapping` TSV applies to both.

Per `results/p2_heldout_preregistration.json`'s `arms.randomdag` block: this arm asks whether the
model does as well on a scrambled taxonomy as on the real one -- if it does, the geometry is
memorising tree SHAPE (depth, branching pattern), not taxonomy. Per `p2_amendment_1_20260924`,
RandomDAG's chance floor is NOT the same as the real tree's (rewiring collapses fan-out, measured
~5.4x easier at chance on mollusca_6447_clean), so any CROSS-TREE comparison against this arm's
output must read `normalized_rank`, never raw MRR -- see `scripts/score_p2_linkpred.py` and
`src/taxembed/eval/preregistration.py::p2_verdict(..., amendment_1=True)`.

--visibility defaults to 0.0 (matches the vis00 arm): the array has only ONE RandomDAG split per
seed (no vis00/vis50 split for this arm, per the 9-element array), and 0.0 is the split every
held-out node's own ancestry survives at both real-tree settings.

Usage (single absolute-path invocation, per CLAUDE.md shell hygiene):
  <python> scripts/build_p2_randomdag_split.py --npz <real_closure.npz> --outdir <dir> --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))

from taxembed.eval.p2_split import DEFAULT_BAND, depth_from_closure  # noqa: E402
from taxembed.eval.randomdag import closure_from_parent, randomize_parents  # noqa: E402
from taxembed.eval.subtree import parent_from_closure  # noqa: E402
from taxembed.utils.training_pairs import TrainingPairs  # noqa: E402

from build_p2_split import build_split  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", required=True, type=Path,
                    help="source (REAL) transitive closure .npz to randomise")
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=0,
                    help="drives BOTH the closure randomisation and the holdout split -- one "
                         "seed, one fully reproducible run")
    ap.add_argument("--visibility", type=float, default=0.0,
                    help="fraction of depth_diff>=2 closure rows visible in training; default "
                         "0.0 (see module docstring)")
    ap.add_argument("--frac-test", type=float, default=0.10)
    ap.add_argument("--frac-val", type=float, default=0.0,
                    help="see build_p2_split.py --frac-val: default 0.0, nothing withheld "
                         "without an explicit flag")
    ap.add_argument("--band", type=int, nargs=2, default=list(DEFAULT_BAND),
                    metavar=("LO", "HI"), help="inclusive per-node depth band")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    real_pairs = TrainingPairs.load(args.npz)
    n = real_pairs.n_nodes
    parent = parent_from_closure(real_pairs.ancestor_idx, real_pairs.descendant_idx,
                                  real_pairs.depth_diff, n)
    depth = depth_from_closure(real_pairs.descendant_idx, real_pairs.descendant_depth,
                               real_pairs.ancestor_idx, real_pairs.ancestor_depth, n)

    rand_parent = randomize_parents(parent, depth, seed=args.seed)
    rand_pairs = closure_from_parent(rand_parent, depth)

    real_clade = args.npz.parent.name
    clade = f"{real_clade}_randomdag"
    closure_name = f"taxonomy_edges_{clade}_seed{args.seed}_transitive.npz"
    closure_path = args.outdir / closure_name
    rand_pairs.save(closure_path)
    print(f"wrote randomised closure: {closure_path}  "
          f"({len(rand_pairs):,} pairs, {n:,} nodes, seed {args.seed})")

    manifest = build_split(rand_pairs, clade=clade, source_npz=closure_path, outdir=args.outdir,
                            visibility=args.visibility, seed=args.seed,
                            frac_test=args.frac_test, frac_val=args.frac_val,
                            band=(args.band[0], args.band[1]))
    manifest["real_source_npz"] = str(args.npz)
    manifest["randomised_closure_npz"] = str(closure_path)
    out = args.outdir / f"p2_{clade}_vis{int(round(args.visibility * 100)):02d}_seed{args.seed}_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
