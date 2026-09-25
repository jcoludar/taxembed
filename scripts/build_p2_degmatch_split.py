#!/usr/bin/env python
"""Build the P2 degree-matched-control split: a fan-out-preserving randomised closure, then the
SAME held-out-leaf split logic used for the real-tree arms (Task 8 brief, "WHAT CHANGED SINCE THE
BRIEF WAS WRITTEN").

p2_amendment_4_20260924 (results/p2_heldout_preregistration.json, USER DESIGN decision 2026-09-24,
BEFORE any P2 array was submitted and BEFORE any P2 outcome data existed): replaces
`scripts/build_p2_randomdag_split.py` as P2's control builder. `randomize_parents` draws each
node's new parent uniformly (i.i.d.) from the level above, which collapses the real taxonomy's
heavy-tailed fan-out toward the mean -- measured on mollusca_6447_clean seed 0
(helpers/p2_randomdag_changes_the_chance_floor.py): mean fan-out 4.61 -> 1.96, max 187 -> 14,
raising the chance floor (mean sibling_chance) 5.4x (p2_amendment_1_20260924). This builder instead
calls `taxembed.eval.randomdag.degree_matched_shuffle`, which permutes the REAL parent-label
multiset per depth level -- every parent keeps EXACTLY the child count it had, only which children
it gets is randomised -- so the fan-out distribution, and therefore the chance floor and the
degree prior, match the real tree by construction. `helpers/p2_degmatch_changes_the_chance_floor.py`
measures this directly and is the artifact this amendment's numbers are drawn from.

Two steps, in order (identical shape to build_p2_randomdag_split.py, sole substitution:
`degree_matched_shuffle` for `randomize_parents`):
  1. Load the real transitive closure, recover its `parent`/`depth` arrays, and rewire every
     non-root node's parent via `taxembed.eval.randomdag.degree_matched_shuffle` -- depth is
     preserved EXACTLY (spec v3 SS.P2.4), so the control arm's own closure pair count equals the
     real tree's, same as the retired RandomDAG construction. The randomised closure is expanded
     via `closure_from_parent` and WRITTEN TO DISK as its own transitive-closure `.npz`, in the
     identical `TrainingPairs` schema as any other closure file.
  2. Hand that randomised closure to `scripts/build_p2_split.py::build_split` -- imported, not
     duplicated -- so the degree-matched arm holds out parent edges from the RANDOMISED tree and
     is later scored (`scripts/score_p2_linkpred.py`) against RANDOMISED parents. Node identities
     (indices 0..n-1) are untouched by rewiring, so they line up 1:1 with the real-tree arms and
     the SAME `--mapping` TSV applies to both.

Per `results/p2_heldout_preregistration.json`'s `p2_amendment_4_20260924` block: this arm asks
whether the model does as well on a scrambled-but-degree-matched taxonomy as on the real one -- if
it does, the geometry is memorising tree SHAPE (depth, branching pattern), not taxonomy, WITHOUT
the confound of an easier chance floor that RandomDAG's uniform rewiring introduced.

--visibility (mirrors p2_amendment_2_20260924's matched-control design, now applied to the
degree-matched control): this script is invoked TWICE per seed, once at --visibility 0.0 and once
at --visibility 0.5, building the two MATCHED controls degmatch_vis00 / degmatch_vis50 -- one per
real arm, so vis00 is read against a control trained the same way (0% visibility) and vis50 against
a control trained the same way (50% visibility), never a control confounded with the OTHER arm's
visibility setting. Both calls for a given seed share the SAME randomised closure
(`degree_matched_shuffle` depends only on `--seed`, never `--visibility`) and the SAME held-out
node set (`select_holdout`, likewise seed-only) -- only the TRAIN split's visibility thinning
differs between the two calls, exactly mirroring how the real-tree vis00/vis50 pair differs.

Usage (single absolute-path invocation, per CLAUDE.md shell hygiene) -- run once per visibility,
same --seed both times, to build a matched pair:
  <python> scripts/build_p2_degmatch_split.py --npz <real_closure.npz> --outdir <dir> --seed 0 \\
      --visibility 0.0
  <python> scripts/build_p2_degmatch_split.py --npz <real_closure.npz> --outdir <dir> --seed 0 \\
      --visibility 0.5
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
from taxembed.eval.randomdag import closure_from_parent, degree_matched_shuffle  # noqa: E402
from taxembed.eval.subtree import parent_from_closure  # noqa: E402
from taxembed.utils.training_pairs import TrainingPairs  # noqa: E402

from build_p2_split import build_split  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", required=True, type=Path,
                    help="source (REAL) transitive closure .npz to randomise")
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=0,
                    help="drives BOTH the closure shuffle and the holdout split -- one seed, one "
                         "fully reproducible run")
    ap.add_argument("--visibility", type=float, default=0.0,
                    help="fraction of depth_diff>=2 closure rows visible in training; default "
                         "0.0. Run this script once at 0.0 and once at 0.5 (same --seed both "
                         "times) to build the matched degmatch_vis00/degmatch_vis50 pair "
                         "(p2_amendment_4_20260924) -- see module docstring")
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

    shuffled_parent = degree_matched_shuffle(parent, depth, seed=args.seed)
    shuffled_pairs = closure_from_parent(shuffled_parent, depth)

    real_clade = args.npz.parent.name
    clade = f"{real_clade}_degmatch"
    closure_name = f"taxonomy_edges_{clade}_seed{args.seed}_transitive.npz"
    closure_path = args.outdir / closure_name
    shuffled_pairs.save(closure_path)
    print(f"wrote degree-matched closure: {closure_path}  "
          f"({len(shuffled_pairs):,} pairs, {n:,} nodes, seed {args.seed})")

    manifest = build_split(shuffled_pairs, clade=clade, source_npz=closure_path, outdir=args.outdir,
                            visibility=args.visibility, seed=args.seed,
                            frac_test=args.frac_test, frac_val=args.frac_val,
                            band=(args.band[0], args.band[1]))
    manifest["real_source_npz"] = str(args.npz)
    manifest["degree_matched_closure_npz"] = str(closure_path)
    out = args.outdir / f"p2_{clade}_vis{int(round(args.visibility * 100)):02d}_seed{args.seed}_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
