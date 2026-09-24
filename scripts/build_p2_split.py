#!/usr/bin/env python
"""Build a P2 held-out split: withhold the parent edge of band-eligible LEAF nodes.

The output is a closure .npz in the IDENTICAL TrainingPairs schema, so the trainer consumes it
unmodified:  taxembed train --file <train.npz> --mapping <clade>.mapping.tsv ...

That is the point of the design. The negative sampler builds its ancestry index from whatever
closure it is handed (train_hierarchical.py:263-294), so feeding it the split file means held-out
relations are invisible to training -- as ancestry knowledge as well as as positive pairs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from taxembed.eval.p2_split import (
    DEFAULT_BAND,
    depth_from_closure,
    eligible_nodes,
    parent_edge_mask,
    select_holdout,
)
from taxembed.eval.subtree import parent_from_closure
from taxembed.utils.training_pairs import TrainingPairs


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_split(pairs: TrainingPairs, clade: str, source_npz: Path, outdir: Path,
                 visibility: float, seed: int, frac_test: float = 0.10, frac_val: float = 0.0,
                 band: tuple[int, int] = DEFAULT_BAND) -> dict:
    """Core split-construction logic: eligible-leaf selection, holdout, visibility thinning,
    and manifest bookkeeping. Factored out so `scripts/build_p2_randomdag_split.py` can reuse it
    unchanged against a RANDOMISED closure instead of duplicating it (Task 8 brief) -- everything
    downstream of "which closure am I holding out from" is identical between the real-tree arms
    and the RandomDAG control.

    `source_npz` is only used for provenance (`source_npz`/`source_md5` in the manifest); the
    caller must have already written `pairs` to that path if it is not the original input file
    (e.g. the RandomDAG builder writes its randomised closure to disk first, then passes that
    path here so the manifest points at the actual file the closure was read from).
    """
    outdir.mkdir(parents=True, exist_ok=True)
    n = pairs.n_nodes

    parent = parent_from_closure(pairs.ancestor_idx, pairs.descendant_idx, pairs.depth_diff, n)
    depth = depth_from_closure(pairs.descendant_idx, pairs.descendant_depth,
                               pairs.ancestor_idx, pairs.ancestor_depth, n)
    band = (band[0], band[1])

    elig_all = eligible_nodes(parent, depth, band=band, leaves_only=False)
    elig_leaf = eligible_nodes(parent, depth, band=band, leaves_only=True)
    split = select_holdout(elig_leaf, frac_test, frac_val, seed)
    held_all = np.concatenate([split["test"], split["val"]])

    drop_parent = parent_edge_mask(pairs, held_all)

    rng = np.random.default_rng(seed + 10_000)  # independent stream from the node split
    held_mask = np.zeros(n, dtype=bool)
    held_mask[held_all] = True
    deep = pairs.depth_diff >= 2
    is_heldout_row = held_mask[pairs.descendant_idx]
    # Held-out nodes are leaves: they are never an ancestor, and their only depth_diff==1
    # row is their own parent edge (always dropped above). Their depth_diff>=2 rows are
    # therefore their ONLY possible source of a trained coordinate -- exempt those rows
    # from visibility thinning so every held-out node keeps at least its ancestry closure.
    hide_deep = deep & ~is_heldout_row & (rng.random(len(pairs)) >= visibility)

    keep = ~(drop_parent | hide_deep)
    train = pairs[keep]

    vis_tag = f"vis{int(round(visibility * 100)):02d}"
    train_name = f"p2_{clade}_{vis_tag}_seed{seed}_train.npz"
    held_name = f"p2_{clade}_seed{seed}_heldout.npz"
    train.save(outdir / train_name)
    np.savez_compressed(outdir / held_name, test=split["test"], val=split["val"])

    manifest = {
        "clade": clade,
        "source_npz": str(source_npz),
        "source_md5": md5(source_npz),
        "visibility": visibility,
        "seed": seed,
        "band": list(band),
        "n_nodes": int(n),
        "n_pairs_source": int(len(pairs)),
        "n_eligible_all": int(len(elig_all)),
        "n_eligible_leaves": int(len(elig_leaf)),
        "n_test": int(len(split["test"])),
        "n_val": int(len(split["val"])),
        "n_train_nodes": int(len(split["train"])),
        "n_pairs_train": int(len(train)),
        "n_pairs_removed_parent_edges": int(drop_parent.sum()),
        "n_pairs_removed_visibility": int(hide_deep.sum()),
        "n_pairs_heldout_ancestry_kept": int((deep & is_heldout_row).sum()),
        "train_npz": train_name,
        "heldout_npz": held_name,
        "train_md5": md5(outdir / train_name),
        "heldout_md5": md5(outdir / held_name),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    out = outdir / f"p2_{clade}_{vis_tag}_seed{seed}_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", required=True, type=Path, help="source transitive closure .npz")
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--visibility", type=float, required=True,
                    help="fraction of depth_diff>=2 closure rows visible in training (0.0 or 0.5)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frac-test", type=float, default=0.10)
    ap.add_argument("--frac-val", type=float, default=0.0,
                    help="fraction of eligible leaves withheld as 'val' -- reserved for "
                         "threshold-tuning, NOT scored by scripts/score_p2_linkpred.py. Default "
                         "0.0 (fix round 1, IMPORTANT #3): the old 0.05 default silently withheld "
                         "5%% of eligible nodes from every reported number whenever this flag was "
                         "forgotten, with no trace in the output. Pass explicitly to withhold val.")
    ap.add_argument("--band", type=int, nargs=2, default=list(DEFAULT_BAND),
                    metavar=("LO", "HI"), help="inclusive per-node depth band")
    args = ap.parse_args()

    pairs = TrainingPairs.load(args.npz)
    clade = args.npz.parent.name
    manifest = build_split(pairs, clade=clade, source_npz=args.npz, outdir=args.outdir,
                            visibility=args.visibility, seed=args.seed,
                            frac_test=args.frac_test, frac_val=args.frac_val,
                            band=(args.band[0], args.band[1]))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
