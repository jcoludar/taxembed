#!/usr/bin/env python3
"""REVIEW WAVE 3 (read-only): I1 -- is `metrics_poincare` a second reading or the same one?

The final review claimed metrics_poincare is BIT-IDENTICAL to metrics_cosine on a real checkpoint.
`p2_amendment_3_20260924`'s `metrics_poincare_is_not_independent_corroboration` block records the
algebra and says it was verified STRUCTURALLY on synthetic equal-radius configurations, explicitly
NOT on a trained checkpoint ("none exists"). One does exist -- results/task9_runs_mollusca -- so this
measures the claim on real embeddings through the production ranking function, and separately checks
whether the equal-radius premise the algebra rests on actually holds on that checkpoint.

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

from taxembed.eval.linkpred import candidate_pool, linkpred_metrics, rank_of_true_parent  # noqa: E402
from taxembed.eval.p2_split import depth_from_closure  # noqa: E402
from taxembed.eval.subtree import parent_from_closure  # noqa: E402
from taxembed.utils.training_pairs import TrainingPairs  # noqa: E402

SPLITS = _REPO / "data" / "p2_splits"
CKPT = _REPO / "results" / "task9_runs_mollusca" / "echino_canonical_s0_epoch200.pth"


def main() -> int:
    manifest = json.loads((SPLITS / "p2_mollusca_6447_clean_vis00_seed0_manifest.json").read_text())
    pairs = TrainingPairs.load(Path(manifest["source_npz"]))
    n = pairs.n_nodes
    parent = parent_from_closure(pairs.ancestor_idx, pairs.descendant_idx, pairs.depth_diff, n)
    depth = depth_from_closure(pairs.descendant_idx, pairs.descendant_depth,
                               pairs.ancestor_idx, pairs.ancestor_depth, n)
    held = np.asarray(np.load(SPLITS / "p2_mollusca_6447_clean_seed0_heldout.npz")["test"],
                      dtype=np.int64)

    import torch
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    emb = ck.get("embeddings")
    if emb is None:
        emb = ck["state_dict"]["lt.weight"]
    emb = emb.detach().float().numpy()
    z = ck.get("z_embeddings")
    radii = np.linalg.norm(z.detach().double().numpy(), axis=1) if z is not None else None
    print(f"z_embeddings present: {z is not None}; "
          f"max |z| {float(radii.max()) if radii is not None else float('nan'):.4f}")

    cands = [candidate_pool(parent, depth, node=int(v)) for v in held]
    n_cand = np.array([len(c) for c in cands], dtype=np.int64)
    tp = parent[held]

    rc = np.array([rank_of_true_parent(emb, int(v), int(p), c, metric="cosine", tie_seed=0)
                   for v, p, c in zip(held, tp, cands)], dtype=np.int64)
    rp = np.array([rank_of_true_parent(emb, int(v), int(p), c, metric="poincare", tie_seed=0,
                                       radii=radii)
                   for v, p, c in zip(held, tp, cands)], dtype=np.int64)

    mc, mp = linkpred_metrics(rc, n_cand), linkpred_metrics(rp, n_cand)
    print(f"\nrank vectors identical: {np.array_equal(rc, rp)}  "
          f"({int((rc != rp).sum())} of {len(rc)} ranks differ)")
    print("metrics_cosine  :", json.dumps({k: (round(v, 17) if isinstance(v, float) else v)
                                           for k, v in mc.items()}))
    print("metrics_poincare:", json.dumps({k: (round(v, 17) if isinstance(v, float) else v)
                                           for k, v in mp.items()}))
    print(f"every metric bit-identical: {mc == mp}")

    # is the equal-radius premise real on this checkpoint?
    if radii is not None:
        spreads = []
        for c in cands:
            if len(c) >= 2:
                r = radii[c]
                spreads.append(float(r.max() - r.min()))
        spreads = np.array(spreads)
        print(f"\nwithin-pool |z| spread over {len(spreads)} scored pools: "
              f"median {np.median(spreads):.6f}  p95 {np.percentile(spreads, 95):.6f}  "
              f"max {spreads.max():.6f}  (mean |z| {radii.mean():.4f})")
        print("  => the equal-radius premise the monotonicity argument rests on is "
              f"{'TIGHT' if spreads.max() < 1e-3 else 'NOT exact'} on real embeddings; the rank "
              "identity above is the measurement that matters either way.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
