"""READ-ONLY review helper (adversarial pre-submit review, 2026-09-26).

Measures, for every P2 metazoa split the 12-element GPU array will train on:
  - the arrays the .npz actually holds and their shapes
  - the edge count that drives per-epoch cost
  - the ratio to the reference run whose wall-clock we know
    (task8_fixed_canonical_s0: /data/taxonomy_edges_metazoa_33208_clean_transitive.npz,
     200 epochs, ~14.7 h measured from checkpoint mtimes on LRZ)

Writes nothing. Reads only files already on disk.
"""
from __future__ import annotations

import os
import numpy as np

REPO = "/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings"
SPLITS = os.path.join(REPO, "data", "p2_splits")

REF_CANDIDATES = [
    os.path.join(REPO, "data", "taxonomy_edges_metazoa_33208_clean_transitive.npz"),
    os.path.join(REPO, "data", "taxopy", "metazoa_33208_clean",
                 "taxonomy_edges_metazoa_33208_clean_transitive.npz"),
]
REF_HOURS = 14.7  # measured: milestone_epoch20 18:00 Sep23 -> epoch200 07:12 Sep24, extrapolated


def describe(path: str) -> dict:
    with np.load(path, allow_pickle=True) as z:
        info = {k: (z[k].shape, str(z[k].dtype)) for k in z.files}
        n_edges = 0
        for key in ("ancestor_idx", "edges", "pairs", "train_edges", "arr_0"):
            if key in z:
                a = z[key]
                n_edges = int(a.shape[0])
                break
        if not n_edges:
            n_edges = max((int(z[k].shape[0]) for k in z.files), default=0)
    return {"arrays": info, "n_edges": n_edges, "bytes": os.path.getsize(path)}


def main() -> None:
    ref = next((p for p in REF_CANDIDATES if os.path.exists(p)), None)
    ref_edges = None
    if ref:
        d = describe(ref)
        ref_edges = d["n_edges"]
        print(f"REFERENCE {ref}")
        print(f"  arrays: {d['arrays']}")
        print(f"  n_edges={ref_edges}  bytes={d['bytes']}  measured 200-ep wall-clock ~{REF_HOURS} h")
    else:
        print("REFERENCE full metazoa transitive .npz NOT FOUND locally; ratios unavailable")
    print()

    rows = []
    for arm in ("vis00", "vis50", "degmatch_vis00", "degmatch_vis50"):
        for seed in (0, 1, 2):
            p = os.path.join(
                SPLITS, f"p2_metazoa_33208_clean_{arm}_seed{seed}_train.npz")
            if not os.path.exists(p):
                print(f"MISSING {p}")
                continue
            d = describe(p)
            rows.append((arm, seed, d))

    print(f"{'arm':<16}{'seed':<6}{'n_edges':>12}{'bytes':>12}{'x_ref':>9}{'est_h':>9}")
    for arm, seed, d in rows:
        ratio = (d["n_edges"] / ref_edges) if (ref_edges and d["n_edges"]) else float("nan")
        est = ratio * REF_HOURS
        print(f"{arm:<16}{seed:<6}{d['n_edges']:>12}{d['bytes']:>12}{ratio:>9.3f}{est:>9.1f}")

    print()
    print("array shapes, one example per arm:")
    seen = set()
    for arm, seed, d in rows:
        if arm in seen:
            continue
        seen.add(arm)
        print(f"  {arm}: {d['arrays']}")


if __name__ == "__main__":
    main()
