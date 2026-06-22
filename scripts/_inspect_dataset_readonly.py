"""Read-only inspector for taxopy dataset npz + mapping + edgelist headers.
Throwaway diagnostic for an external read-only deep-dive. Writes nothing.
"""
import numpy as np
from pathlib import Path

BASE = Path("/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/data/taxopy")

DATASETS = [
    "echinodermata_7586_clean",
    "arthropoda_6656_clean",
    "mollusca_6447_clean",
    "metazoa_33208_clean",
]

for ds in DATASETS:
    d = BASE / ds
    print("=" * 70)
    print("DATASET:", ds)
    npz_path = d / f"taxonomy_edges_{ds}_transitive.npz"
    z = np.load(npz_path, allow_pickle=True)
    print("  npz files/arrays:", list(z.files))
    for k in z.files:
        arr = z[k]
        print(f"    array '{k}': shape={arr.shape} dtype={arr.dtype}", end="")
        try:
            print(f" min={arr.min()} max={arr.max()}", end="")
        except Exception:
            pass
        flat = np.asarray(arr).ravel()
        print(" head=", flat[:6].tolist())
    # mapping head
    mp = d / f"taxonomy_edges_{ds}.mapping.tsv"
    with open(mp) as fh:
        lines = [next(fh) for _ in range(4)]
    n_map = sum(1 for _ in open(mp))
    print(f"  mapping.tsv lines={n_map}; first 4 lines:")
    for ln in lines:
        print("    | " + ln.rstrip("\n"))
    # mapped.edgelist head
    me = d / f"taxonomy_edges_{ds}.mapped.edgelist"
    with open(me) as fh:
        elines = [next(fh) for _ in range(3)]
    n_edge = sum(1 for _ in open(me))
    print(f"  mapped.edgelist lines={n_edge}; first 3 lines:")
    for ln in elines:
        print("    | " + ln.rstrip("\n"))
    # raw edgelist head
    re = d / f"taxonomy_edges_{ds}.edgelist"
    with open(re) as fh:
        rlines = [next(fh) for _ in range(3)]
    print("  edgelist (raw) first 3 lines:")
    for ln in rlines:
        print("    | " + ln.rstrip("\n"))
    z.close()
