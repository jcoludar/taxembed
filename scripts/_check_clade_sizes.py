"""Report node + transitive-pair counts for the local clade datasets (read-only).

Used to decide whether a clade is small enough to train the canonical recipe locally on MPS
(generalization step) vs. needing LRZ. Throwaway probe.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from taxembed.utils.training_pairs import TrainingPairs

DATASETS = [
    "echinodermata_7586_clean",
    "mollusca_6447_clean",
    "arthropoda_6656_clean",
    "metazoa_33208_clean",
]

for ds in DATASETS:
    base = ROOT / "data" / "taxopy" / ds
    npz = base / f"taxonomy_edges_{ds}_transitive.npz"
    if not npz.exists():
        print(f"{ds:32s} MISSING {npz}")
        continue
    pairs = TrainingPairs.load(npz)
    n_pairs = None
    for attr in ("pairs", "edges", "rows"):
        v = getattr(pairs, attr, None)
        if v is not None:
            try:
                n_pairs = len(v)
                break
            except TypeError:
                pass
    pstr = f"{n_pairs:,}" if isinstance(n_pairs, int) else str(n_pairs)
    print(f"{ds:32s} n_nodes={pairs.n_nodes:>8,}  n_transitive_pairs={pstr:>14}")
