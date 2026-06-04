"""Probe the Eukaryota (taxid 2759) build size BEFORE committing to a full build (Rule 10).

Loads the local taxdump, collects the Eukaryota clade, and reports: raw node count, max depth,
estimated transitive pairs (= the training-pair volume), how many nodes the default noise filter
would strip, and the top-level kingdom breakdown. Read-only; writes nothing.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from taxembed.utils.taxdump import load_taxdb
from taxembed.builders.taxopy_clade import (
    _build_children_index, _collect_clade, _compile_noise_filter,
    _is_container_node, DEFAULT_NOISE_PATTERNS,
)

EUK = 2759
DATA = ROOT / "data"

print("Loading taxdump (full NCBI tree)...")
taxdb = load_taxdb(DATA)
parent_map = {int(c): int(p) for c, p in taxdb.taxid2parent.items()}
print(f"  full NCBI nodes: {len(parent_map):,}")

children = _build_children_index(parent_map)
print("Collecting Eukaryota clade...")
depths, edges = _collect_clade(children, EUK)
n = len(depths)
maxd = max(depths.values())
avgd = sum(depths.values()) / n
est_pairs = int(avgd * n * 1.1)
print(f"\nEukaryota raw: nodes={n:,}  max_depth={maxd}  avg_depth={avgd:.1f}")
print(f"  est. transitive pairs ~= {est_pairs:,}  (~{est_pairs*22/1e9:.2f} GB in-RAM arrays @22B/pair)")
print(f"  (metazoa_33208_clean for comparison: 498,246 nodes)")


def name_of(tid):
    return taxdb.taxid2name.get(str(tid)) or taxdb.taxid2name.get(tid)


# How much would --clean strip? Count nodes whose name is noisy / missing / a container.
noisy = _compile_noise_filter(DEFAULT_NOISE_PATTERNS)
n_noise = n_missing = n_container = 0
noise_samples = []
for tid in depths:
    if tid == EUK:
        continue
    nm = name_of(tid)
    if nm is None:
        n_missing += 1
    elif noisy(nm):
        n_noise += 1
        if len(noise_samples) < 15:
            noise_samples.append(nm)
    elif _is_container_node(nm):
        n_container += 1
print(f"\nNoise the --clean filter targets (name-match only; actual prune is iterative bottom-up):")
print(f"  noisy-name nodes : {n_noise:,}")
print(f"  missing-name     : {n_missing:,}")
print(f"  container nodes  : {n_container:,}")
print(f"  -> rough upper-bound strip ~= {100*(n_noise+n_missing)/n:.1f}% (containers collapse only if emptied)")
print(f"  sample noisy names: {noise_samples}")

# Top-level kingdom breakdown (direct children of Eukaryota + their clade sizes).
print(f"\nTop-level groups under Eukaryota:")
for kid in sorted(children.get(EUK, []), key=lambda k: -len(_collect_clade(children, k)[0])):
    ksz = len(_collect_clade(children, kid)[0])
    if ksz >= 100:
        print(f"  {name_of(kid):32s} {ksz:>10,}")
