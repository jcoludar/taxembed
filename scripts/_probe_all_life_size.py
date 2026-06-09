"""Probe the all-Life build size BEFORE committing to a full LRZ run (Rule 10).

Compares two candidate roots:
  * cellular organisms (131567) = Bacteria + Archaea + Eukaryota  -- NO viruses (the "properly
    hierarched" tree of life the user asked for)
  * root (1) = everything incl. Viruses + unclassified top-level junk (for contrast only)

Reports raw node count, max depth, estimated transitive pairs (training-pair volume + rough RAM),
the name-match noise the --clean filter targets, and the top-level domain breakdown. Read-only.
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

DATA = ROOT / "data"
CELLULAR = 131567
ROOT_TAX = 1

print("Loading taxdump (full NCBI tree)...")
taxdb = load_taxdb(DATA)
parent_map = {int(c): int(p) for c, p in taxdb.taxid2parent.items()}
print(f"  full NCBI nodes: {len(parent_map):,}")
children = _build_children_index(parent_map)
noisy = _compile_noise_filter(DEFAULT_NOISE_PATTERNS)


def name_of(tid):
    return taxdb.taxid2name.get(str(tid)) or taxdb.taxid2name.get(tid)


def probe(label, root_taxid):
    depths, edges = _collect_clade(children, root_taxid)
    n = len(depths)
    maxd = max(depths.values())
    avgd = sum(depths.values()) / n
    est_pairs = int(avgd * n * 1.1)
    n_noise = n_missing = n_container = 0
    for tid in depths:
        if tid == root_taxid:
            continue
        nm = name_of(tid)
        if nm is None:
            n_missing += 1
        elif noisy(nm):
            n_noise += 1
        elif _is_container_node(nm):
            n_container += 1
    strip_pct = 100 * (n_noise + n_missing) / n
    print(f"\n=== {label} (taxid {root_taxid}) ===")
    print(f"  raw nodes        : {n:,}")
    print(f"  max depth        : {maxd}   avg depth: {avgd:.1f}")
    print(f"  est. transitive pairs ~= {est_pairs:,}  (~{est_pairs*22/1e9:.1f} GB in-RAM @22B/pair)")
    print(f"  noisy-name nodes : {n_noise:,}")
    print(f"  missing-name     : {n_missing:,}")
    print(f"  container nodes  : {n_container:,}")
    print(f"  rough name-match strip ~= {strip_pct:.1f}% (iterative bottom-up prune removes more)")
    print(f"  est. CLEAN nodes ~= {int(n*(1-strip_pct/100)):,}")
    print(f"  top-level groups under {label}:")
    for kid in sorted(children.get(root_taxid, []), key=lambda k: -len(_collect_clade(children, k)[0])):
        ksz = len(_collect_clade(children, kid)[0])
        if ksz >= 1000:
            print(f"    {str(name_of(kid)):32s} {ksz:>12,}")


print("\n(reference: metazoa_33208_clean = 498,246 nodes; eukaryota_2759_clean = 877,584 nodes, 18.9M pairs, 20h30m V100)")
probe("cellular organisms", CELLULAR)
probe("root (incl Viruses)", ROOT_TAX)
