"""Verify the clean Eukaryota dataset before GPU (Rule 10): junk gone, structure sane, npz intact.

Checks:
  1. npz integrity + n_nodes == mapping rows.
  2. How many node names STILL match the noise patterns (should be ~0; a few noisy *internal* nodes
     can survive if they have real children — report + sample them).
  3. Kingdom balance (nodes per top-level eukaryote group) — sanity that we kept full diversity.
  4. Random name sample to eyeball.
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from taxembed.utils.taxdump import load_taxdb
from taxembed.utils.training_pairs import TrainingPairs
from taxembed.builders.taxopy_clade import _compile_noise_filter, DEFAULT_NOISE_PATTERNS

DS = ROOT / "data" / "taxopy" / "eukaryota_2759_clean"
NPZ = DS / "taxonomy_edges_eukaryota_2759_clean_transitive.npz"
MAP = DS / "taxonomy_edges_eukaryota_2759_clean.mapping.tsv"

# Named top-level eukaryote groups for the kingdom-balance check.
KINGDOMS = {
    "Opisthokonta": 33154, "Viridiplantae": 33090, "Sar": 2698737, "Rhodophyta": 2763,
    "Amoebozoa": 554915, "Discoba": 2611341, "Metamonada": 2611352, "Haptista": 2683617,
    "Metazoa": 33208, "Fungi": 4751,
}

print("1) npz + mapping integrity")
pairs = TrainingPairs.load(NPZ)
mp = pd.read_csv(MAP, sep="\t")
print(f"   pairs.n_nodes = {pairs.n_nodes:,}  mapping rows = {len(mp):,}  match = {pairs.n_nodes == len(mp)}")
print(f"   transitive pairs = {len(pairs):,}  depth_diff max = {int(pairs.depth_diff.max())}")

taxids = mp["taxid"].astype(int).tolist()
taxid_set = set(taxids)

print("\n2) residual noise (post-clean)")
taxdb = load_taxdb(ROOT / "data")
def name_of(t): return taxdb.taxid2name.get(str(t)) or taxdb.taxid2name.get(t)
noisy = _compile_noise_filter(DEFAULT_NOISE_PATTERNS)
resid, samples = 0, []
for t in taxids:
    nm = name_of(t)
    if nm and noisy(nm):
        resid += 1
        if len(samples) < 20:
            samples.append(nm)
print(f"   names still matching noise patterns: {resid:,} / {len(taxids):,} ({100*resid/len(taxids):.3f}%)")
print(f"   (expected near-0; survivors are internal nodes with real children)")
if samples:
    print(f"   sample survivors: {samples}")

print("\n3) kingdom balance (membership by ancestor walk)")
parent = {int(c): int(p) for c, p in taxdb.taxid2parent.items()}
def lineage(t):
    seen = set(); cur = t
    while cur in parent and cur not in seen:
        yield cur; seen.add(cur)
        nxt = parent[cur]
        if nxt == cur: break
        cur = nxt
king_counts = Counter()
for t in taxids:
    lin = set(lineage(t))
    for kname, ktid in KINGDOMS.items():
        if ktid in lin:
            king_counts[kname] += 1
for kname, _ in sorted(KINGDOMS.items(), key=lambda kv: -king_counts[kv[0]]):
    print(f"   {kname:16s} {king_counts[kname]:>9,}")

print("\n4) random name sample")
rng = np.random.default_rng(0)
for t in rng.choice(taxids, 20, replace=False):
    print(f"   {t:>10}  {name_of(int(t))}  [{taxdb.taxid2rank.get(str(t)) or taxdb.taxid2rank.get(int(t))}]")
