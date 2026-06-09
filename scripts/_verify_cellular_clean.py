"""Verify the clean cellular-organisms (131567) dataset before GPU (Rule 10).

Checks (mirrors _verify_eukaryota_clean.py):
  1. npz integrity + n_nodes == mapping rows; transitive pair count + max depth_diff.
  2. Residual noise: how many node names STILL match the noise patterns (should be ~0; a few
     noisy *internal* containers can survive if they have real children — report + sample).
  3. Domain balance (Bacteria / Archaea / Eukaryota) — sanity that all three cellular domains kept.
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

DS = ROOT / "data" / "taxopy" / "cellular_organisms_131567_clean"
NPZ = DS / "taxonomy_edges_cellular_organisms_131567_clean_transitive.npz"
MAP = DS / "taxonomy_edges_cellular_organisms_131567_clean.mapping.tsv"

DOMAINS = {"Bacteria": 2, "Archaea": 2157, "Eukaryota": 2759}

print("1) npz + mapping integrity")
pairs = TrainingPairs.load(NPZ)
mp = pd.read_csv(MAP, sep="\t")
print(f"   pairs.n_nodes = {pairs.n_nodes:,}  mapping rows = {len(mp):,}  match = {pairs.n_nodes == len(mp)}")
print(f"   transitive pairs = {len(pairs):,}  depth_diff max = {int(pairs.depth_diff.max())}")

print("\n2) residual noise (name-match on KEPT nodes; want ~0)")
taxdb = load_taxdb(ROOT / "data")
noisy = _compile_noise_filter(DEFAULT_NOISE_PATTERNS)
taxid_col = "taxid" if "taxid" in mp.columns else mp.columns[0]
kept_taxids = [int(t) for t in mp[taxid_col].tolist() if str(t).isdigit()]


def name_of(tid):
    return taxdb.taxid2name.get(str(tid)) or taxdb.taxid2name.get(tid)


hits = [(t, name_of(t)) for t in kept_taxids if name_of(t) and noisy(name_of(t))]
print(f"   residual noisy-name nodes: {len(hits):,}  ({100*len(hits)/len(kept_taxids):.3f}%)")
print(f"   sample: {[h[1] for h in hits[:12]]}")

print("\n3) domain balance (lineage walk to a domain root)")
parent = {int(c): int(p) for c, p in taxdb.taxid2parent.items()}


def domain_of(tid, _cache={}):
    seen = []
    cur = tid
    while cur not in (1, 131567) and cur in parent:
        for dname, droot in DOMAINS.items():
            if cur == droot:
                return dname
        seen.append(cur)
        nxt = parent.get(cur)
        if nxt == cur or nxt is None:
            break
        cur = nxt
    return "other/root"


counts = Counter(domain_of(t) for t in kept_taxids)
for d in ["Bacteria", "Archaea", "Eukaryota", "other/root"]:
    print(f"   {d:14s} {counts.get(d, 0):>10,}")

print("\n4) random name eyeball (seed 0, 20 names)")
rng = np.random.default_rng(0)
sample = rng.choice(kept_taxids, size=min(20, len(kept_taxids)), replace=False)
for t in sample:
    print(f"   {t:>10}  {name_of(t)}")
