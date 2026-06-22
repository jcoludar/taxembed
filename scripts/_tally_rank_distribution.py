"""Tally rank distribution + flag 'sp.' / unclassified / environmental noise in the
metazoa_33208_clean dataset that v2 was trained on."""

import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path("/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings")
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_hierarchy_hyperbolic import load_mapping, load_taxonomy_with_depth

MAPPING = REPO_ROOT / "data/taxopy/metazoa_33208_clean/taxonomy_edges_metazoa_33208_clean.mapping.tsv"
DATA_DIR = REPO_ROOT / "data"

NOISE_PATTERNS = {
    " sp.": re.compile(r"\bsp\."),
    "unclassified": re.compile(r"unclassified", re.IGNORECASE),
    "environmental": re.compile(r"environmental", re.IGNORECASE),
    "uncultured": re.compile(r"uncultured", re.IGNORECASE),
    "endosymbiont": re.compile(r"endosymbiont", re.IGNORECASE),
    "group/clade/lineage in name": re.compile(r"\b(clade|group|lineage|complex)\b", re.IGNORECASE),
    "cf./aff.": re.compile(r"\b(cf|aff)\.", re.IGNORECASE),
}

idx2tax = load_mapping(MAPPING)
taxonomy = load_taxonomy_with_depth(set(idx2tax.values()), data_dir=DATA_DIR)

total = len(taxonomy)
print(f"\nTotal nodes in trained taxonomy: {total:,}\n")

ranks = Counter(node["rank"] for node in taxonomy.values())
print("Rank distribution:")
for rank, count in ranks.most_common():
    pct = 100 * count / total
    print(f"  {rank:<22}{count:>10,}  {pct:5.2f}%")

print()
print("Noise patterns in names (NOT mutually exclusive):")
all_noise_ids = set()
for label, regex in NOISE_PATTERNS.items():
    matches = [tid for tid, node in taxonomy.items() if regex.search(node["name"])]
    all_noise_ids.update(matches)
    print(f"  {label:<32}{len(matches):>10,}  {100 * len(matches) / total:5.2f}%")
print(f"  {'UNION (any noise pattern)':<32}{len(all_noise_ids):>10,}  {100 * len(all_noise_ids) / total:5.2f}%")

# Cross-tab: noise nodes by rank — what rank are the "sp." entries hiding at?
print()
print("Rank distribution of UNION-of-noise-patterns:")
noise_ranks = Counter(taxonomy[tid]["rank"] for tid in all_noise_ids)
for rank, count in noise_ranks.most_common():
    print(f"  {rank:<22}{count:>10,}")

# How many properly-leaf species nodes do we have, with a full lineage to phylum?
def has_rank_in_lineage(taxid, taxonomy, target_rank):
    visited = set()
    cur = taxid
    while cur in taxonomy and cur not in visited:
        visited.add(cur)
        if taxonomy[cur]["rank"] == target_rank:
            return True
        nxt = taxonomy[cur]["parent"]
        if nxt == cur:
            break
        cur = nxt
    return False

print()
print("Lineage completeness for SPECIES-rank nodes:")
species_ids = [tid for tid, node in taxonomy.items() if node["rank"] == "species"]
print(f"  Total species-rank nodes: {len(species_ids):,}")
for target in ["genus", "family", "order", "class", "phylum"]:
    n = sum(1 for tid in species_ids if has_rank_in_lineage(tid, taxonomy, target))
    print(f"    have {target:<8} ancestor: {n:,}  ({100 * n / len(species_ids):5.2f}%)")

# Of the NOISE-IN-NAME species, do they still have phylum lineage?
noise_species = [tid for tid in species_ids if tid in all_noise_ids]
print()
print(f"Noise-named species (e.g. 'Genus sp.'): {len(noise_species):,}")
for target in ["genus", "family", "order", "class", "phylum"]:
    n = sum(1 for tid in noise_species if has_rank_in_lineage(tid, taxonomy, target))
    print(f"    have {target:<8} ancestor: {n:,}  ({100 * n / max(1, len(noise_species)):5.2f}%)")
