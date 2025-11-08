#!/usr/bin/env python3
"""
Check the composition of organisms in the dataset by taxonomic group.
"""

import pandas as pd
from collections import defaultdict

def load_taxonomy_tree(valid_taxids):
    """Load taxonomy tree filtered to valid TaxIDs."""
    names = {}
    with open("data/names.dmp", "r") as f:
        for line in f:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 4 and parts[3] == "scientific name":
                taxid = int(parts[0])
                if taxid in valid_taxids:
                    names[taxid] = parts[1]
    
    nodes = {}
    with open("data/nodes.dmp", "r") as f:
        for line in f:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 5:
                taxid = int(parts[0])
                if taxid in valid_taxids:
                    parent = int(parts[1])
                    rank = parts[2]
                    nodes[taxid] = {"parent": parent, "rank": rank, "name": names.get(taxid, "")}
    
    return nodes


def find_group_descendants(nodes, root_taxid):
    """Find all descendants of a root taxid."""
    descendants = set()
    
    def find_desc(taxid):
        descendants.add(taxid)
        for child_id, child_info in nodes.items():
            if child_info["parent"] == taxid and child_id not in descendants:
                find_desc(child_id)
    
    if root_taxid in nodes:
        find_desc(root_taxid)
    
    return descendants


def main():
    # Load mapping
    df = pd.read_csv("data/taxonomy_edges_small.mapping.tsv", sep="\t", header=None, names=["idx", "taxid"])
    valid_taxids = set(int(x) for x in df["taxid"] if str(x).isnumeric())
    
    print(f"Total organisms in small dataset: {len(valid_taxids):,}\n")
    
    # Load taxonomy
    nodes = load_taxonomy_tree(valid_taxids)
    
    # Check major taxonomic groups
    groups = {
        "Primates": 9443,
        "Mammals": 40674,
        "Vertebrates": 7742,
        "Bacteria": 2,
        "Archaea": 2157,
        "Fungi": 4751,
        "Plants (Viridiplantae)": 33090,
        "Insects": 50557,
        "Rodents": 9989,
        "Nematodes": 6231,
        "Arthropods": 6656,
        "Metazoa (Animals)": 33208,
    }
    
    print("=" * 70)
    print("DATASET COMPOSITION BY TAXONOMIC GROUP")
    print("=" * 70)
    
    for group_name, root_taxid in groups.items():
        descendants = find_group_descendants(nodes, root_taxid)
        count = len(descendants)
        percentage = (count / len(valid_taxids)) * 100
        print(f"{group_name:30s}: {count:6,} organisms ({percentage:5.2f}%)")
    
    print("=" * 70)
    
    # Check top-level domains
    print("\nTop-level taxonomy distribution:")
    rank_counts = defaultdict(int)
    for taxid, info in nodes.items():
        rank = info.get("rank", "unknown")
        rank_counts[rank] += 1
    
    for rank, count in sorted(rank_counts.items(), key=lambda x: -x[1])[:15]:
        percentage = (count / len(valid_taxids)) * 100
        print(f"  {rank:20s}: {count:6,} ({percentage:5.2f}%)")


if __name__ == "__main__":
    main()
