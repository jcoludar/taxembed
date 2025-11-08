#!/usr/bin/env python3
"""
Phase 1: Build transitive closure of taxonomy for training.

Instead of training on 100K parent-child edges,
we'll train on ALL ancestor-descendant pairs.
This is CRITICAL for hyperbolic embeddings to learn hierarchy.
"""

import pandas as pd
import numpy as np
from collections import defaultdict, deque
import pickle


def load_ncbi_taxonomy():
    """Load full NCBI taxonomy structure."""
    print("Loading NCBI taxonomy...")
    
    # Load taxonomy structure
    taxonomy = {}
    with open("data/nodes.dmp", "r") as f:
        for line in f:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 5:
                taxid = int(parts[0])
                parent = int(parts[1])
                rank = parts[2]
                taxonomy[taxid] = {
                    "parent": parent,
                    "rank": rank,
                    "children": [],
                    "depth": None
                }
    
    # Build children lists
    for taxid, info in taxonomy.items():
        parent = info["parent"]
        if parent != taxid and parent in taxonomy:  # Not root
            taxonomy[parent]["children"].append(taxid)
    
    print(f"  ✓ Loaded {len(taxonomy):,} nodes")
    return taxonomy


def load_small_dataset_taxids():
    """Load TaxIDs that are in the small dataset."""
    print("Loading small dataset TaxIDs...")
    
    # Load mapping
    df = pd.read_csv("data/taxonomy_edges_small.mapping.tsv", 
                     sep="\t", header=None, names=["idx", "taxid"])
    
    # Get valid numeric TaxIDs
    valid_taxids = set()
    for taxid_str in df["taxid"]:
        try:
            valid_taxids.add(int(taxid_str))
        except ValueError:
            pass
    
    print(f"  ✓ Found {len(valid_taxids):,} unique TaxIDs in small dataset")
    return valid_taxids


def compute_depths(taxonomy):
    """Compute depth for each node (distance from root)."""
    print("Computing taxonomic depths...")
    
    def get_depth(taxid, visited=None):
        if visited is None:
            visited = set()
        
        if taxid not in taxonomy:
            return 0
        
        if taxonomy[taxid]["depth"] is not None:
            return taxonomy[taxid]["depth"]
        
        if taxid in visited:  # Cycle detection
            return 0
        
        visited.add(taxid)
        parent = taxonomy[taxid]["parent"]
        
        if parent == taxid:  # Root
            taxonomy[taxid]["depth"] = 0
        else:
            taxonomy[taxid]["depth"] = get_depth(parent, visited) + 1
        
        return taxonomy[taxid]["depth"]
    
    for taxid in taxonomy:
        if taxonomy[taxid]["depth"] is None:
            get_depth(taxid)
    
    depths = [info["depth"] for info in taxonomy.values() if info["depth"] is not None]
    print(f"  ✓ Max depth: {max(depths)}, Mean depth: {np.mean(depths):.1f}")


def find_all_ancestors(taxid, taxonomy):
    """Find all ancestors of a node (path to root)."""
    ancestors = []
    current = taxid
    visited = set()
    
    while current in taxonomy and current not in visited:
        parent = taxonomy[current]["parent"]
        if parent == current:  # Root
            break
        ancestors.append(parent)
        visited.add(current)
        current = parent
    
    return ancestors


def build_transitive_closure(taxonomy, valid_taxids):
    """
    Build transitive closure: ALL (ancestor, descendant) pairs.
    
    For each node in valid_taxids, add edges to ALL its ancestors.
    This ensures the model learns the full hierarchy, not just local parent-child.
    """
    print("\nBuilding transitive closure...")
    print("This creates ALL ancestor-descendant pairs for proper hierarchy learning")
    
    # For each valid node, find all its ancestors
    all_pairs = []
    depth_counts = defaultdict(int)
    
    for i, taxid in enumerate(valid_taxids, 1):
        if i % 10000 == 0:
            print(f"  Processed {i:,}/{len(valid_taxids):,} nodes...")
        
        if taxid not in taxonomy:
            continue
        
        # Find all ancestors
        ancestors = find_all_ancestors(taxid, taxonomy)
        
        # Only keep ancestors that are also in valid_taxids
        valid_ancestors = [a for a in ancestors if a in valid_taxids]
        
        # Add all (ancestor, descendant) pairs
        for ancestor in valid_ancestors:
            all_pairs.append((ancestor, taxid))
            
            # Track depth difference
            depth_diff = taxonomy[taxid]["depth"] - taxonomy[ancestor]["depth"]
            depth_counts[depth_diff] += 1
    
    print(f"\n  ✓ Generated {len(all_pairs):,} ancestor-descendant pairs")
    print(f"  (vs. {len(valid_taxids):,} nodes in dataset)")
    
    # Show depth distribution
    print(f"\n  Depth distribution of pairs:")
    for depth_diff in sorted(depth_counts.keys())[:10]:
        print(f"    Depth diff {depth_diff}: {depth_counts[depth_diff]:,} pairs")
    
    return all_pairs


def create_training_data(all_pairs, valid_taxids, taxonomy):
    """
    Create training dataset with metadata for depth-aware sampling.
    
    Format: (ancestor_idx, descendant_idx, depth_diff, ancestor_depth, descendant_depth)
    """
    print("\nCreating training dataset with metadata...")
    
    # Load TaxID → Index mapping
    # File format: taxid\tidx (header present)
    df = pd.read_csv("data/taxonomy_edges_small.mapping.tsv",
                     sep="\t")
    
    taxid_to_idx = {}
    for _, row in df.iterrows():
        try:
            taxid = int(row["taxid"])
            idx = int(row["idx"])
            taxid_to_idx[taxid] = idx
        except ValueError:
            pass
    
    print(f"  ✓ Loaded {len(taxid_to_idx):,} TaxID to index mappings")
    
    # Convert pairs to indices with metadata
    training_data = []
    skipped = 0
    
    for ancestor_taxid, descendant_taxid in all_pairs:
        # Skip if either TaxID not in mapping
        if ancestor_taxid not in taxid_to_idx or descendant_taxid not in taxid_to_idx:
            skipped += 1
            continue
        
        ancestor_idx = taxid_to_idx[ancestor_taxid]
        descendant_idx = taxid_to_idx[descendant_taxid]
        
        # Get depths
        ancestor_depth = taxonomy[ancestor_taxid]["depth"]
        descendant_depth = taxonomy[descendant_taxid]["depth"]
        depth_diff = descendant_depth - ancestor_depth
        
        training_data.append({
            "ancestor_idx": ancestor_idx,
            "descendant_idx": descendant_idx,
            "depth_diff": depth_diff,
            "ancestor_depth": ancestor_depth,
            "descendant_depth": descendant_depth,
            "ancestor_taxid": ancestor_taxid,
            "descendant_taxid": descendant_taxid
        })
    
    print(f"  ✓ Created {len(training_data):,} training pairs")
    print(f"  ✗ Skipped {skipped:,} pairs (TaxIDs not in mapping)")
    
    return training_data


def save_training_data(training_data, output_file):
    """Save training data in multiple formats."""
    
    # Save as pickle (full metadata)
    pickle_file = output_file.replace('.tsv', '.pkl')
    with open(pickle_file, 'wb') as f:
        pickle.dump(training_data, f)
    print(f"\n  ✓ Saved full data to: {pickle_file}")
    
    # Save as TSV (for compatibility with existing training code)
    df = pd.DataFrame(training_data)
    df.to_csv(output_file, sep='\t', index=False)
    print(f"  ✓ Saved TSV to: {output_file}")
    
    # Save simple edgelist (ancestor_idx descendant_idx)
    edgelist_file = output_file.replace('.tsv', '.edgelist')
    with open(edgelist_file, 'w') as f:
        for item in training_data:
            f.write(f"{item['ancestor_idx']} {item['descendant_idx']}\n")
    print(f"  ✓ Saved edgelist to: {edgelist_file}")
    
    # Print statistics
    depths = [item['depth_diff'] for item in training_data]
    print(f"\n  Statistics:")
    print(f"    Total pairs: {len(training_data):,}")
    print(f"    Depth differences: min={min(depths)}, max={max(depths)}, mean={np.mean(depths):.1f}")
    
    depth_bins = defaultdict(int)
    for d in depths:
        if d == 1:
            depth_bins["1 (parent-child)"] += 1
        elif d == 2:
            depth_bins["2 (grandparent)"] += 1
        elif d <= 5:
            depth_bins["3-5 (ancestors)"] += 1
        elif d <= 10:
            depth_bins["6-10 (distant)"] += 1
        else:
            depth_bins[">10 (very distant)"] += 1
    
    print(f"\n  Breakdown by depth:")
    for key in sorted(depth_bins.keys()):
        pct = depth_bins[key] / len(training_data) * 100
        print(f"    {key:25s}: {depth_bins[key]:8,} ({pct:5.1f}%)")


def main():
    print("="*80)
    print("PHASE 1: TRANSITIVE CLOSURE DATA PREPARATION")
    print("="*80)
    print()
    print("Goal: Build ALL ancestor-descendant pairs for proper hierarchy learning")
    print()
    
    # Step 1: Load NCBI taxonomy
    taxonomy = load_ncbi_taxonomy()
    
    # Step 2: Load small dataset TaxIDs
    valid_taxids = load_small_dataset_taxids()
    
    # Step 3: Compute depths
    compute_depths(taxonomy)
    
    # Step 4: Build transitive closure
    all_pairs = build_transitive_closure(taxonomy, valid_taxids)
    
    # Step 5: Create training data with metadata
    training_data = create_training_data(all_pairs, valid_taxids, taxonomy)
    
    # Step 6: Save
    output_file = "data/taxonomy_edges_small_transitive.tsv"
    save_training_data(training_data, output_file)
    
    print("\n" + "="*80)
    print("✅ PHASE 1 COMPLETE")
    print("="*80)
    print(f"\nTraining data ready:")
    print(f"  - {output_file}")
    print(f"  - data/taxonomy_edges_small_transitive.pkl (with metadata)")
    print(f"  - data/taxonomy_edges_small_transitive.edgelist (for training)")
    print()
    print("Next: Phase 2 - Create training script with depth-aware sampling")


if __name__ == "__main__":
    main()
