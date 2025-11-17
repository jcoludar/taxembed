#!/usr/bin/env python3
"""
Validate data files for Poincaré embedding training.

Checks:
1. Edgelist files have no headers
2. All values are numeric
3. Mapping files are consistent
4. Node indices are sequential
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from taxembed.utils.data_validation import coverage_from_indices, load_mapping


def validate_edgelist(filepath):
    """Validate an edgelist file."""
    print(f"\n📋 Validating edgelist: {filepath}")
    
    issues = []
    edges = []
    nodes = set()
    
    with open(filepath, 'r') as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) != 2:
                issues.append(f"Line {ln}: Expected 2 values, got {len(parts)}")
                continue
            
            # Check if numeric
            try:
                u, v = int(parts[0]), int(parts[1])
                edges.append((u, v))
                nodes.add(u)
                nodes.add(v)
            except ValueError:
                issues.append(f"Line {ln}: Non-numeric values: {parts}")
    
    # Check for headers
    if issues and issues[0].startswith("Line 1"):
        print("  ⚠️  Possible header line detected")
    
    # Statistics
    print(f"  ✓ Total edges: {len(edges):,}")
    print(f"  ✓ Unique nodes: {len(nodes):,}")
    
    if len(nodes) > 0:
        min_node = min(nodes)
        max_node = max(nodes)
        print(f"  ✓ Node range: {min_node} to {max_node}")
        
        # Check if sequential
        if max_node - min_node + 1 == len(nodes):
            print(f"  ✓ Nodes are sequential")
        else:
            expected = max_node - min_node + 1
            print(f"  ⚠️  Nodes are NOT sequential (expected {expected}, got {len(nodes)})")
    
    # Report issues
    if issues:
        print(f"\n  ❌ Found {len(issues)} issues:")
        for issue in issues[:10]:  # Show first 10
            print(f"     {issue}")
        if len(issues) > 10:
            print(f"     ... and {len(issues) - 10} more")
        return False
    else:
        print(f"  ✅ No issues found")
        return True


def validate_mapping(filepath):
    """Validate a mapping file."""
    print(f"\n📋 Validating mapping: {filepath}")
    
    try:
        df = load_mapping(Path(filepath))
    except Exception as e:
        print(f"  ❌ Failed to read: {e}")
        return False
    
    print(f"  ✓ Total mappings: {len(df):,}")
    
    invalid_mask = ~df["taxid"].str.isnumeric()
    non_numeric = df[invalid_mask]

    if not non_numeric.empty:
        print(f"  ❌ Found {len(non_numeric)} non-numeric TaxIDs:")
        for idx, row in non_numeric.head(10).iterrows():
            print(f"     Row {idx}: taxid='{row['taxid']}' is not numeric")
        return False
    
    # Check if indices are sequential
    indices = sorted(df['idx'].values)
    if indices == list(range(len(indices))):
        print(f"  ✓ Indices are sequential (0 to {len(indices)-1})")
    else:
        print(f"  ⚠️  Indices are NOT sequential")
        print(f"     Expected: 0 to {len(indices)-1}")
        print(f"     Got: {indices[0]} to {indices[-1]}")
    
    # Check for duplicates
    dup_taxids = df[df.duplicated('taxid', keep=False)]
    dup_indices = df[df.duplicated('idx', keep=False)]
    
    if len(dup_taxids) > 0:
        print(f"  ❌ Found {len(dup_taxids)} duplicate TaxIDs")
        return False
    if len(dup_indices) > 0:
        print(f"  ❌ Found {len(dup_indices)} duplicate indices")
        return False
    
    print(f"  ✅ No issues found")
    return True


def validate_consistency(edgelist_file, mapping_file):
    """Validate consistency between edgelist and mapping."""
    print(f"\n📋 Validating consistency...")
    
    # Load edgelist nodes
    nodes = set()
    with open(edgelist_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 2:
                try:
                    nodes.add(int(parts[0]))
                    nodes.add(int(parts[1]))
                except ValueError:
                    pass
    
    # Load mapping
    df = load_mapping(Path(mapping_file))
    
    # Filter out non-numeric taxids
    numeric_df = df[df['taxid'].str.isnumeric()]
    mapped_indices = set(numeric_df['idx'].values)
    
    print(f"  Edgelist nodes: {len(nodes):,}")
    print(f"  Mapping indices: {len(mapped_indices):,}")
    
    # Check if all edgelist nodes are in mapping
    unmapped = nodes - mapped_indices
    if unmapped:
        print(f"  ⚠️  {len(unmapped)} nodes in edgelist not in mapping")
        print(f"     Examples: {sorted(list(unmapped))[:10]}")
    else:
        print(f"  ✓ All edgelist nodes are in mapping")
    
    report = coverage_from_indices(numeric_df, nodes)
    if report.is_perfect:
        print(f"  ✓ All mapping indices are used in edgelist")
    else:
        print(
            f"  ⚠️  Coverage gap: {report.missing_count} indices missing "
            f"({report.coverage_ratio * 100:.1f}% covered)"
        )
        sample = sorted(report.missing_indices)[:10]
        if sample:
            print(f"     Examples: {sample}")
    
    return len(unmapped) == 0


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_data.py <dataset_name>")
        print("Example: python validate_data.py small")
        print("         python validate_data.py full")
        sys.exit(1)
    
    dataset = sys.argv[1]
    data_dir = Path(__file__).parent.parent / 'data'
    
    print(f"{'='*60}")
    print(f"DATA VALIDATION - {dataset.upper()} DATASET")
    print(f"{'='*60}")
    
    if dataset == 'full':
        edgelist = data_dir / 'taxonomy_edges.mapped.edgelist'
        mapping = data_dir / 'taxonomy_edges.mapping.tsv'
    elif dataset == 'small':
        edgelist = data_dir / 'taxonomy_edges_small.mapped.edgelist'
        mapping = data_dir / 'taxonomy_edges_small.mapping.tsv'
    else:
        print(f"Unknown dataset: {dataset}")
        sys.exit(1)
    
    # Validate files
    results = []
    
    if edgelist.exists():
        results.append(validate_edgelist(edgelist))
    else:
        print(f"\n❌ File not found: {edgelist}")
        results.append(False)
    
    if mapping.exists():
        results.append(validate_mapping(mapping))
    else:
        print(f"\n❌ File not found: {mapping}")
        results.append(False)
    
    if edgelist.exists() and mapping.exists():
        results.append(validate_consistency(edgelist, mapping))
    
    # Summary
    print(f"\n{'='*60}")
    if all(results):
        print("✅ ALL CHECKS PASSED")
    else:
        print("❌ SOME CHECKS FAILED")
    print(f"{'='*60}\n")
    
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
