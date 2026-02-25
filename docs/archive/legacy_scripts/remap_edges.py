#!/usr/bin/env python3
"""
Remap TaxIDs in edge list to sequential indices (0, 1, 2, ...).

This script reads an edge list file with TaxIDs and creates:
1. A remapped edge list with sequential indices
2. A mapping file (TaxID -> index)
"""
import sys
import collections

def main():
    if len(sys.argv) < 2:
        print("Usage: python remap_edges.py <input_edgelist>")
        sys.exit(1)
    
    in_path = sys.argv[1]
    out_edges = in_path.replace(".edgelist", ".mapped.edgelist")
    out_map = in_path.replace(".edgelist", ".mapping.tsv")
    
    nodes = collections.OrderedDict()  # preserves order of first appearance
    edges = []
    
    print(f"Reading from: {in_path}")
    
    with open(in_path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Skip header lines (common patterns)
            if line.startswith("id1") or line.startswith("taxid") or line.startswith("#"):
                print(f"Skipping header line {ln}: {line}")
                continue
            
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Bad line {ln}: {line!r}")
            
            u, v = parts
            
            # Skip if both are non-numeric (likely header)
            try:
                int(u)
                int(v)
            except ValueError:
                print(f"Skipping non-numeric line {ln}: {line}")
                continue
            
            # Register nodes
            if u not in nodes:
                nodes[u] = len(nodes)
            if v not in nodes:
                nodes[v] = len(nodes)
            edges.append((nodes[u], nodes[v]))
    
    # Write remapped edges (no header)
    print(f"Writing to: {out_edges}")
    with open(out_edges, "w") as fo:
        for u, v in edges:
            fo.write(f"{u} {v}\n")
    
    # Write mapping file (with header)
    print(f"Writing to: {out_map}")
    with open(out_map, "w") as fm:
        fm.write("taxid\tidx\n")
        for taxid, idx in nodes.items():
            fm.write(f"{taxid}\t{idx}\n")
    
    print(f"\n✓ Complete!")
    print(f"  Nodes: {len(nodes):,}")
    print(f"  Edges: {len(edges):,}")
    print(f"  Files: {out_edges}, {out_map}")

if __name__ == "__main__":
    main()
