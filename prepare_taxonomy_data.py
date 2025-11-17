#!/usr/bin/env python3
"""
Parse NCBI taxonomy data and generate edge list for Poincaré embeddings.
Extracts parent-child relationships from nodes.dmp and creates a CSV file.
"""

import sys
import shutil
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd
from tqdm import tqdm

def parse_nodes_dmp(nodes_file):
    """
    Parse nodes.dmp file to extract tax_id and parent_tax_id.
    
    Format: tax_id | parent_tax_id | rank | ...
    Delimiter: \t|\t
    """
    print(f"Parsing {nodes_file}...")
    
    edges = []
    with open(nodes_file, 'r') as f:
        for line in tqdm(f, desc="Reading nodes"):
            # Split by \t|\t delimiter
            parts = line.strip().split('\t|\t')
            if len(parts) >= 2:
                try:
                    tax_id = int(parts[0].strip())
                    parent_tax_id = int(parts[1].strip())
                    
                    # Skip self-loops (root node)
                    if tax_id != parent_tax_id:
                        edges.append({
                            'id1': tax_id,
                            'id2': parent_tax_id,
                            'weight': 1.0
                        })
                except (ValueError, IndexError):
                    continue
    
    return edges

def parse_names_dmp(names_file):
    """
    Parse names.dmp file to extract scientific names.
    
    Format: tax_id | name | unique_name | name_class
    Delimiter: \t|\t
    """
    print(f"Parsing {names_file}...")
    
    names_map = {}
    with open(names_file, 'r') as f:
        for line in tqdm(f, desc="Reading names"):
            parts = line.strip().split('\t|\t')
            if len(parts) >= 4:
                try:
                    tax_id = int(parts[0].strip())
                    name = parts[1].strip()
                    name_class = parts[3].strip()
                    
                    # Prefer scientific names, but keep first occurrence
                    if tax_id not in names_map:
                        names_map[tax_id] = name
                    elif name_class == 'scientific name':
                        names_map[tax_id] = name
                except (ValueError, IndexError):
                    continue
    
    return names_map

TAXDUMP_URL = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/new_taxdump/new_taxdump.tar.gz"


def ensure_taxdump(data_dir: Path) -> tuple[Path, Path, Path | None]:
    """Download/extract the NCBI taxdump files if they are missing."""

    nodes_file = data_dir / "nodes.dmp"
    names_file = data_dir / "names.dmp"
    merged_file = data_dir / "merged.dmp"

    if nodes_file.exists() and names_file.exists():
        return nodes_file, names_file, merged_file if merged_file.exists() else None

    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / "new_taxdump.tar.gz"

    print(f"Downloading NCBI taxonomy → {archive_path}")
    with urllib.request.urlopen(TAXDUMP_URL) as response, archive_path.open("wb") as out_f:
        shutil.copyfileobj(response, out_f)

    print("Extracting taxdump archive...")
    with tarfile.open(archive_path, "r:gz") as tar:
        for member_name in ("nodes.dmp", "names.dmp", "merged.dmp"):
            member = tar.getmember(member_name)
            tar.extract(member, path=data_dir)

    print("  ✓ nodes.dmp, names.dmp, merged.dmp ready")
    return nodes_file, names_file, merged_file


def main():
    data_dir = Path(__file__).parent / "data"
    nodes_file, names_file, _ = ensure_taxdump(data_dir)
    output_file = data_dir / "taxonomy_edges.csv"
    
    # Parse taxonomy data
    edges = parse_nodes_dmp(nodes_file)
    names_map = parse_names_dmp(names_file)
    
    print(f"\nExtracted {len(edges)} edges from taxonomy")
    print(f"Extracted {len(names_map)} taxonomy names")
    
    # Create DataFrame
    df = pd.DataFrame(edges)
    
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Sample edges:\n{df.head(10)}")
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"\n✓ Saved edge list to {output_file}")
    
    # Also create edgelist format (no header, whitespace-separated)
    edgelist_file = output_file.with_suffix('.edgelist')
    with open(edgelist_file, 'w') as f:
        for _, row in df.iterrows():
            f.write(f"{int(row['id1'])} {int(row['id2'])}\n")
    print(f"✓ Saved edgelist to {edgelist_file}")
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  Total edges: {len(df)}")
    print(f"  Unique parent nodes: {df['id2'].nunique()}")
    print(f"  Unique child nodes: {df['id1'].nunique()}")
    unique_nodes = set(df['id1'].unique()) | set(df['id2'].unique())
    print(f"  Total unique nodes: {len(unique_nodes)}")
    
    # Show some example edges with names
    print(f"\nExample edges with names:")
    sample_edges = df.head(10)
    for _, row in sample_edges.iterrows():
        child_id = int(row['id1'])
        parent_id = int(row['id2'])
        child_name = names_map.get(child_id, f"Unknown_{child_id}")
        parent_name = names_map.get(parent_id, f"Unknown_{parent_id}")
        print(f"  {child_name} (id={child_id}) → {parent_name} (id={parent_id})")

if __name__ == '__main__':
    main()
