#!/bin/bash
# Regenerate clean training data from NCBI taxonomy

set -e  # Exit on error

echo "========================================="
echo "REGENERATING CLEAN TRAINING DATA"
echo "========================================="
echo ""

# Check if data directory exists
if [ ! -d "data" ]; then
    echo "❌ data/ directory not found"
    exit 1
fi

# Check if nodes.dmp exists
if [ ! -f "data/nodes.dmp" ]; then
    echo "❌ data/nodes.dmp not found"
    echo "Please download NCBI taxonomy first:"
    echo "  wget https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/new_taxdump/new_taxdump.tar.gz"
    echo "  tar -xzf new_taxdump.tar.gz -C data/"
    exit 1
fi

echo "Step 1: Parsing NCBI taxonomy..."
python prepare_taxonomy_data.py
echo ""

echo "Step 2: Creating small subset..."
# Create small subset (first 100k edges)
if [ -f "data/taxonomy_edges.edgelist" ]; then
    head -n 100001 data/taxonomy_edges.edgelist > data/taxonomy_edges_small.edgelist
    echo "✓ Created data/taxonomy_edges_small.edgelist"
else
    echo "⚠️  data/taxonomy_edges.edgelist not found, skipping small subset"
fi
echo ""

echo "Step 3: Remapping edges (full dataset)..."
python remap_edges.py data/taxonomy_edges.edgelist
echo ""

echo "Step 4: Remapping edges (small dataset)..."
python remap_edges.py data/taxonomy_edges_small.edgelist
echo ""

echo "Step 5: Validating data..."
python scripts/validate_data.py full
python scripts/validate_data.py small
echo ""

echo "========================================="
echo "✅ DATA REGENERATION COMPLETE"
echo "========================================="
echo ""
echo "Files created:"
echo "  - data/taxonomy_edges.csv"
echo "  - data/taxonomy_edges.edgelist"
echo "  - data/taxonomy_edges.mapped.edgelist"
echo "  - data/taxonomy_edges.mapping.tsv"
echo "  - data/taxonomy_edges_small.edgelist"
echo "  - data/taxonomy_edges_small.mapped.edgelist"
echo "  - data/taxonomy_edges_small.mapping.tsv"
echo ""
echo "Ready to train!"
