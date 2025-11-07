#!/bin/bash
# Clean up repository: remove checkpoints, logs, and temporary files

set -e

echo "==========================================="
echo "REPOSITORY CLEANUP"
echo "==========================================="
echo ""

# Count files to be removed
checkpoint_count=$(find . -name "*.pth" -o -name "*.pth.*" | wc -l | tr -d ' ')
log_count=$(find . -maxdepth 1 -name "*.log" | wc -l | tr -d ' ')
png_count=$(find . -maxdepth 1 -name "*.png" | wc -l | tr -d ' ')

echo "Files to be removed:"
echo "  - Checkpoints (*.pth, *.pth.*): $checkpoint_count files"
echo "  - Logs (*.log): $log_count files"
echo "  - Visualizations (*.png): $png_count files"
echo ""

# Ask for confirmation
read -p "Continue with cleanup? (y/N) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cleanup cancelled."
    exit 0
fi

echo ""
echo "Removing checkpoint files..."
find . -name "*.pth" -delete
find . -name "*.pth.*" -delete
echo "✓ Removed $checkpoint_count checkpoint files"

echo ""
echo "Removing log files..."
find . -maxdepth 1 -name "*.log" -delete
echo "✓ Removed $log_count log files"

echo ""
echo "Removing visualization files..."
find . -maxdepth 1 -name "*.png" -delete
echo "✓ Removed $png_count PNG files"

echo ""
echo "Removing redundant scripts..."
# Keep only the consolidated visualization script
rm -f visualize_primates.py
rm -f visualize_primates_proper.py  
rm -f visualize_primates_small_only.py
rm -f visualize_by_taxonomy.py
rm -f visualize_trained_small_dataset.py
echo "✓ Removed redundant visualization scripts"

echo ""
echo "Removing old shell scripts..."
rm -f train-mammals.sh
rm -f train-nouns.sh
rm -f train_taxonomy.sh
rm -f train_taxonomy_quick.sh
echo "✓ Removed old training scripts"

echo ""
echo "Removing nohup.out..."
rm -f nohup.out
echo "✓ Removed nohup.out"

echo ""
echo "==========================================="
echo "✅ CLEANUP COMPLETE"
echo "==========================================="
echo ""
echo "Repository is now clean and organized!"
echo ""
echo "Remaining structure:"
echo "  scripts/          - Organized scripts"
echo "  src/taxembed/     - Source code"
echo "  tests/            - Unit tests"
echo "  data/             - Data files (gitignored)"
echo "  docs/             - Documentation"
echo ""
echo "To train a model:"
echo "  python embed.py -dset data/taxonomy_edges_small.mapped.edgelist ..."
echo ""
echo "To visualize embeddings:"
echo "  python scripts/visualize_embeddings.py <checkpoint.pth> --highlight primates"
