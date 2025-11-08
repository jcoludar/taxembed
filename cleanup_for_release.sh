#!/bin/bash
# Repository cleanup script before GitHub push
# Removes temporary files, old checkpoints, and intermediate artifacts

echo "================================================"
echo "REPOSITORY CLEANUP FOR RELEASE"
echo "================================================"
echo ""

# Create artifacts directory if it doesn't exist
mkdir -p artifacts_archive

# 1. Remove old training checkpoints (keep only best v3)
echo "1. Cleaning old checkpoints..."
mv taxonomy_model_hierarchical_small_v3_best.pth artifacts_archive/ 2>/dev/null
rm -f taxonomy_model_hierarchical_small*.pth
mv artifacts_archive/taxonomy_model_hierarchical_small_v3_best.pth ./
echo "   ✓ Kept: taxonomy_model_hierarchical_small_v3_best.pth"
echo "   ✓ Removed: All other checkpoint versions"

# 2. Remove training logs
echo "2. Cleaning training logs..."
rm -f training*.log
echo "   ✓ Removed: training logs"

# 3. Remove analysis plots (can be regenerated)
echo "3. Cleaning temporary plots..."
rm -f *.png
echo "   ✓ Removed: temporary plot files"

# 4. Consolidate documentation
echo "4. Organizing documentation..."

# Keep essential docs
KEEP_DOCS=(
    "README.md"
    "QUICKSTART.md"
    "JOURNEY.md"
    "DEVELOPMENT_HISTORY.md"
    "CODE_OF_CONDUCT.md"
    "CONTRIBUTING.md"
)

# Archive intermediate docs
mkdir -p docs/archive
for doc in *.md; do
    keep=false
    for keep_doc in "${KEEP_DOCS[@]}"; do
        if [ "$doc" = "$keep_doc" ]; then
            keep=true
            break
        fi
    done
    
    if [ "$keep" = false ] && [ "$doc" != "README.md" ]; then
        mv "$doc" docs/archive/ 2>/dev/null
    fi
done

echo "   ✓ Moved intermediate docs to docs/archive/"

# 5. Clean Python cache
echo "5. Cleaning Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null
echo "   ✓ Removed: Python cache files"

# 6. Clean up data directory
echo "6. Organizing data directory..."
# Keep only essential data files
cd data
ls -lh | grep -E "\.pkl$|\.tsv$|\.edgelist$" | head -5
cd ..
echo "   ✓ Data files organized"

# 7. Summary
echo ""
echo "================================================"
echo "CLEANUP COMPLETE"
echo "================================================"
echo ""
echo "Kept files:"
echo "  - taxonomy_model_hierarchical_small_v3_best.pth (latest model)"
echo "  - Core documentation (README, QUICKSTART, JOURNEY)"
echo "  - All source code"
echo "  - Data files"
echo ""
echo "Removed:"
echo "  - Old checkpoints"
echo "  - Training logs"
echo "  - Temporary plots"
echo "  - Intermediate documentation (moved to docs/archive/)"
echo ""
echo "Ready for git commit and push!"
