#!/bin/bash
# Repository cleanup script
# Removes intermediate files, keeps small model (epoch 28) results

set -e

echo "================================================================================"
echo "REPOSITORY CLEANUP"
echo "================================================================================"
echo ""

# Remove animals model intermediate files (keep only final results for reference)
echo "🗑️  Removing animals model intermediate files..."
rm -f taxonomy_model_animals_epoch*.pth
rm -f animals_taxonomy_*.png
echo "   ✓ Removed animals intermediate files"

# Remove build scripts for full/animals (unsuccessful attempts)
echo ""
echo "🗑️  Removing unsuccessful build/train scripts..."
rm -f build_transitive_closure_full.py
rm -f build_transitive_closure_animals.py
rm -f train_full.py
rm -f train_animals.py
rm -f train_animals_fast.py
rm -f continue_animals_training.py
rm -f visualize_animals.py
rm -f visualize_animals_hyperbolic.py
echo "   ✓ Removed unsuccessful attempt scripts"

# Remove temporary check/plot scripts
echo ""
echo "🗑️  Removing temporary analysis scripts..."
rm -f check_animals_model.py
rm -f plot_best_epoch.py
echo "   ✓ Removed temporary scripts"

# Move/organize visualization files for small model
echo ""
echo "📂 Organizing small model visualizations..."
if [ -f "best_epoch_analysis_epoch28.png" ]; then
    mv best_epoch_analysis_epoch28.png small_model_28epoch/
    echo "   ✓ Moved best_epoch_analysis_epoch28.png to small_model_28epoch/"
fi

if [ -f "umap_taxonomy_model_small_best_mammals_highlighted.png" ]; then
    mv umap_taxonomy_model_small_best_mammals_highlighted.png small_model_28epoch/
    echo "   ✓ Moved mammals UMAP to small_model_28epoch/"
fi

# Keep taxonomy_model_animals_best.pth as a reference (final result)
echo ""
echo "📦 Keeping reference files..."
echo "   ✓ small_model_28epoch/ (complete, epoch 28 best)"
echo "   ✓ taxonomy_model_animals_best.pth (reference: animals epoch 4)"

# List what's preserved
echo ""
echo "================================================================================"
echo "PRESERVED FILES"
echo "================================================================================"
echo ""
echo "📁 small_model_28epoch/"
ls -lh small_model_28epoch/*.pth small_model_28epoch/*.png 2>/dev/null | awk '{print "   " $9 " (" $5 ")"}'

echo ""
echo "📄 Reference files:"
ls -lh taxonomy_model_animals_best.pth 2>/dev/null | awk '{print "   " $9 " (" $5 ")"}'

echo ""
echo "================================================================================"
echo "✅ CLEANUP COMPLETE"
echo "================================================================================"
echo ""
echo "Repository is now clean and ready for commit!"
