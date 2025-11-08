#!/bin/bash
# Simple training monitor - shows last 20 lines every 30 seconds

echo "🔍 Monitoring training progress..."
echo "Press Ctrl+C to stop monitoring (training continues)"
echo ""

while true; do
    clear
    echo "=================================================="
    echo "TRAINING MONITOR - $(date '+%H:%M:%S')"
    echo "=================================================="
    echo ""
    
    # Show last 20 lines of log
    tail -20 training.log
    
    echo ""
    echo "=================================================="
    echo "Checkpoints:"
    ls -lht taxonomy_model_hierarchical_small*.pth 2>/dev/null | head -5
    
    echo ""
    echo "Press Ctrl+C to stop monitoring"
    sleep 30
done
