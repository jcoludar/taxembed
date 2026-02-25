#!/usr/bin/env python3
"""Quick script to check trained model status."""

import torch
import os
from glob import glob

print("="*80)
print("TRAINED MODEL SUMMARY")
print("="*80)

# Find all model files
model_files = sorted(glob("taxonomy_model_small*.pth"))

if not model_files:
    print("❌ No trained models found!")
    exit(1)

print(f"\nFound {len(model_files)} model files:\n")

best_loss = float('inf')
best_file = None

for model_file in model_files:
    size_mb = os.path.getsize(model_file) / (1024**2)
    
    # Load checkpoint
    try:
        ckpt = torch.load(model_file, map_location='cpu')
        
        # Extract info
        epoch = ckpt.get('epoch', '?')
        loss = ckpt.get('loss', None)
        embeddings = ckpt.get('embeddings', None)
        
        if embeddings is not None:
            n_nodes = embeddings.shape[0]
            dim = embeddings.shape[1]
            norms = embeddings.norm(dim=1)
            max_norm = norms.max().item()
            mean_norm = norms.mean().item()
            outside = (norms >= 1.0).sum().item()
        else:
            n_nodes = dim = max_norm = mean_norm = outside = "?"
        
        # Track best
        if loss is not None and loss < best_loss:
            best_loss = loss
            best_file = model_file
        
        # Print info
        status = "✅" if outside == 0 else f"⚠️ {outside} outside"
        loss_str = f"{loss:.6f}" if loss is not None else "N/A"
        
        print(f"  {os.path.basename(model_file):35s} | "
              f"Epoch {epoch:>3} | "
              f"Loss: {loss_str:>10} | "
              f"Nodes: {n_nodes:>6,} | "
              f"Dim: {dim:>2} | "
              f"MaxNorm: {max_norm:.4f} | "
              f"{status}")
        
    except Exception as e:
        print(f"  {os.path.basename(model_file):35s} | ERROR: {e}")

print("\n" + "="*80)
if best_file:
    print(f"🏆 BEST MODEL: {best_file}")
    print(f"   Loss: {best_loss:.6f}")
    
    # Create symlink or copy
    best_link = "taxonomy_model_small_best.pth"
    if os.path.exists(best_link):
        os.remove(best_link)
    
    # Create a copy
    import shutil
    shutil.copy(best_file, best_link)
    print(f"   ✓ Copied to: {best_link}")
else:
    print("⚠️  Could not determine best model")

print("="*80)
print("\nRECOMMENDED NEXT STEPS:")
print("  1. Analyze hierarchy quality:")
print("     python analyze_hierarchy_hyperbolic.py")
print("\n  2. Visualize embeddings:")
print("     python scripts/visualize_embeddings.py taxonomy_model_small_best.pth --highlight mammals")
print("\n  3. Query nearest neighbors:")
print("     python query_embeddings.py taxonomy_model_small_best.pth")
print("="*80)
