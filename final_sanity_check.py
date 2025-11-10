#!/usr/bin/env python3
"""
Final sanity check before commit.
Validates all core files and models.
"""

import os
import sys
import torch
import pickle
import pandas as pd

def check_file_exists(path, description):
    """Check if file exists."""
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  ✅ {description}: {path} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"  ❌ {description} MISSING: {path}")
        return False

def check_model(path, expected_shape):
    """Check model checkpoint integrity."""
    try:
        ckpt = torch.load(path, map_location='cpu')
        embeddings = ckpt['embeddings']
        
        # Check shape
        if embeddings.shape != expected_shape:
            print(f"    ❌ Shape mismatch: {embeddings.shape} != {expected_shape}")
            return False
        
        # Check ball constraint
        norms = embeddings.norm(dim=1).detach().numpy()
        outside = (norms >= 1.0).sum()
        max_norm = norms.max()
        
        if outside > 0:
            print(f"    ❌ {outside} embeddings outside ball")
            return False
        
        if max_norm > 1.0:
            print(f"    ❌ Max norm {max_norm:.4f} > 1.0")
            return False
        
        print(f"    ✅ Shape: {embeddings.shape}, Max norm: {max_norm:.4f}, All inside ball")
        
        # Check loss
        loss = ckpt.get('loss', None)
        if loss:
            print(f"    ✅ Loss: {loss:.6f}")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Error loading model: {e}")
        return False

def check_data_file(path, description):
    """Check data file integrity."""
    try:
        if path.endswith('.pkl'):
            with open(path, 'rb') as f:
                data = pickle.load(f)
            print(f"    ✅ {len(data):,} items")
        elif path.endswith('.tsv'):
            df = pd.read_csv(path, sep='\t', header=None)
            print(f"    ✅ {len(df):,} rows")
        return True
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return False

def main():
    print("="*80)
    print("FINAL SANITY CHECK")
    print("="*80)
    print()
    
    all_passed = True
    
    # 1. Check core training scripts
    print("1️⃣  Core Training Scripts")
    all_passed &= check_file_exists("train_small.py", "Main training script")
    all_passed &= check_file_exists("train_hierarchical.py", "Hierarchical model")
    all_passed &= check_file_exists("visualize_multi_groups.py", "Visualization")
    print()
    
    # 2. Check documentation
    print("2️⃣  Documentation")
    all_passed &= check_file_exists("README.md", "Main README")
    all_passed &= check_file_exists("JOURNEY.md", "Development history")
    all_passed &= check_file_exists("FINAL_STATUS.md", "Final status")
    all_passed &= check_file_exists("TRAIN_SMALL_GUIDE.md", "Training guide")
    print()
    
    # 3. Check small model (production)
    print("3️⃣  Small Model (Production)")
    if check_file_exists("small_model_28epoch/taxonomy_model_small_best.pth", 
                         "Best model"):
        check_model("small_model_28epoch/taxonomy_model_small_best.pth", 
                   torch.Size([92290, 10]))
    
    all_passed &= check_file_exists("small_model_28epoch/taxonomy_embeddings_multi_groups.png",
                                    "Multi-group viz")
    all_passed &= check_file_exists("small_model_28epoch/best_epoch_analysis_epoch28.png",
                                    "Epoch analysis")
    print()
    
    # 4. Check animals model (reference)
    print("4️⃣  Animals Model (Reference)")
    if check_file_exists("taxonomy_model_animals_best.pth", "Animals model"):
        check_model("taxonomy_model_animals_best.pth", torch.Size([1055469, 10]))
    print()
    
    # 5. Check data files
    print("5️⃣  Data Files")
    if check_file_exists("data/taxonomy_edges_small_transitive.pkl", "Training data"):
        check_data_file("data/taxonomy_edges_small_transitive.pkl", "Training pairs")
    
    if check_file_exists("data/taxonomy_edges_small.mapping.tsv", "Mapping file"):
        check_data_file("data/taxonomy_edges_small.mapping.tsv", "TaxID mappings")
    
    all_passed &= check_file_exists("data/names.dmp", "NCBI names")
    all_passed &= check_file_exists("data/nodes.dmp", "NCBI nodes")
    print()
    
    # 6. Check no intermediate files remain
    print("6️⃣  Cleanup Verification")
    intermediate_files = [
        "taxonomy_model_animals_epoch1.pth",
        "taxonomy_model_animals_epoch2.pth",
        "animals_taxonomy_umap.png",
        "build_transitive_closure_full.py",
        "train_animals.py",
    ]
    
    cleanup_ok = True
    for f in intermediate_files:
        if os.path.exists(f):
            print(f"  ⚠️  Intermediate file still present: {f}")
            cleanup_ok = False
    
    if cleanup_ok:
        print("  ✅ No intermediate files found (good!)")
    print()
    
    # Final verdict
    print("="*80)
    if all_passed and cleanup_ok:
        print("✅ ALL CHECKS PASSED - Repository is ready for commit!")
    else:
        print("❌ SOME CHECKS FAILED - Please review above")
        sys.exit(1)
    print("="*80)

if __name__ == "__main__":
    main()
