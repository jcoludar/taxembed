#!/usr/bin/env python3
"""
Utility script to clean up old checkpoint files.
Keeps only the most recent N checkpoints per model.
"""

import os
import re
import argparse
from pathlib import Path
from collections import defaultdict


def find_checkpoints(directory="."):
    """Find all checkpoint files in the directory."""
    checkpoints = defaultdict(list)
    
    # Pattern to match checkpoint files
    # Format: model_name_epochN.pth or model_name.pth.N
    pattern1 = re.compile(r'^(.+)_epoch(\d+)\.pth$')
    pattern2 = re.compile(r'^(.+)\.pth\.(\d+)$')
    
    for file in os.listdir(directory):
        full_path = os.path.join(directory, file)
        if not os.path.isfile(full_path):
            continue
            
        # Try both patterns
        match = pattern1.match(file) or pattern2.match(file)
        if match:
            model_name = match.group(1)
            epoch = int(match.group(2))
            checkpoints[model_name].append({
                'path': full_path,
                'epoch': epoch,
                'mtime': os.path.getmtime(full_path),
                'size': os.path.getsize(full_path)
            })
    
    return checkpoints


def cleanup_checkpoints(checkpoints, keep=20, dry_run=False):
    """Clean up old checkpoints, keeping only the most recent N."""
    total_deleted = 0
    total_size_freed = 0
    
    for model_name, files in checkpoints.items():
        if len(files) <= keep:
            print(f"✓ {model_name}: {len(files)} checkpoints (all kept)")
            continue
        
        # Sort by epoch (newest first)
        files.sort(key=lambda x: x['epoch'], reverse=True)
        
        # Keep the most recent N
        to_keep = files[:keep]
        to_delete = files[keep:]
        
        print(f"\n📁 {model_name}: {len(files)} checkpoints")
        print(f"   Keeping {len(to_keep)} most recent (epochs {to_keep[-1]['epoch']}-{to_keep[0]['epoch']})")
        print(f"   Deleting {len(to_delete)} old checkpoints:")
        
        for checkpoint in to_delete:
            size_mb = checkpoint['size'] / (1024 * 1024)
            print(f"   🗑️  Epoch {checkpoint['epoch']:3d}: {os.path.basename(checkpoint['path'])} ({size_mb:.1f} MB)")
            
            if not dry_run:
                try:
                    os.remove(checkpoint['path'])
                    total_deleted += 1
                    total_size_freed += checkpoint['size']
                except Exception as e:
                    print(f"   ⚠️  Error deleting: {e}")
            else:
                total_deleted += 1
                total_size_freed += checkpoint['size']
    
    return total_deleted, total_size_freed


def main():
    parser = argparse.ArgumentParser(
        description="Clean up old checkpoint files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (see what would be deleted)
  python cleanup_old_checkpoints.py --dry-run
  
  # Keep only 10 most recent checkpoints per model
  python cleanup_old_checkpoints.py --keep 10
  
  # Keep 20 checkpoints in specific directory
  python cleanup_old_checkpoints.py --directory /path/to/checkpoints --keep 20
        """
    )
    parser.add_argument(
        '--directory', '-d',
        default='.',
        help='Directory to search for checkpoints (default: current directory)'
    )
    parser.add_argument(
        '--keep', '-k',
        type=int,
        default=20,
        help='Number of most recent checkpoints to keep per model (default: 20)'
    )
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Show what would be deleted without actually deleting'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("CHECKPOINT CLEANUP")
    print("=" * 80)
    print(f"Directory: {os.path.abspath(args.directory)}")
    print(f"Keep: {args.keep} most recent checkpoints per model")
    print(f"Mode: {'DRY RUN (no files will be deleted)' if args.dry_run else 'LIVE (files will be deleted)'}")
    print("=" * 80)
    print()
    
    # Find checkpoints
    checkpoints = find_checkpoints(args.directory)
    
    if not checkpoints:
        print("No checkpoint files found.")
        return
    
    print(f"Found {len(checkpoints)} model(s) with checkpoints:\n")
    
    # Clean up
    total_deleted, total_size_freed = cleanup_checkpoints(
        checkpoints, 
        keep=args.keep, 
        dry_run=args.dry_run
    )
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    if args.dry_run:
        print(f"Would delete: {total_deleted} checkpoint files")
        print(f"Would free: {total_size_freed / (1024**3):.2f} GB")
    else:
        print(f"Deleted: {total_deleted} checkpoint files")
        print(f"Freed: {total_size_freed / (1024**3):.2f} GB")
    print("=" * 80)


if __name__ == "__main__":
    main()
