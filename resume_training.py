#!/usr/bin/env python3
"""
Resume training from the last checkpoint with early stopping.
"""

import subprocess
import re
import sys
import os
from pathlib import Path
from train_with_early_stopping import train_with_monitoring

def find_latest_checkpoint(checkpoint_base):
    """Find the most recent checkpoint for a model."""
    directory = os.path.dirname(checkpoint_base) or '.'
    base_name = os.path.basename(checkpoint_base).replace('.pth', '')
    
    checkpoints = []
    for file in os.listdir(directory):
        if file.startswith(base_name) and '_epoch' in file and file.endswith('.pth'):
            match = re.search(r'_epoch(\d+)\.pth$', file)
            if match:
                epoch = int(match.group(1))
                checkpoints.append({
                    'path': os.path.join(directory, file),
                    'epoch': epoch
                })
    
    if not checkpoints:
        return None, None
    
    # Return the latest checkpoint
    checkpoints.sort(key=lambda x: x['epoch'], reverse=True)
    return checkpoints[0]['path'], checkpoints[0]['epoch']


def main():
    checkpoint_path = "taxonomy_model_small_early_stop.pth"
    
    # Find latest checkpoint
    latest_checkpoint, last_epoch = find_latest_checkpoint(checkpoint_path)
    
    if latest_checkpoint:
        print(f"Found checkpoint: {os.path.basename(latest_checkpoint)} (epoch {last_epoch})")
        start_epoch = last_epoch + 1
        print(f"Resuming from epoch {start_epoch}...")
        
        # Load the checkpoint and convert format
        import torch
        checkpoint_data = torch.load(latest_checkpoint, map_location='cpu')
        
        # Convert from per-epoch format to embed.py format
        if 'state_dict' in checkpoint_data and 'model' not in checkpoint_data:
            print("Converting checkpoint format...")
            converted_data = {
                'model': checkpoint_data['state_dict'],
                'epoch': checkpoint_data.get('epoch', last_epoch)
            }
            torch.save(converted_data, checkpoint_path)
            print(f"Converted and saved to {os.path.basename(checkpoint_path)}")
        else:
            # Already in correct format, just copy
            import shutil
            shutil.copy(latest_checkpoint, checkpoint_path)
            print(f"Copied {os.path.basename(latest_checkpoint)} -> {os.path.basename(checkpoint_path)}")
    else:
        print("No previous checkpoint found. Starting from scratch...")
        start_epoch = 0
    
    # Training command (without -fresh to resume)
    cmd = [
        "python", "embed.py",
        "-dset", "data/taxonomy_edges_small.mapped.edgelist",
        "-checkpoint", checkpoint_path,
        "-dim", "10",
        "-epochs", "10000",  # Effectively infinite - early stopping is the only limit
        "-negs", "50",
        "-burnin", "10",
        "-batchsize", "32",
        "-model", "distance",
        "-manifold", "poincare",
        "-lr", "0.1",
        "-gpu", "-1",
        "-ndproc", "1",
        "-train_threads", "1",
        "-eval_each", "999999",
    ]
    
    # Only add -fresh if starting from scratch
    if start_epoch == 0:
        cmd.append("-fresh")
    
    print(f"\nResuming training with early stopping (patience=6)...")
    print(f"NO EPOCH LIMIT - Training will continue until loss plateaus for 6 consecutive epochs")
    print()
    
    # Run training with early stopping
    summary = train_with_monitoring(cmd, patience=6, max_checkpoints=20, checkpoint_base=checkpoint_path)
    
    if summary:
        print(f"\n✅ Training completed successfully!")
        print(f"Best model saved to: {checkpoint_path}")
    else:
        print(f"\n⚠️ Training was interrupted")


if __name__ == "__main__":
    main()
