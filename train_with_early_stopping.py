#!/usr/bin/env python3
"""
Training script with early stopping for Poincaré embeddings.
Stops training when loss doesn't improve for N consecutive epochs.
"""

import subprocess
import re
import sys
import signal
import time
import os
from pathlib import Path
from collections import deque

class CheckpointManager:
    """Manages checkpoint files, keeping only N most recent."""
    def __init__(self, max_checkpoints=20):
        self.max_checkpoints = max_checkpoints
        self.checkpoints = deque(maxlen=max_checkpoints)
        
    def add_checkpoint(self, checkpoint_path):
        """Add a new checkpoint and delete old ones if needed."""
        if not os.path.exists(checkpoint_path):
            return
            
        # If we're at capacity, remove the oldest checkpoint
        if len(self.checkpoints) >= self.max_checkpoints:
            old_checkpoint = self.checkpoints[0]
            if os.path.exists(old_checkpoint):
                try:
                    os.remove(old_checkpoint)
                    print(f"🗑️  Deleted old checkpoint: {os.path.basename(old_checkpoint)}")
                except Exception as e:
                    print(f"⚠️  Could not delete {old_checkpoint}: {e}")
        
        self.checkpoints.append(checkpoint_path)
        print(f"💾 Checkpoint saved: {os.path.basename(checkpoint_path)} (keeping {len(self.checkpoints)}/{self.max_checkpoints})")
    
    def cleanup_all(self):
        """Clean up all tracked checkpoints."""
        for checkpoint in self.checkpoints:
            if os.path.exists(checkpoint):
                try:
                    os.remove(checkpoint)
                except Exception:
                    pass


class EarlyStoppingTrainer:
    def __init__(self, patience=6, min_delta=1e-6, max_checkpoints=20):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        self.epoch_losses = []
        self.checkpoint_manager = CheckpointManager(max_checkpoints)
        
    def should_stop(self, current_loss):
        """Check if training should stop."""
        self.epoch_losses.append(current_loss)
        
        # Check if loss improved
        if current_loss < (self.best_loss - self.min_delta):
            print(f"✓ Loss improved: {self.best_loss:.6f} -> {current_loss:.6f}")
            self.best_loss = current_loss
            self.epochs_without_improvement = 0
            return False
        else:
            self.epochs_without_improvement += 1
            print(f"✗ No improvement ({self.epochs_without_improvement}/{self.patience}). Current: {current_loss:.6f}, Best: {self.best_loss:.6f}")
            
            if self.epochs_without_improvement >= self.patience:
                print(f"\n🛑 Early stopping triggered! No improvement for {self.patience} consecutive epochs.")
                return True
        
        return False
    
    def get_summary(self):
        """Get training summary."""
        return {
            'total_epochs': len(self.epoch_losses),
            'best_loss': self.best_loss,
            'final_loss': self.epoch_losses[-1] if self.epoch_losses else None,
            'all_losses': self.epoch_losses
        }


def train_with_monitoring(cmd, patience=6, max_checkpoints=20, checkpoint_base=None):
    """
    Run training command and monitor for early stopping.
    
    Args:
        cmd: Training command as list
        patience: Number of epochs without improvement before stopping
        max_checkpoints: Maximum number of checkpoints to keep
        checkpoint_base: Base name for checkpoint files
    """
    early_stopper = EarlyStoppingTrainer(patience=patience, max_checkpoints=max_checkpoints)
    
    print("=" * 80)
    print("EARLY STOPPING TRAINING")
    print("=" * 80)
    print(f"Command: {' '.join(cmd)}")
    print(f"Patience: {patience} epochs")
    print(f"Max checkpoints: {max_checkpoints}")
    print("=" * 80)
    print()
    
    # Start training process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Pattern to match loss in JSON output
    loss_pattern = re.compile(r'"loss":\s*([\d.]+)')
    epoch_pattern = re.compile(r'"epoch":\s*(\d+)')
    checkpoint_pattern = re.compile(r'Saved checkpoint: (.+\.pth)')
    
    current_epoch = None
    
    try:
        # Monitor output
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
                
            # Print the line
            print(line, end='', flush=True)
            
            # Check for checkpoint saves
            checkpoint_match = checkpoint_pattern.search(line)
            if checkpoint_match and checkpoint_base:
                checkpoint_file = checkpoint_match.group(1)
                early_stopper.checkpoint_manager.add_checkpoint(checkpoint_file)
            
            # Check for epoch number
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
            
            # Check for loss value
            loss_match = loss_pattern.search(line)
            if loss_match and current_epoch is not None:
                loss = float(loss_match.group(1))
                
                # Check early stopping
                if early_stopper.should_stop(loss):
                    print("\n" + "=" * 80)
                    print("STOPPING TRAINING")
                    print("=" * 80)
                    
                    # Send SIGTERM to gracefully stop
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    
                    break
        
        # Wait for process to complete
        process.wait()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Terminating training...")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        return None
    
    # Print summary
    summary = early_stopper.get_summary()
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Total epochs trained: {summary['total_epochs']}")
    print(f"Best loss: {summary['best_loss']:.6f}")
    if summary['final_loss']:
        print(f"Final loss: {summary['final_loss']:.6f}")
    print("\nLoss history:")
    for i, loss in enumerate(summary['all_losses']):
        marker = " ← BEST" if loss == summary['best_loss'] else ""
        print(f"  Epoch {i}: {loss:.6f}{marker}")
    print("=" * 80)
    
    return summary


def main():
    # Configuration for small dataset training
    checkpoint_path = "taxonomy_model_small_early_stop.pth"
    
    # Training command based on working configuration from memories
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
        "-eval_each", "999999",  # Disable evaluation during training
        "-fresh"
    ]
    
    print("Starting training with early stopping (patience=6)...")
    print(f"Model will be saved to: {checkpoint_path}")
    print(f"Maximum checkpoints kept: 20")
    print("NO EPOCH LIMIT - Training will continue until loss plateaus for 6 consecutive epochs")
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
