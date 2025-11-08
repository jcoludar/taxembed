#!/usr/bin/env python3
"""
Assess training progress by examining recent checkpoints.
"""

import torch
import glob
import re
import os
import matplotlib.pyplot as plt
import numpy as np

def extract_loss_history(checkpoint_pattern="taxonomy_model_small_early_stop_epoch*.pth"):
    """Extract loss values from checkpoint files."""
    checkpoints = glob.glob(checkpoint_pattern)
    
    loss_data = []
    for ckpt_path in checkpoints:
        # Extract epoch number from filename
        match = re.search(r'epoch(\d+)\.pth', ckpt_path)
        if not match:
            continue
        epoch = int(match.group(1))
        
        # Load checkpoint and extract loss
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            loss = ckpt.get('loss', None)
            if loss is not None:
                loss_data.append((epoch, loss))
        except Exception as e:
            print(f"Warning: Could not load {ckpt_path}: {e}")
    
    # Sort by epoch
    loss_data.sort(key=lambda x: x[0])
    return loss_data


def analyze_convergence(loss_data, window=20):
    """Analyze if training has converged."""
    if len(loss_data) < window:
        print(f"Not enough data points. Have {len(loss_data)}, need at least {window}")
        return
    
    epochs = [x[0] for x in loss_data]
    losses = [x[1] for x in loss_data]
    
    # Recent losses
    recent_losses = losses[-window:]
    recent_epochs = epochs[-window:]
    
    print(f"\n{'='*80}")
    print(f"TRAINING ASSESSMENT")
    print(f"{'='*80}")
    print(f"Total epochs: {len(loss_data)}")
    print(f"Epoch range: {epochs[0]} - {epochs[-1]}")
    print(f"\nFinal Loss Statistics (last {window} epochs):")
    print(f"  Best loss: {min(recent_losses):.6f} (epoch {recent_epochs[recent_losses.index(min(recent_losses))]})") 
    print(f"  Final loss: {losses[-1]:.6f} (epoch {epochs[-1]})")
    print(f"  Mean: {np.mean(recent_losses):.6f}")
    print(f"  Std: {np.std(recent_losses):.6f}")
    print(f"  Range: {np.ptp(recent_losses):.6f}")
    
    # Check for improvement in recent window
    improvements = 0
    for i in range(len(recent_losses) - 1):
        if recent_losses[i+1] < recent_losses[i]:
            improvements += 1
    
    print(f"\nRecent trend (last {window} epochs):")
    print(f"  Improvements: {improvements}/{window-1} epochs")
    print(f"  % improving: {improvements/(window-1)*100:.1f}%")
    
    # Overall statistics
    print(f"\nOverall Statistics:")
    print(f"  Best loss ever: {min(losses):.6f} (epoch {epochs[losses.index(min(losses))]})")
    print(f"  First loss: {losses[0]:.6f} (epoch {epochs[0]})")
    print(f"  Total improvement: {losses[0] - losses[-1]:.6f} ({(losses[0] - losses[-1])/losses[0]*100:.2f}%)")
    
    # Convergence assessment
    recent_std = np.std(recent_losses)
    if recent_std < 0.001:
        convergence_status = "✅ CONVERGED (loss is very stable)"
    elif recent_std < 0.005:
        convergence_status = "⚠️ NEAR CONVERGENCE (loss is stabilizing)"
    else:
        convergence_status = "🔄 STILL TRAINING (loss is changing)"
    
    print(f"\nConvergence Status: {convergence_status}")
    print(f"{'='*80}\n")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Full training history
    ax1.plot(epochs, losses, 'b-', alpha=0.6, linewidth=1)
    ax1.scatter(epochs, losses, c=range(len(epochs)), cmap='viridis', s=20, alpha=0.7)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Full Training History', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(min(losses) * 0.99, max(losses) * 1.01)
    
    # Recent window
    ax2.plot(recent_epochs, recent_losses, 'r-', alpha=0.6, linewidth=2, label='Loss')
    ax2.scatter(recent_epochs, recent_losses, c='red', s=50, alpha=0.7, zorder=5)
    ax2.axhline(y=min(recent_losses), color='g', linestyle='--', alpha=0.5, label=f'Best: {min(recent_losses):.6f}')
    ax2.axhline(y=np.mean(recent_losses), color='orange', linestyle='--', alpha=0.5, label=f'Mean: {np.mean(recent_losses):.6f}')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title(f'Recent {window} Epochs (Detail)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_assessment.png', dpi=150, bbox_inches='tight')
    print(f"Saved plot to: training_assessment.png")
    plt.close()
    
    return epochs[-1], losses[-1], min(losses)


def main():
    print("Assessing training progress...\n")
    
    loss_data = extract_loss_history()
    
    if not loss_data:
        print("No checkpoint files found!")
        return
    
    final_epoch, final_loss, best_loss = analyze_convergence(loss_data, window=20)
    
    print("\nRecommendation:")
    if abs(final_loss - best_loss) < 0.001:
        print("✅ Training appears to have converged. Model is ready for use.")
    else:
        print("⚠️ Loss is still improving. Consider continuing training.")


if __name__ == "__main__":
    main()
