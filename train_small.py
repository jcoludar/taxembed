#!/usr/bin/env python3
"""
Training script for small dataset with enhanced terminal visualization.

Features:
- Pre-configured for taxonomy_edges_small dataset
- Real-time metrics display showing improvements
- Compact progress visualization
- Automatic comparison with previous epoch
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import pickle
import argparse
import os
import sys
from collections import defaultdict, deque
from datetime import datetime
from tqdm import tqdm

# Import the hierarchical model and components
from train_hierarchical import (
    HierarchicalPoincareEmbedding,
    HierarchicalDataLoader,
    ranking_loss_with_margin,
    radial_regularizer
)


class MetricsTracker:
    """Track and display training metrics with visual improvements."""
    
    def __init__(self):
        self.history = []
        self.best_loss = float('inf')
        self.best_epoch = 0
    
    def update(self, epoch, metrics):
        """Update metrics for current epoch."""
        self.history.append(metrics)
        
        if metrics['loss'] < self.best_loss:
            self.best_loss = metrics['loss']
            self.best_epoch = epoch
    
    def get_previous(self, metric_name):
        """Get previous epoch's metric value."""
        if len(self.history) < 2:
            return None
        return self.history[-2].get(metric_name)
    
    def print_header(self):
        """Print column headers."""
        print("\n" + "="*100)
        print(f"{'Epoch':>6} | {'Loss':>10} | {'ΔLoss':>10} | {'Improve':>8} | "
              f"{'Reg':>8} | {'MaxNorm':>8} | {'Outside':>7} | {'Status':>10}")
        print("="*100)
    
    def print_epoch_summary(self, epoch, metrics, total_epochs):
        """Print compact summary of epoch with improvement indicators."""
        prev_loss = self.get_previous('loss')
        
        # Calculate improvement
        if prev_loss is not None:
            delta = metrics['loss'] - prev_loss
            pct_change = (delta / prev_loss) * 100 if prev_loss != 0 else 0
            
            if delta < 0:
                status = "✓ BETTER"
                delta_str = f"{delta:+.4f}"
                pct_str = f"{pct_change:+.2f}%"
                improve_color = "\033[92m"  # Green
            else:
                status = "✗ WORSE"
                delta_str = f"{delta:+.4f}"
                pct_str = f"{pct_change:+.2f}%"
                improve_color = "\033[91m"  # Red
            
            reset_color = "\033[0m"
        else:
            delta_str = "---"
            pct_str = "---"
            status = "FIRST"
            improve_color = ""
            reset_color = ""
        
        # Format output
        outside_pct = (metrics['outside_count'] / metrics['total_nodes']) * 100 if metrics['total_nodes'] > 0 else 0
        
        print(f"{epoch:6d} | "
              f"{metrics['loss']:10.6f} | "
              f"{improve_color}{delta_str:>10}{reset_color} | "
              f"{improve_color}{pct_str:>8}{reset_color} | "
              f"{metrics['reg_loss']:8.6f} | "
              f"{metrics['max_norm']:8.4f} | "
              f"{outside_pct:6.2f}% | "
              f"{improve_color}{status:>10}{reset_color}")
        
        # Additional info every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            print(f"       └─ Best: {self.best_loss:.6f} @ epoch {self.best_epoch} | "
                  f"Norms: [{metrics['min_norm']:.4f}, {metrics['mean_norm']:.4f}, {metrics['max_norm']:.4f}]")
    
    def print_final_summary(self):
        """Print final training summary."""
        print("\n" + "="*100)
        print("TRAINING SUMMARY")
        print("="*100)
        print(f"Total epochs: {len(self.history)}")
        print(f"Best loss: {self.best_loss:.6f} (epoch {self.best_epoch})")
        
        if len(self.history) >= 2:
            first_loss = self.history[0]['loss']
            last_loss = self.history[-1]['loss']
            total_improvement = first_loss - last_loss
            pct_improvement = (total_improvement / first_loss) * 100
            print(f"Total improvement: {total_improvement:+.6f} ({pct_improvement:+.2f}%)")
        
        print("="*100 + "\n")


def train_with_visualization(model, dataloader, optimizer, n_epochs, 
                             idx_to_depth, max_depth, device, 
                             margin=0.2, lambda_reg=0.1, early_stopping_patience=3,
                             checkpoint_base=None):
    """Train with enhanced terminal visualization."""
    
    model.to(device)
    model.train()
    
    print("\n" + "🚀 "*20)
    print("HIERARCHICAL TRAINING - SMALL DATASET")
    print("🚀 "*20)
    print(f"\nConfiguration:")
    print(f"  Margin: {margin}")
    print(f"  Regularization: λ={lambda_reg}")
    print(f"  Early stopping: {'disabled' if early_stopping_patience == 0 else f'{early_stopping_patience} epochs'}")
    print(f"  Device: {device}")
    print(f"  Max epochs: {n_epochs}")
    
    # Precompute regularizer tensors
    print("\nPrecomputing regularization targets...")
    reg_indices = []
    reg_target_radii = []
    for idx, depth in idx_to_depth.items():
        if idx < model.n_nodes:
            reg_indices.append(idx)
            target_radius = 0.1 + (depth / max_depth) * 0.85
            reg_target_radii.append(target_radius)
    
    reg_indices_tensor = torch.LongTensor(reg_indices).to(device)
    reg_target_radii_tensor = torch.FloatTensor(reg_target_radii).to(device)
    print(f"  ✓ Regularizing {len(reg_indices):,} nodes")
    
    # Metrics tracker
    tracker = MetricsTracker()
    tracker.print_header()
    
    # Early stopping
    epochs_without_improvement = 0
    checkpoint_queue = deque(maxlen=5)
    
    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0
        epoch_reg_loss = 0
        n_batches = 0
        
        # Progress bar for batches
        pbar = tqdm(dataloader, 
                   desc=f"Epoch {epoch:3d}/{n_epochs}",
                   bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                   ncols=100,
                   leave=False,  # Don't leave the bar on screen after completion
                   dynamic_ncols=True,  # Adjust to terminal width
                   mininterval=0.5)  # Update at most every 0.5 seconds
        
        for ancestors, descendants, negatives, depths in pbar:
            ancestors = ancestors.to(device)
            descendants = descendants.to(device)
            negatives = negatives.to(device)
            depths = depths.to(device)
            
            optimizer.zero_grad()
            
            # Ranking loss
            loss = ranking_loss_with_margin(
                model, ancestors, descendants, negatives,
                depths, margin=margin, depth_weight=True
            )
            
            # Radial regularizer
            reg_loss = radial_regularizer(model, reg_indices_tensor, 
                                         reg_target_radii_tensor, lambda_reg)
            
            # Total loss
            total_loss = loss + reg_loss
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Selective projection
            updated_indices = torch.cat([ancestors, descendants, negatives.flatten()])
            updated_indices = torch.unique(updated_indices)
            model.project_to_ball(updated_indices)
            
            # Periodic full projection
            if n_batches % 500 == 0:
                model.project_to_ball(indices=None)
            
            epoch_loss += loss.item()
            epoch_reg_loss += reg_loss.item()
            n_batches += 1
            
            # Update progress bar with current batch metrics
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'reg': f'{reg_loss.item():.4f}'
            })
        
        # Final projection at epoch end
        model.project_to_ball(indices=None)
        
        # Compute metrics
        avg_loss = epoch_loss / n_batches
        avg_reg = epoch_reg_loss / n_batches
        norms = model.embeddings.weight.norm(dim=1)
        outside_count = (norms >= 1.0).sum().item()
        
        metrics = {
            'loss': avg_loss,
            'reg_loss': avg_reg,
            'min_norm': norms.min().item(),
            'mean_norm': norms.mean().item(),
            'max_norm': norms.max().item(),
            'outside_count': outside_count,
            'total_nodes': model.n_nodes
        }
        
        # Early stopping check - MUST happen BEFORE updating tracker
        prev_best_loss = tracker.best_loss
        
        # Update tracker and display
        tracker.update(epoch, metrics)
        tracker.print_epoch_summary(epoch, metrics, n_epochs)
        
        # Save checkpoint
        if checkpoint_base:
            checkpoint_path = checkpoint_base.replace('.pth', f'_epoch{epoch}.pth')
            torch.save({
                'state_dict': {'lt.weight': model.embeddings.weight},
                'embeddings': model.embeddings.weight,
                'epoch': epoch,
                'loss': avg_loss,
                'reg_loss': avg_reg,
                'best_loss': tracker.best_loss,
                'epochs_without_improvement': epochs_without_improvement,
            }, checkpoint_path)
            
            # Manage checkpoint queue
            if len(checkpoint_queue) >= checkpoint_queue.maxlen:
                old_checkpoint = checkpoint_queue[0]
                if os.path.exists(old_checkpoint):
                    os.remove(old_checkpoint)
            
            checkpoint_queue.append(checkpoint_path)
        
        # Check if loss improved (compare against PREVIOUS best, not updated best)
        if avg_loss < prev_best_loss:
            epochs_without_improvement = 0
            
            # Save best model
            if checkpoint_base:
                best_checkpoint = checkpoint_base.replace('.pth', '_best.pth')
                torch.save({
                    'state_dict': {'lt.weight': model.embeddings.weight},
                    'embeddings': model.embeddings.weight,
                    'epoch': epoch,
                    'loss': avg_loss,
                    'reg_loss': avg_reg,
                }, best_checkpoint)
        else:
            epochs_without_improvement += 1
            
            # Only check early stopping if patience > 0 (0 means disabled)
            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                print(f"\n🛑 Early stopping triggered after {epoch} epochs")
                print(f"   Best loss: {tracker.best_loss:.6f} (epoch {tracker.best_epoch})")
                break
    
    # Final summary
    tracker.print_final_summary()


def main():
    parser = argparse.ArgumentParser(description='Train on small dataset with enhanced visualization')
    parser.add_argument('--data', default='data/taxonomy_edges_small_transitive.pkl',
                       help='Training data (default: small dataset)')
    parser.add_argument('--checkpoint', default='taxonomy_model_small.pth',
                       help='Output checkpoint path')
    parser.add_argument('--dim', type=int, default=10,
                       help='Embedding dimension')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Maximum number of epochs')
    parser.add_argument('--early-stopping', type=int, default=5,
                       help='Early stopping patience')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--n-negatives', type=int, default=50,
                       help='Number of negative samples')
    parser.add_argument('--lr', type=float, default=0.005,
                       help='Learning rate')
    parser.add_argument('--margin', type=float, default=0.2,
                       help='Ranking loss margin')
    parser.add_argument('--lambda-reg', type=float, default=0.1,
                       help='Radial regularization weight')
    parser.add_argument('--gpu', type=int, default=-1,
                       help='GPU device (-1 for CPU)')
    
    args = parser.parse_args()
    
    # Device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    
    # Load training data
    print(f"\nLoading training data from {args.data}...")
    if not os.path.exists(args.data):
        print(f"❌ Error: Training data not found at {args.data}")
        print(f"\nPlease run first:")
        print(f"  python build_transitive_closure.py")
        sys.exit(1)
    
    with open(args.data, 'rb') as f:
        training_data = pickle.load(f)
    print(f"  ✓ Loaded {len(training_data):,} training pairs")
    
    # Get dataset info - CRITICAL: use mapping file, not training data!
    # Training data may not include all nodes (e.g., leaf nodes, isolated nodes)
    print("Loading mapping to determine true n_nodes...")
    mapping_df = pd.read_csv("data/taxonomy_edges_small.mapping.tsv",
                             sep="\t", header=None, names=["taxid", "idx"])
    mapping_df['idx'] = pd.to_numeric(mapping_df['idx'], errors='coerce')
    mapping_df['taxid'] = pd.to_numeric(mapping_df['taxid'], errors='coerce')
    mapping_df = mapping_df.dropna()
    mapping_df['idx'] = mapping_df['idx'].astype(int)
    mapping_df['taxid'] = mapping_df['taxid'].astype(int)
    n_nodes = int(mapping_df['idx'].max()) + 1
    
    max_depth = max(item['descendant_depth'] for item in training_data)
    
    print(f"  Nodes: {n_nodes:,}")
    print(f"  Max depth: {max_depth}")
    
    # Build depth mapping from training data
    idx_to_depth = {}
    for item in training_data:
        idx_to_depth[item['descendant_idx']] = item['descendant_depth']
        if item['ancestor_idx'] not in idx_to_depth:
            idx_to_depth[item['ancestor_idx']] = item['ancestor_depth']
    
    # For nodes NOT in training data, load their depths from full taxonomy
    nodes_in_training = set(idx_to_depth.keys())
    missing_nodes = n_nodes - len(nodes_in_training)
    
    if missing_nodes > 0:
        print(f"\n⚠️  {missing_nodes:,} nodes not in training data - loading their depths from taxonomy...")
        
        # Build TaxID -> depth mapping from training data
        taxid_to_depth = {}
        for item in training_data:
            taxid_to_depth[item['ancestor_taxid']] = item['ancestor_depth']
            taxid_to_depth[item['descendant_taxid']] = item['descendant_depth']
        
        # Map missing indices to depths via TaxID lookup
        for idx in range(n_nodes):
            if idx not in idx_to_depth:
                # Find TaxID for this index
                taxid = mapping_df[mapping_df['idx'] == idx]['taxid'].values[0]
                if taxid in taxid_to_depth:
                    idx_to_depth[idx] = taxid_to_depth[taxid]
                else:
                    # Node not in taxonomy at all - likely a leaf, assign max depth
                    idx_to_depth[idx] = max_depth
        
        print(f"  ✓ Assigned depths to {missing_nodes:,} additional nodes")
        print(f"  ✓ Total nodes with depth info: {len(idx_to_depth):,} / {n_nodes:,}")
    
    # Create model
    print("\nInitializing model with depth-aware embeddings...")
    model = HierarchicalPoincareEmbedding(
        n_nodes=n_nodes,
        dim=args.dim,
        max_depth=max_depth,
        init_depth_data=idx_to_depth
    )
    
    # Create dataloader
    print("Creating dataloader with hard negative sampling...")
    dataloader = HierarchicalDataLoader(
        training_data=training_data,
        n_nodes=n_nodes,
        batch_size=args.batch_size,
        n_negatives=args.n_negatives,
        depth_stratify=True
    )
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Train with visualization
    train_with_visualization(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        n_epochs=args.epochs,
        idx_to_depth=idx_to_depth,
        max_depth=max_depth,
        device=device,
        margin=args.margin,
        lambda_reg=args.lambda_reg,
        early_stopping_patience=args.early_stopping,
        checkpoint_base=args.checkpoint
    )
    
    # Save final model
    print(f"Saving final model to {args.checkpoint}...")
    torch.save({
        'state_dict': {'lt.weight': model.embeddings.weight},
        'embeddings': model.embeddings.weight,
        'n_nodes': n_nodes,
        'dim': args.dim,
        'max_depth': max_depth,
    }, args.checkpoint)
    
    print("\n✅ Training complete!")
    print(f"   Model saved: {args.checkpoint}")
    print(f"   Best model: {args.checkpoint.replace('.pth', '_best.pth')}")
    print("\nNext steps:")
    print("  python analyze_hierarchy_hyperbolic.py")
    print(f"  python scripts/visualize_embeddings.py {args.checkpoint.replace('.pth', '_best.pth')}")


if __name__ == "__main__":
    main()
