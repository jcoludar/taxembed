#!/usr/bin/env python3
"""
Phase 2: Hierarchical Training for Poincaré Embeddings

Key improvements:
1. Train on transitive closure (ALL ancestor-descendant pairs)
2. Depth-aware initialization (deeper nodes near boundary)
3. Radial regularizer (enforce depth → radius mapping)
4. Hard negative sampling (cousins at same depth)
5. Depth weighting (deeper pairs matter more)
6. Proper hyperbolic distance and loss
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from collections import defaultdict, deque
from tqdm import tqdm
import argparse
import os


class HierarchicalPoincareEmbedding(nn.Module):
    """Poincaré embeddings with hierarchical structure."""
    
    def __init__(self, n_nodes, dim=10, max_depth=38, init_depth_data=None):
        super().__init__()
        self.n_nodes = n_nodes
        self.dim = dim
        self.max_depth = max_depth
        
        # Embeddings (initialize later with depth info)
        self.embeddings = nn.Embedding(n_nodes, dim)
        
        # Initialize based on depth if available
        if init_depth_data is not None:
            self._initialize_by_depth(init_depth_data)
        else:
            # Default: uniform small initialization
            nn.init.uniform_(self.embeddings.weight, -0.001, 0.001)
    
    def _initialize_by_depth(self, depth_data):
        """
        Initialize embeddings based on taxonomic depth.
        
        Deeper nodes → larger radius (closer to boundary).
        This encodes hierarchy from the start!
        """
        print("Initializing embeddings by depth...")
        
        # Map: idx → depth
        idx_to_depth = depth_data
        
        with torch.no_grad():
            for idx in range(self.n_nodes):
                depth = idx_to_depth.get(idx, 0)
                
                # Radius increases with depth
                # Root (depth 0): r ≈ 0.1
                # Max depth: r ≈ 0.95 (near boundary)
                target_radius = 0.1 + (depth / self.max_depth) * 0.85
                
                # Random direction on sphere
                vec = torch.randn(self.dim)
                vec = vec / vec.norm()
                
                # Scale to target radius
                self.embeddings.weight[idx] = vec * target_radius
        
        norms = self.embeddings.weight.norm(dim=1)
        print(f"  ✓ Initialized: norm range [{norms.min():.3f}, {norms.max():.3f}]")
    
    def forward(self, indices):
        """Get embeddings for indices."""
        return self.embeddings(indices)
    
    def poincare_distance(self, u, v, eps=1e-5):
        """
        Compute Poincaré distance between embeddings.
        
        d(u,v) = arcosh(1 + 2||u-v||²/((1-||u||²)(1-||v||²)))
        """
        # Compute squared norms
        u_norm_sq = (u ** 2).sum(dim=-1)
        v_norm_sq = (v ** 2).sum(dim=-1)
        
        # Clamp to stay inside ball
        u_norm_sq = torch.clamp(u_norm_sq, 0, 1 - eps)
        v_norm_sq = torch.clamp(v_norm_sq, 0, 1 - eps)
        
        # Squared Euclidean distance
        diff_norm_sq = ((u - v) ** 2).sum(dim=-1)
        
        # Poincaré distance
        numerator = 2 * diff_norm_sq
        denominator = (1 - u_norm_sq) * (1 - v_norm_sq)
        
        dist = torch.acosh(1 + numerator / (denominator + eps) + eps)
        
        return dist
    
    def project_to_ball(self, indices=None, eps=1e-5):
        """
        Project embeddings back into Poincaré ball with HARD constraint.
        
        Args:
            indices: If provided, only project these indices (more efficient).
                    If None, project all embeddings.
        """
        with torch.no_grad():
            if indices is not None:
                # Only project updated embeddings
                embs = self.embeddings.weight[indices]
                norms = embs.norm(dim=1, keepdim=True)
                # Hard projection: if norm >= 1-eps, scale it down
                # Use where to only scale embeddings that need it
                needs_projection = norms >= (1 - eps)
                scale = torch.where(
                    needs_projection,
                    (1 - eps) / (norms + eps),
                    torch.ones_like(norms)
                )
                self.embeddings.weight[indices] = embs * scale
            else:
                # Project all embeddings
                norms = self.embeddings.weight.norm(dim=1, keepdim=True)
                needs_projection = norms >= (1 - eps)
                scale = torch.where(
                    needs_projection,
                    (1 - eps) / (norms + eps),
                    torch.ones_like(norms)
                )
                self.embeddings.weight.mul_(scale)


class HierarchicalDataLoader:
    """
    Data loader with depth-aware sampling and hard negatives.
    """
    
    def __init__(self, training_data, n_nodes, batch_size=32, 
                 n_negatives=50, depth_stratify=True):
        self.training_data = training_data  # List of dicts with metadata
        self.n_nodes = n_nodes
        self.batch_size = batch_size
        self.n_negatives = n_negatives
        self.depth_stratify = depth_stratify
        
        # Build index by depth for stratified sampling
        if depth_stratify:
            self.depth_buckets = defaultdict(list)
            for i, item in enumerate(training_data):
                depth_diff = item['depth_diff']
                self.depth_buckets[depth_diff].append(i)
            print(f"  ✓ Created {len(self.depth_buckets)} depth buckets for sampling")
        
        # Build node → siblings map for hard negatives
        self._build_sibling_map()
    
    def _build_sibling_map(self):
        """Build map: node → nodes at same depth (for hard negatives)."""
        print("  Building sibling map for hard negatives...")
        
        depth_to_nodes = defaultdict(set)
        for item in self.training_data:
            depth_to_nodes[item['descendant_depth']].add(item['descendant_idx'])
        
        self.sibling_map = {}
        for depth, nodes in depth_to_nodes.items():
            nodes_list = list(nodes)
            for node in nodes_list:
                # Siblings = other nodes at same depth
                self.sibling_map[node] = [n for n in nodes_list if n != node]
        
        print(f"  ✓ Built sibling map for hard negatives")
    
    def __len__(self):
        return len(self.training_data) // self.batch_size
    
    def __iter__(self):
        """Iterate over batches with depth-aware sampling."""
        indices = list(range(len(self.training_data)))
        
        if self.depth_stratify:
            # Stratified sampling: mix shallow and deep pairs
            np.random.shuffle(indices)
        else:
            # Random sampling
            np.random.shuffle(indices)
        
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch = [self.training_data[idx] for idx in batch_indices]
            
            # Extract data (vectorized for speed)
            batch_size = len(batch)
            ancestors = np.zeros(batch_size, dtype=np.int64)
            descendants = np.zeros(batch_size, dtype=np.int64)
            depths = np.zeros(batch_size, dtype=np.float32)
            
            for j, item in enumerate(batch):
                ancestors[j] = item['ancestor_idx']
                descendants[j] = item['descendant_idx']
                depths[j] = item['depth_diff']
            
            # Convert to tensors once
            ancestors = torch.from_numpy(ancestors)
            descendants = torch.from_numpy(descendants)
            depths = torch.from_numpy(depths)
            
            # Sample hard negatives (cousins at same depth)
            negatives = np.zeros((batch_size, self.n_negatives), dtype=np.int64)
            for j, item in enumerate(batch):
                desc_idx = item['descendant_idx']
                
                # Get siblings (nodes at same depth)
                siblings = self.sibling_map.get(desc_idx, [])
                
                if len(siblings) >= self.n_negatives:
                    # Sample from siblings (hard negatives)
                    negatives[j] = np.random.choice(siblings, self.n_negatives, replace=False)
                else:
                    # Mix siblings + random negatives
                    n_sibling = len(siblings)
                    n_random = self.n_negatives - n_sibling
                    if n_sibling > 0:
                        negatives[j, :n_sibling] = siblings
                        negatives[j, n_sibling:] = np.random.choice(self.n_nodes, n_random, replace=False)
                    else:
                        negatives[j] = np.random.choice(self.n_nodes, self.n_negatives, replace=False)
            
            negatives = torch.from_numpy(negatives)
            
            yield ancestors, descendants, negatives, depths


def ranking_loss_with_margin(model, ancestors, descendants, negatives, 
                             depths, margin=0.1, depth_weight=True):
    """
    Ranking loss with margin and optional depth weighting.
    
    Loss encourages:
    - d(ancestor, descendant) < d(ancestor, negative) + margin
    - Deeper pairs get higher weight (they're more informative)
    """
    # Get embeddings
    anc_emb = model(ancestors)  # (batch, dim)
    desc_emb = model(descendants)  # (batch, dim)
    neg_emb = model(negatives)  # (batch, n_neg, dim)
    
    # Positive distances (ancestor → descendant)
    pos_dist = model.poincare_distance(anc_emb, desc_emb)  # (batch,)
    
    # Negative distances (ancestor → each negative)
    # Expand anc_emb to match negatives shape
    anc_emb_expanded = anc_emb.unsqueeze(1).expand_as(neg_emb)  # (batch, n_neg, dim)
    neg_dist = model.poincare_distance(anc_emb_expanded, neg_emb)  # (batch, n_neg)
    
    # Margin ranking loss: max(0, pos_dist - neg_dist + margin)
    losses = torch.relu(pos_dist.unsqueeze(1) - neg_dist + margin)  # (batch, n_neg)
    loss = losses.mean(dim=1)  # Average over negatives: (batch,)
    
    # Depth weighting: deeper pairs are more important
    if depth_weight:
        # Weight = sqrt(depth) to emphasize deep pairs without over-weighting
        weights = torch.sqrt(depths + 1)  # +1 to avoid zero weight
        weights = weights / weights.mean()  # Normalize
        loss = loss * weights
    
    return loss.mean()


def radial_regularizer(model, idx_to_depth_tensor, target_radii_tensor, lambda_reg=0.01):
    """
    Vectorized radial regularizer to keep nodes at expected radius based on depth.
    
    Encourages: ||embedding|| ≈ f(depth)
    where f(depth) = 0.1 + (depth/max_depth) * 0.85
    
    Args:
        idx_to_depth_tensor: Tensor of indices to regularize
        target_radii_tensor: Tensor of target radii for each index
    """
    if len(idx_to_depth_tensor) == 0:
        return torch.tensor(0.0, device=model.embeddings.weight.device)
    
    # Get embeddings for these indices
    embs = model.embeddings.weight[idx_to_depth_tensor]  # (n, dim)
    
    # Compute actual radii
    actual_radii = embs.norm(dim=1)  # (n,)
    
    # L2 penalty
    reg_loss = ((actual_radii - target_radii_tensor) ** 2).mean()
    
    return lambda_reg * reg_loss


def train_hierarchical(model, dataloader, optimizer, n_epochs, 
                       idx_to_depth, max_depth, device, 
                       margin=0.2, lambda_reg=0.01, early_stopping_patience=3,
                       checkpoint_base=None):
    """Train with hierarchical constraints and early stopping."""
    
    model.to(device)
    model.train()
    
    print(f"\nStarting hierarchical training...")
    print(f"  Margin: {margin}")
    print(f"  Radial regularization: λ={lambda_reg}")
    print(f"  Early stopping patience: {early_stopping_patience} epochs")
    print(f"  Device: {device}")
    print()
    
    # Precompute regularizer tensors (once, not every batch!)
    print("Precomputing radial regularization tensors...")
    reg_indices = []
    reg_target_radii = []
    for idx, depth in idx_to_depth.items():
        if idx < model.n_nodes:
            reg_indices.append(idx)
            target_radius = 0.1 + (depth / max_depth) * 0.85
            reg_target_radii.append(target_radius)
    
    reg_indices_tensor = torch.LongTensor(reg_indices).to(device)
    reg_target_radii_tensor = torch.FloatTensor(reg_target_radii).to(device)
    print(f"  ✓ Will regularize {len(reg_indices):,} nodes")
    
    # Early stopping tracking
    best_loss = float('inf')
    epochs_without_improvement = 0
    
    # Checkpoint management (keep only last 5 epoch checkpoints)
    checkpoint_queue = deque(maxlen=5)
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        epoch_reg_loss = 0
        n_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}")
        
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
            
            # Radial regularizer (using precomputed tensors)
            reg_loss = radial_regularizer(model, reg_indices_tensor, reg_target_radii_tensor, lambda_reg)
            
            # Total loss
            total_loss = loss + reg_loss
            
            # Backward
            total_loss.backward()
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Project back to ball (only modified embeddings for efficiency)
            # Collect all unique indices that were updated
            updated_indices = torch.cat([ancestors, descendants, negatives.flatten()])
            updated_indices = torch.unique(updated_indices)
            model.project_to_ball(updated_indices)
            
            # Periodic full projection to catch any stragglers
            # Every 500 batches, project ALL embeddings
            if n_batches % 500 == 0:
                model.project_to_ball(indices=None)  # Project all
            
            # Track
            epoch_loss += loss.item()
            epoch_reg_loss += reg_loss.item()
            n_batches += 1
            
            # Update progress
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'reg': f'{reg_loss.item():.4f}'
            })
        
        # Epoch summary
        avg_loss = epoch_loss / n_batches
        avg_reg = epoch_reg_loss / n_batches
        
        # FINAL projection: enforce ALL embeddings are inside ball at epoch end
        model.project_to_ball(indices=None)
        
        print(f"Epoch {epoch+1:3d}: Loss={avg_loss:.4f}, Reg={avg_reg:.4f}")
        
        # Check norms (after final projection)
        norms = model.embeddings.weight.norm(dim=1)
        outside_count = (norms >= 1.0).sum().item()
        print(f"  Norms: min={norms.min():.4f}, mean={norms.mean():.4f}, max={norms.max():.4f}")
        if outside_count > 0:
            print(f"  ⚠️  {outside_count} embeddings still outside ball (should be 0!)")
        
        # Save checkpoint every epoch
        if checkpoint_base:
            checkpoint_path = checkpoint_base.replace('.pth', f'_epoch{epoch+1}.pth')
            torch.save({
                'state_dict': {'lt.weight': model.embeddings.weight},
                'embeddings': model.embeddings.weight,
                'epoch': epoch + 1,
                'loss': avg_loss,
                'reg_loss': avg_reg,
                'best_loss': best_loss,
                'epochs_without_improvement': epochs_without_improvement,
            }, checkpoint_path)
            
            # Manage checkpoint queue (keep only last 5)
            if len(checkpoint_queue) >= checkpoint_queue.maxlen:
                old_checkpoint = checkpoint_queue[0]
                if os.path.exists(old_checkpoint):
                    os.remove(old_checkpoint)
                    print(f"  🗑️  Deleted old: {os.path.basename(old_checkpoint)}")
            
            checkpoint_queue.append(checkpoint_path)
            print(f"  💾 Saved: {os.path.basename(checkpoint_path)} (keeping last {len(checkpoint_queue)})")
        
        # Early stopping check
        if avg_loss < best_loss:
            improvement = best_loss - avg_loss
            print(f"  ✓ Loss improved by {improvement:.6f}")
            best_loss = avg_loss
            epochs_without_improvement = 0
            
            # Save best model
            if checkpoint_base:
                best_checkpoint = checkpoint_base.replace('.pth', '_best.pth')
                torch.save({
                    'state_dict': {'lt.weight': model.embeddings.weight},
                    'embeddings': model.embeddings.weight,
                    'epoch': epoch + 1,
                    'loss': avg_loss,
                    'reg_loss': avg_reg,
                }, best_checkpoint)
                print(f"  💾 Best model saved: {best_checkpoint}")
        else:
            epochs_without_improvement += 1
            print(f"  ✗ No improvement ({epochs_without_improvement}/{early_stopping_patience})")
            
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\n🛑 Early stopping triggered after {epoch+1} epochs")
                print(f"   Best loss: {best_loss:.6f}")
                break


def main():
    parser = argparse.ArgumentParser(description='Hierarchical Poincaré Training')
    parser.add_argument('--data', default='data/taxonomy_edges_small_transitive.pkl',
                       help='Training data (pickle file with metadata)')
    parser.add_argument('--checkpoint', default='taxonomy_model_hierarchical.pth',
                       help='Output checkpoint path')
    parser.add_argument('--dim', type=int, default=10,
                       help='Embedding dimension')
    parser.add_argument('--epochs', type=int, default=10000,
                       help='Maximum number of epochs (early stopping will trigger)')
    parser.add_argument('--early-stopping', type=int, default=3,
                       help='Early stopping patience (stop if no improvement for N epochs)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--n-negatives', type=int, default=50,
                       help='Number of negative samples')
    parser.add_argument('--lr', type=float, default=0.005,
                       help='Learning rate (reduced to prevent escaping ball)')
    parser.add_argument('--margin', type=float, default=0.2,
                       help='Ranking loss margin')
    parser.add_argument('--lambda-reg', type=float, default=0.1,
                       help='Radial regularization weight (increased from 0.01 to keep embeddings in ball)')
    parser.add_argument('--gpu', type=int, default=-1,
                       help='GPU device (-1 for CPU)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("HIERARCHICAL POINCARÉ TRAINING")
    print("="*80)
    print()
    
    # Device (force CPU for stability on macOS - MPS can hang with custom ops)
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU: cuda:{args.gpu}")
    else:
        device = torch.device('cpu')
        print(f"Using CPU (recommended for macOS)")
    
    # Note: MPS disabled because it can hang with hyperbolic distance operations
    
    # Load training data
    print(f"Loading training data from {args.data}...")
    with open(args.data, 'rb') as f:
        training_data = pickle.load(f)
    print(f"  ✓ Loaded {len(training_data):,} training pairs")
    
    # Get number of nodes and max depth
    n_nodes = max(max(item['ancestor_idx'], item['descendant_idx']) 
                  for item in training_data) + 1
    max_depth = max(item['descendant_depth'] for item in training_data)
    
    print(f"  Nodes: {n_nodes:,}")
    print(f"  Max depth: {max_depth}")
    
    # Build idx → depth mapping for initialization
    idx_to_depth = {}
    for item in training_data:
        idx_to_depth[item['descendant_idx']] = item['descendant_depth']
        if item['ancestor_idx'] not in idx_to_depth:
            idx_to_depth[item['ancestor_idx']] = item['ancestor_depth']
    
    # Create model with depth-aware initialization
    print("\nCreating model...")
    model = HierarchicalPoincareEmbedding(
        n_nodes=n_nodes,
        dim=args.dim,
        max_depth=max_depth,
        init_depth_data=idx_to_depth
    )
    
    # Create dataloader
    print("\nCreating dataloader...")
    dataloader = HierarchicalDataLoader(
        training_data=training_data,
        n_nodes=n_nodes,
        batch_size=args.batch_size,
        n_negatives=args.n_negatives,
        depth_stratify=True
    )
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Train
    train_hierarchical(
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
    
    # Save
    print(f"\nSaving model to {args.checkpoint}...")
    torch.save({
        'state_dict': {'lt.weight': model.embeddings.weight},
        'embeddings': model.embeddings.weight,
        'n_nodes': n_nodes,
        'dim': args.dim,
        'max_depth': max_depth,
    }, args.checkpoint)
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)
    print(f"Model saved to: {args.checkpoint}")
    print()
    print("Next: Run analyze_hierarchy_hyperbolic.py to verify improvements!")


if __name__ == "__main__":
    main()
