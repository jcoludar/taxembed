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
from scipy import stats
from tqdm import tqdm

from taxembed.optim import RiemannianAdam

# Import the hierarchical model and components
from train_hierarchical import (
    HierarchicalPoincareEmbedding,
    HierarchicalDataLoader,
    TrainingPairs,
    ranking_loss_with_margin,
    radial_regularizer
)


def _pairwise_poincare_distance(embs, eps=1e-5):
    """(N, D) Poincare embeddings -> (N, N) distance matrix."""
    norm_sq = (embs ** 2).sum(dim=1)
    norm_sq = norm_sq.clamp(0, 1 - eps)
    diff_sq = torch.cdist(embs, embs) ** 2
    denom = (1 - norm_sq).unsqueeze(1) * (1 - norm_sq).unsqueeze(0)
    return torch.acosh((1 + 2 * diff_sq / (denom + eps)).clamp(min=1 + eps))


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
        print("\n" + "="*140)
        print(f"{'Epoch':>6} | {'Loss':>10} | {'ΔLoss':>10} | {'Improve':>8} | "
              f"{'Reg':>8} | {'MaxNorm':>8} | {'Outside':>7} | "
              f"{'DepthCorr':>9} | {'Hier%':>6} | {'kNN%':>5} | {'Sep':>5} | {'Status':>10}")
        print("="*140)
    
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

        # Depth-norm correlation coloring
        depth_corr = metrics.get('depth_norm_corr', 0.0)
        if depth_corr > 0.3:
            corr_color = "\033[92m"  # Green
        elif depth_corr > 0.0:
            corr_color = "\033[93m"  # Yellow
        else:
            corr_color = "\033[91m"  # Red
        reset = "\033[0m"

        hier_pct = metrics.get('hierarchy_pct', 0.0)
        knn_purity = metrics.get('knn_purity', 0.0)
        sep_ratio = metrics.get('class_sep_ratio', 0.0)

        # Class separation coloring
        if sep_ratio > 1.2:
            sep_color = "\033[92m"  # Green
        elif sep_ratio > 1.0:
            sep_color = "\033[93m"  # Yellow
        else:
            sep_color = "\033[91m"  # Red

        print(f"{epoch:6d} | "
              f"{metrics['loss']:10.6f} | "
              f"{improve_color}{delta_str:>10}{reset_color} | "
              f"{improve_color}{pct_str:>8}{reset_color} | "
              f"{metrics['reg_loss']:8.6f} | "
              f"{metrics['max_norm']:8.4f} | "
              f"{outside_pct:6.2f}% | "
              f"{corr_color}{depth_corr:+8.3f}{reset} | "
              f"{hier_pct:5.1f}% | "
              f"{knn_purity*100:4.1f}% | "
              f"{sep_color}{sep_ratio:5.2f}{reset} | "
              f"{improve_color}{status:>10}{reset_color}")

        # Additional info every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            rank_loss = metrics.get('ranking_loss', metrics['loss'])
            reg_loss = metrics['reg_loss']
            ratio_str = f"{rank_loss / (reg_loss + 1e-8):.1f}" if reg_loss > 1e-8 else "inf"
            print(f"       └─ Best: {self.best_loss:.6f} @ epoch {self.best_epoch} | "
                  f"Norms: [{metrics['min_norm']:.4f}, {metrics['mean_norm']:.4f}, {metrics['max_norm']:.4f}] | "
                  f"Rank/Reg: {ratio_str}")
    
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


def compute_hierarchy_metrics(model, idx_to_depth, training_data, device, n_sample=1000):
    """Compute depth-norm correlation and hierarchy preservation percentage.

    Uses Poincare-space embeddings (via model.get_poincare_embeddings()) so
    norms are correct even with euclidean_param.

    Supports both TrainingPairs and legacy list-of-dicts as training_data.
    """
    with torch.no_grad():
        # Get Poincare-space embeddings
        poincare_embs = model.get_poincare_embeddings()

        # Depth-norm Pearson correlation
        indices = sorted(idx_to_depth.keys())
        depths = np.array([idx_to_depth[i] for i in indices])
        norms = poincare_embs[indices].norm(dim=1).cpu().numpy()
        if len(depths) > 1 and np.std(depths) > 0 and np.std(norms) > 0:
            corr, _ = stats.pearsonr(depths, norms)
        else:
            corr = 0.0

        # Hierarchy preservation: sample pairs, check ancestor_norm < descendant_norm
        n_pairs = len(training_data)
        if n_pairs == 0:
            return corr, 0.0

        sample_size = min(n_sample, n_pairs)
        sample_idx = np.random.choice(n_pairs, sample_size, replace=False)

        if isinstance(training_data, TrainingPairs):
            anc_indices = training_data.ancestor_idx[sample_idx].astype(np.int64)
            desc_indices = training_data.descendant_idx[sample_idx].astype(np.int64)
            anc_norms = poincare_embs[torch.from_numpy(anc_indices)].norm(dim=1).cpu().numpy()
            desc_norms = poincare_embs[torch.from_numpy(desc_indices)].norm(dim=1).cpu().numpy()
            correct = int(np.sum(anc_norms < desc_norms))
        else:
            correct = 0
            for si in sample_idx:
                item = training_data[si]
                anc_norm = poincare_embs[item['ancestor_idx']].norm().item()
                desc_norm = poincare_embs[item['descendant_idx']].norm().item()
                if anc_norm < desc_norm:
                    correct += 1
        hier_pct = (correct / sample_size) * 100

    return corr, hier_pct


def _build_class_labels(training_data, n_nodes):
    """Assign each node to its top-level ancestor (depth=1) for class separation.

    Reconstructs the parent-child tree from dd=1 pairs, finds root children,
    then propagates labels downward.  Returns node_idx → class_label array
    (class_label = the depth-1 ancestor idx, or -1 for unlabeled).
    """
    pairs = training_data if isinstance(training_data, TrainingPairs) else TrainingPairs.from_list(training_data)

    # Extract parent-child edges (dd=1)
    dd1_mask = pairs.depth_diff == 1
    parents = pairs.ancestor_idx[dd1_mask]
    children = pairs.descendant_idx[dd1_mask]

    # Build child → parent map
    child_to_parent = {}
    for i in range(len(parents)):
        c, p = int(children[i]), int(parents[i])
        child_to_parent[c] = p

    # Find root (depth=0) and its direct children (depth=1) = class anchors
    depth0_nodes = set()
    for i in range(len(pairs)):
        if int(pairs.ancestor_depth[i]) == 0:
            depth0_nodes.add(int(pairs.ancestor_idx[i]))

    root_children = set()
    for c, p in child_to_parent.items():
        if p in depth0_nodes:
            root_children.add(c)

    if len(root_children) < 2:
        return None  # Can't compute class separation with <2 classes

    # Trace each node up to its depth-1 ancestor
    labels = np.full(n_nodes, -1, dtype=np.int32)
    for rc in root_children:
        labels[rc] = rc

    # BFS downward: build parent → children map
    parent_to_children = defaultdict(list)
    for c, p in child_to_parent.items():
        parent_to_children[p].append(c)

    from collections import deque
    for rc in root_children:
        queue = deque([rc])
        while queue:
            node = queue.popleft()
            for child in parent_to_children.get(node, []):
                if labels[child] == -1:
                    labels[child] = rc
                    queue.append(child)

    n_labeled = int(np.sum(labels >= 0))
    n_classes = len(root_children)
    return labels, n_labeled, n_classes


def compute_class_separation(model, class_labels_info, device, n_sample=500, k=10):
    """Compute kNN taxonomic purity and class separation ratio.

    Uses Poincare distance (not Euclidean) for correct evaluation in
    hyperbolic space.

    kNN purity: for each sampled node, what fraction of its k nearest
    neighbors share the same top-level class?  Directly measures clustering.

    Class separation ratio: mean(inter-class dist) / mean(intra-class dist).
    Values > 1 mean classes are separated in embedding space.
    """
    if class_labels_info is None:
        return 0.0, 0.0

    labels, n_labeled, n_classes = class_labels_info
    if n_classes < 2:
        return 0.0, 0.0

    with torch.no_grad():
        # Sample labeled nodes
        labeled_mask = labels >= 0
        labeled_indices = np.where(labeled_mask)[0]
        if len(labeled_indices) < 4:
            return 0.0, 0.0

        sample_size = min(n_sample, len(labeled_indices))
        sampled = np.random.choice(labeled_indices, sample_size, replace=False)
        sampled_labels = labels[sampled]

        # Get Poincare-space embeddings for sampled nodes
        poincare_embs = model.get_poincare_embeddings()
        embs = poincare_embs[torch.from_numpy(sampled.astype(np.int64))].cpu()

        # Compute pairwise Poincare distances
        dists = _pairwise_poincare_distance(embs)

        # kNN purity
        k_actual = min(k, sample_size - 1)
        _, knn_indices = dists.topk(k_actual + 1, largest=False, dim=1)  # +1 for self
        knn_indices = knn_indices[:, 1:]  # exclude self

        correct = 0
        total = 0
        for i in range(sample_size):
            my_label = sampled_labels[i]
            neighbor_labels = sampled_labels[knn_indices[i].numpy()]
            correct += int(np.sum(neighbor_labels == my_label))
            total += k_actual
        knn_purity = correct / total if total > 0 else 0.0

        # Class separation ratio: inter / intra distances
        dists_np = dists.numpy()
        same_class = sampled_labels[:, None] == sampled_labels[None, :]
        # Exclude diagonal
        np.fill_diagonal(same_class, False)
        diff_class = ~same_class
        np.fill_diagonal(diff_class, False)

        intra_dists = dists_np[same_class]
        inter_dists = dists_np[diff_class]

        if len(intra_dists) > 0 and len(inter_dists) > 0:
            mean_intra = float(np.mean(intra_dists))
            mean_inter = float(np.mean(inter_dists))
            sep_ratio = mean_inter / (mean_intra + 1e-8)
        else:
            sep_ratio = 0.0

    return knn_purity, sep_ratio


def compute_multiscale_knn(model, training_data, device, n_sample=500, k=10):
    """Compute kNN purity at multiple depth levels (phylum, class, order, family).

    Assigns labels by ancestor at each target depth and computes kNN purity
    for each level. Useful for evaluating large-scale embeddings where
    single top-level class metrics are too coarse.

    Returns dict mapping depth → kNN purity.
    """
    pairs = training_data if isinstance(training_data, TrainingPairs) else TrainingPairs.from_list(training_data)
    n_nodes = pairs.n_nodes

    # Build child→parent from dd=1 pairs
    dd1_mask = pairs.depth_diff == 1
    parents = pairs.ancestor_idx[dd1_mask]
    children = pairs.descendant_idx[dd1_mask]
    child_to_parent: dict[int, int] = {}
    for i in range(len(parents)):
        child_to_parent[int(children[i])] = int(parents[i])

    # Build node depths
    node_depth: dict[int, int] = {}
    for i in range(len(pairs)):
        node_depth[int(pairs.descendant_idx[i])] = int(pairs.descendant_depth[i])
        anc = int(pairs.ancestor_idx[i])
        if anc not in node_depth:
            node_depth[anc] = int(pairs.ancestor_depth[i])

    max_depth = max(node_depth.values()) if node_depth else 0
    # Evaluate at depths 1, 2, 3, 4 (phylum/class/order/family-ish)
    eval_depths = [d for d in [1, 2, 3, 4] if d <= max_depth]
    if not eval_depths:
        return {}

    # For each eval depth, trace each node up to its ancestor at that depth
    results: dict[int, float] = {}
    for target_depth in eval_depths:
        labels = np.full(n_nodes, -1, dtype=np.int32)
        for node_idx, depth in node_depth.items():
            if node_idx >= n_nodes:
                continue
            current = node_idx
            current_depth = depth
            while current_depth > target_depth:
                parent = child_to_parent.get(current)
                if parent is None:
                    break
                current = parent
                current_depth = node_depth.get(current, 0)
            if current_depth == target_depth:
                labels[node_idx] = current

        # Compute kNN purity for this depth level
        labeled_mask = labels >= 0
        labeled_indices = np.where(labeled_mask)[0]
        n_classes = len(set(labels[labeled_indices]))
        if len(labeled_indices) < 4 or n_classes < 2:
            results[target_depth] = 0.0
            continue

        sample_size = min(n_sample, len(labeled_indices))
        sampled = np.random.choice(labeled_indices, sample_size, replace=False)
        sampled_labels = labels[sampled]

        with torch.no_grad():
            poincare_embs = model.get_poincare_embeddings()
            embs = poincare_embs[torch.from_numpy(sampled.astype(np.int64))].cpu()
            dists = _pairwise_poincare_distance(embs)
            k_actual = min(k, sample_size - 1)
            _, knn_indices = dists.topk(k_actual + 1, largest=False, dim=1)
            knn_indices = knn_indices[:, 1:]

            correct = 0
            total = 0
            for i in range(sample_size):
                my_label = sampled_labels[i]
                neighbor_labels = sampled_labels[knn_indices[i].numpy()]
                correct += int(np.sum(neighbor_labels == my_label))
                total += k_actual

        results[target_depth] = correct / total if total > 0 else 0.0

    return results


def parse_curriculum_phases(spec):
    """Parse curriculum phase spec like '1:1,20:3,50:None' into [(epoch, max_depth_diff)]."""
    phases = []
    for part in spec.split(","):
        epoch_str, val_str = part.strip().split(":")
        epoch = int(epoch_str)
        val = None if val_str.strip() == "None" else int(val_str)
        phases.append((epoch, val))
    return sorted(phases, key=lambda x: x[0])


def auto_curriculum_phases(max_depth: int, n_epochs: int) -> list[tuple[int, int | None]]:
    """Generate curriculum phases scaled to tree depth and epoch budget.

    Strategy:
    - First 20% of epochs: depth_diff <= 1 (parent-child only)
    - Next 20%: depth_diff <= max_depth // 4
    - Next 20%: depth_diff <= max_depth // 2
    - Final 40%: full dataset (all pairs)
    """
    e1 = 1
    e2 = max(2, int(n_epochs * 0.2))
    e3 = max(e2 + 1, int(n_epochs * 0.4))
    e4 = max(e3 + 1, int(n_epochs * 0.6))
    return [
        (e1, 1),
        (e2, max(1, max_depth // 4)),
        (e3, max(2, max_depth // 2)),
        (e4, None),  # full dataset
    ]


def _build_class_weights(training_data, n_nodes, device):
    """Build per-node class weights: inverse sqrt of class frequency.

    Returns a tensor of shape [n_nodes] where each node's weight is
    sqrt(max_class_count / its_class_count).  This upweights minority
    classes (Bivalvia ~2x, Cephalopoda ~4.5x) without the extreme
    oversampling that strict class-balanced does.
    """
    info = _build_class_labels(training_data, n_nodes)
    if info is None:
        return None

    labels, n_labeled, n_classes = info
    if n_classes < 2:
        return None

    # Count nodes per class
    from collections import Counter
    labeled_mask = labels >= 0
    counts = Counter(int(labels[i]) for i in range(n_nodes) if labeled_mask[i])
    max_count = max(counts.values())

    # Assign weight = sqrt(max_count / class_count) per node, capped at 10x
    max_weight = 10.0
    weights = torch.ones(n_nodes, device=device)
    for cls, cnt in counts.items():
        w = min((max_count / cnt) ** 0.5, max_weight)
        cls_mask = labels == cls
        weights[torch.from_numpy(np.where(cls_mask)[0])] = w

    # Report
    print(f"  ✓ Class weights (sqrt inverse freq, capped at {max_weight:.0f}x):")
    for cls, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        w = min((max_count / cnt) ** 0.5, max_weight)
        print(f"      class {cls} ({cnt:>6,} nodes): weight={w:.2f}x")

    return weights


def train_with_visualization(model, dataloader, optimizer, n_epochs,
                             idx_to_depth, max_depth, device,
                             margin=0.2, lambda_reg=0.1, early_stopping_patience=15,
                             checkpoint_base=None, training_data=None,
                             use_riemannian=False,
                             curriculum=False, curriculum_phases=None,
                             burnin=0, burnin_multiplier=0.1,
                             radial_nudge=0.05,
                             depth_scale_margin=False, margin_min=0.05,
                             margin_max=1.0, radial_schedule='linear',
                             grad_accum_steps=1, use_amp=False,
                             class_weighted_loss=False):
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
    print(f"  Optimizer: {'Riemannian' if use_riemannian else 'Euclidean'}")
    if radial_nudge > 0:
        print(f"  Radial nudge: {radial_nudge}")
    if burnin > 0:
        print(f"  Burn-in: {burnin} epochs at {burnin_multiplier}x LR")
    if curriculum and curriculum_phases:
        print(f"  Curriculum: {curriculum_phases}")
    if depth_scale_margin:
        print(f"  Depth-scaled margin: [{margin_min}, {margin_max}]")
    if radial_schedule != 'linear':
        print(f"  Radial schedule: {radial_schedule}")

    # Import target_radius helper
    from train_hierarchical import target_radius as _target_radius

    # Precompute regularizer tensors
    print("\nPrecomputing regularization targets...")
    reg_indices = []
    reg_target_radii = []
    for idx, depth in idx_to_depth.items():
        if idx < model.n_nodes:
            reg_indices.append(idx)
            reg_target_radii.append(_target_radius(depth, max_depth, radial_schedule))

    reg_indices_tensor = torch.LongTensor(reg_indices).to(device)
    reg_target_radii_tensor = torch.FloatTensor(reg_target_radii).to(device)
    print(f"  ✓ Regularizing {len(reg_indices):,} nodes")

    # Pre-build radial target lookup for nudge (one target radius per node)
    radial_target_lookup = torch.zeros(model.n_nodes, device=device)
    for idx, depth in idx_to_depth.items():
        if idx < model.n_nodes:
            radial_target_lookup[idx] = _target_radius(depth, max_depth, radial_schedule)

    # Build class labels for separation metrics
    class_labels_info = None
    if training_data is not None:
        class_labels_info = _build_class_labels(training_data, model.n_nodes)
        if class_labels_info is not None:
            _, n_labeled, n_classes = class_labels_info
            print(f"  ✓ Class labels: {n_classes} top-level classes, {n_labeled:,} labeled nodes")
        else:
            print(f"  ⚠️  Could not build class labels (<2 top-level classes)")

    # Build class weight tensor for loss weighting
    class_weight_tensor = None
    if class_weighted_loss and training_data is not None:
        class_weight_tensor = _build_class_weights(training_data, model.n_nodes, device)
        if class_weight_tensor is None:
            print(f"  ⚠️  Could not build class weights")

    # Metrics tracker
    tracker = MetricsTracker()
    tracker.print_header()

    # Early stopping — use quality score (not raw loss) when curriculum is active.
    # Raw loss across curriculum phases isn't comparable.
    best_quality = -float('inf')
    best_quality_epoch = 0
    epochs_without_improvement = 0
    checkpoint_queue = deque(maxlen=5)

    # Mixed precision (AMP) setup for GPU training
    amp_enabled = use_amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if amp_enabled else None
    if amp_enabled:
        print(f"  Mixed precision: enabled (GradScaler)")
    if grad_accum_steps > 1:
        print(f"  Gradient accumulation: {grad_accum_steps} steps (effective batch = {grad_accum_steps * dataloader.batch_size})")

    # Parse curriculum phases
    parsed_phases = None
    if curriculum and curriculum_phases:
        if curriculum_phases == "auto":
            parsed_phases = auto_curriculum_phases(max_depth, n_epochs)
            phase_str = ", ".join(
                f"epoch {e}: dd<={d if d is not None else 'all'}" for e, d in parsed_phases
            )
            print(f"  Auto curriculum: {phase_str}")
        else:
            parsed_phases = parse_curriculum_phases(curriculum_phases)

    using_curriculum = parsed_phases is not None

    # Store base LR for burn-in
    base_lr = optimizer.param_groups[0]['lr']

    for epoch in range(1, n_epochs + 1):
        # --- Burn-in LR adjustment ---
        if burnin > 0 and epoch <= burnin:
            for pg in optimizer.param_groups:
                pg['lr'] = base_lr * burnin_multiplier
        elif burnin > 0 and epoch == burnin + 1:
            for pg in optimizer.param_groups:
                pg['lr'] = base_lr

        # --- Curriculum phase switching ---
        if parsed_phases is not None:
            # Find the active phase for this epoch
            active_max_dd = None
            for phase_epoch, max_dd in parsed_phases:
                if epoch >= phase_epoch:
                    active_max_dd = max_dd
            if active_max_dd is None:
                dataloader.clear_curriculum()
            else:
                dataloader.set_curriculum_phase(active_max_dd)

        epoch_loss = 0
        epoch_ranking_loss = 0
        epoch_reg_loss = 0
        n_batches = 0

        # Progress bar for batches
        pbar = tqdm(dataloader,
                   desc=f"Epoch {epoch:3d}/{n_epochs}",
                   bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                   ncols=100,
                   leave=False,
                   dynamic_ncols=True,
                   mininterval=0.5)

        for ancestors, descendants, negatives, depths in pbar:
            ancestors = ancestors.to(device)
            descendants = descendants.to(device)
            negatives = negatives.to(device)
            depths = depths.to(device)

            if n_batches % grad_accum_steps == 0:
                optimizer.zero_grad()

            # Forward pass (with optional AMP autocast)
            with torch.amp.autocast('cuda', enabled=amp_enabled):
                # Ranking loss
                loss = ranking_loss_with_margin(
                    model, ancestors, descendants, negatives,
                    depths, margin=margin, depth_weight=True,
                    depth_scale_margin=depth_scale_margin,
                    margin_min=margin_min, margin_max=margin_max,
                    max_depth_diff=max_depth,
                    class_weights=class_weight_tensor,
                )

                # Radial regularizer
                reg_loss = radial_regularizer(model, reg_indices_tensor,
                                             reg_target_radii_tensor, lambda_reg)

                # Total loss (scaled for gradient accumulation)
                total_loss = (loss + reg_loss) / grad_accum_steps

            # Backward pass
            if scaler is not None:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            # Step only every grad_accum_steps batches
            if (n_batches + 1) % grad_accum_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            # Per-batch projection to keep embeddings inside the Poincaré ball
            updated_indices = torch.cat([ancestors, descendants, negatives.flatten()])
            updated_indices = torch.unique(updated_indices)
            model.project_to_ball(updated_indices)

            if n_batches % 500 == 0:
                model.project_to_ball(indices=None)

            # Radial nudge: gently push norms toward depth-based targets
            # Only changes norms, never directions — preserves angular structure
            if radial_nudge > 0:
                with torch.no_grad():
                    if model.euclidean_param:
                        # Nudge in Poincaré space, then map back to z-space
                        # 1. Get current z and compute Poincaré norms
                        z = model.embeddings.weight[updated_indices]
                        z_norm = z.norm(dim=1, keepdim=True).clamp(min=1e-8)
                        z_dir = z / z_norm
                        poincare_norm = torch.tanh(z_norm / 2)
                        # 2. Nudge Poincaré norm toward target
                        target_norms = radial_target_lookup[updated_indices].unsqueeze(1)
                        new_poincare_norm = poincare_norm * (1 - radial_nudge) + target_norms * radial_nudge
                        new_poincare_norm = new_poincare_norm.clamp(1e-6, 1 - 1e-6)
                        # 3. Inverse tanh map: ||z_new|| = 2 * arctanh(||x_new||)
                        new_z_norm = 2 * torch.arctanh(new_poincare_norm)
                        # 4. Keep z direction, update z norm
                        model.embeddings.weight[updated_indices] = z_dir * new_z_norm
                    else:
                        embs = model.embeddings.weight[updated_indices]
                        norms = embs.norm(dim=1, keepdim=True).clamp(min=1e-8)
                        directions = embs / norms
                        target_norms = radial_target_lookup[updated_indices].unsqueeze(1)
                        new_norms = norms * (1 - radial_nudge) + target_norms * radial_nudge
                        model.embeddings.weight[updated_indices] = directions * new_norms

            epoch_loss += loss.item()
            epoch_ranking_loss += loss.item()
            epoch_reg_loss += reg_loss.item()
            n_batches += 1

            # Update progress bar with current batch metrics
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'reg': f'{reg_loss.item():.4f}'
            })

        # Epoch-end safety projection (always)
        model.project_to_ball(indices=None)

        # Compute metrics (use Poincare-space norms for display)
        avg_loss = epoch_loss / n_batches
        avg_ranking = epoch_ranking_loss / n_batches
        avg_reg = epoch_reg_loss / n_batches
        with torch.no_grad():
            poincare_embs_for_norms = model.get_poincare_embeddings()
            norms = poincare_embs_for_norms.norm(dim=1)
        outside_count = (norms >= 1.0).sum().item()

        # Compute hierarchy quality metrics
        depth_corr, hier_pct = compute_hierarchy_metrics(
            model, idx_to_depth, training_data or [], device
        )

        # Compute class separation metrics
        knn_purity, class_sep = compute_class_separation(
            model, class_labels_info, device, n_sample=2000
        )

        metrics = {
            'loss': avg_loss,
            'ranking_loss': avg_ranking,
            'reg_loss': avg_reg,
            'min_norm': norms.min().item(),
            'mean_norm': norms.mean().item(),
            'max_norm': norms.max().item(),
            'outside_count': outside_count,
            'total_nodes': model.n_nodes,
            'depth_norm_corr': depth_corr,
            'hierarchy_pct': hier_pct,
            'knn_purity': knn_purity,
            'class_sep_ratio': class_sep,
        }

        # Quality score for early stopping / best model selection.
        # Combines depth correlation, hierarchy preservation, and class separation.
        # This is curriculum-phase-invariant (unlike raw loss).
        quality = depth_corr + (hier_pct / 100) + knn_purity + (class_sep - 1.0)

        # Early stopping check - MUST happen BEFORE updating tracker
        prev_best_loss = tracker.best_loss

        # Update tracker and display
        tracker.update(epoch, metrics)
        tracker.print_epoch_summary(epoch, metrics, n_epochs)

        # Save checkpoint (always save Poincare-space embeddings for downstream compat)
        if checkpoint_base:
            with torch.no_grad():
                poincare_weight = model.get_poincare_embeddings().detach()
            ckpt_data = {
                'state_dict': {'lt.weight': poincare_weight},
                'embeddings': poincare_weight,
                'epoch': epoch,
                'loss': avg_loss,
                'reg_loss': avg_reg,
                'best_loss': tracker.best_loss,
                'epochs_without_improvement': epochs_without_improvement,
                'depth_norm_corr': depth_corr,
                'hierarchy_pct': hier_pct,
                'knn_purity': knn_purity,
                'class_sep_ratio': class_sep,
            }
            if model.euclidean_param:
                ckpt_data['z_embeddings'] = model.embeddings.weight.detach()
            checkpoint_path = checkpoint_base.replace('.pth', f'_epoch{epoch}.pth')
            torch.save(ckpt_data, checkpoint_path)

            # Manage checkpoint queue
            if len(checkpoint_queue) >= checkpoint_queue.maxlen:
                old_checkpoint = checkpoint_queue[0]
                if os.path.exists(old_checkpoint):
                    os.remove(old_checkpoint)

            checkpoint_queue.append(checkpoint_path)

        # Best model selection: use quality score when curriculum is active,
        # raw loss otherwise.  This prevents saving "best" during an easy
        # curriculum phase that hasn't learned cross-branch structure.
        # Skip burn-in epochs: quality is trivially high during burn-in
        # (depth_corr=+1.0, hier=100%) because the model barely moved
        # from depth initialization.
        improved = False
        in_burnin = burnin > 0 and epoch <= burnin
        if in_burnin:
            pass  # Never treat burn-in as "improved"
        elif using_curriculum:
            if quality > best_quality:
                improved = True
                best_quality = quality
                best_quality_epoch = epoch
        else:
            if avg_loss < prev_best_loss:
                improved = True

        if improved:
            epochs_without_improvement = 0

            # Save best model (reuse poincare_weight from epoch checkpoint)
            if checkpoint_base:
                best_checkpoint = checkpoint_base.replace('.pth', '_best.pth')
                best_ckpt_data = {
                    'state_dict': {'lt.weight': poincare_weight},
                    'embeddings': poincare_weight,
                    'epoch': epoch,
                    'loss': avg_loss,
                    'reg_loss': avg_reg,
                    'depth_norm_corr': depth_corr,
                    'hierarchy_pct': hier_pct,
                    'knn_purity': knn_purity,
                    'class_sep_ratio': class_sep,
                    'quality_score': quality,
                }
                if model.euclidean_param:
                    best_ckpt_data['z_embeddings'] = model.embeddings.weight.detach()
                torch.save(best_ckpt_data, best_checkpoint)
        else:
            epochs_without_improvement += 1

            # Only check early stopping if patience > 0 (0 means disabled)
            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                info = f"quality={best_quality:.4f} (epoch {best_quality_epoch})" if using_curriculum else f"loss={tracker.best_loss:.6f} (epoch {tracker.best_epoch})"
                print(f"\n🛑 Early stopping triggered after {epoch} epochs")
                print(f"   Best: {info}")
                break

    # Final summary
    tracker.print_final_summary()


def main():
    parser = argparse.ArgumentParser(description='Train on small dataset with enhanced visualization')
    parser.add_argument('--data', default='data/taxonomy_edges_small_transitive.pkl',
                       help='Training data (default: small dataset)')
    parser.add_argument('--checkpoint', default='taxonomy_model_small.pth',
                       help='Output checkpoint path')
    parser.add_argument('--mapping', default='data/taxonomy_edges_small.mapping.tsv',
                       help='Mapping file path aligned with training data')
    parser.add_argument('--dim', type=int, default=10,
                       help='Embedding dimension')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Maximum number of epochs')
    parser.add_argument('--early-stopping', type=int, default=15,
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
    parser.add_argument('--optimizer', choices=['radam', 'adam'], default='adam',
                       help='Optimizer: adam (Euclidean, default) or radam (Riemannian Adam)')
    parser.add_argument('--curriculum', action='store_true',
                       help='Enable curriculum learning (shallow pairs first)')
    parser.add_argument('--curriculum-phases', default='1:1,20:3,50:None',
                       help='Curriculum schedule as epoch:max_depth_diff pairs (default: "1:1,20:3,50:None")')
    parser.add_argument('--burnin', type=int, default=0,
                       help='Burn-in epochs with reduced LR (0 to disable)')
    parser.add_argument('--burnin-multiplier', type=float, default=0.1,
                       help='LR multiplier during burn-in (default: 0.1)')
    parser.add_argument('--radial-nudge', type=float, default=0.05,
                       help='Post-step radial correction strength (0 to disable)')
    parser.add_argument('--epoch-fraction', type=float, default=1.0,
                       help='Fraction of pairs to train per epoch (default: 1.0 = all)')
    parser.add_argument('--depth-scale-margin', action='store_true',
                       help='Scale margin by depth_diff (small for local, large for global)')
    parser.add_argument('--margin-min', type=float, default=0.05,
                       help='Minimum margin for depth-scaled mode (default: 0.05)')
    parser.add_argument('--margin-max', type=float, default=1.0,
                       help='Maximum margin for depth-scaled mode (default: 1.0)')
    parser.add_argument('--radial-schedule', choices=['linear', 'log'], default='linear',
                       help='Radial target schedule: linear (default) or log')
    parser.add_argument('--grad-accum-steps', type=int, default=1,
                       help='Gradient accumulation steps (default: 1, no accumulation)')
    parser.add_argument('--amp', action='store_true',
                       help='Enable mixed precision training (CUDA only)')
    parser.add_argument('--tiered-negatives', action='store_true',
                       help='Use tiered negative sampling (hard/medium/easy); best for large clades')
    parser.add_argument('--class-balanced', action='store_true',
                       help='Class-balanced pair sampling: draw equal pairs from each top-level class per epoch')
    parser.add_argument('--class-weighted-loss', action='store_true',
                       help='Upweight minority class pair losses by inverse sqrt frequency')
    parser.add_argument('--euclidean-param', action='store_true',
                       help='Learn in R^d with tanh map to Poincare ball (fixes gradient vanishing)')

    args = parser.parse_args()

    # Validate: radam + euclidean-param is incompatible
    if getattr(args, 'euclidean_param', False) and args.optimizer == 'radam':
        print("❌ Error: --euclidean-param is incompatible with --optimizer radam")
        print("   Riemannian corrections assume parameters live on the manifold.")
        print("   Use --optimizer adam (default) with --euclidean-param.")
        sys.exit(1)
    
    # Device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    
    # Load training data (npz or pickle)
    print(f"\nLoading training data from {args.data}...")
    if not os.path.exists(args.data):
        print(f"❌ Error: Training data not found at {args.data}")
        print(f"\nPlease run first:")
        print(f"  python build_transitive_closure.py")
        sys.exit(1)

    from pathlib import Path as _Path
    data_path = _Path(args.data)
    if data_path.suffix == ".npz":
        training_data = TrainingPairs.load(data_path)
    else:
        with open(data_path, 'rb') as f:
            training_data = TrainingPairs.from_list(pickle.load(f))
    print(f"  ✓ Loaded {len(training_data):,} training pairs")
    
    # Get dataset info - CRITICAL: use mapping file, not training data!
    # Training data may not include all nodes (e.g., leaf nodes, isolated nodes)
    print(f"Loading mapping to determine true n_nodes from {args.mapping}...")
    if not os.path.exists(args.mapping):
        print(f"❌ Error: Mapping file not found at {args.mapping}")
        sys.exit(1)
    
    mapping_df = pd.read_csv(args.mapping, sep="\t", dtype=str)
    if "taxid" not in mapping_df.columns or "idx" not in mapping_df.columns:
        # Re-read assuming headerless file
        mapping_df = pd.read_csv(
            args.mapping,
            sep="\t",
            dtype=str,
            header=None,
            names=["taxid", "idx"],
        )
    else:
        mapping_df = mapping_df[["taxid", "idx"]]
    
    mapping_df['idx'] = pd.to_numeric(mapping_df['idx'], errors='coerce')
    mapping_df['taxid'] = pd.to_numeric(mapping_df['taxid'], errors='coerce')
    mapping_df = mapping_df.dropna(subset=['idx', 'taxid'])
    mapping_df['idx'] = mapping_df['idx'].astype(int)
    mapping_df['taxid'] = mapping_df['taxid'].astype(int)
    n_nodes = int(mapping_df['idx'].max()) + 1

    max_depth = training_data.max_depth

    print(f"  Nodes: {n_nodes:,}")
    print(f"  Max depth: {max_depth}")

    # Build depth mapping from training data
    idx_to_depth = training_data.idx_to_depth_dict()

    # For nodes NOT in training data, load their depths from full taxonomy
    nodes_in_training = set(idx_to_depth.keys())
    missing_nodes = n_nodes - len(nodes_in_training)

    if missing_nodes > 0:
        print(f"\n⚠️  {missing_nodes:,} nodes not in training data - loading their depths from taxonomy...")

        # Build TaxID -> depth mapping from training data
        taxid_to_depth = {}
        for i in range(len(training_data)):
            taxid_to_depth[int(training_data.ancestor_taxid[i])] = int(training_data.ancestor_depth[i])
            taxid_to_depth[int(training_data.descendant_taxid[i])] = int(training_data.descendant_depth[i])

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
    
    # Auto-dim: select embedding dimension automatically
    dim = args.dim
    if dim == 0:
        from taxembed.analysis.dimension import recommend_dim
        rec = recommend_dim(n_nodes, max_cosine=0.3)
        dim = rec["recommended"]
        print(f"\n  Auto-selected dim={dim} (N={n_nodes:,}, angular packing={rec['angular_packing']})")

    # Create model
    euclidean_param = getattr(args, 'euclidean_param', False)
    print("\nInitializing model with depth-aware embeddings...")
    if euclidean_param:
        print("  Euclidean parametrization: ON (z in R^d, tanh map to ball)")
    model = HierarchicalPoincareEmbedding(
        n_nodes=n_nodes,
        dim=dim,
        max_depth=max_depth,
        init_depth_data=idx_to_depth,
        radial_schedule=args.radial_schedule,
        euclidean_param=euclidean_param,
    )
    
    # Create dataloader
    print("Creating dataloader with hard negative sampling...")
    dataloader = HierarchicalDataLoader(
        training_data=training_data,
        n_nodes=n_nodes,
        batch_size=args.batch_size,
        n_negatives=args.n_negatives,
        depth_stratify=True,
        epoch_fraction=args.epoch_fraction,
        tiered_negatives=getattr(args, 'tiered_negatives', False),
        class_balanced=getattr(args, 'class_balanced', False),
    )
    
    # Optimizer selection
    use_riemannian = args.optimizer == 'radam'
    if use_riemannian:
        optimizer = RiemannianAdam(model.parameters(), lr=args.lr)
        print(f"  Optimizer: RiemannianAdam (Poincare-aware)")
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        print(f"  Optimizer: Adam (Euclidean)")

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
        checkpoint_base=args.checkpoint,
        training_data=training_data,
        use_riemannian=use_riemannian,
        curriculum=args.curriculum,
        curriculum_phases=args.curriculum_phases if args.curriculum else None,
        burnin=args.burnin,
        burnin_multiplier=args.burnin_multiplier,
        radial_nudge=args.radial_nudge,
        depth_scale_margin=args.depth_scale_margin,
        margin_min=args.margin_min,
        margin_max=args.margin_max,
        radial_schedule=args.radial_schedule,
        grad_accum_steps=args.grad_accum_steps,
        use_amp=args.amp,
        class_weighted_loss=getattr(args, 'class_weighted_loss', False),
    )
    
    # Save final model (Poincare-space for downstream compat)
    print(f"Saving final model to {args.checkpoint}...")
    with torch.no_grad():
        poincare_weight = model.get_poincare_embeddings().detach()
    final_ckpt = {
        'state_dict': {'lt.weight': poincare_weight},
        'embeddings': poincare_weight,
        'n_nodes': n_nodes,
        'dim': dim,
        'max_depth': max_depth,
    }
    if model.euclidean_param:
        final_ckpt['z_embeddings'] = model.embeddings.weight.detach()
    torch.save(final_ckpt, args.checkpoint)
    
    print("\n✅ Training complete!")
    print(f"   Model saved: {args.checkpoint}")
    print(f"   Best model: {args.checkpoint.replace('.pth', '_best.pth')}")
    print("\nNext steps:")
    print("  python analyze_hierarchy_hyperbolic.py")
    print(f"  python scripts/visualize_embeddings.py {args.checkpoint.replace('.pth', '_best.pth')}")


if __name__ == "__main__":
    main()
