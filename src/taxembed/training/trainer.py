"""Main training loop for Poincaré embeddings."""

import os
from collections import deque

import torch
from tqdm import tqdm

from ..models import MetricsTracker
from .loss import radial_regularizer, ranking_loss_with_margin


def train_model(
    model,
    dataloader,
    optimizer,
    n_epochs: int,
    idx_to_depth: dict[int, int],
    max_depth: int,
    device: torch.device,
    margin: float = 0.2,
    lambda_reg: float = 0.1,
    early_stopping_patience: int = 3,
    checkpoint_base: str | None = None,
):
    """Train Poincaré embeddings with hierarchical features.

    Args:
        model: HierarchicalPoincareEmbedding model
        dataloader: HierarchicalDataLoader instance
        optimizer: PyTorch optimizer
        n_epochs: Number of training epochs
        idx_to_depth: Dictionary mapping node index to depth
        max_depth: Maximum depth in hierarchy
        device: PyTorch device (cpu or cuda)
        margin: Margin for ranking loss
        lambda_reg: Regularization strength
        early_stopping_patience: Patience for early stopping (0 to disable)
        checkpoint_base: Base path for saving checkpoints

    Returns:
        Trained model
    """
    model.to(device)
    model.train()

    print("\n" + "🚀 " * 20)
    print("HIERARCHICAL TRAINING - POINCARÉ EMBEDDINGS")
    print("🚀 " * 20)
    print("\nConfiguration:")
    print(f"  Margin: {margin}")
    print(f"  Regularization: λ={lambda_reg}")
    print(
        f"  Early stopping: {'disabled' if early_stopping_patience == 0 else f'{early_stopping_patience} epochs'}"
    )
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
    checkpoint_queue: deque = deque(maxlen=5)

    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0.0
        epoch_reg_loss = 0.0
        n_batches = 0

        # Progress bar for batches
        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch:3d}/{n_epochs}",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            ncols=100,
            leave=False,
            dynamic_ncols=True,
            mininterval=0.5,
        )

        for ancestors, descendants, negatives, depths in pbar:
            ancestors = ancestors.to(device)
            descendants = descendants.to(device)
            negatives = negatives.to(device)
            depths = depths.to(device)

            optimizer.zero_grad()

            # Ranking loss
            loss = ranking_loss_with_margin(
                model,
                ancestors,
                descendants,
                negatives,
                depths,
                margin=margin,
                depth_weight=True,
            )

            # Radial regularizer
            reg_loss = radial_regularizer(
                model, reg_indices_tensor, reg_target_radii_tensor, lambda_reg
            )

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
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "reg": f"{reg_loss.item():.4f}"})

        # Final projection at epoch end
        model.project_to_ball(indices=None)

        # Compute metrics
        avg_loss = epoch_loss / n_batches
        avg_reg = epoch_reg_loss / n_batches
        norms = model.embeddings.weight.norm(dim=1)
        outside_count = (norms >= 1.0).sum().item()

        metrics = {
            "loss": avg_loss,
            "reg_loss": avg_reg,
            "min_norm": norms.min().item(),
            "mean_norm": norms.mean().item(),
            "max_norm": norms.max().item(),
            "outside_count": outside_count,
            "total_nodes": model.n_nodes,
        }

        # Early stopping check - MUST happen BEFORE updating tracker
        prev_best_loss = tracker.best_loss

        # Update tracker and display
        tracker.update(epoch, metrics)
        tracker.print_epoch_summary(epoch, metrics, n_epochs)

        # Save checkpoint
        if checkpoint_base:
            checkpoint_path = checkpoint_base.replace(".pth", f"_epoch{epoch}.pth")
            torch.save(
                {
                    "state_dict": {"lt.weight": model.embeddings.weight},
                    "embeddings": model.embeddings.weight,
                    "epoch": epoch,
                    "loss": avg_loss,
                    "reg_loss": avg_reg,
                    "best_loss": tracker.best_loss,
                    "epochs_without_improvement": epochs_without_improvement,
                },
                checkpoint_path,
            )

            # Manage checkpoint queue
            if checkpoint_queue.maxlen and len(checkpoint_queue) >= checkpoint_queue.maxlen:
                old_checkpoint = checkpoint_queue[0]
                if os.path.exists(old_checkpoint):
                    os.remove(old_checkpoint)

            checkpoint_queue.append(checkpoint_path)

        # Check if loss improved (compare against PREVIOUS best, not updated best)
        if avg_loss < prev_best_loss:
            epochs_without_improvement = 0

            # Save best model
            if checkpoint_base:
                best_checkpoint = checkpoint_base.replace(".pth", "_best.pth")
                torch.save(
                    {
                        "state_dict": {"lt.weight": model.embeddings.weight},
                        "embeddings": model.embeddings.weight,
                        "epoch": epoch,
                        "loss": avg_loss,
                        "reg_loss": avg_reg,
                    },
                    best_checkpoint,
                )
        else:
            epochs_without_improvement += 1

            # Only check early stopping if patience > 0 (0 means disabled)
            if (
                early_stopping_patience > 0
                and epochs_without_improvement >= early_stopping_patience
            ):
                print(f"\n🛑 Early stopping triggered after {epoch} epochs")
                print(f"   Best loss: {tracker.best_loss:.6f} (epoch {tracker.best_epoch})")
                break

    # Final summary
    tracker.print_final_summary()

    return model
