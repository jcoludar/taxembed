"""Loss functions for hierarchical Poincaré embeddings."""

import torch


def ranking_loss_with_margin(
    model,
    ancestors: torch.Tensor,
    descendants: torch.Tensor,
    negatives: torch.Tensor,
    depths: torch.Tensor,
    margin: float = 0.1,
    depth_weight: bool = True,
) -> torch.Tensor:
    """Ranking loss with margin and optional depth weighting.

    Loss encourages:
    - d(ancestor, descendant) < d(ancestor, negative) + margin
    - Deeper pairs get higher weight (they're more informative)

    Args:
        model: Poincaré embedding model
        ancestors: Tensor of ancestor indices (batch,)
        descendants: Tensor of descendant indices (batch,)
        negatives: Tensor of negative sample indices (batch, n_neg)
        depths: Tensor of depth differences (batch,)
        margin: Margin for ranking loss
        depth_weight: Whether to weight loss by depth

    Returns:
        Scalar loss value
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


def radial_regularizer(
    model,
    idx_to_depth_tensor: torch.Tensor,
    target_radii_tensor: torch.Tensor,
    lambda_reg: float = 0.01,
) -> torch.Tensor:
    """Vectorized radial regularizer to keep nodes at expected radius based on depth.

    Encourages: ||embedding|| ≈ f(depth)
    where f(depth) = 0.1 + (depth/max_depth) * 0.85

    Args:
        model: Poincaré embedding model
        idx_to_depth_tensor: Tensor of indices to regularize
        target_radii_tensor: Tensor of target radii for each index
        lambda_reg: Regularization strength

    Returns:
        Scalar regularization loss
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
