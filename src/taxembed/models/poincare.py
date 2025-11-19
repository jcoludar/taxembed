"""Poincaré embeddings model with hierarchical features."""

import torch
import torch.nn as nn


class HierarchicalPoincareEmbedding(nn.Module):
    """Poincaré embeddings with hierarchical structure.

    Key features:
    - Depth-aware initialization (deeper nodes near boundary)
    - Hyperbolic distance computation
    - Ball constraint projection

    Args:
        n_nodes: Number of nodes to embed
        dim: Embedding dimensionality
        max_depth: Maximum depth in hierarchy (for initialization)
        init_depth_data: Optional dict mapping node idx -> depth for initialization
    """

    def __init__(
        self,
        n_nodes: int,
        dim: int = 10,
        max_depth: int = 38,
        init_depth_data: dict | None = None,
    ):
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

    def _initialize_by_depth(self, depth_data: dict):
        """Initialize embeddings based on taxonomic depth.

        Deeper nodes → larger radius (closer to boundary).
        This encodes hierarchy from the start!

        Args:
            depth_data: Dict mapping idx -> depth
        """
        print("Initializing embeddings by depth...")

        # Map: idx → depth
        idx_to_depth = depth_data

        with torch.no_grad():
            for idx in range(self.n_nodes):
                depth = idx_to_depth.get(idx, 0)

                # Radius increases with depth
                # Root (depth 0): r ≈ 0.1
                # Max depth: r ≈ 0.95
                target_radius = 0.1 + (depth / self.max_depth) * 0.85

                # Random direction on sphere
                vec = torch.randn(self.dim)
                vec = vec / vec.norm()

                # Scale to target radius
                self.embeddings.weight[idx] = vec * target_radius

        norms = self.embeddings.weight.norm(dim=1)
        print(f"  ✓ Initialized: norm range [{norms.min():.3f}, {norms.max():.3f}]")

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Get embeddings for indices.

        Args:
            indices: Tensor of node indices

        Returns:
            Embeddings tensor of shape (batch_size, dim)
        """
        return self.embeddings(indices)

    def poincare_distance(
        self, u: torch.Tensor, v: torch.Tensor, eps: float = 1e-5
    ) -> torch.Tensor:
        """Compute Poincaré distance between embeddings.

        Formula: d(u,v) = arcosh(1 + 2||u-v||²/((1-||u||²)(1-||v||²)))

        Args:
            u: First embedding tensor
            v: Second embedding tensor
            eps: Small value for numerical stability

        Returns:
            Poincaré distances
        """
        # Compute squared norms
        u_norm_sq = (u**2).sum(dim=-1)
        v_norm_sq = (v**2).sum(dim=-1)

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

    def project_to_ball(self, indices: torch.Tensor | None = None, max_norm: float = 0.999):
        """Project embeddings back into Poincaré ball with HARD constraint.

        Args:
            indices: If provided, only project these indices (more efficient).
                    If None, project all embeddings.
            max_norm: Maximum allowed norm (default 0.999, essentially at boundary)
        """
        with torch.no_grad():
            if indices is not None:
                # Only project updated embeddings
                embs = self.embeddings.weight[indices]
                norms = embs.norm(dim=1, keepdim=True)
                # Hard projection: if norm >= max_norm, scale it down
                # Use where to only scale embeddings that need it
                needs_projection = norms >= max_norm
                scale = torch.where(
                    needs_projection, max_norm / (norms + 1e-8), torch.ones_like(norms)
                )
                self.embeddings.weight[indices] = embs * scale
            else:
                # Project all embeddings
                norms = self.embeddings.weight.norm(dim=1, keepdim=True)
                needs_projection = norms >= max_norm
                scale = torch.where(
                    needs_projection, max_norm / (norms + 1e-8), torch.ones_like(norms)
                )
                self.embeddings.weight.mul_(scale)
