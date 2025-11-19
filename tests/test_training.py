"""Tests for training utilities."""

import torch

from taxembed.training import (
    HierarchicalDataLoader,
    radial_regularizer,
    ranking_loss_with_margin,
)


class TestHierarchicalDataLoader:
    """Test the hierarchical data loader."""

    def test_initialization(self, sample_training_data):
        """Test data loader initialization."""
        loader = HierarchicalDataLoader(
            training_data=sample_training_data, n_nodes=10, batch_size=2, n_negatives=5
        )
        assert loader.batch_size == 2
        assert loader.n_negatives == 5

    def test_length(self, sample_training_data):
        """Test data loader length."""
        loader = HierarchicalDataLoader(
            training_data=sample_training_data, n_nodes=10, batch_size=2, n_negatives=5
        )
        # 3 items with batch_size 2 = 1 batch (3 // 2)
        assert len(loader) == 1

    def test_iteration(self, sample_training_data):
        """Test iterating through data loader."""
        loader = HierarchicalDataLoader(
            training_data=sample_training_data, n_nodes=10, batch_size=2, n_negatives=5
        )

        for ancestors, descendants, negatives, depths in loader:
            assert ancestors.shape[0] <= 2  # Batch size
            assert descendants.shape[0] <= 2
            assert negatives.shape == (ancestors.shape[0], 5)  # n_negatives
            assert depths.shape[0] <= 2
            break  # Just test one batch


class TestLossFunctions:
    """Test loss functions."""

    def test_ranking_loss_with_margin(self, simple_model):
        """Test ranking loss computation."""
        batch_size = 4
        n_negatives = 3

        ancestors = torch.randint(0, 10, (batch_size,))
        descendants = torch.randint(0, 10, (batch_size,))
        negatives = torch.randint(0, 10, (batch_size, n_negatives))
        depths = torch.rand(batch_size)

        loss = ranking_loss_with_margin(
            simple_model,
            ancestors,
            descendants,
            negatives,
            depths,
            margin=0.1,
            depth_weight=True,
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss >= 0  # Loss should be non-negative

    def test_radial_regularizer(self, simple_model, sample_depth_map):
        """Test radial regularization."""
        idx_tensor = torch.tensor([0, 1, 2])
        target_radii = torch.tensor([0.1, 0.5, 0.9])

        reg_loss = radial_regularizer(simple_model, idx_tensor, target_radii, lambda_reg=0.1)

        assert isinstance(reg_loss, torch.Tensor)
        assert reg_loss.ndim == 0  # Scalar
        assert reg_loss >= 0

    def test_radial_regularizer_empty(self, simple_model):
        """Test radial regularizer with empty indices."""
        idx_tensor = torch.tensor([], dtype=torch.long)
        target_radii = torch.tensor([])

        reg_loss = radial_regularizer(simple_model, idx_tensor, target_radii, lambda_reg=0.1)

        assert reg_loss == 0.0
