"""Tests for Poincaré embedding models."""

import torch

from taxembed.models import HierarchicalPoincareEmbedding, MetricsTracker


class TestHierarchicalPoincareEmbedding:
    """Test the Poincaré embedding model."""

    def test_initialization(self):
        """Test model initialization."""
        model = HierarchicalPoincareEmbedding(n_nodes=100, dim=10, max_depth=5)
        assert model.n_nodes == 100
        assert model.dim == 10
        assert model.max_depth == 5
        assert model.embeddings.weight.shape == (100, 10)

    def test_depth_initialization(self, sample_depth_map):
        """Test depth-aware initialization."""
        model = HierarchicalPoincareEmbedding(
            n_nodes=4, dim=5, max_depth=2, init_depth_data=sample_depth_map
        )

        # Check that norms increase with depth
        norms = model.embeddings.weight.norm(dim=1)
        assert norms[0] < norms[1] < norms[2]  # Depth 0 < 1 < 2

    def test_forward(self, simple_model):
        """Test forward pass."""
        indices = torch.tensor([0, 1, 2])
        embeddings = simple_model(indices)
        assert embeddings.shape == (3, 5)

    def test_poincare_distance(self, simple_model):
        """Test Poincaré distance computation."""
        u = torch.randn(10, 5) * 0.5  # Keep inside ball
        v = torch.randn(10, 5) * 0.5

        distances = simple_model.poincare_distance(u, v)
        assert distances.shape == (10,)
        assert torch.all(distances >= 0)  # Distances are non-negative

    def test_project_to_ball(self, simple_model):
        """Test ball projection."""
        # Set some embeddings outside the ball
        with torch.no_grad():
            simple_model.embeddings.weight[0] = torch.ones(5) * 1.5

        # Project back
        simple_model.project_to_ball()

        # Check all embeddings are inside ball
        norms = simple_model.embeddings.weight.norm(dim=1)
        assert torch.all(norms < 1.0)

    def test_selective_projection(self, simple_model):
        """Test selective projection of specific indices."""
        # Set one embedding outside
        with torch.no_grad():
            simple_model.embeddings.weight[5] = torch.ones(5) * 1.5

        # Project only that index
        simple_model.project_to_ball(indices=torch.tensor([5]))

        # Check it's inside
        norm = simple_model.embeddings.weight[5].norm()
        assert norm < 1.0


class TestMetricsTracker:
    """Test the metrics tracker."""

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = MetricsTracker()
        assert len(tracker.history) == 0
        assert tracker.best_loss == float("inf")
        assert tracker.best_epoch == 0

    def test_update(self):
        """Test metric updates."""
        tracker = MetricsTracker()
        metrics = {"loss": 0.5, "reg_loss": 0.1}

        tracker.update(1, metrics)
        assert len(tracker.history) == 1
        assert tracker.best_loss == 0.5
        assert tracker.best_epoch == 1

    def test_best_loss_tracking(self):
        """Test that best loss is tracked correctly."""
        tracker = MetricsTracker()

        tracker.update(1, {"loss": 0.5})
        tracker.update(2, {"loss": 0.3})  # Better
        tracker.update(3, {"loss": 0.4})  # Worse

        assert tracker.best_loss == 0.3
        assert tracker.best_epoch == 2

    def test_get_previous(self):
        """Test getting previous metrics."""
        tracker = MetricsTracker()

        tracker.update(1, {"loss": 0.5})
        assert tracker.get_previous("loss") is None  # No previous

        tracker.update(2, {"loss": 0.3})
        assert tracker.get_previous("loss") == 0.5
