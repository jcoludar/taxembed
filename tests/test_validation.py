"""Tests for validation utilities."""

import torch

from taxembed.models import HierarchicalPoincareEmbedding


class TestValidation:
    """Test validation functions."""

    def test_ball_constraint_validation(self, simple_model):
        """Test that ball constraints are maintained."""
        # All embeddings should be inside unit ball after projection
        simple_model.project_to_ball()

        norms = simple_model.embeddings.weight.norm(dim=1)
        assert torch.all(norms < 1.0), "Some embeddings are outside the unit ball"

    def test_depth_ordering(self, sample_depth_map):
        """Test that deeper nodes have larger norms."""
        model = HierarchicalPoincareEmbedding(
            n_nodes=4, dim=5, max_depth=2, init_depth_data=sample_depth_map
        )

        norms = model.embeddings.weight.norm(dim=1)

        # Root should have smallest norm
        assert norms[0] < norms[1]
        assert norms[0] < norms[2]
