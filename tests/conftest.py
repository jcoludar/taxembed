"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture
def sample_training_data():
    """Create sample training data for tests."""
    return [
        {
            "ancestor_idx": 0,
            "descendant_idx": 1,
            "ancestor_depth": 0,
            "descendant_depth": 1,
            "depth_diff": 1,
        },
        {
            "ancestor_idx": 0,
            "descendant_idx": 2,
            "ancestor_depth": 0,
            "descendant_depth": 2,
            "depth_diff": 2,
        },
        {
            "ancestor_idx": 1,
            "descendant_idx": 2,
            "ancestor_depth": 1,
            "descendant_depth": 2,
            "depth_diff": 1,
        },
    ]


@pytest.fixture
def sample_depth_map():
    """Create sample depth mapping."""
    return {
        0: 0,  # Root
        1: 1,  # Level 1
        2: 2,  # Level 2
        3: 2,  # Level 2
    }


@pytest.fixture
def simple_model():
    """Create a simple Poincaré model for testing."""
    from taxembed.models import HierarchicalPoincareEmbedding

    return HierarchicalPoincareEmbedding(n_nodes=10, dim=5, max_depth=3)


@pytest.fixture
def data_dir(tmp_path):
    """Create temporary data directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir
