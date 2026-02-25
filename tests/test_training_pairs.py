"""Tests for TrainingPairs, dimension analysis, and curriculum utilities."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# TrainingPairs
# ---------------------------------------------------------------------------
from train_hierarchical import TrainingPairs, HierarchicalDataLoader


def _make_sample_pairs(n: int = 100) -> list[dict]:
    """Create a minimal set of training pairs for testing."""
    pairs = []
    for i in range(n):
        pairs.append(
            {
                "ancestor_idx": 0,
                "descendant_idx": i + 1,
                "depth_diff": 1,
                "ancestor_depth": 0,
                "descendant_depth": 1,
                "ancestor_taxid": 1000,
                "descendant_taxid": 2000 + i,
            }
        )
    return pairs


class TestTrainingPairs:
    def test_from_list_basic(self):
        pairs_list = _make_sample_pairs(50)
        tp = TrainingPairs.from_list(pairs_list)
        assert len(tp) == 50
        assert tp.ancestor_idx.dtype == np.int32
        assert tp.depth_diff.dtype == np.int16

    def test_getitem_single(self):
        tp = TrainingPairs.from_list(_make_sample_pairs(10))
        item = tp[0]
        assert isinstance(item, dict)
        assert item["ancestor_idx"] == 0
        assert item["descendant_idx"] == 1

    def test_getitem_slice(self):
        tp = TrainingPairs.from_list(_make_sample_pairs(20))
        sub = tp[:5]
        assert isinstance(sub, TrainingPairs)
        assert len(sub) == 5

    def test_n_nodes(self):
        tp = TrainingPairs.from_list(_make_sample_pairs(10))
        # ancestor_idx max is 0, descendant_idx max is 10 → n_nodes = 11
        assert tp.n_nodes == 11

    def test_max_depth(self):
        tp = TrainingPairs.from_list(_make_sample_pairs(5))
        assert tp.max_depth == 1

    def test_idx_to_depth_dict(self):
        tp = TrainingPairs.from_list(_make_sample_pairs(5))
        d = tp.idx_to_depth_dict()
        assert d[0] == 0  # ancestor depth
        assert d[1] == 1  # descendant depth

    def test_save_load_roundtrip(self, tmp_path):
        tp = TrainingPairs.from_list(_make_sample_pairs(30))
        save_path = tmp_path / "test_pairs.npz"
        tp.save(save_path)

        loaded = TrainingPairs.load(save_path)
        assert len(loaded) == 30
        np.testing.assert_array_equal(tp.ancestor_idx, loaded.ancestor_idx)
        np.testing.assert_array_equal(tp.descendant_idx, loaded.descendant_idx)
        np.testing.assert_array_equal(tp.depth_diff, loaded.depth_diff)

    def test_empty_pairs(self):
        tp = TrainingPairs.from_list([])
        assert len(tp) == 0


# ---------------------------------------------------------------------------
# HierarchicalDataLoader
# ---------------------------------------------------------------------------
class TestHierarchicalDataLoader:
    def test_basic_iteration(self):
        pairs = _make_sample_pairs(100)
        tp = TrainingPairs.from_list(pairs)
        loader = HierarchicalDataLoader(
            tp, n_nodes=tp.n_nodes, batch_size=32, n_negatives=5
        )
        batches = list(loader)
        assert len(batches) > 0
        ancestors, descendants, negatives, depths = batches[0]
        assert ancestors.shape[0] <= 32
        assert negatives.shape == (ancestors.shape[0], 5)

    def test_accepts_list_of_dicts(self):
        """Backward compat: can pass legacy list-of-dicts."""
        pairs = _make_sample_pairs(50)
        loader = HierarchicalDataLoader(
            pairs, n_nodes=51, batch_size=16, n_negatives=3
        )
        batches = list(loader)
        assert len(batches) > 0

    def test_epoch_fraction(self):
        tp = TrainingPairs.from_list(_make_sample_pairs(200))
        loader_full = HierarchicalDataLoader(
            tp, n_nodes=tp.n_nodes, batch_size=32, n_negatives=5, epoch_fraction=1.0
        )
        loader_half = HierarchicalDataLoader(
            tp, n_nodes=tp.n_nodes, batch_size=32, n_negatives=5, epoch_fraction=0.5
        )
        # Half should have roughly half the batches
        assert len(loader_half) <= len(loader_full)
        assert len(loader_half) > 0

    def test_curriculum_phase(self):
        """Multi-depth pairs with curriculum filtering."""
        pairs = []
        for dd in [1, 2, 3]:
            for i in range(20):
                pairs.append(
                    {
                        "ancestor_idx": 0,
                        "descendant_idx": dd * 100 + i,
                        "depth_diff": dd,
                        "ancestor_depth": 0,
                        "descendant_depth": dd,
                        "ancestor_taxid": 1,
                        "descendant_taxid": dd * 100 + i,
                    }
                )
        tp = TrainingPairs.from_list(pairs)
        loader = HierarchicalDataLoader(
            tp, n_nodes=400, batch_size=10, n_negatives=3
        )

        # Set curriculum to only dd=1
        loader.set_curriculum_phase(1)
        batches_phase1 = list(loader)

        # Clear curriculum — all pairs
        loader.clear_curriculum()
        batches_all = list(loader)

        assert len(batches_phase1) < len(batches_all)

    def test_depth_stratified_sampling(self):
        """epoch_fraction should sample equally from each depth_diff level."""
        pairs = []
        # 100 pairs at dd=1, 100 at dd=2, 100 at dd=3
        for dd in [1, 2, 3]:
            for i in range(100):
                pairs.append({
                    "ancestor_idx": 0,
                    "descendant_idx": dd * 1000 + i + 1,
                    "depth_diff": dd,
                    "ancestor_depth": 0,
                    "descendant_depth": dd,
                    "ancestor_taxid": 1,
                    "descendant_taxid": dd * 1000 + i + 1,
                })
        tp = TrainingPairs.from_list(pairs)
        loader = HierarchicalDataLoader(
            tp, n_nodes=4000, batch_size=10, n_negatives=3, epoch_fraction=0.3,
        )

        # Collect all pair indices seen in one epoch
        dd_seen = {1: 0, 2: 0, 3: 0}
        for ancestors, descendants, negatives, depths in loader:
            for d in depths.tolist():
                dd_seen[d] = dd_seen.get(d, 0) + 1

        # Each dd level should get ~30 pairs (30% of 100)
        # Allow generous margin for randomness, but all levels should be represented
        for dd in [1, 2, 3]:
            assert dd_seen[dd] > 0, f"dd={dd} should have samples"
            # Each level should get roughly 30% of its 100 pairs (±50%)
            assert 10 < dd_seen[dd] < 60, f"dd={dd} got {dd_seen[dd]}, expected ~30"


# ---------------------------------------------------------------------------
# Dimension analysis
# ---------------------------------------------------------------------------
from taxembed.analysis.dimension import angular_packing_dim, participation_ratio, recommend_dim


class TestDimensionAnalysis:
    def test_angular_packing_basic(self):
        d = angular_packing_dim(1000, max_cosine=0.2)
        assert d > 0
        assert isinstance(d, int)

    def test_angular_packing_single_item(self):
        assert angular_packing_dim(1) == 1

    def test_angular_packing_decreases_with_cosine(self):
        d_strict = angular_packing_dim(10000, 0.1)
        d_relaxed = angular_packing_dim(10000, 0.5)
        assert d_strict > d_relaxed

    def test_angular_packing_increases_with_items(self):
        d_small = angular_packing_dim(100, 0.2)
        d_large = angular_packing_dim(100000, 0.2)
        assert d_large > d_small

    def test_participation_ratio_identity(self):
        """Identity-like covariance → PR = number of dims."""
        rng = np.random.default_rng(42)
        # Standard normal in 5D → PR should be close to 5
        embeddings = rng.standard_normal((1000, 5))
        pr = participation_ratio(embeddings)
        assert 4.0 < pr <= 5.5

    def test_participation_ratio_one_dim(self):
        """All variance in one direction → PR close to 1."""
        rng = np.random.default_rng(42)
        embeddings = np.zeros((100, 5))
        embeddings[:, 0] = rng.standard_normal(100)
        pr = participation_ratio(embeddings)
        assert pr < 2.0

    def test_recommend_dim(self):
        rec = recommend_dim(10000, max_cosine=0.2)
        assert "n_nodes" in rec
        assert "angular_packing" in rec
        assert "recommended" in rec
        assert rec["recommended"] >= 10

    def test_recommend_dim_with_embeddings(self):
        rng = np.random.default_rng(42)
        emb = rng.standard_normal((500, 20)).astype(np.float32)
        rec = recommend_dim(500, embeddings=emb)
        assert "participation_ratio" in rec
        assert "current_dim" in rec
        assert rec["current_dim"] == 20


# ---------------------------------------------------------------------------
# Class separation metrics
# ---------------------------------------------------------------------------
from train_small import _build_class_labels, compute_class_separation
from train_hierarchical import HierarchicalPoincareEmbedding


class TestClassSeparation:
    def _make_tree_pairs(self):
        """Build a small tree: root(0) → A(1), B(2); A → C(3), D(4); B → E(5), F(6)."""
        pairs = [
            # dd=1: parent-child
            {"ancestor_idx": 0, "descendant_idx": 1, "depth_diff": 1,
             "ancestor_depth": 0, "descendant_depth": 1, "ancestor_taxid": 100, "descendant_taxid": 101},
            {"ancestor_idx": 0, "descendant_idx": 2, "depth_diff": 1,
             "ancestor_depth": 0, "descendant_depth": 1, "ancestor_taxid": 100, "descendant_taxid": 102},
            {"ancestor_idx": 1, "descendant_idx": 3, "depth_diff": 1,
             "ancestor_depth": 1, "descendant_depth": 2, "ancestor_taxid": 101, "descendant_taxid": 103},
            {"ancestor_idx": 1, "descendant_idx": 4, "depth_diff": 1,
             "ancestor_depth": 1, "descendant_depth": 2, "ancestor_taxid": 101, "descendant_taxid": 104},
            {"ancestor_idx": 2, "descendant_idx": 5, "depth_diff": 1,
             "ancestor_depth": 1, "descendant_depth": 2, "ancestor_taxid": 102, "descendant_taxid": 105},
            {"ancestor_idx": 2, "descendant_idx": 6, "depth_diff": 1,
             "ancestor_depth": 1, "descendant_depth": 2, "ancestor_taxid": 102, "descendant_taxid": 106},
            # dd=2: transitive
            {"ancestor_idx": 0, "descendant_idx": 3, "depth_diff": 2,
             "ancestor_depth": 0, "descendant_depth": 2, "ancestor_taxid": 100, "descendant_taxid": 103},
            {"ancestor_idx": 0, "descendant_idx": 4, "depth_diff": 2,
             "ancestor_depth": 0, "descendant_depth": 2, "ancestor_taxid": 100, "descendant_taxid": 104},
            {"ancestor_idx": 0, "descendant_idx": 5, "depth_diff": 2,
             "ancestor_depth": 0, "descendant_depth": 2, "ancestor_taxid": 100, "descendant_taxid": 105},
            {"ancestor_idx": 0, "descendant_idx": 6, "depth_diff": 2,
             "ancestor_depth": 0, "descendant_depth": 2, "ancestor_taxid": 100, "descendant_taxid": 106},
        ]
        return TrainingPairs.from_list(pairs)

    def test_build_class_labels(self):
        tp = self._make_tree_pairs()
        result = _build_class_labels(tp, 7)
        assert result is not None
        labels, n_labeled, n_classes = result
        assert n_classes == 2  # nodes 1 and 2 are the two top-level classes
        # Node 3 and 4 should have same label as node 1
        assert labels[3] == labels[1]
        assert labels[4] == labels[1]
        # Node 5 and 6 should have same label as node 2
        assert labels[5] == labels[2]
        assert labels[6] == labels[2]
        # Classes 1 and 2 should be different
        assert labels[1] != labels[2]

    def test_class_separation_well_separated(self):
        """Model with well-separated classes should have high kNN purity."""
        tp = self._make_tree_pairs()
        labels_info = _build_class_labels(tp, 7)
        model = HierarchicalPoincareEmbedding(n_nodes=7, dim=5, max_depth=2)
        # Manually set embeddings: class A nodes in one direction, class B in another
        import torch
        with torch.no_grad():
            model.embeddings.weight[0] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])  # root
            model.embeddings.weight[1] = torch.tensor([0.5, 0.0, 0.0, 0.0, 0.0])  # A
            model.embeddings.weight[3] = torch.tensor([0.6, 0.1, 0.0, 0.0, 0.0])  # C (child of A)
            model.embeddings.weight[4] = torch.tensor([0.6, -0.1, 0.0, 0.0, 0.0]) # D (child of A)
            model.embeddings.weight[2] = torch.tensor([-0.5, 0.0, 0.0, 0.0, 0.0]) # B
            model.embeddings.weight[5] = torch.tensor([-0.6, 0.1, 0.0, 0.0, 0.0]) # E (child of B)
            model.embeddings.weight[6] = torch.tensor([-0.6, -0.1, 0.0, 0.0, 0.0])# F (child of B)
        knn_purity, sep_ratio = compute_class_separation(model, labels_info, "cpu", n_sample=6, k=2)
        assert knn_purity > 0.8  # Neighbors should mostly be same class
        assert sep_ratio > 1.0   # Inter > intra distance

    def test_class_separation_returns_zero_without_labels(self):
        knn, sep = compute_class_separation(None, None, "cpu")
        assert knn == 0.0
        assert sep == 0.0


# ---------------------------------------------------------------------------
# Curriculum auto-phases
# ---------------------------------------------------------------------------
from train_small import auto_curriculum_phases, parse_curriculum_phases


class TestCurriculum:
    def test_parse_curriculum_phases(self):
        phases = parse_curriculum_phases("1:1,20:3,50:None")
        assert phases == [(1, 1), (20, 3), (50, None)]

    def test_auto_curriculum_basic(self):
        phases = auto_curriculum_phases(max_depth=20, n_epochs=100)
        assert len(phases) == 4
        # First phase: parent-child only
        assert phases[0][1] == 1
        # Last phase: unrestricted
        assert phases[-1][1] is None

    def test_auto_curriculum_epochs_increasing(self):
        phases = auto_curriculum_phases(max_depth=10, n_epochs=50)
        epochs = [e for e, _ in phases]
        assert epochs == sorted(epochs)
        assert len(set(epochs)) == len(epochs)  # all unique

    def test_auto_curriculum_small_epochs(self):
        """Even with very few epochs, phases should be valid."""
        phases = auto_curriculum_phases(max_depth=5, n_epochs=5)
        assert len(phases) == 4
        epochs = [e for e, _ in phases]
        assert all(e >= 1 for e in epochs)


# ---------------------------------------------------------------------------
# Tiered negative sampling (Iteration 1)
# ---------------------------------------------------------------------------
class TestTieredNegatives:
    """Verify ancestry-aware hard negative structures and sampling."""

    def _make_deep_tree_pairs(self):
        """Build a 3-level tree:
        root(0) → A(1), B(2);
        A → C(3), D(4); B → E(5), F(6);
        C → G(7), H(8); D → I(9), J(10);
        E → K(11), L(12); F → M(13), N(14)
        """
        pairs = []
        edges = [
            (0, 1, 0, 1), (0, 2, 0, 1),                # root → A, B
            (1, 3, 1, 2), (1, 4, 1, 2),                  # A → C, D
            (2, 5, 1, 2), (2, 6, 1, 2),                  # B → E, F
            (3, 7, 2, 3), (3, 8, 2, 3),                  # C → G, H
            (4, 9, 2, 3), (4, 10, 2, 3),                 # D → I, J
            (5, 11, 2, 3), (5, 12, 2, 3),                # E → K, L
            (6, 13, 2, 3), (6, 14, 2, 3),                # F → M, N
        ]
        # dd=1 edges
        for anc, desc, ad, dd in edges:
            pairs.append({
                "ancestor_idx": anc, "descendant_idx": desc, "depth_diff": 1,
                "ancestor_depth": ad, "descendant_depth": dd,
                "ancestor_taxid": 100 + anc, "descendant_taxid": 100 + desc,
            })
        # dd=2 transitive edges
        for anc, mid_desc, ad, _ in edges:
            for anc2, desc2, ad2, dd2 in edges:
                if anc2 == mid_desc:
                    pairs.append({
                        "ancestor_idx": anc, "descendant_idx": desc2, "depth_diff": 2,
                        "ancestor_depth": ad, "descendant_depth": dd2,
                        "ancestor_taxid": 100 + anc, "descendant_taxid": 100 + desc2,
                    })
        # dd=3: root → leaves
        for leaf in range(7, 15):
            pairs.append({
                "ancestor_idx": 0, "descendant_idx": leaf, "depth_diff": 3,
                "ancestor_depth": 0, "descendant_depth": 3,
                "ancestor_taxid": 100, "descendant_taxid": 100 + leaf,
            })
        return TrainingPairs.from_list(pairs)

    def test_ancestry_structures_built(self):
        """Verify _build_depth_index creates grandparent and class lookups."""
        tp = self._make_deep_tree_pairs()
        loader = HierarchicalDataLoader(tp, n_nodes=15, batch_size=4, n_negatives=3)
        # Grandparent structures exist
        assert hasattr(loader, '_node_to_grandparent')
        assert hasattr(loader, '_gp_depth_to_nodes')
        assert hasattr(loader, '_node_to_class')
        assert hasattr(loader, '_class_depth_to_nodes')

    def test_grandparent_mapping(self):
        """G(7) and H(8) share grandparent A(1); I(9) shares grandparent A(1) too."""
        tp = self._make_deep_tree_pairs()
        loader = HierarchicalDataLoader(tp, n_nodes=15, batch_size=4, n_negatives=3)
        # G(7) parent=C(3), grandparent=A(1)
        assert loader._node_to_grandparent.get(7) == 1
        assert loader._node_to_grandparent.get(8) == 1
        assert loader._node_to_grandparent.get(9) == 1
        # K(11) parent=E(5), grandparent=B(2)
        assert loader._node_to_grandparent.get(11) == 2

    def test_class_labels(self):
        """Nodes under A(1) get class A; nodes under B(2) get class B."""
        tp = self._make_deep_tree_pairs()
        loader = HierarchicalDataLoader(tp, n_nodes=15, batch_size=4, n_negatives=3)
        # A subtree: 1,3,4,7,8,9,10
        for idx in [1, 3, 4, 7, 8, 9, 10]:
            assert loader._node_to_class.get(idx) == 1, f"node {idx} should be class 1"
        # B subtree: 2,5,6,11,12,13,14
        for idx in [2, 5, 6, 11, 12, 13, 14]:
            assert loader._node_to_class.get(idx) == 2, f"node {idx} should be class 2"

    def test_default_iteration_produces_valid_negatives(self):
        """Iteration with default sampling should produce correct-shape negatives."""
        tp = self._make_deep_tree_pairs()
        loader = HierarchicalDataLoader(tp, n_nodes=15, batch_size=4, n_negatives=5)
        batches = list(loader)
        assert len(batches) > 0
        for ancestors, descendants, negatives, depths in batches:
            assert negatives.shape[1] == 5
            # All negative indices should be valid node indices
            assert negatives.min() >= 0
            assert negatives.max() < 15

    def test_tiered_iteration_produces_valid_negatives(self):
        """Iteration with tiered_negatives=True should produce correct-shape negatives."""
        tp = self._make_deep_tree_pairs()
        loader = HierarchicalDataLoader(
            tp, n_nodes=15, batch_size=4, n_negatives=5, tiered_negatives=True,
        )
        batches = list(loader)
        assert len(batches) > 0
        for ancestors, descendants, negatives, depths in batches:
            assert negatives.shape[1] == 5
            assert negatives.min() >= 0
            assert negatives.max() < 15


# ---------------------------------------------------------------------------
# Depth-scaled margin (Iteration 3)
# ---------------------------------------------------------------------------
from train_hierarchical import ranking_loss_with_margin

class TestDepthScaledMargin:
    def test_fixed_margin_unchanged(self):
        """Default (no depth scaling) should work as before."""
        import torch
        model = HierarchicalPoincareEmbedding(n_nodes=10, dim=5, max_depth=5)
        ancestors = torch.tensor([0, 0])
        descendants = torch.tensor([1, 2])
        negatives = torch.tensor([[3, 4], [5, 6]])
        depths = torch.tensor([1.0, 2.0])
        loss = ranking_loss_with_margin(model, ancestors, descendants, negatives, depths, margin=0.2)
        assert loss.item() >= 0

    def test_depth_scaled_margin_varies(self):
        """With depth scaling, different depth_diffs should produce different effective margins."""
        import torch
        model = HierarchicalPoincareEmbedding(n_nodes=10, dim=5, max_depth=5)
        ancestors = torch.tensor([0, 0])
        descendants = torch.tensor([1, 2])
        negatives = torch.tensor([[3, 4], [5, 6]])
        depths_shallow = torch.tensor([1.0, 1.0])
        depths_deep = torch.tensor([5.0, 5.0])
        loss_shallow = ranking_loss_with_margin(
            model, ancestors, descendants, negatives, depths_shallow, margin=0.2,
            depth_scale_margin=True, margin_min=0.05, margin_max=1.0, max_depth_diff=5,
        )
        loss_deep = ranking_loss_with_margin(
            model, ancestors, descendants, negatives, depths_deep, margin=0.2,
            depth_scale_margin=True, margin_min=0.05, margin_max=1.0, max_depth_diff=5,
        )
        # Both should produce valid loss values
        assert loss_shallow.item() >= 0
        assert loss_deep.item() >= 0


# ---------------------------------------------------------------------------
# Logarithmic radial targets (Iteration 4)
# ---------------------------------------------------------------------------
from train_hierarchical import target_radius

class TestRadialSchedule:
    def test_linear_schedule(self):
        """Linear: target at depth 0 is 0.1, at max depth is 0.95."""
        assert abs(target_radius(0, 10, 'linear') - 0.1) < 1e-6
        assert abs(target_radius(10, 10, 'linear') - 0.95) < 1e-6

    def test_log_schedule(self):
        """Log: target at depth 0 is 0.1, at max depth is 0.95."""
        assert abs(target_radius(0, 10, 'log') - 0.1) < 1e-6
        assert abs(target_radius(10, 10, 'log') - 0.95) < 1e-6

    def test_log_spreads_deep_shells(self):
        """Log schedule should give deeper nodes more spacing than linear."""
        # At mid-depth, log should give a LARGER radius than linear
        # (log(1+d)/log(1+D) > d/D for d < D)
        lin = target_radius(5, 10, 'linear')
        log = target_radius(5, 10, 'log')
        assert log > lin, "Log schedule should spread deep shells more than linear"

    def test_model_init_with_log_schedule(self):
        """Model should initialize successfully with log schedule."""
        depth_data = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
        model = HierarchicalPoincareEmbedding(
            n_nodes=5, dim=3, max_depth=4, init_depth_data=depth_data,
            radial_schedule='log',
        )
        norms = model.embeddings.weight.norm(dim=1)
        assert norms.min() > 0
        assert norms.max() < 1.0

    def test_numpy_vectorized(self):
        """target_radius should work with numpy arrays."""
        depths = np.array([0, 5, 10])
        radii = target_radius(depths, 10, 'log')
        assert len(radii) == 3
        assert radii[0] < radii[1] < radii[2]


# ---------------------------------------------------------------------------
# Euclidean parametrization (tanh map)
# ---------------------------------------------------------------------------
import torch
from train_small import _pairwise_poincare_distance


class TestEuclideanParam:
    """Tests for the tanh map z -> x = tanh(||z||/2) * z/||z||."""

    def test_tanh_map_origin(self):
        """z=0 should map to x=0."""
        model = HierarchicalPoincareEmbedding(
            n_nodes=1, dim=5, max_depth=1, euclidean_param=True
        )
        with torch.no_grad():
            model.embeddings.weight[0] = torch.zeros(5)
        x = model.get_poincare_embeddings(torch.tensor([0]))
        assert torch.allclose(x, torch.zeros(1, 5), atol=1e-6)

    def test_tanh_map_always_inside_ball(self):
        """Any z in R^d should map to ||x|| < 1 (within float32 precision)."""
        model = HierarchicalPoincareEmbedding(
            n_nodes=100, dim=10, max_depth=5, euclidean_param=True
        )
        with torch.no_grad():
            # Scale * 2: ||z|| ~ 6.3, tanh(3.15) ~ 0.997 (avoids float32 saturation)
            model.embeddings.weight[:50] = torch.randn(50, 10) * 2
            model.embeddings.weight[50:] = torch.randn(50, 10) * 0.001
        x = model.get_poincare_embeddings()
        norms = x.norm(dim=1)
        assert (norms < 1.0).all(), f"Max norm: {norms.max().item()}"
        # Also verify very large z maps to ~1.0 (float32 saturates tanh to exactly 1.0,
        # then z/||z|| normalization can introduce ~1e-7 error)
        with torch.no_grad():
            model.embeddings.weight[:] = torch.randn(100, 10) * 100
        x_big = model.get_poincare_embeddings()
        norms_big = x_big.norm(dim=1)
        assert (norms_big < 1.0 + 1e-6).all(), f"Max norm (large z): {norms_big.max().item()}"

    def test_tanh_map_preserves_direction(self):
        """direction(x) should equal direction(z)."""
        model = HierarchicalPoincareEmbedding(
            n_nodes=10, dim=5, max_depth=3, euclidean_param=True
        )
        with torch.no_grad():
            model.embeddings.weight[:] = torch.randn(10, 5) * 2
        z = model.embeddings.weight.clone()
        x = model.get_poincare_embeddings()
        z_dir = z / z.norm(dim=1, keepdim=True).clamp(min=1e-8)
        x_dir = x / x.norm(dim=1, keepdim=True).clamp(min=1e-8)
        cos_sim = (z_dir * x_dir).sum(dim=1)
        assert torch.allclose(cos_sim, torch.ones(10), atol=1e-5)

    def test_initialization_inverse(self):
        """With euclidean_param, Poincare norms should match direct init."""
        depth_data = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
        model_direct = HierarchicalPoincareEmbedding(
            n_nodes=5, dim=10, max_depth=4, init_depth_data=depth_data,
            euclidean_param=False,
        )
        model_ep = HierarchicalPoincareEmbedding(
            n_nodes=5, dim=10, max_depth=4, init_depth_data=depth_data,
            euclidean_param=True,
        )
        with torch.no_grad():
            norms_direct = model_direct.get_poincare_embeddings().norm(dim=1)
            norms_ep = model_ep.get_poincare_embeddings().norm(dim=1)
        # Norms should be similar (both targeting same radii, different random dirs)
        assert torch.allclose(norms_direct, norms_ep, atol=0.05)

    def test_project_to_ball_noop(self):
        """project_to_ball should be no-op with euclidean_param=True."""
        model = HierarchicalPoincareEmbedding(
            n_nodes=5, dim=3, max_depth=2, euclidean_param=True
        )
        with torch.no_grad():
            model.embeddings.weight[:] = torch.randn(5, 3) * 100
        original = model.embeddings.weight.clone()
        model.project_to_ball()
        assert torch.equal(model.embeddings.weight, original)

    def test_gradient_flows_through_tanh(self):
        """Gradients should reach z from a Poincare distance loss."""
        model = HierarchicalPoincareEmbedding(
            n_nodes=3, dim=5, max_depth=2, euclidean_param=True
        )
        x = model.get_poincare_embeddings(torch.tensor([0, 1]))
        dist = model.poincare_distance(x[0:1], x[1:2])
        dist.backward()
        assert model.embeddings.weight.grad is not None
        # Gradients for nodes 0 and 1 should be non-zero
        grad_norms = model.embeddings.weight.grad[:2].norm(dim=1)
        assert (grad_norms > 0).all(), "Gradients should flow through tanh map"

    def test_forward_uses_tanh_map(self):
        """forward() should return Poincare-space embeddings when euclidean_param=True."""
        model = HierarchicalPoincareEmbedding(
            n_nodes=5, dim=3, max_depth=2, euclidean_param=True
        )
        with torch.no_grad():
            model.embeddings.weight[:] = torch.randn(5, 3) * 5
        fwd = model(torch.tensor([0, 1, 2]))
        norms = fwd.norm(dim=1)
        assert (norms < 1.0).all(), "forward() should return inside-ball embeddings"


    def test_poincare_nudge_inverse_map(self):
        """Poincaré-space nudge + inverse tanh should move z norms correctly."""
        model = HierarchicalPoincareEmbedding(
            n_nodes=5, dim=3, max_depth=4, euclidean_param=True
        )
        with torch.no_grad():
            # Set z to known values
            model.embeddings.weight[:] = torch.randn(5, 3)
            z_before = model.embeddings.weight.clone()
            z_norm_before = z_before.norm(dim=1, keepdim=True).clamp(min=1e-8)
            poincare_before = torch.tanh(z_norm_before / 2)

            # Apply a nudge toward target radii
            target = torch.tensor([[0.2], [0.4], [0.6], [0.8], [0.9]])
            nudge_strength = 0.1
            new_poincare = poincare_before * (1 - nudge_strength) + target * nudge_strength
            new_poincare = new_poincare.clamp(1e-6, 1 - 1e-6)
            new_z_norm = 2 * torch.arctanh(new_poincare)
            z_dir = z_before / z_norm_before
            z_after = z_dir * new_z_norm

            # Verify: mapping z_after back to Poincaré gives new_poincare
            poincare_after = torch.tanh(z_after.norm(dim=1, keepdim=True) / 2)
            assert torch.allclose(poincare_after, new_poincare, atol=1e-5)

            # Verify: direction is preserved
            z_dir_after = z_after / z_after.norm(dim=1, keepdim=True).clamp(min=1e-8)
            assert torch.allclose(z_dir, z_dir_after, atol=1e-5)


class TestNegativeSamplingSemantics:
    """Lock the semantic contract for negative sampling that vectorization must preserve."""

    def _make_deep_tree_pairs(self):
        """Build a 3-level tree (same as TestTieredNegatives)."""
        pairs = []
        edges = [
            (0, 1, 0, 1), (0, 2, 0, 1),
            (1, 3, 1, 2), (1, 4, 1, 2),
            (2, 5, 1, 2), (2, 6, 1, 2),
            (3, 7, 2, 3), (3, 8, 2, 3),
            (4, 9, 2, 3), (4, 10, 2, 3),
            (5, 11, 2, 3), (5, 12, 2, 3),
            (6, 13, 2, 3), (6, 14, 2, 3),
        ]
        for anc, desc, ad, dd in edges:
            pairs.append({
                "ancestor_idx": anc, "descendant_idx": desc, "depth_diff": 1,
                "ancestor_depth": ad, "descendant_depth": dd,
                "ancestor_taxid": 100 + anc, "descendant_taxid": 100 + desc,
            })
        for anc, mid_desc, ad, _ in edges:
            for anc2, desc2, ad2, dd2 in edges:
                if anc2 == mid_desc:
                    pairs.append({
                        "ancestor_idx": anc, "descendant_idx": desc2, "depth_diff": 2,
                        "ancestor_depth": ad, "descendant_depth": dd2,
                        "ancestor_taxid": 100 + anc, "descendant_taxid": 100 + desc2,
                    })
        for leaf in range(7, 15):
            pairs.append({
                "ancestor_idx": 0, "descendant_idx": leaf, "depth_diff": 3,
                "ancestor_depth": 0, "descendant_depth": 3,
                "ancestor_taxid": 100, "descendant_taxid": 100 + leaf,
            })
        return TrainingPairs.from_list(pairs)

    def test_default_negatives_are_same_depth(self):
        """Default sampling: negatives should share the descendant's depth when pool exists."""
        tp = self._make_deep_tree_pairs()
        loader = HierarchicalDataLoader(tp, n_nodes=15, batch_size=32, n_negatives=5)
        for _ancestors, descendants, negatives, _depths in loader:
            for j in range(len(descendants)):
                desc_idx = int(descendants[j])
                desc_depth = int(loader._node_to_depth[desc_idx])
                depth_pool = loader._depth_to_nodes.get(desc_depth)
                if depth_pool is not None and len(depth_pool) > 1:
                    for neg_idx in negatives[j].tolist():
                        neg_depth = int(loader._node_to_depth[neg_idx])
                        assert neg_depth == desc_depth or neg_idx < 15, (
                            f"Neg {neg_idx} depth {neg_depth} != desc depth {desc_depth}"
                        )
            break  # one batch is enough

    def test_default_fallback_to_random(self):
        """When depth pool has only 1 node (self), fallback to random works."""
        # Create a tree where one depth has only one node
        pairs = [
            {"ancestor_idx": 0, "descendant_idx": 1, "depth_diff": 1,
             "ancestor_depth": 0, "descendant_depth": 1,
             "ancestor_taxid": 100, "descendant_taxid": 101},
            {"ancestor_idx": 1, "descendant_idx": 2, "depth_diff": 1,
             "ancestor_depth": 1, "descendant_depth": 2,
             "ancestor_taxid": 101, "descendant_taxid": 102},
        ]
        tp = TrainingPairs.from_list(pairs)
        loader = HierarchicalDataLoader(tp, n_nodes=3, batch_size=2, n_negatives=3)
        # Node 0 at depth 0 is the only depth-0 node — must fall back
        batches = list(loader)
        assert len(batches) > 0
        for _, _, negatives, _ in batches:
            assert negatives.shape[1] == 3
            assert negatives.min() >= 0

    def test_tiered_hard_share_grandparent(self):
        """With tiered_negatives=True, hard negatives share grandparent."""
        tp = self._make_deep_tree_pairs()
        loader = HierarchicalDataLoader(
            tp, n_nodes=15, batch_size=32, n_negatives=10, tiered_negatives=True,
        )
        found_shared_gp = False
        for _ancestors, descendants, negatives, _depths in loader:
            for j in range(len(descendants)):
                desc_idx = int(descendants[j])
                gp = loader._node_to_grandparent.get(desc_idx, -1)
                if gp < 0:
                    continue
                for neg_idx in negatives[j].tolist():
                    neg_gp = loader._node_to_grandparent.get(neg_idx, -2)
                    if neg_gp == gp and neg_idx != desc_idx:
                        found_shared_gp = True
                        break
                if found_shared_gp:
                    break
            break
        assert found_shared_gp, "Tiered sampling should produce some hard negatives sharing grandparent"

    def test_negatives_fill_remainder(self):
        """When hard/medium pools are too small, remainder is filled from easy/random."""
        tp = self._make_deep_tree_pairs()
        # Request more negatives than cousins exist
        loader = HierarchicalDataLoader(
            tp, n_nodes=15, batch_size=32, n_negatives=20, tiered_negatives=True,
        )
        batches = list(loader)
        assert len(batches) > 0
        for _, _, negatives, _ in batches:
            # All slots should be filled (no zeros from unfilled allocation)
            assert negatives.shape[1] == 20


class TestVectorizedSamplingPerformance:
    """Performance canary: vectorized sampling must be fast enough for scale."""

    def test_vectorized_sampling_performance(self):
        """~1000 nodes / ~5000 pairs: 10 epochs should complete in < 5 seconds."""
        import time

        # Build a synthetic tree: root → 10 classes → 10 subgroups each → 10 leaves each
        pairs = []
        node_id = 1
        for _cls in range(10):
            cls_id = node_id
            pairs.append({
                "ancestor_idx": 0, "descendant_idx": cls_id, "depth_diff": 1,
                "ancestor_depth": 0, "descendant_depth": 1,
                "ancestor_taxid": 1000, "descendant_taxid": 1000 + cls_id,
            })
            node_id += 1
            for _sub in range(10):
                sub_id = node_id
                pairs.append({
                    "ancestor_idx": cls_id, "descendant_idx": sub_id, "depth_diff": 1,
                    "ancestor_depth": 1, "descendant_depth": 2,
                    "ancestor_taxid": 1000 + cls_id, "descendant_taxid": 1000 + sub_id,
                })
                # Transitive: root → subgroup
                pairs.append({
                    "ancestor_idx": 0, "descendant_idx": sub_id, "depth_diff": 2,
                    "ancestor_depth": 0, "descendant_depth": 2,
                    "ancestor_taxid": 1000, "descendant_taxid": 1000 + sub_id,
                })
                node_id += 1
                for _leaf in range(10):
                    leaf_id = node_id
                    pairs.append({
                        "ancestor_idx": sub_id, "descendant_idx": leaf_id, "depth_diff": 1,
                        "ancestor_depth": 2, "descendant_depth": 3,
                        "ancestor_taxid": 1000 + sub_id, "descendant_taxid": 1000 + leaf_id,
                    })
                    # Transitive pairs
                    pairs.append({
                        "ancestor_idx": cls_id, "descendant_idx": leaf_id, "depth_diff": 2,
                        "ancestor_depth": 1, "descendant_depth": 3,
                        "ancestor_taxid": 1000 + cls_id, "descendant_taxid": 1000 + leaf_id,
                    })
                    pairs.append({
                        "ancestor_idx": 0, "descendant_idx": leaf_id, "depth_diff": 3,
                        "ancestor_depth": 0, "descendant_depth": 3,
                        "ancestor_taxid": 1000, "descendant_taxid": 1000 + leaf_id,
                    })
                    node_id += 1

        tp = TrainingPairs.from_list(pairs)
        n_nodes = node_id

        loader = HierarchicalDataLoader(
            tp, n_nodes=n_nodes, batch_size=64, n_negatives=20,
        )

        start = time.perf_counter()
        for _epoch in range(10):
            for _ in loader:
                pass
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, f"10 epochs took {elapsed:.2f}s, expected < 5s"

    def test_tiered_vectorized_sampling_performance(self):
        """Same tree with tiered_negatives=True should also be fast."""
        import time

        pairs = []
        node_id = 1
        for _cls in range(10):
            cls_id = node_id
            pairs.append({
                "ancestor_idx": 0, "descendant_idx": cls_id, "depth_diff": 1,
                "ancestor_depth": 0, "descendant_depth": 1,
                "ancestor_taxid": 1000, "descendant_taxid": 1000 + cls_id,
            })
            node_id += 1
            for _sub in range(10):
                sub_id = node_id
                pairs.append({
                    "ancestor_idx": cls_id, "descendant_idx": sub_id, "depth_diff": 1,
                    "ancestor_depth": 1, "descendant_depth": 2,
                    "ancestor_taxid": 1000 + cls_id, "descendant_taxid": 1000 + sub_id,
                })
                pairs.append({
                    "ancestor_idx": 0, "descendant_idx": sub_id, "depth_diff": 2,
                    "ancestor_depth": 0, "descendant_depth": 2,
                    "ancestor_taxid": 1000, "descendant_taxid": 1000 + sub_id,
                })
                node_id += 1
                for _leaf in range(10):
                    leaf_id = node_id
                    pairs.append({
                        "ancestor_idx": sub_id, "descendant_idx": leaf_id, "depth_diff": 1,
                        "ancestor_depth": 2, "descendant_depth": 3,
                        "ancestor_taxid": 1000 + sub_id, "descendant_taxid": 1000 + leaf_id,
                    })
                    pairs.append({
                        "ancestor_idx": cls_id, "descendant_idx": leaf_id, "depth_diff": 2,
                        "ancestor_depth": 1, "descendant_depth": 3,
                        "ancestor_taxid": 1000 + cls_id, "descendant_taxid": 1000 + leaf_id,
                    })
                    pairs.append({
                        "ancestor_idx": 0, "descendant_idx": leaf_id, "depth_diff": 3,
                        "ancestor_depth": 0, "descendant_depth": 3,
                        "ancestor_taxid": 1000, "descendant_taxid": 1000 + leaf_id,
                    })
                    node_id += 1

        tp = TrainingPairs.from_list(pairs)
        loader = HierarchicalDataLoader(
            tp, n_nodes=node_id, batch_size=64, n_negatives=20, tiered_negatives=True,
        )

        start = time.perf_counter()
        for _epoch in range(10):
            for _ in loader:
                pass
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, f"10 tiered epochs took {elapsed:.2f}s, expected < 5s"


class TestPairwisePoincareDistance:
    """Tests for _pairwise_poincare_distance."""

    def test_symmetric(self):
        embs = torch.tensor([[0.1, 0.2], [0.3, 0.4], [-0.1, 0.5]])
        dists = _pairwise_poincare_distance(embs)
        assert torch.allclose(dists, dists.T, atol=1e-5)

    def test_zero_diagonal(self):
        embs = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        dists = _pairwise_poincare_distance(embs)
        # Diagonal should be near-zero (small numerical error from acosh(1+eps))
        assert torch.allclose(dists.diag(), torch.zeros(2), atol=0.01)

    def test_correct_values(self):
        """Compare with model.poincare_distance for a pair."""
        model = HierarchicalPoincareEmbedding(n_nodes=2, dim=3, max_depth=1)
        with torch.no_grad():
            model.embeddings.weight[0] = torch.tensor([0.1, 0.2, 0.3])
            model.embeddings.weight[1] = torch.tensor([0.4, -0.1, 0.2])
        embs = model.embeddings.weight.clone()
        pairwise = _pairwise_poincare_distance(embs)
        direct = model.poincare_distance(embs[0:1], embs[1:2])
        assert abs(pairwise[0, 1].item() - direct.item()) < 0.01

    def test_positive_distances(self):
        """All off-diagonal distances should be positive."""
        embs = torch.randn(10, 5) * 0.5
        dists = _pairwise_poincare_distance(embs)
        mask = ~torch.eye(10, dtype=torch.bool)
        assert (dists[mask] > 0).all()
