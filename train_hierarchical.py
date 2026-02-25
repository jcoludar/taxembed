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
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import argparse
import os


@dataclass
class TrainingPairs:
    """Columnar storage for training pairs. O(N) memory instead of O(N * dict_overhead).

    Replaces the legacy List[Dict] format with numpy arrays.
    For 48M pairs: ~480MB (columnar) vs ~16GB (list of dicts).
    """

    ancestor_idx: np.ndarray      # int32
    descendant_idx: np.ndarray    # int32
    depth_diff: np.ndarray        # int16
    ancestor_depth: np.ndarray    # int16
    descendant_depth: np.ndarray  # int16
    ancestor_taxid: np.ndarray    # int32 (for debugging / provenance)
    descendant_taxid: np.ndarray  # int32

    def __len__(self) -> int:
        return len(self.ancestor_idx)

    def __getitem__(self, idx):
        """Support integer and slice/array indexing (returns dict for single, TrainingPairs for batch)."""
        if isinstance(idx, (int, np.integer)):
            return {
                "ancestor_idx": int(self.ancestor_idx[idx]),
                "descendant_idx": int(self.descendant_idx[idx]),
                "depth_diff": int(self.depth_diff[idx]),
                "ancestor_depth": int(self.ancestor_depth[idx]),
                "descendant_depth": int(self.descendant_depth[idx]),
                "ancestor_taxid": int(self.ancestor_taxid[idx]),
                "descendant_taxid": int(self.descendant_taxid[idx]),
            }
        return TrainingPairs(
            ancestor_idx=self.ancestor_idx[idx],
            descendant_idx=self.descendant_idx[idx],
            depth_diff=self.depth_diff[idx],
            ancestor_depth=self.ancestor_depth[idx],
            descendant_depth=self.descendant_depth[idx],
            ancestor_taxid=self.ancestor_taxid[idx],
            descendant_taxid=self.descendant_taxid[idx],
        )

    @classmethod
    def from_list(cls, pairs: list[dict]) -> "TrainingPairs":
        """Convert legacy list-of-dicts format to columnar arrays."""
        n = len(pairs)
        ancestor_idx = np.empty(n, dtype=np.int32)
        descendant_idx = np.empty(n, dtype=np.int32)
        depth_diff = np.empty(n, dtype=np.int16)
        ancestor_depth = np.empty(n, dtype=np.int16)
        descendant_depth = np.empty(n, dtype=np.int16)
        ancestor_taxid = np.empty(n, dtype=np.int32)
        descendant_taxid = np.empty(n, dtype=np.int32)

        for i, p in enumerate(pairs):
            ancestor_idx[i] = p["ancestor_idx"]
            descendant_idx[i] = p["descendant_idx"]
            depth_diff[i] = p["depth_diff"]
            ancestor_depth[i] = p["ancestor_depth"]
            descendant_depth[i] = p["descendant_depth"]
            ancestor_taxid[i] = p["ancestor_taxid"]
            descendant_taxid[i] = p["descendant_taxid"]

        return cls(
            ancestor_idx=ancestor_idx,
            descendant_idx=descendant_idx,
            depth_diff=depth_diff,
            ancestor_depth=ancestor_depth,
            descendant_depth=descendant_depth,
            ancestor_taxid=ancestor_taxid,
            descendant_taxid=descendant_taxid,
        )

    @classmethod
    def load(cls, path: Path) -> "TrainingPairs":
        """Load from .npz file."""
        data = np.load(path)
        return cls(
            ancestor_idx=data["ancestor_idx"],
            descendant_idx=data["descendant_idx"],
            depth_diff=data["depth_diff"],
            ancestor_depth=data["ancestor_depth"],
            descendant_depth=data["descendant_depth"],
            ancestor_taxid=data["ancestor_taxid"],
            descendant_taxid=data["descendant_taxid"],
        )

    def save(self, path: Path) -> None:
        """Save as .npz file."""
        np.savez_compressed(
            path,
            ancestor_idx=self.ancestor_idx,
            descendant_idx=self.descendant_idx,
            depth_diff=self.depth_diff,
            ancestor_depth=self.ancestor_depth,
            descendant_depth=self.descendant_depth,
            ancestor_taxid=self.ancestor_taxid,
            descendant_taxid=self.descendant_taxid,
        )

    @property
    def n_nodes(self) -> int:
        return int(max(self.ancestor_idx.max(), self.descendant_idx.max())) + 1

    @property
    def max_depth(self) -> int:
        return int(self.descendant_depth.max())

    def idx_to_depth_dict(self) -> dict[int, int]:
        """Build idx → depth mapping (for model initialization)."""
        result: dict[int, int] = {}
        for i in range(len(self)):
            desc_idx = int(self.descendant_idx[i])
            result[desc_idx] = int(self.descendant_depth[i])
            anc_idx = int(self.ancestor_idx[i])
            if anc_idx not in result:
                result[anc_idx] = int(self.ancestor_depth[i])
        return result


def target_radius(depth, max_depth, schedule='linear'):
    """Compute target radius for a node at the given depth.

    Works with scalars, numpy arrays, and torch tensors.

    Args:
        depth: Node depth.
        max_depth: Maximum depth in the tree.
        schedule: 'linear' (default) or 'log'.
    """
    if schedule == 'log':
        return 0.1 + 0.85 * (np.log1p(depth) / np.log1p(max_depth))
    return 0.1 + (depth / max_depth) * 0.85


class HierarchicalPoincareEmbedding(nn.Module):
    """Poincaré embeddings with hierarchical structure."""

    def __init__(self, n_nodes, dim=10, max_depth=38, init_depth_data=None,
                 radial_schedule='linear', euclidean_param=False):
        super().__init__()
        self.n_nodes = n_nodes
        self.dim = dim
        self.max_depth = max_depth
        self.radial_schedule = radial_schedule
        self.euclidean_param = euclidean_param

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

                # Radius increases with depth (schedule-aware)
                tr = target_radius(depth, self.max_depth, self.radial_schedule)

                # Random direction on sphere
                vec = torch.randn(self.dim)
                vec = vec / vec.norm()

                # Scale to target radius (or inverse-tanh for euclidean_param)
                if self.euclidean_param:
                    z_norm = 2 * torch.arctanh(torch.tensor(tr).clamp(max=0.999))
                    self.embeddings.weight[idx] = vec * z_norm
                else:
                    self.embeddings.weight[idx] = vec * tr

        norms = self.embeddings.weight.norm(dim=1)
        print(f"  ✓ Initialized ({self.radial_schedule}): norm range [{norms.min():.3f}, {norms.max():.3f}]")
    
    def get_poincare_embeddings(self, indices=None):
        """Map z in R^d to x in B^d via x = tanh(||z||/2) * z/||z||.

        When euclidean_param=False, returns raw embeddings (already in ball).
        """
        z = self.embeddings(indices) if indices is not None else self.embeddings.weight
        if not self.euclidean_param:
            return z
        z_norm = z.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return torch.tanh(z_norm / 2) * (z / z_norm)

    def forward(self, indices):
        """Get Poincare-space embeddings for indices."""
        return self.get_poincare_embeddings(indices)
    
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
    
    def project_to_ball(self, indices=None, max_norm=0.999):
        """
        Project embeddings back into Poincaré ball with HARD constraint.

        No-op when euclidean_param=True (tanh map guarantees ||x|| < 1).

        Args:
            indices: If provided, only project these indices (more efficient).
                    If None, project all embeddings.
            max_norm: Maximum allowed norm (default 0.999, essentially at boundary)
        """
        if self.euclidean_param:
            return
        with torch.no_grad():
            if indices is not None:
                # Only project updated embeddings
                embs = self.embeddings.weight[indices]
                norms = embs.norm(dim=1, keepdim=True)
                # Hard projection: if norm >= max_norm, scale it down
                # Use where to only scale embeddings that need it
                needs_projection = norms >= max_norm
                scale = torch.where(
                    needs_projection,
                    max_norm / (norms + 1e-8),
                    torch.ones_like(norms)
                )
                self.embeddings.weight[indices] = embs * scale
            else:
                # Project all embeddings
                norms = self.embeddings.weight.norm(dim=1, keepdim=True)
                needs_projection = norms >= max_norm
                scale = torch.where(
                    needs_projection,
                    max_norm / (norms + 1e-8),
                    torch.ones_like(norms)
                )
                self.embeddings.weight.mul_(scale)


class HierarchicalDataLoader:
    """Data loader with depth-aware sampling and hard negatives.

    Supports both legacy List[Dict] and columnar TrainingPairs input.
    Memory-efficient: uses depth-indexed arrays instead of per-node sibling lists.
    """

    def __init__(self, training_data, n_nodes, batch_size=32,
                 n_negatives=50, depth_stratify=True, epoch_fraction=1.0,
                 tiered_negatives=False, class_balanced=False):
        # Normalize to TrainingPairs if needed
        if isinstance(training_data, TrainingPairs):
            self.pairs = training_data
        else:
            self.pairs = TrainingPairs.from_list(training_data)

        self.n_nodes = n_nodes
        self.batch_size = batch_size
        self.n_negatives = n_negatives
        self.depth_stratify = depth_stratify
        self.epoch_fraction = epoch_fraction
        self.tiered_negatives = tiered_negatives
        self.class_balanced = class_balanced

        # Build depth-indexed arrays for O(N) memory hard negative sampling
        self._build_depth_index()

        # Pre-index by depth_diff for curriculum filtering (vectorized)
        self._unique_depth_diffs = np.unique(self.pairs.depth_diff)
        self._depth_diff_masks = {
            int(dd): np.where(self.pairs.depth_diff == dd)[0]
            for dd in self._unique_depth_diffs
        }
        if depth_stratify:
            print(f"  ✓ Created {len(self._depth_diff_masks)} depth buckets for sampling")

        # Class-balanced pair indexing
        if class_balanced:
            self._build_class_pair_index()

        # Curriculum state: None means full dataset
        self._curriculum_max_depth_diff = None
        self._curriculum_indices = None

    def _build_depth_index(self):
        """Build depth-to-nodes arrays for hard negative sampling.

        O(N) memory: one array per depth level, shared across all nodes.
        Replaces the old O(N^2) sibling_map.
        """
        print("  Building depth-indexed arrays for hard negatives...")

        # Collect unique (descendant_idx, descendant_depth) pairs
        all_desc_idx = self.pairs.descendant_idx
        all_desc_depth = self.pairs.descendant_depth
        all_anc_idx = self.pairs.ancestor_idx
        all_anc_depth = self.pairs.ancestor_depth

        # Build node_to_depth from all observed nodes
        node_depth = {}
        for i in range(len(self.pairs)):
            node_depth[int(all_desc_idx[i])] = int(all_desc_depth[i])
            anc = int(all_anc_idx[i])
            if anc not in node_depth:
                node_depth[anc] = int(all_anc_depth[i])

        # Group nodes by depth
        depth_to_nodes: dict[int, list[int]] = defaultdict(list)
        for node_idx, depth in node_depth.items():
            depth_to_nodes[depth].append(node_idx)

        self._depth_to_nodes: dict[int, np.ndarray] = {
            d: np.array(nodes, dtype=np.int64)
            for d, nodes in depth_to_nodes.items()
        }

        # Fast lookup: node_idx → depth
        self._node_to_depth = np.zeros(self.n_nodes, dtype=np.int16)
        for node_idx, depth in node_depth.items():
            if node_idx < self.n_nodes:
                self._node_to_depth[node_idx] = depth

        total_entries = sum(len(arr) for arr in self._depth_to_nodes.values())
        print(f"  ✓ Built depth index: {len(self._depth_to_nodes)} levels, {total_entries:,} entries")

        # --- Ancestry structures for tiered negative sampling ---
        # Build child→parent map from dd=1 (parent-child) pairs
        dd1_mask = self.pairs.depth_diff == 1
        dd1_parents = self.pairs.ancestor_idx[dd1_mask]
        dd1_children = self.pairs.descendant_idx[dd1_mask]
        self._child_to_parent: dict[int, int] = {}
        for i in range(len(dd1_parents)):
            self._child_to_parent[int(dd1_children[i])] = int(dd1_parents[i])

        # Build grandparent lookup and (grandparent, depth) → nodes index
        self._node_to_grandparent: dict[int, int] = {}
        gp_depth_list: dict[tuple[int, int], list[int]] = defaultdict(list)
        for node_idx, depth in node_depth.items():
            parent = self._child_to_parent.get(node_idx)
            if parent is not None:
                gp = self._child_to_parent.get(parent)
                if gp is not None:
                    self._node_to_grandparent[node_idx] = gp
                    gp_depth_list[(gp, depth)].append(node_idx)

        self._gp_depth_to_nodes: dict[tuple[int, int], np.ndarray] = {
            key: np.array(nodes, dtype=np.int64)
            for key, nodes in gp_depth_list.items()
        }

        # Build class (depth-1 ancestor) labels via BFS from root children
        depth0_nodes = {idx for idx, d in node_depth.items() if d == 0}
        parent_to_children_map: dict[int, list[int]] = defaultdict(list)
        for child, parent in self._child_to_parent.items():
            parent_to_children_map[parent].append(child)

        root_children: set[int] = set()
        for child, parent in self._child_to_parent.items():
            if parent in depth0_nodes:
                root_children.add(child)

        self._node_to_class: dict[int, int] = {}
        for rc in root_children:
            self._node_to_class[rc] = rc
            queue = deque([rc])
            while queue:
                node = queue.popleft()
                for child in parent_to_children_map.get(node, []):
                    if child not in self._node_to_class:
                        self._node_to_class[child] = rc
                        queue.append(child)

        # Build (class, depth) → nodes index
        class_depth_list: dict[tuple[int, int], list[int]] = defaultdict(list)
        for node_idx, class_idx in self._node_to_class.items():
            d = node_depth.get(node_idx, 0)
            class_depth_list[(class_idx, d)].append(node_idx)

        self._class_depth_to_nodes: dict[tuple[int, int], np.ndarray] = {
            key: np.array(nodes, dtype=np.int64)
            for key, nodes in class_depth_list.items()
        }

        # Vectorized array forms for fast batch-level lookups (Iteration 6 scaling)
        self._node_gp_arr = np.full(self.n_nodes, -1, dtype=np.int64)
        for node, gp in self._node_to_grandparent.items():
            if node < self.n_nodes:
                self._node_gp_arr[node] = gp

        self._node_class_arr = np.full(self.n_nodes, -1, dtype=np.int64)
        for node, cls in self._node_to_class.items():
            if node < self.n_nodes:
                self._node_class_arr[node] = cls

        n_with_gp = len(self._node_to_grandparent)
        n_with_class = len(self._node_to_class)
        n_classes = len(root_children)
        print(f"  ✓ Tiered negatives: {n_with_gp:,} with grandparent, "
              f"{n_with_class:,} with class ({n_classes} classes)")

    def _build_class_pair_index(self):
        """Group pair indices by the descendant's top-level class.

        Used for class-balanced sampling: each epoch draws equal numbers of
        pairs from each class bucket, so minority classes (Bivalvia, Cephalopoda)
        get proportional gradient signal regardless of their size.
        """
        # Assign each pair to the class of its descendant node
        n_pairs = len(self.pairs)
        pair_classes = np.full(n_pairs, -1, dtype=np.int64)
        for i in range(n_pairs):
            desc_idx = int(self.pairs.descendant_idx[i])
            if desc_idx < self.n_nodes:
                pair_classes[i] = self._node_class_arr[desc_idx]

        # Group pair indices by class
        self._class_to_pair_indices: dict[int, np.ndarray] = {}
        unique_classes = np.unique(pair_classes)
        for cls in unique_classes:
            if cls < 0:
                continue
            mask = pair_classes == cls
            self._class_to_pair_indices[cls] = np.where(mask)[0]

        # Report
        total_labeled = sum(len(v) for v in self._class_to_pair_indices.values())
        class_names = list(self._class_to_pair_indices.keys())
        sizes = [len(self._class_to_pair_indices[c]) for c in class_names]
        print(f"  ✓ Class-balanced: {len(class_names)} classes, {total_labeled:,} labeled pairs")
        for cls, sz in sorted(zip(class_names, sizes), key=lambda x: -x[1]):
            pct = sz / total_labeled * 100 if total_labeled > 0 else 0
            print(f"      class {cls}: {sz:>8,} pairs ({pct:5.1f}%)")

    def set_curriculum_phase(self, max_depth_diff):
        """Restrict iteration to pairs with depth_diff <= max_depth_diff."""
        parts = []
        for dd, idx_arr in self._depth_diff_masks.items():
            if dd <= max_depth_diff:
                parts.append(idx_arr)
        self._curriculum_max_depth_diff = max_depth_diff
        self._curriculum_indices = np.concatenate(parts) if parts else np.array([], dtype=np.int64)

    def clear_curriculum(self):
        """Remove curriculum filter; iterate over the full dataset."""
        self._curriculum_max_depth_diff = None
        self._curriculum_indices = None

    def _sample_negatives_default_vectorized(
        self, desc_depths: np.ndarray, desc_idxs: np.ndarray, batch_size: int
    ) -> np.ndarray:
        """Vectorized default negative sampling grouped by depth.

        Instead of O(batch_size) Python iterations, groups items by unique depth
        and samples O(unique_depths) bulk arrays (~15-40 iterations for typical trees).
        """
        negatives = np.zeros((batch_size, self.n_negatives), dtype=np.int64)
        unique_depths, inverse = np.unique(desc_depths, return_inverse=True)

        for di, d in enumerate(unique_depths):
            mask = inverse == di  # items at this depth
            n_items = int(mask.sum())
            pool = self._depth_to_nodes.get(int(d))

            if pool is not None and len(pool) > self.n_negatives:
                # Fast path: pool large enough for bulk sampling
                rand_positions = np.random.randint(0, len(pool), size=(n_items, self.n_negatives))
                negatives[mask] = pool[rand_positions]
            elif pool is not None and len(pool) > 1:
                # Medium path: pool exists but small — per-item with replacement awareness
                items_at_depth = np.where(mask)[0]
                self_idxs = desc_idxs[mask]
                for item_pos, self_idx in zip(items_at_depth, self_idxs, strict=False):
                    cands = pool[pool != self_idx]
                    if len(cands) >= self.n_negatives:
                        negatives[item_pos] = np.random.choice(cands, self.n_negatives, replace=False)
                    elif len(cands) > 0:
                        negatives[item_pos, :len(cands)] = cands
                        negatives[item_pos, len(cands):] = np.random.choice(
                            self.n_nodes, self.n_negatives - len(cands)
                        )
                    else:
                        negatives[item_pos] = np.random.choice(self.n_nodes, self.n_negatives)
            else:
                # Fallback: no depth pool — random from all nodes
                negatives[mask] = np.random.randint(0, self.n_nodes, size=(n_items, self.n_negatives))

        return negatives

    def _sample_negatives_tiered_vectorized(
        self, desc_depths: np.ndarray, desc_idxs: np.ndarray, batch_size: int
    ) -> np.ndarray:
        """Vectorized tiered negative sampling: 50% hard, 30% medium, 20% easy.

        Groups batch items by (grandparent, depth) and (class, depth) pairs
        to minimize Python iterations.
        """
        n_hard = self.n_negatives // 2
        n_medium = int(self.n_negatives * 0.3)
        negatives = np.zeros((batch_size, self.n_negatives), dtype=np.int64)
        filled = np.zeros(batch_size, dtype=np.int32)

        # Vectorized lookups
        gps = self._node_gp_arr[desc_idxs.astype(np.int64)]
        classes = self._node_class_arr[desc_idxs.astype(np.int64)]

        # --- Hard tier (50%): group by (grandparent, depth) ---
        gp_depth_keys = np.stack([gps, desc_depths.astype(np.int64)], axis=1)
        # Find unique (gp, depth) combos (excluding gp=-1)
        valid_gp = gps >= 0
        if valid_gp.any():
            valid_indices = np.where(valid_gp)[0]
            valid_keys = gp_depth_keys[valid_indices]
            unique_keys, key_inverse = np.unique(valid_keys, axis=0, return_inverse=True)

            for ki in range(len(unique_keys)):
                gp_val, d_val = int(unique_keys[ki, 0]), int(unique_keys[ki, 1])
                item_mask = key_inverse == ki
                item_positions = valid_indices[item_mask]
                pool = self._gp_depth_to_nodes.get((gp_val, d_val))
                if pool is None or len(pool) < 2:
                    continue

                n_items = len(item_positions)
                n_want = min(n_hard, len(pool) - 1)  # reserve 1 for self-exclusion
                if n_want <= 0:
                    continue

                if len(pool) > n_want + 5:
                    # Rejection sampling: sample and rely on low collision probability
                    rand_pos = np.random.randint(0, len(pool), size=(n_items, n_want))
                    sampled = pool[rand_pos]
                else:
                    # Small pool: per-item careful sampling
                    sampled = np.zeros((n_items, n_want), dtype=np.int64)
                    self_idxs = desc_idxs[item_positions]
                    for li in range(n_items):
                        cands = pool[pool != self_idxs[li]]
                        n_take = min(n_want, len(cands))
                        if n_take > 0:
                            sampled[li, :n_take] = np.random.choice(cands, n_take, replace=False)

                for li, pos in enumerate(item_positions):
                    n_got = min(n_want, self.n_negatives)
                    negatives[pos, :n_got] = sampled[li, :n_got]
                    filled[pos] = n_got

        # --- Medium tier (30%): group by (class, depth) ---
        valid_cls = classes >= 0
        if valid_cls.any():
            cls_depth_keys = np.stack([classes, desc_depths.astype(np.int64)], axis=1)
            valid_cls_indices = np.where(valid_cls)[0]
            valid_cls_keys = cls_depth_keys[valid_cls_indices]
            unique_cls_keys, cls_inverse = np.unique(valid_cls_keys, axis=0, return_inverse=True)

            for ki in range(len(unique_cls_keys)):
                cls_val, d_val = int(unique_cls_keys[ki, 0]), int(unique_cls_keys[ki, 1])
                item_mask = cls_inverse == ki
                item_positions = valid_cls_indices[item_mask]
                pool = self._class_depth_to_nodes.get((cls_val, d_val))
                if pool is None or len(pool) < 2:
                    continue

                for pos in item_positions:
                    already_filled = int(filled[pos])
                    remaining_medium = min(n_medium, self.n_negatives - already_filled)
                    if remaining_medium <= 0:
                        continue
                    self_idx = int(desc_idxs[pos])
                    cands = pool[pool != self_idx]
                    if len(cands) <= 0:
                        continue
                    n_take = min(remaining_medium, len(cands))
                    chosen = np.random.choice(cands, n_take, replace=False)
                    negatives[pos, already_filled:already_filled + n_take] = chosen
                    filled[pos] = already_filled + n_take

        # --- Easy tier (remainder): same-depth random ---
        unique_depths, depth_inverse = np.unique(desc_depths, return_inverse=True)
        for di, d in enumerate(unique_depths):
            d_int = int(d)
            mask = depth_inverse == di
            positions = np.where(mask)[0]
            pool = self._depth_to_nodes.get(d_int)

            for pos in positions:
                remaining = self.n_negatives - int(filled[pos])
                if remaining <= 0:
                    continue
                self_idx = int(desc_idxs[pos])
                start = int(filled[pos])

                if pool is not None and len(pool) > 1:
                    cands = pool[pool != self_idx]
                    if len(cands) >= remaining:
                        negatives[pos, start:start + remaining] = np.random.choice(
                            cands, remaining, replace=False
                        )
                        filled[pos] = self.n_negatives
                    elif len(cands) > 0:
                        negatives[pos, start:start + len(cands)] = cands
                        filled[pos] += len(cands)
                        leftover = remaining - len(cands)
                        negatives[pos, start + len(cands):start + len(cands) + leftover] = (
                            np.random.choice(self.n_nodes, leftover)
                        )
                        filled[pos] = self.n_negatives
                    else:
                        negatives[pos, start:] = np.random.choice(self.n_nodes, remaining)
                        filled[pos] = self.n_negatives
                else:
                    negatives[pos, start:] = np.random.choice(self.n_nodes, remaining)
                    filled[pos] = self.n_negatives

        return negatives

    def _active_indices(self) -> np.ndarray:
        if self._curriculum_indices is not None:
            return self._curriculum_indices.copy()
        return np.arange(len(self.pairs), dtype=np.int64)

    def __len__(self):
        n = len(self._active_indices())
        if self.epoch_fraction < 1.0:
            n = int(n * self.epoch_fraction)
        return n // self.batch_size

    def __iter__(self):
        """Iterate over batches with vectorized construction and epoch subsampling."""
        indices = self._active_indices()

        # Class-balanced sampling: draw equal numbers from each class bucket
        if self.class_balanced and hasattr(self, '_class_to_pair_indices') and self._class_to_pair_indices:
            # Filter class buckets to only include active indices
            active_set = set(indices) if self._curriculum_indices is not None else None
            class_buckets = {}
            for cls, pair_idx in self._class_to_pair_indices.items():
                if active_set is not None:
                    filtered = pair_idx[np.isin(pair_idx, indices)]
                else:
                    filtered = pair_idx
                if len(filtered) > 0:
                    class_buckets[cls] = filtered

            if class_buckets:
                n_classes = len(class_buckets)
                # Total pairs to draw this epoch
                total_target = len(indices)
                if self.epoch_fraction < 1.0:
                    total_target = int(total_target * self.epoch_fraction)
                per_class = max(1, total_target // n_classes)

                # Sample per_class pairs from each bucket (with replacement if bucket is small)
                parts = []
                for cls, bucket in class_buckets.items():
                    replace = len(bucket) < per_class
                    sampled = np.random.choice(bucket, per_class, replace=replace)
                    parts.append(sampled)
                indices = np.concatenate(parts)
                np.random.shuffle(indices)
            else:
                np.random.shuffle(indices)
                if self.epoch_fraction < 1.0:
                    n_sample = int(len(indices) * self.epoch_fraction)
                    if n_sample > 0:
                        indices = np.random.choice(indices, n_sample, replace=False)
        else:
            # Epoch subsampling: train on a fraction each epoch
            if self.epoch_fraction < 1.0:
                # Depth-stratified sampling: sample equally from each depth_diff
                # level to prevent deeper pairs from diluting shallower ones.
                # Each dd level contributes epoch_fraction of its pairs.
                active_set = set(indices) if self._curriculum_indices is not None else None
                parts = []
                for dd in sorted(self._depth_diff_masks.keys()):
                    dd_indices = self._depth_diff_masks[dd]
                    if active_set is not None:
                        dd_indices = dd_indices[np.isin(dd_indices, indices)]
                    if len(dd_indices) == 0:
                        continue
                    n_sample = max(1, int(len(dd_indices) * self.epoch_fraction))
                    sampled = np.random.choice(dd_indices, n_sample, replace=False)
                    parts.append(sampled)
                if parts:
                    indices = np.concatenate(parts)
                np.random.shuffle(indices)
            else:
                np.random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_size = len(batch_indices)

            # Vectorized extraction from columnar arrays
            ancestors = torch.from_numpy(self.pairs.ancestor_idx[batch_indices].astype(np.int64))
            descendants = torch.from_numpy(self.pairs.descendant_idx[batch_indices].astype(np.int64))
            depths = torch.from_numpy(self.pairs.depth_diff[batch_indices].astype(np.float32))

            # Sample hard negatives using vectorized depth-grouped operations
            desc_depths = self.pairs.descendant_depth[batch_indices]
            desc_idxs = self.pairs.descendant_idx[batch_indices]

            if self.tiered_negatives:
                negatives = self._sample_negatives_tiered_vectorized(desc_depths, desc_idxs, batch_size)
            else:
                negatives = self._sample_negatives_default_vectorized(desc_depths, desc_idxs, batch_size)

            negatives = torch.from_numpy(negatives)

            yield ancestors, descendants, negatives, depths


def ranking_loss_with_margin(model, ancestors, descendants, negatives,
                             depths, margin=0.1, depth_weight=True,
                             depth_scale_margin=False, margin_min=0.05,
                             margin_max=1.0, max_depth_diff=None,
                             class_weights=None):
    """
    Ranking loss with margin and optional depth weighting.

    Loss encourages:
    - d(ancestor, descendant) < d(ancestor, negative) + margin
    - Deeper pairs get higher weight (they're more informative)

    When depth_scale_margin is True, the margin scales with depth_diff:
        margin = margin_min + (margin_max - margin_min) * (depth / max_depth_diff)
    Small margins for local pairs (dd=1) produce tight clustering;
    large margins for distant pairs produce strong global separation.

    When class_weights is provided (tensor of shape [n_nodes]), each pair's
    loss is multiplied by the weight of its descendant's class. This upweights
    minority classes to counter imbalanced sampling.
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

    # Margin: either fixed or depth-scaled
    if depth_scale_margin and max_depth_diff is not None and max_depth_diff > 0:
        scaled_margin = margin_min + (margin_max - margin_min) * (depths / max_depth_diff)
        losses = torch.relu(pos_dist.unsqueeze(1) - neg_dist + scaled_margin.unsqueeze(1))
    else:
        losses = torch.relu(pos_dist.unsqueeze(1) - neg_dist + margin)  # (batch, n_neg)
    loss = losses.mean(dim=1)  # Average over negatives: (batch,)

    # Depth weighting: deeper pairs are more important
    if depth_weight:
        # Weight = sqrt(depth) to emphasize deep pairs without over-weighting
        weights = torch.sqrt(depths + 1)  # +1 to avoid zero weight
        weights = weights / weights.mean()  # Normalize
        loss = loss * weights

    # Class weighting: upweight minority class pairs
    if class_weights is not None:
        cw = class_weights[descendants]  # (batch,)
        loss = loss * cw

    return loss.mean()


def radial_regularizer(model, idx_to_depth_tensor, target_radii_tensor, lambda_reg=0.01):
    """
    Vectorized radial regularizer to keep nodes at expected radius based on depth.

    Encourages: ||embedding|| ≈ f(depth)
    where f(depth) = 0.1 + (depth/max_depth) * 0.85

    When euclidean_param=True, computes Poincaré norms as tanh(||z||/2)
    so gradients flow through tanh back to z.

    Args:
        idx_to_depth_tensor: Tensor of indices to regularize
        target_radii_tensor: Tensor of target radii for each index
    """
    if len(idx_to_depth_tensor) == 0:
        return torch.tensor(0.0, device=model.embeddings.weight.device)

    if model.euclidean_param:
        # Compute Poincaré-space norms via tanh map (gradient flows through)
        z = model.embeddings.weight[idx_to_depth_tensor]  # (n, dim)
        z_norm = z.norm(dim=1).clamp(min=1e-8)  # (n,)
        actual_radii = torch.tanh(z_norm / 2)  # (n,)
    else:
        # Get embeddings for these indices
        embs = model.embeddings.weight[idx_to_depth_tensor]  # (n, dim)
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
    schedule = getattr(model, 'radial_schedule', 'linear')
    for idx, depth in idx_to_depth.items():
        if idx < model.n_nodes:
            reg_indices.append(idx)
            reg_target_radii.append(target_radius(depth, max_depth, schedule))
    
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
    
    # Load training data (npz or pickle)
    print(f"Loading training data from {args.data}...")
    data_path = Path(args.data)
    if data_path.suffix == ".npz":
        pairs = TrainingPairs.load(data_path)
    else:
        with open(data_path, 'rb') as f:
            pairs = TrainingPairs.from_list(pickle.load(f))
    training_data = pairs
    print(f"  ✓ Loaded {len(pairs):,} training pairs")

    # Get number of nodes and max depth
    n_nodes = pairs.n_nodes
    max_depth = pairs.max_depth

    print(f"  Nodes: {n_nodes:,}")
    print(f"  Max depth: {max_depth}")

    # Build idx → depth mapping for initialization
    idx_to_depth = pairs.idx_to_depth_dict()
    
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
