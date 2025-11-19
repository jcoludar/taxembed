"""Hierarchical data loader with depth-aware sampling."""

from collections import defaultdict
from collections.abc import Iterator

import numpy as np
import torch


class HierarchicalDataLoader:
    """Data loader with depth-aware sampling and hard negatives.

    Args:
        training_data: List of dicts with metadata (ancestor_idx, descendant_idx, etc.)
        n_nodes: Total number of nodes
        batch_size: Batch size for training
        n_negatives: Number of negative samples per positive pair
        depth_stratify: Whether to use depth-aware stratification
    """

    def __init__(
        self,
        training_data: list[dict],
        n_nodes: int,
        batch_size: int = 32,
        n_negatives: int = 50,
        depth_stratify: bool = True,
    ):
        self.training_data = training_data
        self.n_nodes = n_nodes
        self.batch_size = batch_size
        self.n_negatives = n_negatives
        self.depth_stratify = depth_stratify

        # Build index by depth for stratified sampling
        if depth_stratify:
            self.depth_buckets = defaultdict(list)
            for i, item in enumerate(training_data):
                depth_diff = item["depth_diff"]
                self.depth_buckets[depth_diff].append(i)
            print(f"  ✓ Created {len(self.depth_buckets)} depth buckets for sampling")

        # Build node → siblings map for hard negatives
        self._build_sibling_map()

    def _build_sibling_map(self):
        """Build map: node → nodes at same depth (for hard negatives)."""
        print("  Building sibling map for hard negatives...")

        depth_to_nodes = defaultdict(set)
        for item in self.training_data:
            depth_to_nodes[item["descendant_depth"]].add(item["descendant_idx"])

        self.sibling_map = {}
        for _depth, nodes in depth_to_nodes.items():
            nodes_list = list(nodes)
            for node in nodes_list:
                # Siblings = other nodes at same depth
                self.sibling_map[node] = [n for n in nodes_list if n != node]

        print("  ✓ Built sibling map for hard negatives")

    def __len__(self) -> int:
        return len(self.training_data) // self.batch_size

    def __iter__(
        self,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Iterate over batches with depth-aware sampling.

        Yields:
            Tuple of (ancestors, descendants, negatives, depths) tensors
        """
        indices = list(range(len(self.training_data)))

        if self.depth_stratify:
            # Stratified sampling: mix shallow and deep pairs
            np.random.shuffle(indices)
        else:
            # Random sampling
            np.random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            batch = [self.training_data[idx] for idx in batch_indices]

            # Extract data (vectorized for speed)
            batch_size = len(batch)
            ancestors_np = np.zeros(batch_size, dtype=np.int64)
            descendants_np = np.zeros(batch_size, dtype=np.int64)
            depths_np = np.zeros(batch_size, dtype=np.float32)

            for j, item in enumerate(batch):
                ancestors_np[j] = item["ancestor_idx"]
                descendants_np[j] = item["descendant_idx"]
                depths_np[j] = item["depth_diff"]

            # Convert to tensors once
            ancestors = torch.from_numpy(ancestors_np)
            descendants = torch.from_numpy(descendants_np)
            depths = torch.from_numpy(depths_np)

            # Sample hard negatives (cousins at same depth)
            negatives_np = np.zeros((batch_size, self.n_negatives), dtype=np.int64)
            for j, item in enumerate(batch):
                desc_idx = item["descendant_idx"]

                # Get siblings (nodes at same depth)
                siblings = self.sibling_map.get(desc_idx, [])

                if len(siblings) >= self.n_negatives:
                    # Sample from siblings (hard negatives)
                    negatives_np[j] = np.random.choice(siblings, self.n_negatives, replace=False)
                else:
                    # Mix siblings + random negatives
                    n_sibling = len(siblings)
                    n_random = self.n_negatives - n_sibling
                    if n_sibling > 0:
                        negatives_np[j, :n_sibling] = siblings
                        negatives_np[j, n_sibling:] = np.random.choice(
                            self.n_nodes, n_random, replace=False
                        )
                    else:
                        negatives_np[j] = np.random.choice(
                            self.n_nodes, self.n_negatives, replace=False
                        )

            negatives = torch.from_numpy(negatives_np)

            yield ancestors, descendants, negatives, depths
