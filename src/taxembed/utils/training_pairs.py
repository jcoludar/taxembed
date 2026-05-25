"""Columnar storage for ancestor-descendant training pairs.

Replaces the legacy List[Dict] format with numpy arrays. For 60M pairs the
columnar form is ~1.2 GB instead of ~26 GB.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TrainingPairs:
    """Columnar storage for training pairs.

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
        """Integer indexing returns a dict; slice/array indexing returns TrainingPairs."""
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

    @classmethod
    def concat(cls, *parts: "TrainingPairs") -> "TrainingPairs":
        """Concatenate multiple TrainingPairs into one. Empty inputs are skipped."""
        parts = tuple(p for p in parts if len(p) > 0)
        if not parts:
            raise ValueError("concat requires at least one non-empty TrainingPairs")
        if len(parts) == 1:
            return parts[0]
        return cls(
            ancestor_idx=np.concatenate([p.ancestor_idx for p in parts]),
            descendant_idx=np.concatenate([p.descendant_idx for p in parts]),
            depth_diff=np.concatenate([p.depth_diff for p in parts]),
            ancestor_depth=np.concatenate([p.ancestor_depth for p in parts]),
            descendant_depth=np.concatenate([p.descendant_depth for p in parts]),
            ancestor_taxid=np.concatenate([p.ancestor_taxid for p in parts]),
            descendant_taxid=np.concatenate([p.descendant_taxid for p in parts]),
        )

    def write_edgelist(self, path: Path) -> None:
        """Write `ancestor_idx descendant_idx\\n` edgelist directly from arrays."""
        with Path(path).open("w") as handle:
            anc = self.ancestor_idx
            dsc = self.descendant_idx
            for i in range(len(anc)):
                handle.write(f"{int(anc[i])} {int(dsc[i])}\n")

    @property
    def n_nodes(self) -> int:
        return int(max(self.ancestor_idx.max(), self.descendant_idx.max())) + 1

    @property
    def max_depth(self) -> int:
        return int(self.descendant_depth.max())

    def idx_to_depth_dict(self) -> dict[int, int]:
        """Build idx -> depth mapping (for model initialization)."""
        result: dict[int, int] = {}
        for i in range(len(self)):
            desc_idx = int(self.descendant_idx[i])
            result[desc_idx] = int(self.descendant_depth[i])
            anc_idx = int(self.ancestor_idx[i])
            if anc_idx not in result:
                result[anc_idx] = int(self.ancestor_depth[i])
        return result
