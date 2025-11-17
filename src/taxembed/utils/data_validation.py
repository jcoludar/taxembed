"""Shared helpers for validating mapping coverage and node indices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Set

import pandas as pd


@dataclass(frozen=True)
class CoverageReport:
    """Summary of how many mapped nodes appear in downstream artifacts."""

    total_nodes: int
    used_nodes: int
    missing_indices: Set[int]

    @property
    def missing_count(self) -> int:
        return len(self.missing_indices)

    @property
    def coverage_ratio(self) -> float:
        if self.total_nodes == 0:
            return 1.0
        return self.used_nodes / self.total_nodes

    @property
    def is_perfect(self) -> bool:
        return self.missing_count == 0


def load_mapping(mapping_path: Path) -> pd.DataFrame:
    """Load a mapping TSV that contains ``taxid`` and ``idx`` columns."""

    if not mapping_path.exists():
        raise FileNotFoundError(mapping_path)

    df = pd.read_csv(mapping_path, sep="\t", dtype={"taxid": str, "idx": str})

    # Handle files that already provide headers vs. headerless TSVs.
    if "taxid" not in df.columns or "idx" not in df.columns:
        df.columns = ["taxid", "idx"]

    df["idx"] = pd.to_numeric(df["idx"], errors="raise").astype(int)
    # Preserve TaxIDs as strings to avoid accidental truncation.
    return df


def mapping_indices(df: pd.DataFrame) -> Set[int]:
    """Return the set of sequential indices present in a mapping dataframe."""

    return set(df["idx"].astype(int).tolist())


def coverage_from_indices(df: pd.DataFrame, used_indices: Iterable[int]) -> CoverageReport:
    """Compare mapped indices against a collection of indices that appear in data."""

    mapped = mapping_indices(df)
    used = {int(idx) for idx in used_indices}
    missing = mapped - used
    return CoverageReport(
        total_nodes=len(mapped),
        used_nodes=len(used),
        missing_indices=missing,
    )


__all__ = [
    "CoverageReport",
    "coverage_from_indices",
    "load_mapping",
    "mapping_indices",
]

