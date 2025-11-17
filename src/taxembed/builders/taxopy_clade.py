"""Build datasets for arbitrary clades using TaxoPy."""

from __future__ import annotations

import json
import pickle
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import taxopy
from tqdm.auto import tqdm

from taxembed.utils.data_validation import coverage_from_indices

TaxId = int
Edge = Tuple[TaxId, TaxId]


@dataclass
class CladeBuildResult:
    """Summary returned once a clade dataset has been materialized."""

    dataset_name: str
    root_taxid: int
    node_count: int
    edge_count: int
    max_depth: int
    pairs_count: int
    output_dir: Path
    files: Dict[str, Path]


def slugify(text: str) -> str:
    """Convert arbitrary taxon names to filesystem-friendly slugs."""

    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "clade"


def _build_children_index(parent_map: Dict[int, int]) -> Dict[int, List[int]]:
    children: Dict[int, List[int]] = defaultdict(list)
    for child, parent in parent_map.items():
        if child == parent:
            continue
        children[parent].append(child)
    return children


def _collect_clade(
    children_map: Dict[int, List[int]],
    root_taxid: int,
    max_depth: Optional[int] = None,
) -> Tuple[Dict[int, int], List[Edge]]:
    depth_by_taxid: Dict[int, int] = {}
    edges: List[Edge] = []

    queue: deque[Tuple[int, int]] = deque([(root_taxid, 0)])
    visited: set[int] = set()

    while queue:
        taxid, depth = queue.popleft()
        if taxid in visited:
            continue
        visited.add(taxid)
        depth_by_taxid[taxid] = depth

        if max_depth is not None and depth >= max_depth:
            continue

        for child in children_map.get(taxid, []):
            edges.append((taxid, child))
            queue.append((child, depth + 1))

    return depth_by_taxid, edges


def _build_mapping(depths: Dict[int, int]) -> Tuple[pd.DataFrame, Dict[int, int], Dict[int, int]]:
    sorted_taxids = sorted(depths.keys())
    mapping_df = pd.DataFrame(
        {
            "taxid": sorted_taxids,
            "idx": list(range(len(sorted_taxids))),
        }
    )
    taxid_to_idx = dict(zip(mapping_df["taxid"], mapping_df["idx"]))
    idx_to_taxid = {idx: taxid for taxid, idx in taxid_to_idx.items()}
    return mapping_df, taxid_to_idx, idx_to_taxid


def _build_transitive_pairs(
    depths: Dict[int, int],
    parent_map: Dict[int, int],
    taxid_to_idx: Dict[int, int],
) -> List[Dict[str, int]]:
    pairs: List[Dict[str, int]] = []

    iterator = depths.items()
    for taxid, depth in tqdm(iterator, desc="Building ancestor-descendant pairs", unit="node"):
        descendant_idx = taxid_to_idx[taxid]
        ancestor = parent_map.get(taxid)

        while ancestor is not None and ancestor in depths:
            ancestor_depth = depths[ancestor]
            pairs.append(
                {
                    "ancestor_idx": taxid_to_idx[ancestor],
                    "descendant_idx": descendant_idx,
                    "depth_diff": depth - ancestor_depth,
                    "ancestor_depth": ancestor_depth,
                    "descendant_depth": depth,
                    "ancestor_taxid": ancestor,
                    "descendant_taxid": taxid,
                }
            )
            ancestor = parent_map.get(ancestor)

    return pairs


def _ensure_coverage(
    pairs: List[Dict[str, int]],
    mapping_df: pd.DataFrame,
    depths: Dict[int, int],
    parent_map: Dict[int, int],
) -> None:
    idx_to_taxid = dict(zip(mapping_df["idx"], mapping_df["taxid"]))
    taxid_to_idx = dict(zip(mapping_df["taxid"], mapping_df["idx"]))

    covered = {entry["ancestor_idx"] for entry in pairs}
    covered.update(entry["descendant_idx"] for entry in pairs)

    all_indices = set(mapping_df["idx"])
    missing = sorted(all_indices - covered)

    for idx in missing:
        taxid = idx_to_taxid[idx]
        parent_taxid = parent_map.get(taxid)

        if parent_taxid is None or parent_taxid not in taxid_to_idx:
            parent_idx = idx
            parent_taxid = taxid
            parent_depth = depths.get(taxid, 0)
        else:
            parent_idx = taxid_to_idx[parent_taxid]
            parent_depth = depths[parent_taxid]

        node_depth = depths.get(taxid, parent_depth)
        pairs.append(
            {
                "ancestor_idx": parent_idx,
                "descendant_idx": idx,
                "depth_diff": max(node_depth - parent_depth, 0),
                "ancestor_depth": parent_depth,
                "descendant_depth": node_depth,
                "ancestor_taxid": parent_taxid,
                "descendant_taxid": taxid,
            }
        )


def _write_edges(edges: Sequence[Edge], path: Path) -> None:
    with path.open("w") as handle:
        for parent, child in edges:
            handle.write(f"{parent} {child}\n")


def _write_mapped_edges(edges: Sequence[Edge], mapping: Dict[int, int], path: Path) -> None:
    with path.open("w") as handle:
        for parent, child in edges:
            if parent in mapping and child in mapping:
                handle.write(f"{mapping[parent]} {mapping[child]}\n")


def _write_mapping(mapping_df: pd.DataFrame, path: Path) -> None:
    mapping_df.to_csv(path, sep="\t", index=False)


def _write_transitive(training_pairs: List[Dict[str, int]], prefix: Path) -> Dict[str, Path]:
    tsv_path = prefix.with_name(f"{prefix.name}.tsv")
    edgelist_path = prefix.with_name(f"{prefix.name}.edgelist")
    pkl_path = prefix.with_name(f"{prefix.name}.pkl")

    df = pd.DataFrame(training_pairs)
    df.to_csv(tsv_path, sep="\t", index=False)

    with edgelist_path.open("w") as handle:
        for item in training_pairs:
            handle.write(f"{item['ancestor_idx']} {item['descendant_idx']}\n")

    with pkl_path.open("wb") as handle:
        pickle.dump(training_pairs, handle)

    return {
        "transitive_tsv": tsv_path,
        "transitive_edgelist": edgelist_path,
        "transitive_pickle": pkl_path,
    }


def _write_manifest(
    manifest_path: Path,
    *,
    dataset_name: str,
    root_taxid: int,
    root_name: str,
    node_count: int,
    edge_count: int,
    max_depth: int,
    pairs_count: int,
    max_depth_requested: Optional[int],
) -> None:
    manifest = {
        "dataset_name": dataset_name,
        "root_taxid": root_taxid,
        "root_name": root_name,
        "nodes": node_count,
        "edges": edge_count,
        "max_depth_observed": max_depth,
        "max_depth_requested": max_depth_requested,
        "transitive_pairs": pairs_count,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))


def build_clade_dataset(
    root_taxid: int,
    *,
    dataset_name: Optional[str] = None,
    output_dir: Path | str = Path("data") / "taxopy",
    taxdump_dir: Path | str = Path("data"),
    max_depth: Optional[int] = None,
) -> CladeBuildResult:
    """Materialize a dataset for ``root_taxid`` and its descendants."""

    output_dir = Path(output_dir)
    taxdump_dir = Path(taxdump_dir)

    taxdb = taxopy.TaxDb(taxdb_dir=str(taxdump_dir))
    parent_map = {int(child): int(parent) for child, parent in taxdb.taxid2parent.items()}
    if root_taxid not in parent_map:
        raise ValueError(f"TaxID {root_taxid} not found in taxonomy data at {taxdump_dir}")
    children_map = _build_children_index(parent_map)

    depths, edges = _collect_clade(children_map, root_taxid, max_depth=max_depth)
    if root_taxid not in depths:
        depths[root_taxid] = 0

    mapping_df, taxid_to_idx, idx_to_taxid = _build_mapping(depths)

    if not dataset_name:
        root_name = taxdb.taxid2name.get(str(root_taxid)) or taxdb.taxid2name.get(root_taxid, str(root_taxid))
        dataset_name = slugify(root_name)
    else:
        root_name = taxdb.taxid2name.get(str(root_taxid)) or taxdb.taxid2name.get(root_taxid, str(root_taxid))

    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    prefix = dataset_dir / f"taxonomy_edges_{dataset_name}"

    raw_edges_path = prefix.with_name(f"{prefix.name}.edgelist")
    mapped_edges_path = prefix.with_name(f"{prefix.name}.mapped.edgelist")
    mapping_path = prefix.with_name(f"{prefix.name}.mapping.tsv")

    _write_edges(edges, raw_edges_path)
    _write_mapped_edges(edges, taxid_to_idx, mapped_edges_path)
    _write_mapping(mapping_df, mapping_path)

    training_pairs = _build_transitive_pairs(depths, parent_map, taxid_to_idx)
    _ensure_coverage(training_pairs, mapping_df, depths, parent_map)

    used_indices = {
        entry["ancestor_idx"] for entry in training_pairs
    } | {entry["descendant_idx"] for entry in training_pairs}
    coverage = coverage_from_indices(mapping_df, used_indices)
    if not coverage.is_perfect:
        missing_taxids = [idx_to_taxid[idx] for idx in sorted(coverage.missing_indices)]
        raise RuntimeError(
            "Coverage validation failed even after remediation. "
            f"Missing indices: {missing_taxids[:10]}"
        )

    transitive_paths = _write_transitive(training_pairs, prefix.with_name(f"{prefix.name}_transitive"))

    manifest_path = prefix.with_name(f"{prefix.name}_manifest.json")
    _write_manifest(
        manifest_path,
        dataset_name=dataset_name,
        root_taxid=root_taxid,
        root_name=str(root_name),
        node_count=len(mapping_df),
        edge_count=len(edges),
        max_depth=max(depths.values()) if depths else 0,
        pairs_count=len(training_pairs),
        max_depth_requested=max_depth,
    )

    files = {
        "raw_edgelist": raw_edges_path,
        "mapped_edgelist": mapped_edges_path,
        "mapping": mapping_path,
        "manifest": manifest_path,
        **transitive_paths,
    }

    return CladeBuildResult(
        dataset_name=dataset_name,
        root_taxid=root_taxid,
        node_count=len(mapping_df),
        edge_count=len(edges),
        max_depth=max(depths.values()) if depths else 0,
        pairs_count=len(training_pairs),
        output_dir=dataset_dir,
        files=files,
    )

