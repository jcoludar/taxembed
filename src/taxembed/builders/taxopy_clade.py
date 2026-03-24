"""Build datasets for arbitrary clades using TaxoPy."""

from __future__ import annotations

import json
import pickle
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import taxopy
from tqdm.auto import tqdm

from taxembed.utils.data_validation import coverage_from_indices

TaxId = int
Edge = Tuple[TaxId, TaxId]

# Default noise patterns for taxonomy cleanup.
# These match noisy LEAF nodes (unnamed species, uncertain IDs, etc.).
DEFAULT_NOISE_PATTERNS: list[str] = [
    r"\bsp\.",                      # Genus sp. (unnamed species, barcoding vouchers)
    r"\b(?:nr|cf|aff)\.\s",        # Near/confer/affinity (uncertain IDs)
    r"\benvironmental\b",           # Environmental samples
    r"\buncultured\b",              # Uncultured organisms
    r"\bunidentified\b",            # Unidentified
    r"\b[Hh]ybrid\b",              # Hybrid taxa
]

# Container nodes that group junk — only removed bottom-up when all children are gone.
CONTAINER_PATTERNS: list[tuple[str, str]] = [
    ("unclassified", r"^unclassified\s"),
    ("incertae sedis", r"\bincertae\s+sedis\b"),
    ("environmental samples", r"^environmental\s+samples$"),
]


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


def _compile_noise_filter(
    patterns: list[str],
) -> Callable[[str], bool]:
    """Compile regex patterns into a single filter function.

    Returns a callable that returns True if the name is noisy.
    """
    compiled = [re.compile(p, re.IGNORECASE) for p in patterns]

    def _is_noisy(name: str) -> bool:
        return any(rx.search(name) for rx in compiled)

    return _is_noisy


def _is_container_node(name: str) -> bool:
    """Return True if name matches a container pattern (unclassified X, incertae sedis)."""
    return any(re.search(pat, name, re.IGNORECASE) for _, pat in CONTAINER_PATTERNS)


def _filter_clade_noise(
    depths: Dict[int, int],
    edges: list[Edge],
    taxid2name: dict,
    noise_filter: Callable[[str], bool],
    root_taxid: int,
) -> Tuple[Dict[int, int], list[Edge], int]:
    """Remove noisy leaf nodes via bottom-up pruning, then collapse empty containers.

    Strategy:
    1. Mark leaves whose scientific name matches noise patterns -> remove
    2. Mark leaves with no name in taxid2name (stale taxids) -> remove
    3. After removing noisy leaves, if a container node has ALL children removed -> remove
    4. Repeat until stable

    Returns (filtered_depths, filtered_edges, n_removed).
    """
    all_taxids = set(depths.keys())

    # Build mutable children-of from edge list
    children_of: Dict[int, set[int]] = defaultdict(set)
    parent_of: Dict[int, int] = {}
    for parent, child in edges:
        if parent in all_taxids and child in all_taxids:
            children_of[parent].add(child)
            parent_of[child] = parent

    removed: set[int] = set()

    # Resolve names for all taxids (handle both str and int keys)
    def _get_name(tid: int) -> str | None:
        return taxid2name.get(str(tid)) or taxid2name.get(tid)

    # Iterative bottom-up pruning
    changed = True
    while changed:
        changed = False
        for tid in list(all_taxids - removed):
            if tid == root_taxid:
                continue
            # Skip if has live children (not a leaf)
            alive_kids = children_of.get(tid, set()) - removed
            if alive_kids:
                continue

            name = _get_name(tid)

            # Stale taxid: not in NCBI dump
            if name is None:
                removed.add(tid)
                changed = True
                continue

            # Name matches noise pattern
            if noise_filter(name):
                removed.add(tid)
                changed = True
                continue

        # Container collapse pass
        for tid in list(all_taxids - removed):
            if tid == root_taxid:
                continue
            name = _get_name(tid)
            if name is None:
                continue
            if not _is_container_node(name):
                continue
            alive_kids = children_of.get(tid, set()) - removed
            if not alive_kids:
                removed.add(tid)
                changed = True

    # Rebuild depths and edges from kept nodes
    kept = all_taxids - removed
    filtered_depths = {tid: d for tid, d in depths.items() if tid in kept}

    # Re-parent through removed internal nodes
    new_parent: Dict[int, int] = {}
    for tid in kept:
        if tid == root_taxid:
            continue
        p = parent_of.get(tid)
        while p is not None and p not in kept:
            p = parent_of.get(p)
        if p is not None and p != tid:
            new_parent[tid] = p

    # Rebuild edges and recompute depths via BFS from root
    new_children: Dict[int, list[int]] = defaultdict(list)
    for child, parent in new_parent.items():
        new_children[parent].append(child)

    bfs_depths: Dict[int, int] = {root_taxid: 0}
    filtered_edges: list[Edge] = []
    queue: deque[int] = deque([root_taxid])
    while queue:
        node = queue.popleft()
        for child in new_children.get(node, []):
            if child not in bfs_depths:
                bfs_depths[child] = bfs_depths[node] + 1
                filtered_edges.append((node, child))
                queue.append(child)

    return bfs_depths, filtered_edges, len(removed)


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
    max_pairs: int | None = None,
) -> List[Dict[str, int]]:
    """Build all ancestor-descendant pairs via parent-chain traversal.

    For very large clades (>500K nodes), uses a columnar pre-allocation
    strategy to avoid list-of-dicts memory overhead.
    """
    n_nodes = len(depths)

    # For large clades, estimate pairs and pre-allocate numpy arrays
    if n_nodes > 500_000:
        return _build_transitive_pairs_columnar(depths, parent_map, taxid_to_idx, max_pairs)

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


def _build_transitive_pairs_columnar(
    depths: Dict[int, int],
    parent_map: Dict[int, int],
    taxid_to_idx: Dict[int, int],
    max_pairs: int | None = None,
) -> List[Dict[str, int]]:
    """Memory-efficient columnar pair builder for large clades (>500K nodes).

    Pre-computes ancestor chains and writes directly to numpy arrays,
    then converts to list-of-dicts at the end. For 2.4M nodes, this avoids
    the ~16GB dict overhead of the naive approach.
    """
    n_nodes = len(depths)
    # Estimate total pairs: avg_depth * n_nodes (conservative upper bound)
    avg_depth = sum(depths.values()) / max(n_nodes, 1)
    est_pairs = int(avg_depth * n_nodes * 1.1)
    if max_pairs is not None:
        est_pairs = min(est_pairs, max_pairs * 2)  # allocate extra, trim later
    print(f"  Columnar builder: {n_nodes:,} nodes, est. {est_pairs:,} pairs")

    # Pre-allocate columnar arrays
    ancestor_idx_arr = np.empty(est_pairs, dtype=np.int32)
    descendant_idx_arr = np.empty(est_pairs, dtype=np.int32)
    depth_diff_arr = np.empty(est_pairs, dtype=np.int16)
    ancestor_depth_arr = np.empty(est_pairs, dtype=np.int16)
    descendant_depth_arr = np.empty(est_pairs, dtype=np.int16)
    ancestor_taxid_arr = np.empty(est_pairs, dtype=np.int32)
    descendant_taxid_arr = np.empty(est_pairs, dtype=np.int32)

    idx = 0
    for taxid, depth in tqdm(depths.items(), desc="Building pairs (columnar)", unit="node", total=n_nodes):
        desc_idx = taxid_to_idx[taxid]
        ancestor = parent_map.get(taxid)

        while ancestor is not None and ancestor in depths:
            if idx >= len(ancestor_idx_arr):
                # Grow arrays (double capacity)
                new_size = len(ancestor_idx_arr) * 2
                ancestor_idx_arr = np.resize(ancestor_idx_arr, new_size)
                descendant_idx_arr = np.resize(descendant_idx_arr, new_size)
                depth_diff_arr = np.resize(depth_diff_arr, new_size)
                ancestor_depth_arr = np.resize(ancestor_depth_arr, new_size)
                descendant_depth_arr = np.resize(descendant_depth_arr, new_size)
                ancestor_taxid_arr = np.resize(ancestor_taxid_arr, new_size)
                descendant_taxid_arr = np.resize(descendant_taxid_arr, new_size)

            anc_depth = depths[ancestor]
            ancestor_idx_arr[idx] = taxid_to_idx[ancestor]
            descendant_idx_arr[idx] = desc_idx
            depth_diff_arr[idx] = depth - anc_depth
            ancestor_depth_arr[idx] = anc_depth
            descendant_depth_arr[idx] = depth
            ancestor_taxid_arr[idx] = ancestor
            descendant_taxid_arr[idx] = taxid
            idx += 1
            ancestor = parent_map.get(ancestor)

    # Trim to actual size
    total = idx
    print(f"  ✓ Built {total:,} pairs (columnar)")

    # Convert to list-of-dicts for backward compatibility with downstream code
    pairs = []
    for i in range(total):
        pairs.append({
            "ancestor_idx": int(ancestor_idx_arr[i]),
            "descendant_idx": int(descendant_idx_arr[i]),
            "depth_diff": int(depth_diff_arr[i]),
            "ancestor_depth": int(ancestor_depth_arr[i]),
            "descendant_depth": int(descendant_depth_arr[i]),
            "ancestor_taxid": int(ancestor_taxid_arr[i]),
            "descendant_taxid": int(descendant_taxid_arr[i]),
        })

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


def _write_transitive_npz(training_pairs: List[Dict[str, int]], prefix: Path) -> Dict[str, Path]:
    """Write training pairs in memory-efficient .npz format (columnar arrays)."""
    npz_path = prefix.with_name(f"{prefix.name}.npz")
    n = len(training_pairs)

    ancestor_idx = np.empty(n, dtype=np.int32)
    descendant_idx = np.empty(n, dtype=np.int32)
    depth_diff = np.empty(n, dtype=np.int16)
    ancestor_depth = np.empty(n, dtype=np.int16)
    descendant_depth = np.empty(n, dtype=np.int16)
    ancestor_taxid = np.empty(n, dtype=np.int32)
    descendant_taxid = np.empty(n, dtype=np.int32)

    for i, p in enumerate(training_pairs):
        ancestor_idx[i] = p["ancestor_idx"]
        descendant_idx[i] = p["descendant_idx"]
        depth_diff[i] = p["depth_diff"]
        ancestor_depth[i] = p["ancestor_depth"]
        descendant_depth[i] = p["descendant_depth"]
        ancestor_taxid[i] = p["ancestor_taxid"]
        descendant_taxid[i] = p["descendant_taxid"]

    np.savez_compressed(
        npz_path,
        ancestor_idx=ancestor_idx,
        descendant_idx=descendant_idx,
        depth_diff=depth_diff,
        ancestor_depth=ancestor_depth,
        descendant_depth=descendant_depth,
        ancestor_taxid=ancestor_taxid,
        descendant_taxid=descendant_taxid,
    )
    return {"transitive_npz": npz_path}


def _stratified_subsample(
    training_pairs: List[Dict[str, int]], max_pairs: int
) -> List[Dict[str, int]]:
    """Subsample pairs while preserving the depth_diff distribution."""
    if len(training_pairs) <= max_pairs:
        return training_pairs

    # Group by depth_diff
    buckets: Dict[int, List[int]] = defaultdict(list)
    for i, p in enumerate(training_pairs):
        buckets[p["depth_diff"]].append(i)

    total = len(training_pairs)
    selected: List[int] = []

    for dd, indices in buckets.items():
        # Proportional allocation: this bucket gets (bucket_size / total) * max_pairs
        n_select = max(1, int(len(indices) / total * max_pairs))
        n_select = min(n_select, len(indices))
        chosen = np.random.choice(indices, n_select, replace=False).tolist()
        selected.extend(chosen)

    # If rounding caused us to get too many, trim; too few, add random extras
    if len(selected) > max_pairs:
        selected = list(np.random.choice(selected, max_pairs, replace=False))
    elif len(selected) < max_pairs:
        remaining = set(range(total)) - set(selected)
        extra_needed = max_pairs - len(selected)
        if remaining:
            extra = list(np.random.choice(list(remaining), min(extra_needed, len(remaining)), replace=False))
            selected.extend(extra)

    return [training_pairs[i] for i in sorted(selected)]


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
    clean: bool = False,
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
        "clean": clean,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))


def build_clade_dataset(
    root_taxid: int,
    *,
    dataset_name: Optional[str] = None,
    output_dir: Path | str = Path("data") / "taxopy",
    taxdump_dir: Path | str = Path("data"),
    max_depth: Optional[int] = None,
    max_pairs: Optional[int] = None,
    clean: bool = False,
    exclude_patterns: list[str] | None = None,
) -> CladeBuildResult:
    """Materialize a dataset for ``root_taxid`` and its descendants.

    Parameters
    ----------
    clean : bool
        When True, apply default noise filtering (sp., cf., environmental, etc.)
        via bottom-up leaf pruning. Removes ~30-50% of nodes in typical NCBI clades.
    exclude_patterns : list[str] | None
        Custom regex patterns for additional noise filtering. Merged with defaults
        when ``clean=True``, or used standalone otherwise.
    """

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

    # Noise filtering
    if clean or exclude_patterns:
        patterns = list(DEFAULT_NOISE_PATTERNS) if clean else []
        if exclude_patterns:
            patterns.extend(exclude_patterns)
        noise_filter = _compile_noise_filter(patterns)
        n_before = len(depths)
        depths, edges, n_removed = _filter_clade_noise(
            depths, edges, taxdb.taxid2name, noise_filter, root_taxid
        )
        print(
            f"  Noise filtering: {n_before:,} -> {len(depths):,} nodes "
            f"({n_removed:,} removed, {100 * n_removed / n_before:.1f}%)"
        )

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

    # Stratified subsampling if max_pairs is set
    if max_pairs and len(training_pairs) > max_pairs:
        print(f"  Subsampling {len(training_pairs):,} pairs to {max_pairs:,} (stratified by depth_diff)")
        training_pairs = _stratified_subsample(training_pairs, max_pairs)

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
    npz_paths = _write_transitive_npz(training_pairs, prefix.with_name(f"{prefix.name}_transitive"))
    transitive_paths.update(npz_paths)

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
        clean=clean or bool(exclude_patterns),
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

