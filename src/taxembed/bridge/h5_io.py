"""HDF5 read/write for per-residue and per-protein embeddings."""

from pathlib import Path

import h5py
import numpy as np


def save_embeddings(
    embeddings: dict[str, np.ndarray],
    h5_path: Path | str,
) -> None:
    """Save embeddings to H5 (gzip-compressed float32).

    Works for both per-residue (L, D) and per-protein (D,) shapes.
    """
    h5_path = Path(h5_path)
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(h5_path), "w") as f:
        for sid, emb in embeddings.items():
            f.create_dataset(sid, data=emb.astype(np.float32), compression="gzip", compression_opts=4)
    print(f"Saved {len(embeddings)} embeddings to {h5_path}")


def load_embeddings(h5_path: Path | str) -> dict[str, np.ndarray]:
    """Load embeddings from H5. Returns {protein_id: ndarray}."""
    embeddings = {}
    with h5py.File(str(h5_path), "r") as f:
        for key in f.keys():
            embeddings[key] = np.array(f[key], dtype=np.float32)
    print(f"Loaded {len(embeddings)} embeddings from {h5_path}")
    return embeddings


def save_embedding_single(h5_path: Path | str, protein_id: str, embedding: np.ndarray) -> None:
    """Append one embedding to H5 in append mode (resumable extraction)."""
    h5_path = Path(h5_path)
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(h5_path), "a") as f:
        if protein_id not in f:
            f.create_dataset(
                protein_id,
                data=embedding.astype(np.float32),
                compression="gzip",
                compression_opts=4,
            )


def get_existing_ids(h5_path: Path | str) -> set[str]:
    """Return set of protein IDs already in H5 (for resume support)."""
    h5_path = Path(h5_path)
    if not h5_path.exists():
        return set()
    with h5py.File(str(h5_path), "r") as f:
        return set(f.keys())


def mean_pool(embeddings: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Reduce per-residue (L, D) embeddings to per-protein (D,) via mean pooling."""
    reduced = {}
    for sid, emb in embeddings.items():
        if emb.ndim == 2:
            reduced[sid] = emb.mean(axis=0).astype(np.float32)
        else:
            reduced[sid] = emb.astype(np.float32)
    return reduced
