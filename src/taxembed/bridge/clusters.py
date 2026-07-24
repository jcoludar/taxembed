"""Identity-cluster sequences with mmseqs2 easy-cluster (spec §5: ortholog rows in the same
identity cluster must move together across CV folds, so no near-duplicate leaks train->test).

Single entry point `mmseqs_cluster(fasta, ...)` -> {seq_id: cluster_rep}. mmseqs writes a pile of
temp files; we point it at a caller-supplied tmpdir and clean up the run outputs by default.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


def have_mmseqs() -> bool:
    """True iff the mmseqs binary is on PATH."""
    return shutil.which("mmseqs") is not None


def parse_cluster_tsv(tsv_path: Path | str) -> dict[str, str]:
    """Parse mmseqs `<prefix>_cluster.tsv` (two columns: representative<TAB>member) into
    {member_id: representative_id}. Representatives map to themselves."""
    mapping: dict[str, str] = {}
    for line in Path(tsv_path).read_text().splitlines():
        if not line.strip():
            continue
        rep, member = line.split("\t")[:2]
        mapping[member] = rep
    return mapping


def mmseqs_cluster(
    fasta: Path | str,
    out_prefix: Path | str | None = None,
    tmp_dir: Path | str | None = None,
    min_seq_id: float = 0.9,
    extra_args: list[str] | None = None,
    cleanup: bool = True,
) -> dict[str, str]:
    """Run `mmseqs easy-cluster <fasta> <prefix> <tmpdir> --min-seq-id <id>` and return
    {seq_id: cluster_rep}. Raises if mmseqs is absent or the run fails — callers decide whether
    to fall back to one-cluster-per-sequence.

    out_prefix / tmp_dir default to a private temp dir; pass explicit paths to inspect outputs.
    """
    if not have_mmseqs():
        raise RuntimeError("mmseqs not found on PATH")
    fasta = Path(fasta)
    if not fasta.exists():
        raise FileNotFoundError(fasta)

    owns_scratch = out_prefix is None or tmp_dir is None
    scratch = Path(tempfile.mkdtemp(prefix="mmseqs_cluster_")) if owns_scratch else None
    prefix = Path(out_prefix) if out_prefix is not None else scratch / "clust"
    tmp = Path(tmp_dir) if tmp_dir is not None else scratch / "tmp"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)

    cmd = [
        "mmseqs", "easy-cluster", str(fasta), str(prefix), str(tmp),
        "--min-seq-id", str(min_seq_id),
    ] + (extra_args or [])
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    cluster_tsv = Path(f"{prefix}_cluster.tsv")
    if not cluster_tsv.exists():
        raise RuntimeError(f"mmseqs produced no {cluster_tsv}")
    mapping = parse_cluster_tsv(cluster_tsv)

    if cleanup and scratch is not None:
        shutil.rmtree(scratch, ignore_errors=True)
    return mapping
