"""Stage the closure files beside the released tensor so the Hugging Face bundle is self-sufficient.

2026-09-30. The manuscript's Data availability promises the parent-child edgelist and the training
closure alongside the embedding. Until now the HF repo carried only the tensor, the taxid->row mapping,
the card and the licence, so tree distances and depths could not be computed from the release alone.

What this does (read-only on the sources; copies into release/taxembed-cellular-v1/):
  data/taxopy/cellular_organisms_131567_clean/*.mapped.edgelist   -> edges_parent_child.tsv
  data/taxopy/cellular_organisms_131567_clean/*_transitive.npz    -> training_closure.npz
  data/taxopy/cellular_organisms_131567_clean/*_manifest.json     -> closure_manifest.json
and prints: npz keys + shapes, edge count, md5 of the bundle's taxid_to_index.tsv against the closure's
mapping.tsv (they must be identical: release row i == closure node i), and the root row.

Run:  <submodule>/.venv/bin/python helpers/_hf_bundle_prep_20260930.py
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/taxopy/cellular_organisms_131567_clean"
DST = ROOT / "release/taxembed-cellular-v1"
STEM = "taxonomy_edges_cellular_organisms_131567_clean"

COPIES = {
    SRC / f"{STEM}.mapped.edgelist": DST / "edges_parent_child.tsv",
    SRC / f"{STEM}_transitive.npz": DST / "training_closure.npz",
    SRC / f"{STEM}_manifest.json": DST / "closure_manifest.json",
}


def md5(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    for src, dst in COPIES.items():
        assert src.exists(), src
        if dst.exists():
            print(f"exists, left as is: {dst.name} ({dst.stat().st_size:,} B)")
            continue
        shutil.copy2(src, dst)
        print(f"copied {src.name} -> {dst.name} ({dst.stat().st_size:,} B)")

    z = np.load(DST / "training_closure.npz")
    print("\ntraining_closure.npz keys:")
    for k in z.files:
        print(f"  {k:<18} {z[k].dtype!s:<8} {z[k].shape}")

    edges = np.loadtxt(DST / "edges_parent_child.tsv", dtype=np.int64)
    n_nodes = int(edges.max()) + 1
    children = set(edges[:, 1].tolist())
    roots = [i for i in range(n_nodes) if i not in children]
    print(f"\nedges_parent_child.tsv: {len(edges):,} edges, {n_nodes:,} nodes, root row(s) {roots}")
    assert len(edges) == 1_102_162 and n_nodes == 1_102_163, (len(edges), n_nodes)

    m_release = md5(DST / "taxid_to_index.tsv")
    m_closure = md5(SRC / f"{STEM}.mapping.tsv")
    print(f"\nmd5 taxid_to_index.tsv (release) {m_release}")
    print(f"md5 mapping.tsv        (closure) {m_closure}")
    assert m_release == m_closure, "release rows and closure nodes are NOT the same indexing"
    print("OK: release row i == closure node i")


if __name__ == "__main__":
    main()
