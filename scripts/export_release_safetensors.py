#!/usr/bin/env python3
"""Export the released TaxEmbed cellular embedding to a portable safetensors bundle.

Reads the trained checkpoint (`lt.weight`, the 1,102,163 x 100 Poincare-ball
coordinate table), validates it against the taxid<->index mapping, and writes a
clean, universally loadable release bundle:

    <out_dir>/cellular_embedding.safetensors   (tensor key: "embedding", float32)
    <out_dir>/taxid_to_index.tsv               (copy of the mapping)
    <out_dir>/LICENSE                          (copy of the Apache-2.0 license)

The .pth is read-only here (never modified). Run from the repo root:

    python scripts/export_release_safetensors.py

All paths default relative to the repository; override with --ckpt/--mapping/--out-dir.
"""
import argparse
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parent.parent
DEF_CKPT = REPO / "artifacts/tags/cellular_canonical/cellular_canonical.pth"
DEF_MAPPING = REPO / "data/taxonomy_edges_cellular_organisms_131567_clean.mapping.tsv"
DEF_OUT = REPO / "release/taxembed-cellular-v1"
DEF_LICENSE = REPO / "LICENSE"

TAXDUMP_DATE = "2026-06-09"
MODEL_NAME = "TaxEmbed-cellular-v1"


def find_lt_weight(obj):
    """Locate the embedding table `lt.weight` inside a torch checkpoint."""
    if torch.is_tensor(obj):
        return "<root-tensor>", obj
    if isinstance(obj, dict):
        if "lt.weight" in obj and torch.is_tensor(obj["lt.weight"]):
            return "lt.weight", obj["lt.weight"]
        for container in ("model", "state_dict", "model_state_dict", "net", "module"):
            sub = obj.get(container)
            if isinstance(sub, dict) and torch.is_tensor(sub.get("lt.weight")):
                return f"{container}.lt.weight", sub["lt.weight"]
        # any key ending in lt.weight, one level deep
        for k, v in obj.items():
            if torch.is_tensor(v) and k.endswith("lt.weight"):
                return k, v
        for k, v in obj.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if torch.is_tensor(vv) and kk.endswith("lt.weight"):
                        return f"{k}.{kk}", vv
    raise SystemExit(
        "Could not find 'lt.weight'. Top-level type/keys: "
        + (str(list(obj.keys())) if isinstance(obj, dict) else str(type(obj)))
    )


def count_mapping_rows(mapping: Path) -> int:
    with open(mapping) as fh:
        header = fh.readline()  # 'taxid\tidx'
        assert header.strip().split("\t")[:2] == ["taxid", "idx"], f"unexpected header: {header!r}"
        return sum(1 for _ in fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=DEF_CKPT)
    ap.add_argument("--mapping", type=Path, default=DEF_MAPPING)
    ap.add_argument("--out-dir", type=Path, default=DEF_OUT)
    ap.add_argument("--license", type=Path, default=DEF_LICENSE)
    args = ap.parse_args()

    for p in (args.ckpt, args.mapping):
        if not p.exists():
            raise SystemExit(f"missing input: {p}")

    print(f"Loading checkpoint: {args.ckpt}  ({args.ckpt.stat().st_size/1e6:.0f} MB)")
    obj = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    key, W = find_lt_weight(obj)
    print(f"  found embedding at '{key}': shape={tuple(W.shape)} dtype={W.dtype}")

    orig_dtype = str(W.dtype).replace("torch.", "")
    W = W.detach().to(torch.float32).contiguous()

    # --- sanity checks ---
    n_rows, dim = W.shape
    n_map = count_mapping_rows(args.mapping)
    print(f"  mapping rows: {n_map:,}   embedding rows: {n_rows:,}   dim: {dim}")
    assert n_rows == n_map, f"row-count mismatch: embedding {n_rows} vs mapping {n_map}"
    norms = W.norm(dim=1)
    max_norm = float(norms.max())
    n_outside = int((norms >= 1.0).sum())
    print(f"  Poincare-ball norms: max={max_norm:.6f}  mean={float(norms.mean()):.4f}"
          f"  #(norm>=1)={n_outside}")
    if n_outside:
        print("  WARNING: some points lie on/outside the unit ball (norm >= 1).")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_st = args.out_dir / "cellular_embedding.safetensors"
    metadata = {
        "model": MODEL_NAME,
        "description": "Hyperbolic (Poincare-ball) embedding of the NCBI tree of cellular life",
        "geometry": "poincare_ball",
        "dim": str(dim),
        "n_taxa": str(n_rows),
        "row_order": "index (idx) column of taxid_to_index.tsv; row i is the taxon with idx==i",
        "radius_meaning": "embedding norm ||x|| tracks taxonomic depth (root center, tips near rim)",
        "angle_meaning": "direction encodes learned lineage",
        "taxdump_release": TAXDUMP_DATE,
        "index_mapping_file": "taxid_to_index.tsv",
        "tensor_key": "embedding",
        "orig_dtype": orig_dtype,
        "license": "Apache-2.0",
    }
    print(f"Writing {out_st} ...")
    save_file({"embedding": W}, str(out_st), metadata=metadata)
    print(f"  wrote {out_st.stat().st_size/1e6:.0f} MB")

    out_map = args.out_dir / "taxid_to_index.tsv"
    shutil.copy2(args.mapping, out_map)
    print(f"  copied mapping -> {out_map} ({out_map.stat().st_size/1e6:.1f} MB)")

    if args.license.exists():
        shutil.copy2(args.license, args.out_dir / "LICENSE")
        print(f"  copied LICENSE -> {args.out_dir / 'LICENSE'}")

    print("\nDONE. Bundle contents:")
    for f in sorted(args.out_dir.iterdir()):
        print(f"  {f.name:32s} {f.stat().st_size/1e6:>9.1f} MB")


if __name__ == "__main__":
    main()
