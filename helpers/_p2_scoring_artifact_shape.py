#!/usr/bin/env python3
"""What per-query detail do the P2 scoring artifacts retain?

Read-only. Determines whether the below-chance diagnosis can be done from the banked
artifacts alone (per-query ranks + node ids) or whether it needs the LRZ checkpoints.

Written 2026-09-29 for the P2 below-chance-MRR diagnosis.
"""
import json
from pathlib import Path

ROOT = Path("/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings")
ART = ROOT / "artifacts" / "p2_scoring" / "p2_real_s0_20260928_193929.json"


def shape(obj, depth=0, maxdepth=2):
    if depth > maxdepth:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, list):
                inner = ""
                if v and not isinstance(v[0], (dict, list)):
                    inner = f" e.g. {v[:4]}"
                elif v and isinstance(v[0], dict):
                    inner = f" keys={list(v[0].keys())[:12]}"
                print(f"{'  ' * depth}{k}: list len={len(v)}{inner}")
                if v and isinstance(v[0], dict) and depth < maxdepth:
                    shape(v[0], depth + 2, maxdepth)
            elif isinstance(v, dict):
                print(f"{'  ' * depth}{k}: dict keys={list(v.keys())[:12]}")
                shape(v, depth + 1, maxdepth)
            else:
                print(f"{'  ' * depth}{k}: {type(v).__name__} = {str(v)[:80]}")


def main():
    d = json.loads(ART.read_text())
    print("=" * 78)
    print(f"SHAPE of {ART.name}  ({ART.stat().st_size / 1e6:.1f} MB)")
    print("=" * 78)
    shape(d)


if __name__ == "__main__":
    main()
