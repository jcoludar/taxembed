"""Run the closed-form negative-sampler audit over a closure npz; write JSON to results/.

Usage (single absolute-path invocation, per CLAUDE.md shell hygiene):
  <venv-python> scripts/audit_negative_sampling.py --npz <path> --out results/negative_sampler_audit.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from taxembed.eval.sampler_audit import false_negative_audit  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Closed-form negative-sampler false-negative audit")
    ap.add_argument("--npz", required=True, help="transitive closure .npz")
    ap.add_argument("--out", required=True, help="output JSON path")
    args = ap.parse_args()

    d = np.load(args.npz)
    result = false_negative_audit(
        d["ancestor_idx"], d["descendant_idx"], d["ancestor_depth"], d["descendant_depth"]
    )
    result["source_npz"] = str(Path(args.npz).resolve())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print(f"pairs                 : {result['n_pairs']:,}")
    print(f"overall FN rate       : {result['overall_rate']:.6%}")
    print(f"zero-valid-neg pairs  : {result['n_zero_pool']:,} ({result['frac_zero_pool']:.4%})")
    print(f"root-anchored pairs   : {result['n_root_anchored']:,} ({result['frac_root_anchored']:.4%})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
