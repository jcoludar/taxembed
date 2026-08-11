"""Prove --seed actually controls the training run, in both directions.

`train_small.py --help` showing a --seed flag proves only that the flag is ACCEPTED.
A seed the training path ignored would look identical. This runs the real entrypoint
three times on a small clade and checks BOTH directions:

  same seed  -> identical best loss   (the flag does something)
  diff seed  -> different best loss   (it is not a constant / no-op)

Reports the observed losses either way, so a failure says what actually happened
rather than just "assertion failed".

Usage (single absolute-path invocation, per CLAUDE.md shell hygiene):
  <venv-python> scripts/verify_seed_reproducibility.py --npz <closure.npz> --mapping <map.tsv>
      [--out results/seed_reproducibility.json]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BEST_LOSS = re.compile(r"^Best loss:\s+([0-9.]+)", re.MULTILINE)


def run_once(npz: str, mapping: str, seed: int, workdir: Path, tag: str,
             epochs: int, dim: int) -> float:
    """Run the real entrypoint once; return its reported best loss."""
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "train_small.py"),
        "--data", npz,
        "--mapping", mapping,
        "--checkpoint", str(workdir / f"{tag}.pth"),
        "--dim", str(dim),
        "--epochs", str(epochs),
        "--batch-size", "256",
        "--n-negatives", "50",
        "--lr", "0.001",
        "--loss", "softmax",
        "--euclidean-param",
        "--seed", str(seed),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    match = _BEST_LOSS.search(proc.stdout)
    if match is None:
        raise RuntimeError(f"could not parse a best loss from run {tag}")
    if f"seeded: {seed}" not in proc.stdout:
        raise RuntimeError(f"run {tag} never reported being seeded -- flag not wired")
    return float(match.group(1))


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify --seed controls training, both directions")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--dim", type=int, default=50)
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        a = run_once(args.npz, args.mapping, 0, workdir, "seed0_a", args.epochs, args.dim)
        b = run_once(args.npz, args.mapping, 0, workdir, "seed0_b", args.epochs, args.dim)
        c = run_once(args.npz, args.mapping, 1, workdir, "seed1", args.epochs, args.dim)

    same_seed_reproducible = a == b
    different_seed_differs = a != c

    result = {
        "source_npz": str(Path(args.npz).resolve()),
        "epochs": args.epochs,
        "dim": args.dim,
        "best_loss": {"seed0_run_a": a, "seed0_run_b": b, "seed1": c},
        "same_seed_reproducible": same_seed_reproducible,
        "different_seed_differs": different_seed_differs,
        "passed": same_seed_reproducible and different_seed_differs,
        "note": (
            "CPU run. On GPU, --amp plus CUDA scatter nondeterminism gives "
            "near-reproducibility, not bitwise equality."
        ),
    }

    print(f"seed 0, run a : {a:.6f}")
    print(f"seed 0, run b : {b:.6f}")
    print(f"seed 1        : {c:.6f}")
    print(f"same seed reproducible : {same_seed_reproducible}")
    print(f"different seed differs : {different_seed_differs}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"wrote {out}")

    if not result["passed"]:
        raise SystemExit("SEED VERIFICATION FAILED -- see the losses above")


if __name__ == "__main__":
    main()
