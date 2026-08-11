"""Task 9: run the canonical-vs-prior recipe contrast at clade scale, seeded.

Six runs -- two arms x three seeds. The arms are taken verbatim from the two recorded
run.json files (artifacts/tags/metazoa_softmax and .../metazoa_lower_lr_bigger_batch),
NOT from the manuscript's one-line description of them, because they differ in four
substantive factors rather than one. See results/recipe_angular_comparison.json.

Captures both metric families from the trainer's own per-epoch table:
  planted : DepthCorr  (quote against the Task 4 initialization floor, never alone)
  learned : kNN%, Sep  (angular structure -- the open question)

Full stdout per run is kept so the trajectory can be re-parsed later, and checkpoints
are kept so the proper analysis pipeline can score retrieval@10 and purity properly.

Usage (single absolute-path invocation, per CLAUDE.md shell hygiene):
  <venv-python> scripts/run_recipe_contrast.py --npz <closure.npz> --mapping <map.tsv>
      --outdir results/task9_runs [--epochs 200] [--seeds 0 1 2]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_BEST_LOSS = re.compile(r"^Best loss:\s+([0-9.]+)", re.MULTILINE)

# Flags shared by both arms, verbatim from the recorded configs.
_SHARED = [
    "--dim", "100",
    "--margin", "0.2",
    "--lambda-reg", "0.1",
    "--early-stopping", "999",
    "--optimizer", "adam",
    "--burnin", "0",
    "--burnin-multiplier", "0.1",
    "--radial-nudge", "0.05",
    "--curriculum",
    "--curriculum-phases", "auto",
    "--epoch-fraction", "0.3",
    "--depth-scale-margin",
    "--margin-min", "0.05",
    "--margin-max", "1.0",
    "--radial-schedule", "log",
    "--euclidean-param",
    "--loss", "softmax",
]

# The four factors that actually separate the arms.
_ARMS = {
    "canonical": [
        "--batch-size", "256",
        "--n-negatives", "300",
        "--lr", "0.001",
        "--grad-accum-steps", "8",
        "--lr-schedule", "cosine_warmrestart",
        "--warm-restart-on-phase",
        "--lr-min-multiplier", "0.01",
    ],
    "prior": [
        "--batch-size", "128",
        "--n-negatives", "100",
        "--lr", "0.005",
        "--grad-accum-steps", "4",
        # no --lr-schedule: const, as recorded
    ],
}


def parse_final_epoch(stdout: str) -> dict | None:
    """Pull the last per-epoch row from the trainer's table.

    Columns: Epoch | Loss | dLoss | Improve | Reg | MaxNorm | Outside | DepthCorr |
             Hier% | kNN% | Sep | Status
    """
    rows = []
    for line in _ANSI.sub("", stdout).splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 12:
            continue
        try:
            epoch = int(parts[0])
        except ValueError:
            continue
        try:
            rows.append({
                "epoch": epoch,
                "loss": float(parts[1]),
                "depth_corr": float(parts[7]),
                "hier_pct": float(parts[8].rstrip("%")),
                "knn_pct": float(parts[9].rstrip("%")),
                "separation": float(parts[10]),
            })
        except ValueError:
            continue
    return rows[-1] if rows else None


def run_one(npz: str, mapping: str, arm: str, seed: int, outdir: Path,
            epochs: int) -> dict:
    tag = f"echino_{arm}_s{seed}"
    cmd = [
        sys.executable, str(_REPO_ROOT / "train_small.py"),
        "--data", npz,
        "--mapping", mapping,
        "--checkpoint", str(outdir / f"{tag}.pth"),
        "--epochs", str(epochs),
        "--seed", str(seed),
        *_SHARED, *_ARMS[arm],
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    (outdir / f"{tag}.log").write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr)

    if proc.returncode != 0:
        return {"tag": tag, "arm": arm, "seed": seed, "failed": True,
                "returncode": proc.returncode}

    best = _BEST_LOSS.search(proc.stdout)
    return {
        "tag": tag,
        "arm": arm,
        "seed": seed,
        "failed": False,
        "flags": _ARMS[arm],
        "best_loss": float(best.group(1)) if best else None,
        "final_epoch": parse_final_epoch(proc.stdout),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Task 9 recipe contrast")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    runs = []
    for arm in ("canonical", "prior"):
        for seed in args.seeds:
            print(f"[running] {arm} seed {seed} ...", flush=True)
            result = run_one(args.npz, args.mapping, arm, seed, outdir, args.epochs)
            runs.append(result)
            if result["failed"]:
                print(f"  FAILED rc={result['returncode']} -- see {result['tag']}.log",
                      flush=True)
            else:
                fe = result["final_epoch"] or {}
                print(f"  depth_corr={fe.get('depth_corr')} "
                      f"knn={fe.get('knn_pct')} sep={fe.get('separation')}", flush=True)

    (outdir / "runs.json").write_text(json.dumps({"runs": runs}, indent=2))
    print(f"\nwrote {outdir / 'runs.json'}")

    for arm in ("canonical", "prior"):
        ok = [r for r in runs if r["arm"] == arm and not r["failed"] and r["final_epoch"]]
        if not ok:
            continue
        print(f"\n{arm}:")
        for r in ok:
            fe = r["final_epoch"]
            print(f"  seed {r['seed']}: depth_corr {fe['depth_corr']:+.3f} | "
                  f"kNN {fe['knn_pct']:.1f}% | sep {fe['separation']:.2f}")


if __name__ == "__main__":
    main()
