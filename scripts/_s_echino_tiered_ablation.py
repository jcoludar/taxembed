#!/usr/bin/env python
"""Controlled echino A/B: winning softmax recipe WITH vs WITHOUT --tiered-negatives.

Motivation (2026-06-03 analysis): the metazoa-scale collapse is hypothesised to be a
negative-sampling problem — at echino scale the default same-depth negatives are
already within-clade (hard), but at 498k scale "same depth" spans foreign clades, so
the within-clade angular gradient starves. Hard/within-subtree negatives were NEVER
tried at metazoa scale. Before spending V100 time, validate echino-first (RED-LINE):
does turning on --tiered-negatives preserve the winning recipe's strong separation, or
break it? Identical command both runs; the ONLY difference is --tiered-negatives.

Runs sequentially (single MPS device). Writes two new tags (no overwrite):
  echino_softmax_abctrl  (baseline, default same-depth negatives)
  echino_softmax_tiered  (+ --tiered-negatives)
then analyses each (class/order/family separation) and captures stdout per tag.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings")
PY = str(ROOT / ".venv" / "bin" / "python")
TAXEMBED = str(ROOT / ".venv" / "bin" / "taxembed")
DS = ROOT / "data" / "taxopy" / "echinodermata_7586_clean"
NPZ = str(DS / "taxonomy_edges_echinodermata_7586_clean_transitive.npz")
MAPPING = str(DS / "taxonomy_edges_echinodermata_7586_clean.mapping.tsv")
ANALYZE = str(ROOT / "scripts" / "analyze_hierarchy_hyperbolic.py")

BASE = [
    TAXEMBED, "train",
    "--file", NPZ, "--mapping", MAPPING,
    "--dim", "100", "--epochs", "200",
    "--loss", "softmax", "--euclidean-param",
    "--radial-nudge", "0.05", "--lambda-reg", "0.1",
    "--early-stopping", "0",  # full 200 ep; default es=15 fires ~ep49 on loss-plateau (under-trains)
    "--gpu", "0",
]

RUNS = [
    ("echino_softmax_abctrl", []),
    ("echino_softmax_tiered", ["--tiered-negatives"]),
]


def run(label, cmd):
    print(f"\n{'='*80}\n[RUNNER] {label}\n{'='*80}", flush=True)
    print("[RUNNER] " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    print(f"[RUNNER] {label} exit={r.returncode}", flush=True)
    return r.returncode


def main():
    for tag, extra in RUNS:
        rc = run(f"TRAIN {tag}", BASE + extra + ["-as", tag])
        if rc != 0:
            print(f"[RUNNER] ABORT — training {tag} failed", flush=True)
            sys.exit(1)

    for tag, _ in RUNS:
        outdir = ROOT / "artifacts" / "tags" / tag / "analysis"
        ckpt = ROOT / "artifacts" / "tags" / tag / f"{tag}_best.pth"
        cmd = [
            PY, ANALYZE,
            "--checkpoint", str(ckpt),
            "--mapping", MAPPING,
            "--ranks", "class", "order", "family",
            "-o", str(outdir),
        ]
        print(f"\n{'='*80}\n[RUNNER] ANALYZE {tag}\n{'='*80}", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        cap = outdir / "analyze_stdout.txt"
        cap.parent.mkdir(parents=True, exist_ok=True)
        cap.write_text(r.stdout + "\n----- STDERR -----\n" + r.stderr)
        print(f"[RUNNER] ANALYZE {tag} exit={r.returncode} → {cap}", flush=True)
        print(r.stdout, flush=True)

    print("\n[RUNNER] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
