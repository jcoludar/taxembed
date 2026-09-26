#!/usr/bin/env python3
"""Verify the 12 uploaded P2 splits on LRZ against the manifests that describe them.

Why against the MANIFEST and not merely against the local file: the manifest's `train_md5` is the
number the split's own bookkeeping recorded at build time. Comparing remote-vs-local proves the
transfer worked; comparing remote-vs-manifest proves the transfer worked AND that the file on LRZ is
the one the manifest describes -- so the provenance chain (source closure md5 -> split -> uploaded
artifact) closes end to end. A transfer check alone would pass happily on a file whose manifest had
drifted.

This is the guard review finding C5 existed for: an input problem that surfaces only INSIDE the
container, AFTER the GPU queue wait. Cheap here, expensive there.

Also checks the exact filenames `scripts/p2_lrz_train.sh` will construct from its array index, so a
naming mismatch cannot be discovered by a job at 03:00.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOCAL_SPLITS = REPO / "data/p2_splits"
REMOTE_BASE = "/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz"
REMOTE_SPLITS = f"{REMOTE_BASE}/data/p2_splits"
REMOTE_MAP = f"{REMOTE_BASE}/data/taxonomy_edges_metazoa_33208_clean.mapping.tsv"
CLADE = "metazoa_33208_clean"
ARMS = ("vis00", "vis50", "degmatch_vis00", "degmatch_vis50")
SEEDS = (0, 1, 2)


def expected_files() -> list[tuple[str, int, str]]:
    """Exactly what p2_lrz_train.sh's `case "${ARM}"` block will look for, index 0-11."""
    out = []
    for arm in ARMS:
        for seed in SEEDS:
            out.append((arm, seed, f"p2_{CLADE}_{arm}_seed{seed}_train.npz"))
    return out


def remote_md5(paths: list[str]) -> dict[str, str]:
    r = subprocess.run(["ssh", "ai", "md5sum", *paths], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"remote md5sum failed:\n{r.stderr}")
    got = {}
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            got[Path(parts[-1]).name] = parts[0]
    return got


def main() -> int:
    wanted = expected_files()
    names = [n for _, _, n in wanted]

    # the manifests are the source of truth for what each file SHOULD hash to
    manifest_md5: dict[str, str] = {}
    missing_manifest = []
    for arm, seed, name in wanted:
        man = LOCAL_SPLITS / f"p2_{CLADE}_{arm}_seed{seed}_manifest.json"
        if not man.exists():
            missing_manifest.append(man.name)
            continue
        manifest_md5[name] = json.loads(man.read_text())["train_md5"]

    print(f"remote : {REMOTE_SPLITS}")
    print(f"checking {len(names)} train files the array will open by exact name\n")

    got = remote_md5([f"{REMOTE_SPLITS}/{n}" for n in names])

    problems: list[str] = []
    for arm, seed, name in wanted:
        idx = ARMS.index(arm) * 3 + seed
        r_md5 = got.get(name)
        m_md5 = manifest_md5.get(name)
        if r_md5 is None:
            print(f"  idx {idx:>2}  {arm:16s} seed{seed}  MISSING ON LRZ")
            problems.append(f"{name}: absent on LRZ -- array task {idx} would exit 1")
            continue
        if m_md5 is None:
            print(f"  idx {idx:>2}  {arm:16s} seed{seed}  no local manifest")
            problems.append(f"{name}: no manifest to verify against")
            continue
        if r_md5 == m_md5:
            print(f"  idx {idx:>2}  {arm:16s} seed{seed}  ok   {r_md5[:16]}…")
        else:
            print(f"  idx {idx:>2}  {arm:16s} seed{seed}  MISMATCH remote={r_md5[:16]}… "
                  f"manifest={m_md5[:16]}…")
            problems.append(f"{name}: remote md5 != manifest train_md5")

    # the mapping TSV the job also requires
    print("\nthe mapping TSV the job requires:")
    r = subprocess.run(["ssh", "ai", "ls", "-la", REMOTE_MAP], capture_output=True, text=True)
    if r.returncode == 0:
        print(f"  ok   {r.stdout.strip()}")
    else:
        print(f"  MISSING: {REMOTE_MAP}")
        problems.append("the metazoa mapping TSV is absent -- every array task would exit 1")

    print(f"\n{'=' * 78}")
    if missing_manifest:
        print(f"⚠ manifests absent locally: {missing_manifest}")
    if problems:
        print(f"🛑 {len(problems)} PROBLEM(S) -- DO NOT SUBMIT:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("✅ all 12 splits present on LRZ, every md5 matches its manifest, mapping TSV present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
