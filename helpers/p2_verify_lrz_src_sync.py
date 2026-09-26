#!/usr/bin/env python3
"""B1 (2026-09-26): is the LRZ `src` mount byte-identical to the local reviewed tree?

Incident of record: the mount was STALE FROM 2026-08-11 and held NONE of the five P2 eval
modules, so the array as written would have failed on import or trained on August code. It went
stale again on 2026-09-26 -- staged at e602c28 while two further commits of fixes landed locally,
so the cluster's copies of both job scripts and of preregistration.py were the PRE-FIX versions
and the comment in the job script claiming "the engine now REFUSES a wrong-length roll set" was
false on the cluster.

Compares md5 of every file the P2 chain actually executes. ssh is invoked with an ARGV LIST and
no shell metacharacters -- `ssh ai md5sum a b c` is safe, `ssh ai "grep -E 'a|b' f"` is not,
because ssh re-splits its arguments through the REMOTE shell.

Read-only on both sides. Exit 0 only if every file matches.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
LRZ_ROOT = "/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/src"

# Everything the train job or the score job imports or executes.
FILES = [
    "scripts/p2_lrz_train.sh",
    "scripts/p2_lrz_score.sh",
    "scripts/p2_lrz_train_smoke.sh",
    "scripts/score_p2_linkpred.py",
    "scripts/apply_preregistration.py",
    "scripts/build_p2_split.py",
    "scripts/build_p2_degmatch_split.py",
    "src/taxembed/eval/preregistration.py",
    "src/taxembed/eval/baselines.py",
    "src/taxembed/eval/linkpred.py",
    "src/taxembed/eval/p2_split.py",
    "src/taxembed/eval/randomdag.py",
    "src/taxembed/eval/subtree.py",
    "src/taxembed/cli/main.py",
    "src/taxembed/utils/training_pairs.py",
    "train_small.py",
    "pyproject.toml",
]


def local_md5(rel: str) -> str | None:
    p = _REPO / rel
    if not p.exists():
        return None
    return hashlib.md5(p.read_bytes()).hexdigest()


def remote_md5s(rels: list[str]) -> dict[str, str]:
    cmd = ["ssh", "ai", "md5sum"] + [f"{LRZ_ROOT}/{r}" for r in rels]
    rc = subprocess.run(cmd, capture_output=True, text=True)
    out: dict[str, str] = {}
    for line in rc.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            digest, path = parts
            out[path[len(LRZ_ROOT) + 1:]] = digest
    if rc.returncode != 0:
        # md5sum exits non-zero if ANY file is missing; the ones it did hash are still in stdout
        print("  (md5sum reported missing files:)")
        for line in rc.stderr.strip().splitlines():
            print(f"    {line}")
    return out


def main() -> int:
    remote = remote_md5s(FILES)
    bad, missing = [], []
    print(f"{'file':<45} {'local':<34} {'lrz'}")
    for rel in FILES:
        lo = local_md5(rel)
        ro = remote.get(rel)
        if lo is None:
            print(f"  LOCAL-MISSING {rel}")
            missing.append(rel)
            continue
        if ro is None:
            print(f"  LRZ-MISSING   {rel}")
            missing.append(rel)
            continue
        mark = "ok  " if lo == ro else "DIFF"
        if lo != ro:
            bad.append(rel)
        print(f"  {mark} {rel:<40} {lo}  {ro}")

    print(f"\n{'=' * 78}")
    if bad or missing:
        print(f"🛑 NOT IN SYNC: {len(bad)} differ, {len(missing)} missing. DO NOT SUBMIT -- the "
              f"cluster would run different code from the one that was reviewed.")
        for rel in bad:
            print(f"  DIFF    {rel}")
        for rel in missing:
            print(f"  MISSING {rel}")
        return 1
    print(f"✅ all {len(FILES)} executed files are byte-identical on LRZ and locally")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
