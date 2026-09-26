"""READ-ONLY review helper (adversarial pre-submit review, 2026-09-26).

Walks ALL 12 array indices of scripts/p2_lrz_train.sh by EXECUTING the script's own
index -> (ARM, SEED, TAG, TRAIN) block with bash, rather than re-reading it by eye.
The block is extracted from the script text between the `IDX=` line and the line that
closes the `case` statement, so it cannot drift from the file under review.

Checks: 12 distinct (arm, seed) pairs, 12 distinct tags, 12 distinct TRAIN files, and
that each TRAIN file exists locally (the LRZ copies are byte-compared by
helpers/p2_review4_verify_uploads.py).

Writes nothing outside /tmp scratch used as bash stdin.
"""
from __future__ import annotations

import os
import subprocess

REPO = "/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings"
SCRIPT = os.path.join(REPO, "scripts", "p2_lrz_train.sh")
LOCAL_SPLITS = os.path.join(REPO, "data", "p2_splits")


def extract_block(text: str) -> str:
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("IDX="))
    end = next(i for i, l in enumerate(lines) if i > start and l.strip() == "esac")
    return "\n".join(lines[start:end + 1])


def main() -> None:
    with open(SCRIPT) as fh:
        text = fh.read()
    block = extract_block(text)
    # the block references SPLITDIR, defined above it in the real script
    splitdir_line = next(l for l in text.splitlines() if l.startswith("SPLITDIR="))
    print("extracted mapping block from", SCRIPT)
    print("-" * 70)
    print(block)
    print("-" * 70)
    print()

    seen_pairs, seen_tags, seen_files = {}, {}, {}
    print(f"{'IDX':<5}{'ARM':<18}{'SEED':<6}{'TAG':<26}{'TRAIN basename'}")
    for idx in range(12):
        prog = "\n".join([
            "set -euo pipefail",
            splitdir_line,
            f'SLURM_ARRAY_TASK_ID={idx}',
            block,
            'echo "${ARM}|${SEED}|${TAG}|${TRAIN}"',
        ])
        out = subprocess.run(["bash", "-c", prog], capture_output=True, text=True)
        if out.returncode != 0:
            print(f"{idx:<5}BASH FAILED: {out.stderr.strip()}")
            continue
        arm, seed, tag, train = out.stdout.strip().split("|")
        print(f"{idx:<5}{arm:<18}{seed:<6}{tag:<26}{os.path.basename(train)}")
        seen_pairs.setdefault((arm, seed), []).append(idx)
        seen_tags.setdefault(tag, []).append(idx)
        seen_files.setdefault(train, []).append(idx)

    print()
    dup_pairs = {k: v for k, v in seen_pairs.items() if len(v) > 1}
    dup_tags = {k: v for k, v in seen_tags.items() if len(v) > 1}
    dup_files = {k: v for k, v in seen_files.items() if len(v) > 1}
    print(f"distinct (arm,seed): {len(seen_pairs)}/12   duplicates: {dup_pairs or 'none'}")
    print(f"distinct tags      : {len(seen_tags)}/12   duplicates: {dup_tags or 'none'}")
    print(f"distinct TRAIN     : {len(seen_files)}/12   duplicates: {dup_files or 'none'}")

    print()
    print("local existence of each TRAIN (container /data/p2_splits -> repo data/p2_splits):")
    n_ok = 0
    for train in sorted(seen_files):
        local = os.path.join(LOCAL_SPLITS, os.path.basename(train))
        ok = os.path.exists(local)
        n_ok += ok
        if not ok:
            print(f"  MISSING locally: {local}")
    print(f"  {n_ok}/{len(seen_files)} present locally")

    # tag-prefix collision: could one element's `find -name "${TAG}_epoch*.pth"` ever
    # match a SIBLING element's files if the tag dirs were shared?
    print()
    print("tag-prefix collisions (would matter if the tag dirs were ever shared):")
    tags = sorted(seen_tags)
    collisions = [(a, b) for a in tags for b in tags
                  if a != b and b.startswith(a + "_epoch") is False and b.startswith(a)]
    print(f"  {collisions or 'none -- no tag is a prefix of another'}")


if __name__ == "__main__":
    main()
