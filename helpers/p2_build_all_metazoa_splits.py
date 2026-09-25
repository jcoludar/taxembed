#!/usr/bin/env python3
"""Build every metazoa split the 12-arm P2 array consumes, and CHECK the invariants while doing it.

`scripts/p2_lrz_train.sh` maps array index -> arm -> a train .npz by exact filename (see its
`case "${ARM}"` block). Twelve files are required:

    p2_metazoa_33208_clean_{vis00,vis50,degmatch_vis00,degmatch_vis50}_seed{0,1,2}_train.npz

A missing one is review finding C6 ("10 of the 12 training inputs did not exist"), and it fails only
AFTER the GPU queue wait -- which is what Rule 16 exists to prevent.

Why a helper rather than twelve shell invocations: CLAUDE.md forbids compound Bash commands, and a
loop plus per-build assertions is exactly the "write it to helpers/ and run it as one call" case. It
also means the invariant checks below are a reproducible artifact rather than something I eyeballed.

INVARIANTS CHECKED (each would otherwise be an assumption)
----------------------------------------------------------
1. **Rule 1 / never overwrite blind.** Every output path is listed BEFORE any build. A file that
   already exists is SKIPPED unless --rebuild, and the skip is reported, not silent.
2. **The holdout depends only on the seed, never on visibility or on the control.** This is asserted
   by `p2_lrz_train.sh` itself ("degree_matched_shuffle and select_holdout depend only on seed"), and
   the whole matched-control design rests on it: vis00 and degmatch_vis00 must be read against the
   SAME held-out nodes or the comparison mixes the control with the split. So `heldout_md5` must be
   IDENTICAL across all four arms at a given seed. Measured, not trusted.
3. **The manifest arithmetic closes**, in the amended vis-0 form that Task 2's plan defect broke:
       n_pairs_source == n_pairs_train + n_pairs_removed_parent_edges + n_pairs_removed_visibility
   and, at visibility 0.0 only,
       n_pairs_train == (n_nodes - 1 - n_test - n_val) + n_pairs_heldout_ancestry_kept
   The second is the one that encoded the defect when its last term was omitted.
4. **The held-out exemption actually fired**: n_pairs_heldout_ancestry_kept > 0 at EVERY visibility.
   Zero here is the dead-arm condition -- held-out nodes with no training row carry an untrained
   random embedding, and the arm would measure initialization.
5. **Seeds are genuinely different draws.** The three seeds' heldout_md5 must differ from each other.
   A builder that ignored `seed` would give three identical "independent" arms; that exact lesion
   was live in `randomize_parents` until 2026-09-25 (fix wave 3 item 3).
6. 🧨 **THE CONTROL MUST NOT BE THE REAL TREE.** `degree_matched_shuffle` preserves each node's depth
   and child count, so the degmatch arm's pair COUNT is identical to the real arm's at the same
   (visibility, seed) -- measured: degmatch_vis00 seed0 and vis00 seed0 both report exactly
   1,071,269 train pairs. That agreement is structural and therefore proves NOTHING about whether
   the closure was actually rewired. The falsifiable check is that `train_md5` DIFFERS. If it did
   not, the "control" would be the real tree and every below-control comparison would be vacuous --
   the single worst failure available to this experiment, and one the count table would have
   reported as a clean row. (Same shape as
   [[feedback_agreement_is_evidence_only_if_it_could_have_disagreed]].)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CLADE = "metazoa_33208_clean"
SOURCE = REPO / f"data/taxopy/{CLADE}/taxonomy_edges_{CLADE}_transitive.npz"
OUTDIR = REPO / "data/p2_splits"
PY = REPO / ".venv/bin/python"
SEEDS = (0, 1, 2)

# arm -> (builder script, visibility)
ARMS = {
    "vis00": ("scripts/build_p2_split.py", 0.0),
    "vis50": ("scripts/build_p2_split.py", 0.5),
    "degmatch_vis00": ("scripts/build_p2_degmatch_split.py", 0.0),
    "degmatch_vis50": ("scripts/build_p2_degmatch_split.py", 0.5),
}


def train_name(arm: str, seed: int) -> str:
    return f"p2_{CLADE}_{arm}_seed{seed}_train.npz"


def manifest_name(arm: str, seed: int) -> str:
    return f"p2_{CLADE}_{arm}_seed{seed}_manifest.json"


def build(arm: str, seed: int, rebuild: bool) -> tuple[str, dict | None]:
    script, visibility = ARMS[arm]
    out = OUTDIR / train_name(arm, seed)
    if out.exists() and not rebuild:
        man = OUTDIR / manifest_name(arm, seed)
        if man.exists():
            return "skipped (exists)", json.loads(man.read_text())
        return "PRESENT BUT NO MANIFEST", None
    cmd = [str(PY), str(REPO / script), "--npz", str(SOURCE), "--outdir", str(OUTDIR),
           "--visibility", str(visibility), "--seed", str(seed),
           "--frac-test", "0.10", "--frac-val", "0.0"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  BUILD FAILED {arm} seed{seed}:\n{r.stderr}", file=sys.stderr)
        return "FAILED", None
    man = OUTDIR / manifest_name(arm, seed)
    return "built", json.loads(man.read_text())


def check_manifest(arm: str, seed: int, m: dict) -> list[str]:
    bad = []
    lhs = m["n_pairs_source"]
    rhs = m["n_pairs_train"] + m["n_pairs_removed_parent_edges"] + m["n_pairs_removed_visibility"]
    if lhs != rhs:
        bad.append(f"pair accounting does not close: {lhs} != {rhs}")
    if m["n_pairs_heldout_ancestry_kept"] <= 0:
        bad.append("n_pairs_heldout_ancestry_kept == 0 -- DEAD ARM, held-out nodes have no "
                   "training row and would carry an untrained embedding")
    if float(m["visibility"]) == 0.0:
        expect = (m["n_nodes"] - 1 - m["n_test"] - m["n_val"]) + m["n_pairs_heldout_ancestry_kept"]
        if m["n_pairs_train"] != expect:
            bad.append(f"vis-0 identity fails: n_pairs_train {m['n_pairs_train']} != {expect}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild files that already exist (Rule 1: off by default)")
    args = ap.parse_args()

    if not SOURCE.exists():
        raise SystemExit(f"source closure missing: {SOURCE}")

    print(f"source  : {SOURCE}")
    print(f"outdir  : {OUTDIR}")
    print(f"required: {len(ARMS) * len(SEEDS)} train files\n")

    # INVARIANT 1 -- list every target path and its current state BEFORE building anything
    print("Rule 1 pre-flight -- what exists now:")
    for arm in ARMS:
        for seed in SEEDS:
            p = OUTDIR / train_name(arm, seed)
            print(f"  {'EXISTS ' if p.exists() else 'absent '} {train_name(arm, seed)}")
    print()

    manifests: dict[tuple[str, int], dict] = {}
    problems: list[str] = []
    for arm in ARMS:
        for seed in SEEDS:
            status, m = build(arm, seed, args.rebuild)
            if m is None:
                problems.append(f"{arm} seed{seed}: {status}")
                print(f"  {arm:16s} seed{seed}  {status}")
                continue
            manifests[(arm, seed)] = m
            bad = check_manifest(arm, seed, m)
            problems += [f"{arm} seed{seed}: {b}" for b in bad]
            flag = "  ok" if not bad else "  BAD"
            print(f"  {arm:16s} seed{seed}  {status:18s} train_pairs={m['n_pairs_train']:>9,} "
                  f"heldout_anc={m['n_pairs_heldout_ancestry_kept']:>8,}{flag}")

    # INVARIANT 2 -- one holdout per seed, shared by all four arms
    print("\nINVARIANT: heldout_md5 identical across all 4 arms at each seed")
    for seed in SEEDS:
        md5s = {arm: manifests[(arm, seed)]["heldout_md5"]
                for arm in ARMS if (arm, seed) in manifests}
        uniq = set(md5s.values())
        if len(uniq) == 1:
            print(f"  seed{seed}  ok   {uniq.pop()[:16]}…  ({len(md5s)} arms agree)")
        else:
            print(f"  seed{seed}  BAD  {md5s}")
            problems.append(f"seed{seed}: arms disagree on the held-out set -- the matched-control "
                            f"comparison would mix control with split")

    # INVARIANT 5 -- the three seeds are different draws
    print("\nINVARIANT: the 3 seeds are DIFFERENT draws")
    per_seed = {seed: manifests[("vis00", seed)]["heldout_md5"]
                for seed in SEEDS if ("vis00", seed) in manifests}
    if len(set(per_seed.values())) == len(per_seed):
        print(f"  ok   {len(per_seed)} distinct heldout_md5")
    else:
        print(f"  BAD  {per_seed}")
        problems.append("two seeds produced the SAME held-out set -- seeds are not independent draws")

    # INVARIANT 6 -- the control is not the real tree
    print("\nINVARIANT: degmatch train_md5 DIFFERS from the real arm's (counts agree by construction)")
    for vis in ("vis00", "vis50"):
        for seed in SEEDS:
            real, ctrl = (f"{vis}", f"degmatch_{vis}")
            if (real, seed) not in manifests or (ctrl, seed) not in manifests:
                continue
            m_r, m_c = manifests[(real, seed)], manifests[(ctrl, seed)]
            same_count = m_r["n_pairs_train"] == m_c["n_pairs_train"]
            if m_r["train_md5"] == m_c["train_md5"]:
                print(f"  {vis} seed{seed}  BAD  train_md5 IDENTICAL -- the control IS the real tree")
                problems.append(f"{ctrl} seed{seed}: train_md5 identical to {real} -- control is the "
                                f"real tree, every below-control comparison would be vacuous")
            else:
                note = "counts equal (expected)" if same_count else "counts also differ"
                print(f"  {vis} seed{seed}  ok   md5 differs, {note}")

    print(f"\n{'=' * 78}")
    if problems:
        print(f"🛑 {len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"✅ all {len(manifests)} splits present and every invariant holds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
