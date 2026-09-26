#!/usr/bin/env python3
"""Rule 16 static gate for the P2 jobs, run LOCALLY before any queue time is spent.

Rule 16: never submit untested code. A job that dies 30 s in after a multi-hour queue wait is a
wasted cycle; on a depleted GPU fairshare it can cost a day. Incident of record: ESMFold job 5710076
died 37 s after finally being granted a scarce H100, on a one-line error `bash -n` would have caught.

Checks
------
1. `bash -n` on all three job scripts.
2. `py_compile` on every module the jobs import.
3. 🎯 **FLAG REALITY, WITH THE FLAG LIST DERIVED FROM THE JOB SCRIPT ITSELF** -- every `--flag` that
   `p2_lrz_train.sh` actually passes to `taxembed.cli.main train` must be a real option, and likewise
   for `p2_lrz_score.sh` against `scripts/score_p2_linkpred.py`.

   Why derive rather than copy: `p2_lrz_train_smoke.sh` already runs a flag-reality check, but against
   a **hardcoded list** of flag names. A hardcoded list is a SECOND copy of the job's flags, and two
   copies drift. If the job gained a flag the smoke's list does not mention, the smoke would pass and
   the real job would still fail -- the guard would be silent on exactly the change it exists to
   catch. This check reads the flags out of the job script, so it cannot drift from it.

4. **The smoke's hardcoded list is then compared against the derived set**, and any flag the job passes
   but the smoke does not check is reported. That is a gap in the canary, not in the job.

Nothing here contacts LRZ and nothing is submitted.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = REPO / ".venv/bin/python"

JOBS = ["scripts/p2_lrz_train.sh", "scripts/p2_lrz_train_smoke.sh", "scripts/p2_lrz_score.sh"]
MODULES = [
    "src/taxembed/cli/main.py", "src/taxembed/eval/p2_split.py", "src/taxembed/eval/randomdag.py",
    "src/taxembed/eval/linkpred.py", "src/taxembed/eval/baselines.py",
    "src/taxembed/eval/preregistration.py", "scripts/build_p2_split.py",
    "scripts/build_p2_degmatch_split.py", "scripts/build_p2_randomdag_split.py",
    "scripts/score_p2_linkpred.py",
]

FLAG = re.compile(r"(?<![\w-])(--[a-z][a-z0-9-]*)")


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def flags_passed_to(script: Path, invocation: str) -> set[str]:
    """Flags appearing in the continued command line that begins with `invocation`.

    A job line is `python -m taxembed.cli.main train \\` followed by backslash-continued lines, so the
    invocation is reassembled by following the continuations rather than scanning the whole file --
    otherwise flags named in comments or in a DIFFERENT command would be swept in.
    """
    text = script.read_text().splitlines()
    out: set[str] = set()
    i = 0
    while i < len(text):
        line = text[i]
        stripped = line.strip()
        if invocation in stripped and not stripped.startswith("#"):
            block = [stripped]
            while block[-1].endswith("\\") and i + 1 < len(text):
                i += 1
                block.append(text[i].strip())
            joined = " ".join(b.rstrip("\\") for b in block)
            # drop any trailing pipeline (e.g. `| tee ...`) so its flags are not attributed
            joined = joined.split("|")[0]
            out |= set(FLAG.findall(joined))
        i += 1
    return out


def help_flags(cmd: list[str]) -> set[str]:
    r = run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"could not get --help from {' '.join(cmd)}:\n{r.stderr[-2000:]}")
    return set(FLAG.findall(r.stdout))


def main() -> int:
    problems: list[str] = []

    print("1. bash -n on the job scripts")
    for j in JOBS:
        r = run(["bash", "-n", str(REPO / j)])
        ok = r.returncode == 0
        print(f"   {'ok  ' if ok else 'FAIL'} {j}")
        if not ok:
            problems.append(f"{j}: bash -n failed: {r.stderr.strip()}")

    print("\n2. py_compile on the modules the jobs import")
    r = run([str(PY), "-m", "py_compile", *[str(REPO / m) for m in MODULES]])
    if r.returncode == 0:
        print(f"   ok   {len(MODULES)} modules compile")
    else:
        print(f"   FAIL {r.stderr.strip()}")
        problems.append(f"py_compile failed: {r.stderr.strip()}")

    print("\n3. flag reality -- flags DERIVED from the job script, not copied")
    train_real = help_flags([str(PY), "-m", "taxembed.cli.main", "train", "--help"])
    train_used = flags_passed_to(REPO / "scripts/p2_lrz_train.sh", "taxembed.cli.main train")
    missing = sorted(f for f in train_used if f not in train_real)
    print(f"   p2_lrz_train.sh passes {len(train_used)} flags to `taxembed.cli.main train`")
    if missing:
        print(f"   FAIL not real options: {missing}")
        problems.append(f"p2_lrz_train.sh passes non-existent train flags: {missing}")
    else:
        print(f"   ok   all {len(train_used)} are real options")

    score_real = help_flags([str(PY), str(REPO / "scripts/score_p2_linkpred.py"), "--help"])
    score_used = flags_passed_to(REPO / "scripts/p2_lrz_score.sh", "score_p2_linkpred.py")
    missing_s = sorted(f for f in score_used if f not in score_real)
    print(f"   p2_lrz_score.sh passes {len(score_used)} flags to score_p2_linkpred.py")
    if missing_s:
        print(f"   FAIL not real options: {missing_s}")
        problems.append(f"p2_lrz_score.sh passes non-existent score flags: {missing_s}")
    else:
        print(f"   ok   all {len(score_used)} are real options")

    print("\n4. does the SMOKE's hardcoded flag list cover what the job actually passes?")
    smoke = (REPO / "scripts/p2_lrz_train_smoke.sh").read_text()
    # the smoke's own `for flag in ... ; do` lists
    smoke_listed = set(FLAG.findall(smoke))
    uncovered = sorted(f for f in train_used if f not in smoke_listed)
    if uncovered:
        print(f"   ⚠ the smoke does NOT check {len(uncovered)}: {uncovered}")
        print("     Not a defect in the job -- a GAP IN THE CANARY. The smoke's list is a second")
        print("     copy of the job's flags and has drifted from it.")
    else:
        print(f"   ok   the smoke's list covers every flag the job passes")

    # 5. C-C (2026-09-26): the shell's roll-window count must equal the Python constant.
    # `p2_lrz_score.sh` grew a `P2_ROLL_WINDOW_EXPECTED` pre-flight, which is a SECOND COPY of
    # `preregistration.P2_ROLL_WINDOW` -- exactly the shape check 4 exists to police. Two copies
    # drift, and drift here is silent: the shell would wave through a roll set the engine then
    # refuses, after the scoring CPU is already spent, or worse pass a count the engine accepts
    # for a different reason. Derived comparison, not a second hardcoded number in a third place.
    print("\n5. does the shell's roll-window count match the engine's P2_ROLL_WINDOW?")
    sys.path.insert(0, str(REPO / "src"))
    from taxembed.eval.preregistration import P2_ROLL_WINDOW  # noqa: E402
    score_text = (REPO / "scripts/p2_lrz_score.sh").read_text()
    m = re.search(r"^P2_ROLL_WINDOW_EXPECTED=(\d+)", score_text, re.MULTILINE)
    if m is None:
        problems.append("p2_lrz_score.sh has no P2_ROLL_WINDOW_EXPECTED pre-flight -- the "
                        "roll-window count check (C-C) is missing from the job")
        print("   🛑 P2_ROLL_WINDOW_EXPECTED not found in p2_lrz_score.sh")
    elif int(m.group(1)) != P2_ROLL_WINDOW:
        problems.append(f"p2_lrz_score.sh P2_ROLL_WINDOW_EXPECTED={m.group(1)} but "
                        f"preregistration.P2_ROLL_WINDOW={P2_ROLL_WINDOW} -- the shell pre-flight "
                        f"and the engine disagree about the rolling window")
        print(f"   🛑 shell {m.group(1)} vs engine {P2_ROLL_WINDOW}")
    else:
        print(f"   ok   both say {P2_ROLL_WINDOW}")

    # 6. C-C: the train job must clear its tag directory before training.
    print("\n6. does p2_lrz_train.sh clear its tag directory before training?")
    train_text = (REPO / "scripts/p2_lrz_train.sh").read_text()
    if "_epoch*.pth\" -delete" not in train_text:
        problems.append("p2_lrz_train.sh does not clear ${TAG}'s rolling checkpoints before "
                        "training -- a resubmitted array element would leave the previous "
                        "attempt's orphans for the scorer's glob to pick up (C-C)")
        print("   🛑 no pre-training clear found")
    else:
        print("   ok   the tag directory is cleared before each attempt")

    print(f"\n{'=' * 78}")
    if problems:
        print(f"🛑 {len(problems)} PROBLEM(S) -- DO NOT SUBMIT:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("✅ static gate PASSED -- syntax, compilation and flag reality all clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
