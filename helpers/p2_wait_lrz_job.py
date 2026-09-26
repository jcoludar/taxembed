#!/usr/bin/env python3
"""Poll an LRZ job (or array) until every element reaches a terminal state, then report.

Usage:
    python3 helpers/p2_wait_lrz_job.py 5811340 --label p2_smoke --max-min 60

Exists because CLAUDE.md forbids compound Bash commands -- an `until ...; do sleep; done` poll loop
with pipes matches no allowlist rule and would prompt, which strands an unattended run. The polling
logic lives here and runs as ONE venv-python call.

🛑 Rule 16 posture, deliberately built in: **this reports STATE, it does not certify success.**
`COMPLETED` / exit 0 is not evidence that a job did what it was for -- the ESMFold and DRY_RUN
incidents both had clean exits. The caller must still READ THE LOG. So the final line says what to
read next rather than declaring a pass.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
            "BOOT_FAIL", "DEADLINE", "PREEMPTED", "REVOKED", "SPECIAL_EXIT"}


def sacct(jobid: str) -> list[tuple[str, str, str, str]]:
    """(JobID, State, Elapsed, ExitCode) for the job and its steps, main rows only."""
    r = subprocess.run(
        ["ssh", "ai", "sacct", "-j", jobid, "-n", "-P",
         "--format=JobID,State,Elapsed,ExitCode,MaxRSS,TotalCPU"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return []
    rows = []
    for line in r.stdout.strip().splitlines():
        f = line.split("|")
        if len(f) >= 4 and ".batch" not in f[0] and ".extern" not in f[0]:
            rows.append(tuple(f[:4]))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobid")
    ap.add_argument("--label", default="job")
    ap.add_argument("--max-min", type=int, default=60)
    ap.add_argument("--interval", type=int, default=30)
    args = ap.parse_args()

    deadline = time.time() + args.max_min * 60
    last = None
    while time.time() < deadline:
        rows = sacct(args.jobid)
        if rows:
            states = {r[1].split()[0] for r in rows}
            summary = ", ".join(f"{r[0]}={r[1]}" for r in rows)
            if summary != last:
                print(f"[{args.label} {args.jobid}] {summary}", flush=True)
                last = summary
            if states and states <= TERMINAL:
                print(f"\n[{args.label} {args.jobid}] ALL TERMINAL", flush=True)
                for jid, state, elapsed, exitcode in rows:
                    print(f"  {jid}  {state}  elapsed={elapsed}  exit={exitcode}", flush=True)
                bad = sorted(s for s in states if s != "COMPLETED")
                if bad:
                    print(f"🛑 non-COMPLETED state(s): {bad}", flush=True)
                    return 1
                print("State is COMPLETED. 🛑 That is NOT a pass -- READ THE LOG before "
                      "concluding anything (Rule 16).", flush=True)
                return 0
        time.sleep(args.interval)

    print(f"[{args.label} {args.jobid}] still not terminal after {args.max_min} min", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
