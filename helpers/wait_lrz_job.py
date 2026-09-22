"""Poll an LRZ SLURM job until it leaves the queue, then print its sacct row and log tail.

Exists because a shell polling loop (while/sleep/&&) is forbidden by the repo's shell-hygiene
rule; this is one python invocation, safe to run in the background.

Usage:
  <python> helpers/wait_lrz_job.py --job 5802041 --log <abs remote .out path> [--every 60] [--max-min 120]
"""
from __future__ import annotations

import argparse
import subprocess
import time


def _ssh(*args: str) -> str:
    return subprocess.run(["ssh", "ai", *args], capture_output=True, text=True).stdout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--log", required=True, help="remote .out path (%j already substituted)")
    ap.add_argument("--every", type=int, default=60)
    ap.add_argument("--max-min", type=int, default=120)
    ap.add_argument("--tail", type=int, default=40)
    args = ap.parse_args()

    deadline = time.time() + 60 * args.max_min
    while time.time() < deadline:
        if args.job not in _ssh("squeue", "-h", "-j", args.job, "-o", "%i"):
            break
        time.sleep(args.every)
    else:
        print(f"TIMEOUT: job {args.job} still queued/running after {args.max_min} min")
        print(_ssh("squeue", "-j", args.job))
        return

    print(_ssh("sacct", "-j", args.job, "--format=JobID%20,State,Elapsed,ExitCode,MaxRSS"))
    print(_ssh("tail", "-n", str(args.tail), args.log))


if __name__ == "__main__":
    main()
