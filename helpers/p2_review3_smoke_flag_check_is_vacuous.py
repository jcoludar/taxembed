#!/usr/bin/env python3
"""REVIEW WAVE 3 (read-only): is the Rule 16 smoke's flag check able to fail?

`scripts/p2_lrz_train_smoke.sh:84-92` is the ONLY guard standing between a wrong flag and a
20-hour CPU scoring job / a 12-element GPU array. It does:

    python /app/scripts/score_p2_linkpred.py --help > /tmp/p2_score_help.txt
    for flag in --manifest ... --closure --baselines-key; do
        if ! grep -q -- "${flag}" /tmp/p2_score_help.txt; then  ... exit 1 ; fi

`score_p2_linkpred.py` builds its parser as `ArgumentParser(description=__doc__)`, and its module
docstring NAMES most of those flags in prose (the Usage block and the CORRECTION notes). argparse
prints the description inside `--help`. So a substring grep over `--help` can be satisfied by the
DOCSTRING rather than by a real option.

The falsifier: grep the same `--help` text for flags that are definitely NOT options of this
script. If they are "found", the guard cannot distinguish a real option from a mention of one.
Same question asked of `taxembed.cli.main train --help`.

Writes nothing except stdout.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = REPO / ".venv" / "bin" / "python"

SMOKE_SCORE_FLAGS = ["--manifest", "--heldout", "--checkpoints", "--out", "--metric",
                     "--max-checkpoints", "--seed", "--closure", "--baselines-key"]
SMOKE_TRAIN_FLAGS = ["--file", "--mapping", "--dim", "--gpu", "--amp", "--curriculum",
                     "--curriculum-phases", "--early-stopping", "--radial-nudge",
                     "--radial-schedule", "--depth-scale-margin", "--margin-min", "--margin-max",
                     "--epoch-fraction", "--euclidean-param", "--loss", "--save-every", "--epochs",
                     "--seed", "--batch-size", "--n-negatives", "--lr", "--grad-accum-steps",
                     "--lr-schedule", "--warm-restart-on-phase", "--lr-min-multiplier", "-as"]
# flags that are NOT options of score_p2_linkpred.py but ARE named in its docstring / neighbours
NOT_SCORE_OPTIONS = ["--frac-val", "--visibility", "--npz", "--outdir", "--band", "--amendment-1"]
NOT_TRAIN_OPTIONS = ["--frac-val", "--baselines-key", "--manifest", "--heldout", "--closure"]


def help_text(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    return r.stdout + r.stderr


def real_options(help_txt: str, flag: str) -> bool:
    """A flag is a REAL option only if it begins an option line (argparse indents them) or appears
    immediately after a comma in an option line -- i.e. token-anchored, not substring."""
    import re
    pat = re.compile(rf"^\s+(?:-\w,\s+)?{re.escape(flag)}(?:[ ,=]|$)", re.M)
    return bool(pat.search(help_txt))


def audit(label: str, cmd: list[str], claimed: list[str], not_options: list[str]) -> int:
    txt = help_text(cmd)
    print(f"\n===== {label} =====")
    print(f"  --help length {len(txt)} chars")
    bad = 0
    print("  flags the SMOKE asserts, checked the smoke's way (substring) vs token-anchored:")
    for f in claimed:
        sub = f in txt
        tok = real_options(txt, f)
        mark = "" if sub == tok else "   <-- SUBSTRING-ONLY: the smoke passes, the option is NOT real"
        if sub != tok:
            bad += 1
        print(f"    {f:<26} substring={sub!s:<5} real_option={tok!s:<5}{mark}")
    print("  CONTROL -- flags that are NOT options here; the smoke's grep must NOT find them:")
    for f in not_options:
        sub = f in txt
        tok = real_options(txt, f)
        mark = "   <-- 🛑 the smoke's grep WOULD PASS for a flag that does not exist" if sub and not tok else ""
        if sub and not tok:
            bad += 1
        print(f"    {f:<26} substring={sub!s:<5} real_option={tok!s:<5}{mark}")
    return bad


def main() -> int:
    bad = 0
    bad += audit("scripts/score_p2_linkpred.py --help",
                 [str(PY), str(REPO / "scripts" / "score_p2_linkpred.py"), "--help"],
                 SMOKE_SCORE_FLAGS, NOT_SCORE_OPTIONS)
    bad += audit("python -m taxembed.cli.main train --help",
                 [str(PY), "-m", "taxembed.cli.main", "train", "--help"],
                 SMOKE_TRAIN_FLAGS, NOT_TRAIN_OPTIONS)
    print(f"\n{bad} flag(s) where the smoke's substring grep disagrees with a real option check.")
    print("Any non-zero count on a CONTROL line means the Rule 16 flag guard CANNOT FAIL for that "
          "flag: it is satisfied by the docstring argparse prints, not by the option existing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
