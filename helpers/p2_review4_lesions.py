"""READ-ONLY PROBE (review 4, 2026-09-26): lesions the shipped harness does NOT register.

Reuses `helpers/p2_lesion_harness.py`'s shadow-tree machinery (build_shadow / pytest_env /
assert_shadow_imports / run_tests) WITHOUT adding anything to its MUTATIONS table, and applies
four lesions of my own to a throwaway copy of the tracked tree. The real repo is never written to.

  L1  amendment_6's equivalence band loses its `2 * pooled_sd` term.
      Every TestP2Amendment6 fixture is built with `jitter_nr=0.0`, so the pooled SD is exactly
      zero in all 12 of them -- the term can never be the binding one there.

  L2  the C-B fix APPLIED TO ITS UN-TRAVELLED SIBLING, `_p2_single_shared_control`, whose chain
      still ends at the real tree's "baselines". Here "caught" is the interesting answer: it
      would mean the suite PINS the defective fallback, i.e. the fixtures encode it -- the exact
      condition the commit fixed for `_p2_matched_controls` and not for this one.

  L3  p2_lrz_score.sh's roll-window count pre-flight silently disabled by deleting the ROLLING
      clear from p2_lrz_train.sh while keeping the MILESTONE one -- does static gate check 6
      (a substring test) notice?

  L4  p2_lrz_score.sh's P2_ROLL_WINDOW_EXPECTED drifts from the engine's constant -- does static
      gate check 5 notice? (The paired positive control for L3.)
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("lh", REPO / "helpers" / "p2_lesion_harness.py")
lh = importlib.util.module_from_spec(spec)
sys.modules["lh"] = lh          # dataclasses resolves cls.__module__ through sys.modules
spec.loader.exec_module(lh)

PY_LESIONS = [
    dict(
        name="L1 amendment6_band_loses_the_pooled_sd_term",
        rel="src/taxembed/eval/preregistration.py",
        old=("        equiv_band_nr = max(P2_EQUIV_REL_FRACTION * abs(float(c_cmp.mean())),\n"
             "                            P2_FLOOR_SD_MULTIPLE * pooled_vs_control_nr)"),
        new="        equiv_band_nr = P2_EQUIV_REL_FRACTION * abs(float(c_cmp.mean()))",
        tests=["tests/eval/test_preregistration.py::TestP2Amendment6"],
    ),
    dict(
        name="L1b same lesion, WHOLE SUITE",
        rel="src/taxembed/eval/preregistration.py",
        old=("        equiv_band_nr = max(P2_EQUIV_REL_FRACTION * abs(float(c_cmp.mean())),\n"
             "                            P2_FLOOR_SD_MULTIPLE * pooled_vs_control_nr)"),
        new="        equiv_band_nr = P2_EQUIV_REL_FRACTION * abs(float(c_cmp.mean()))",
        tests=["tests"],
    ),
    dict(
        name="L2 C-B's fix applied to _p2_single_shared_control (WHOLE SUITE)",
        rel="src/taxembed/eval/preregistration.py",
        old='    prefixes = ["baselines_randomdag", "baselines"]',
        new='    prefixes = ["baselines_randomdag"]',
        tests=["tests"],
    ),
]

SHELL_LESIONS = [
    dict(
        name="L3 remove ONLY the rolling-checkpoint clear from p2_lrz_train.sh",
        rel="scripts/p2_lrz_train.sh",
        old='    find "${TAGDIR}" -maxdepth 1 -name "${TAG}_epoch*.pth" -delete\n',
        new="",
    ),
    dict(
        name="L4 drift P2_ROLL_WINDOW_EXPECTED 5 -> 7 in p2_lrz_score.sh",
        rel="scripts/p2_lrz_score.sh",
        old="P2_ROLL_WINDOW_EXPECTED=5",
        new="P2_ROLL_WINDOW_EXPECTED=7",
    ),
]


def py_lesion(les: dict) -> None:
    work = Path(tempfile.mkdtemp(prefix="p2rev4_"))
    shadow = work / "tree"
    try:
        lh.build_shadow(shadow, ["data", ".venv"], "worktree")
        lh.assert_shadow_imports(shadow)
        base, base_tail = lh.run_tests(shadow, les["tests"])
        if base.get("failed") or base.get("error") or not base.get("passed"):
            print(f"  SELF-CHECK 4 FAILED on the unmutated shadow: {base}\n{base_tail}")
            return
        target = shadow / les["rel"]
        text = target.read_text()
        if les["old"] not in text:
            print(f"  SELF-CHECK 1 FAILED: pattern not found in {les['rel']}")
            return
        target.write_text(text.replace(les["old"], les["new"]))
        mut, tail = lh.run_tests(shadow, les["tests"])
        caught = bool(mut.get("failed") or mut.get("error"))
        print(f"  baseline {base}  ->  mutant {mut}")
        print(f"  {'CAUGHT' if caught else 'NOT CAUGHT'}")
        if caught:
            for line in tail.splitlines():
                if line.startswith(("FAILED", "ERROR")):
                    print(f"      {line[:150]}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def shell_lesion(les: dict) -> None:
    work = Path(tempfile.mkdtemp(prefix="p2rev4sh_"))
    shadow = work / "tree"
    try:
        lh.build_shadow(shadow, ["data", ".venv"], "worktree")
        base = subprocess.run([sys.executable, str(shadow / "helpers" / "p2_static_gate.py")],
                              capture_output=True, text=True, cwd=str(shadow),
                              env=lh.pytest_env(shadow))
        target = shadow / les["rel"]
        text = target.read_text()
        if les["old"] not in text:
            print(f"  SELF-CHECK 1 FAILED: pattern not found in {les['rel']}")
            return
        target.write_text(text.replace(les["old"], les["new"]))
        mut = subprocess.run([sys.executable, str(shadow / "helpers" / "p2_static_gate.py")],
                             capture_output=True, text=True, cwd=str(shadow),
                             env=lh.pytest_env(shadow))
        print(f"  static gate on the UNMUTATED shadow : exit {base.returncode}")
        print(f"  static gate on the MUTANT           : exit {mut.returncode}  "
              f"{'CAUGHT' if mut.returncode != 0 else 'NOT CAUGHT'}")
        for line in mut.stdout.splitlines():
            if line.strip().startswith(("🛑", "PROBLEM")) or "PROBLEM(S)" in line:
                print(f"      {line[:160]}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    for les in PY_LESIONS:
        print(f"\n{'=' * 90}\n{les['name']}\n{'=' * 90}")
        py_lesion(les)
    for les in SHELL_LESIONS:
        print(f"\n{'=' * 90}\n{les['name']}\n{'=' * 90}")
        shell_lesion(les)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
