#!/usr/bin/env python3
"""Lesion harness for P2 fix wave 3: prove a test CAN FAIL by constructing the defect it is named for.

Why this exists
---------------
The P2 build shipped, by the final review's count, TEN tests that could not fail -- four found during
the build, six more by the review. Every one of them looked like coverage. The lesson the project keeps
re-learning is: *before trusting any guard, construct the lesion and confirm the test fails.*

This harness makes that cheap and, more importantly, makes it HONEST. Doing it by hand in the real
working tree raises a question you cannot answer from the outside -- "did the restore actually work?"
-- so every mutation here runs in a `git archive` SHADOW TREE and the real repo is never touched.
(Method carried from syntile Task 11, where it replaced hand-mutation after four fix rounds.)

Four self-checks, because a harness is a verifier and a verifier you wrote shares your blind spot
------------------------------------------------------------------------------------------------
Each is a way this harness could have printed a confident, meaningless verdict:

1. **The mutation must actually apply.** A `str.replace` that matches nothing is a silent no-op, and
   the run then "proves" the test survives a lesion that was never introduced. A miss is a hard error.
2. **The code under test must be the SHADOW's code.** `taxembed` resolves from the real
   `<repo>/src` via a site-packages path entry, so a shadow-tree run imports the REAL package unless
   `PYTHONPATH` overrides it. We assert `taxembed.__file__` lives inside the shadow before running.
3. **The target test must not SKIP.** Every P2 split test is `skipif(not MOLLUSCA.exists())`, and
   `data/` is untracked -- so a shadow tree without the symlink skips everything and reports 0
   failures, which reads exactly like "the test passed". Skips are counted and refused.
4. **The UNMUTATED shadow must PASS.** Otherwise a failure under mutation proves only that the shadow
   is broken. This is the "re-derive both ways" rule: apply to BASE (must survive), apply to TREE
   (must be caught).

Usage
-----
    python3 helpers/p2_lesion_harness.py --list
    python3 helpers/p2_lesion_harness.py --mutation heldout_exemption
    python3 helpers/p2_lesion_harness.py --all

Exit code 0 means every requested lesion was CAUGHT (and every baseline passed). Non-zero means at
least one test could not fail -- i.e. a real finding, not a harness error.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


@dataclass
class Mutation:
    """One lesion: a single edit that MUST make `tests` fail."""

    name: str
    rel_path: str
    old: str
    new: str
    tests: list[str]
    why: str
    # Usually True: the lesion MUST be caught or the test is not evidence. Set False for a lesion
    # deliberately planted OUTSIDE the code path `tests` covers -- then "not caught" is the CORRECT
    # result and documents a path boundary. Without this field a later reader sees `NOT CAUGHT` and
    # records a coverage hole that does not exist.
    expect_caught: bool = True
    # Untracked paths symlinked into the shadow tree so `tests` actually RUN there.
    #   data/  -- every P2 split test is skipif(not MOLLUSCA.exists()); without it they all skip
    #   .venv/ -- several CLI tests resolve the interpreter as `ROOT/".venv"/"bin"/"python"`
    #             (e.g. tests/eval/test_cophenetic_cli.py:10), so in a shadow tree the subprocess
    #             has no interpreter and the test fails for a reason unrelated to any lesion.
    #             Found by self-check 4: 6 CLI tests failed in an unmutated shadow.
    needs: list[str] = field(default_factory=lambda: ["data", ".venv"])


MUTATIONS: dict[str, Mutation] = {
    # ---- Fix-wave-3 item 1 -------------------------------------------------------------------
    "heldout_exemption": Mutation(
        name="heldout_exemption",
        rel_path="scripts/build_p2_split.py",
        old="hide_deep = deep & ~is_heldout_row & (rng.random(len(pairs)) >= visibility)",
        new="hide_deep = deep & (rng.random(len(pairs)) >= visibility)",
        tests=[
            "tests/eval/test_build_p2_split.py::test_visibility_zero_keeps_parent_edges_plus_heldout_ancestry",
        ],
        why=(
            "Removes the Task-2 plan-defect fix: held-out nodes are no longer exempt from visibility "
            "thinning, so at visibility 0.0 all 28,418 held-out metazoa nodes appear in ZERO training "
            "rows and carry an untrained random embedding -- the arm would measure INITIALIZATION. "
            "The test NAMED for this fix passed with the fix removed, because its assertion was "
            "`set(deep_descendants) <= held_all` and with the fix gone `deep` is empty."
        ),
    ),
    # ---- The same guard's OTHER axes. One lesion per guard is not an enumeration: syntile Task 11
    # ---- built 16 mask mutants and 8 survived the whole suite, of which only 1 was the known one.
    # ---- A guard has an EXISTENCE axis and an EXTENT axis, in both directions.
    "heldout_exemption_partial": Mutation(
        name="heldout_exemption_partial",
        rel_path="scripts/build_p2_split.py",
        old="hide_deep = deep & ~is_heldout_row & (rng.random(len(pairs)) >= visibility)",
        new=("hide_deep = deep & ~(is_heldout_row & (pairs.depth_diff <= 5)) "
             "& (rng.random(len(pairs)) >= visibility)"),
        tests=[
            "tests/eval/test_build_p2_split.py::test_visibility_zero_keeps_parent_edges_plus_heldout_ancestry",
        ],
        why=(
            "UNDER-application at the EXTENT axis: the exemption still exists and still fires only on "
            "held-out rows, but it now covers only depth_diff <= 5, so each held-out node keeps 4 "
            "ancestry rows instead of depth-1. Every held-out node still has a coordinate, so the "
            "existence-style assertions (n_deep > 0, deep_nodes == held_all) all still hold. Only the "
            "manifest identity and the per-node depth-1 count can catch this."
        ),
    ),
    "heldout_exemption_overbroad": Mutation(
        name="heldout_exemption_overbroad",
        rel_path="scripts/build_p2_split.py",
        old="hide_deep = deep & ~is_heldout_row & (rng.random(len(pairs)) >= visibility)",
        new="hide_deep = np.zeros(len(pairs), dtype=bool)",
        tests=[
            "tests/eval/test_build_p2_split.py::test_visibility_zero_keeps_parent_edges_plus_heldout_ancestry",
        ],
        why=(
            "OVER-application: nothing is thinned at all, so RETAINED nodes keep their dd>=2 rows too. "
            "The visibility-0 arm silently becomes a full-closure arm -- the confound P2 exists to "
            "measure. n_deep > 0 passes happily; only the EQUALITY assertion (deep_nodes == held_all) "
            "rejects it. This is the direction `issubset` was blind to in the opposite way."
        ),
    ),
    # ---- Fix-wave-3 item 2 -------------------------------------------------------------------
    # The review flagged `p2_verdict(res)` vs `p2_verdict(res, amendment_2=False)` as a tautology:
    # amendment_2's default IS False, so both sides are byte-identical invocations. The test's stated
    # claim is "the new flag must not rewrite the old one's behaviour" -- these two lesions ask
    # whether it can actually detect the old behaviour being rewritten.
    "p2_frozen_gate_b_amended": Mutation(
        name="p2_frozen_gate_b_amended",
        rel_path="src/taxembed/eval/preregistration.py",
        old="gate_b = bool(loss_drop > -loss_jitter) if amendment_2 else bool(loss_drop > loss_jitter)",
        new="gate_b = bool(loss_drop > -loss_jitter)",
        tests=[
            "tests/eval/test_preregistration.py::TestP2Amendment2"
            "::test_amendment_2_default_false_reproduces_the_original_nine_arm_reading",
        ],
        why=(
            "THE lesion the test claims to catch: the FROZEN (amendment_2=False) reading silently "
            "adopts amendment_2's looser gate (b) -- `loss_drop > -jitter` instead of "
            "`> +jitter`. That is precisely 'the new flag rewriting the old one's behaviour', and it "
            "would admit runs the frozen pre-registration refuses. A tautological "
            "default-vs-explicit-False comparison cannot see it, because BOTH sides move together."
        ),
    ),
    "p2_frozen_control_selection_rewritten": Mutation(
        name="p2_frozen_control_selection_rewritten",
        rel_path="src/taxembed/eval/preregistration.py",
        old="        controls, control_for_arm = _p2_single_shared_control(result, seeds)",
        new="        controls, control_for_arm = _p2_matched_controls(result, seeds, P2_MATCHED_CONTROL)",
        tests=[
            "tests/eval/test_preregistration.py::TestP2Amendment2"
            "::test_amendment_2_default_false_reproduces_the_original_nine_arm_reading",
        ],
        why=(
            "THE CORRECT PROBE for this test's docstring claim. Inside `p2_verdict`, amendment_2 does "
            "NOT touch validity gating -- P2 has its own `p2_validity_gate` and no loss gate at all "
            "(see test_a_rising_training_loss_alone_does_NOT_invalidate_a_run). All amendment_2 does "
            "here is choose matched controls over one shared control. So 'the new flag must not "
            "rewrite the old one's behaviour' means exactly: the FROZEN branch must still select a "
            "single shared control. This lesion rewrites that branch."
        ),
    ),
    "p2_amendment2_default_true": Mutation(
        name="p2_amendment2_default_true",
        rel_path="src/taxembed/eval/preregistration.py",
        # 2026-09-26: re-anchored after amendment_6 was added to the signature. Self-check 1
        # caught the stale text -- a `str.replace` matching nothing would otherwise have
        # "proved" the test survives a lesion that was never introduced.
        old="               amendment_2: bool = False, amendment_4: bool = False,",
        new="               amendment_2: bool = True, amendment_4: bool = False,",
        tests=[
            "tests/eval/test_preregistration.py::TestP2Amendment2"
            "::test_amendment_2_default_false_reproduces_the_original_nine_arm_reading",
        ],
        why=(
            "The other half of the same claim: amendment_2 stops defaulting to False, so every "
            "caller that omits the flag silently gets the 12-arm amended reading. Unlike the lesion "
            "above this one DOES make the two sides of the comparison differ, so the tautology is "
            "not total -- measuring tells us which half of the claim is actually covered."
        ),
    ),
    # ---- Fix-wave-3 item 3 (review I5) ------------------------------------------------------
    # "test_randomdag.py never proves `seed` is used at all -- hard-code default_rng(0) and all its
    # tests pass." Asked separately of BOTH randomisers, because fix wave 2 added a different-seeds
    # test and its own lesion check to `degree_matched_shuffle` only, and the review predates wave 2.
    # Which one is load-bearing matters: amendment 4 swapped the control, so the 12 weekend arms use
    # `degree_matched_shuffle`. `randomize_parents` serves the frozen/amendment_2 RandomDAG path.
    "randomize_parents_ignores_seed": Mutation(
        name="randomize_parents_ignores_seed",
        rel_path="src/taxembed/eval/randomdag.py",
        old=('    """Uniformly resample each non-root node\'s parent from the nodes one level above it."""\n'
             "    parent = np.asarray(parent, dtype=np.int64)\n"
             "    depth = np.asarray(depth, dtype=np.int64)\n"
             "    rng = np.random.default_rng(seed)"),
        new=('    """Uniformly resample each non-root node\'s parent from the nodes one level above it."""\n'
             "    parent = np.asarray(parent, dtype=np.int64)\n"
             "    depth = np.asarray(depth, dtype=np.int64)\n"
             "    rng = np.random.default_rng(0)"),
        tests=["tests/eval/test_randomdag.py"],
        why=(
            "`randomize_parents` silently ignores its `seed` argument and always draws the same "
            "randomisation. Every 'preserves depth / pair count / root' invariant still holds, and "
            "same-seed stability still holds trivially -- so only a DIFFERENT-seeds assertion can "
            "catch it. If this is NOT caught, the three RandomDAG control seeds were never "
            "independent draws, and three 'seeds' would be one run reported three times."
        ),
    ),
    "degree_matched_shuffle_ignores_seed": Mutation(
        name="degree_matched_shuffle_ignores_seed",
        rel_path="src/taxembed/eval/randomdag.py",
        old=("    special case carved out of the general rule.\n"
             '    """\n'
             "    parent = np.asarray(parent, dtype=np.int64)\n"
             "    depth = np.asarray(depth, dtype=np.int64)\n"
             "    rng = np.random.default_rng(seed)"),
        new=("    special case carved out of the general rule.\n"
             '    """\n'
             "    parent = np.asarray(parent, dtype=np.int64)\n"
             "    depth = np.asarray(depth, dtype=np.int64)\n"
             "    rng = np.random.default_rng(0)"),
        tests=["tests/eval/test_randomdag.py"],
        why=(
            "The same lesion on the randomiser the 12 WEEKEND ARMS actually use (amendment 4's "
            "degree-matched control). Expected CAUGHT: fix wave 2 added "
            "test_degree_matched_shuffle_different_seeds_give_different_results plus its own "
            "hard-coded-seed lesion check. This measurement confirms wave 2's claim rather than "
            "trusting the ledger's word for it."
        ),
    ),
    # ---- amendment_6 (2026-09-26, C-A). Five lesions, one per clause the amendment changed:
    # ---- reverting ANY of them silently restores the confound the amendment exists to remove,
    # ---- and each failure mode is different, so one lesion would not enumerate the guard.
    "amendment6_ratio_reverted_to_raw": Mutation(
        name="amendment6_ratio_reverted_to_raw",
        rel_path="src/taxembed/eval/preregistration.py",
        old="        g_cmp, c_cmp = g_nr / g_prior_nr, c_nr / c_prior_nr",
        new="        g_cmp, c_cmp = g_nr, c_nr",
        tests=["tests/eval/test_preregistration.py::TestP2Amendment6"],
        why=(
            "THE amendment_6 lesion: the cross-tree comparison goes back to raw normalized_rank "
            "head-to-head, so the real arm must again beat a number its ~3.8x-easier control gets "
            "for free and GENERALISES becomes structurally unreachable. The flag would still "
            "report amendment_6_applied=True -- the failure is silent in every field except the "
            "verdict itself."
        ),
    ),
    "amendment6_equivalence_clause_dropped": Mutation(
        name="amendment6_equivalence_clause_dropped",
        rel_path="src/taxembed/eval/preregistration.py",
        old=("            and below_chance_level and sign_consistent_nr\n"
             "            and not equivalent_to_control_nr):"),
        new="            and below_chance_level and sign_consistent_nr):",
        tests=["tests/eval/test_preregistration.py::TestP2Amendment6"],
        why=(
            "amendment_6's SECOND correction removed. GENERALISES can then be published on an arm "
            "its own reading calls equivalent_to_control, because `below_control_all_seeds` is a "
            "strict max<min that settles a mathematical tie on the last bit of floating point. "
            "This is the review's EXPECTED case (control improves on its own prior as much as the "
            "arm does), so the lesion is not a corner."
        ),
    ),
    "amendment6_band_reverted_to_absolute": Mutation(
        name="amendment6_band_reverted_to_absolute",
        rel_path="src/taxembed/eval/preregistration.py",
        old=("        equiv_band_nr = max(P2_EQUIV_REL_FRACTION * abs(float(c_cmp.mean())),\n"
             "                            P2_FLOOR_SD_MULTIPLE * pooled_vs_control_nr)"),
        new=("        equiv_band_nr = max(P2_EQUIV_FLOOR,\n"
             "                            P2_FLOOR_SD_MULTIPLE * pooled_vs_control_nr)"),
        tests=["tests/eval/test_preregistration.py::TestP2Amendment6"],
        why=(
            "C-A's second mechanism restored: the MRR-scale 0.01 floor applied to a quantity whose "
            "production values are ~0.04, i.e. a ~25% equivalence band. Measured live in review "
            "scenario S3, where two arms differing by 0.2% read equivalent_to_control=True."
        ),
    ),
    "amendment6_strata_not_scaled": Mutation(
        name="amendment6_strata_not_scaled",
        rel_path="src/taxembed/eval/preregistration.py",
        old=('        diffs_nr = {k: depth_nr["means"][k] / g_prior_nr '
             '- depth_control_nr["means"][k] / c_prior_nr\n'
             "                    for k in shared_nr}"),
        new=('        diffs_nr = {k: depth_nr["means"][k] - depth_control_nr["means"][k]\n'
             "                    for k in shared_nr}"),
        tests=["tests/eval/test_preregistration.py::TestP2Amendment6"],
        why=(
            "A FIX THAT DOES NOT TRAVEL TO ITS SIBLING -- the session's recurring shape. The "
            "aggregate comparison moves to the own-prior ratio scale while the per-stratum sign "
            "test keeps the raw cross-tree difference, so sign_consistent still carries the full "
            "difficulty confound and GENERALISES stays unreachable through that clause alone, "
            "with every aggregate field looking correct."
        ),
    ),
    # ---- C-B (2026-09-26). The FIRST of these is the reviewer's lesion VERBATIM -- C4 #3
    # ---- restored -- which survived the whole 335-test suite on 2026-09-25.
    "control_baselines_fallback_to_the_real_tree": Mutation(
        name="control_baselines_fallback_to_the_real_tree",
        rel_path="src/taxembed/eval/preregistration.py",
        old='        prefixes = [f"baselines_{control_name}", "baselines_randomdag"]',
        new='        prefixes = ["baselines"]',
        tests=["tests/eval/test_preregistration.py", "tests/eval/test_score_p2_cli.py"],
        why=(
            "C4 #3 VERBATIM: every matched control resolves its chance floor and degree prior "
            "from the REAL tree's block. Measured in production terms by review scenario S5 -- "
            "the control publishes 0.13126 (the real tree's) instead of its own 0.18833, with "
            "no error and no _baselines_disagreements, and its gate (b) is then tested against "
            "a floor 23% too low. On 2026-09-25 this was NOT CAUGHT by the named boundary test "
            "NOR by all 335 tests; nothing in the suite protected the control-baselines "
            "resolution. This mutation is the measurement that C-B is actually closed."
        ),
    ),
    "control_baselines_existence_check_removed": Mutation(
        name="control_baselines_existence_check_removed",
        rel_path="src/taxembed/eval/preregistration.py",
        old='    merged["_control_baselines_present"] = assert_control_baselines_exist(merged)',
        new='    merged["_control_baselines_present"] = {}',
        tests=["tests/eval/test_preregistration.py::TestControlBaselinesExistence"],
        why=(
            "The merge-level EXISTENCE check removed. `assert_baselines_agree_across_seeds` "
            "cannot substitute for it: an ABSENT family has nothing to disagree with, which is "
            "precisely why a missing control block sailed through the merge and produced a "
            "computed verdict carrying the wrong tree's floor."
        ),
    ),
    # ---- C-C (2026-09-26): the rolling window.
    "roll_window_not_applied": Mutation(
        name="roll_window_not_applied",
        rel_path="src/taxembed/eval/preregistration.py",
        old="    roll = all_roll[-P2_ROLL_WINDOW:]",
        new="    roll = all_roll",
        tests=["tests/eval/test_preregistration.py::TestP2RollWindowIsApplied"],
        why=(
            "The exact pre-2026-09-26 state: P2_ROLL_WINDOW declared, documented in the "
            "docstring, applied by the TEST FIXTURE, and never applied in production. "
            "`per_run_value` reverts to the mean over every checkpoint the scorer's glob "
            "returned. Measured on a 10-element roll set: 0.68503 instead of 0.82002 and "
            "jitter_sd 0.142284 instead of 0.000171 -> gate (a) fails -> the WHOLE ARRAY reads "
            "UNINFORMATIVE. The old fixture guaranteed exactly 5, so no test could see it."
        ),
    ),
    "roll_window_fix_fully_reverted": Mutation(
        name="roll_window_fix_fully_reverted",
        rel_path="src/taxembed/eval/preregistration.py",
        old=("    roll = all_roll[-P2_ROLL_WINDOW:]\n"
             "    if len(all_roll) != P2_ROLL_WINDOW:"),
        new=("    roll = all_roll\n"
             "    if False:"),
        tests=["tests/eval/test_preregistration.py::TestP2RollWindowIsApplied"],
        why=(
            "THE C-C lesion: BOTH halves reverted together, reproducing the exact "
            "pre-2026-09-26 production state. This is the one that matters -- see "
            "`roll_window_not_applied` below for why the slice alone proves nothing."
        ),
    ),
    "roll_window_count_assert_removed": Mutation(
        name="roll_window_count_assert_removed",
        rel_path="src/taxembed/eval/preregistration.py",
        old="    if len(all_roll) != P2_ROLL_WINDOW:",
        new="    if False:",
        tests=["tests/eval/test_preregistration.py::TestP2RollWindowIsApplied"],
        why=(
            "The window is still applied but the COUNT assert is gone, so a polluted tag "
            "directory is silently trimmed to its trailing 5 instead of refusing. That averages "
            "the right checkpoints while concealing that the directory was never cleared -- the "
            "run is then read as clean when its provenance is not."
        ),
    ),
    "amendment6_zero_prior_guard_removed": Mutation(
        name="amendment6_zero_prior_guard_removed",
        rel_path="src/taxembed/eval/preregistration.py",
        old="    if not np.isfinite(prior) or prior <= 0.0:",
        new="    if False:",
        tests=["tests/eval/test_preregistration.py::TestP2Amendment6"],
        why=(
            "The denominator guard removed. A 0.0 or missing own-prior then divides to inf/nan, "
            "and a NaN comparison is uniformly False -- so every clause reads False and the arm "
            "reads MEMORISES with no error raised. A silent wrong verdict, not a crash."
        ),
    ),
}

# 🧨 PATH-BOUNDARY MARKER, and the reason this field exists at all. `p2_frozen_gate_b_amended` mutates
# `validity_gate`'s loss-based gate (b), which belongs to the TASK 8/9 verdict path. `p2_verdict` never
# calls it -- P2 has its own `p2_validity_gate` and, by ruling, NO loss gate whatsoever. So the P2 test
# CORRECTLY does not catch it, and the only test in the whole suite that does is
# `TestValidityGate::test_a_flat_final_phase_loss_fails_gate_b_under_the_FROZEN_rule` -- a Task 8/9
# test, exactly as it should be.
#
# I got this wrong first time round and it is worth the warning: I planted the lesion believing it
# probed the P2 test's docstring claim, read "6 failed, 327 passed" through a case-sensitive grep that
# hid a `SELF-CHECK 4 FAILED` line, and recorded "the whole suite catches it -- 6 tests fail". The 6
# were the BASELINE's unrelated CLI failures; the mutant had never run. A lesion outside the code path
# under test proves nothing about that test, and a count lifted from the wrong run proves less.
MUTATIONS["p2_frozen_gate_b_amended"] = replace(
    MUTATIONS["p2_frozen_gate_b_amended"], expect_caught=False
)

# 🧨 SECOND PATH-BOUNDARY MARKER (2026-09-26), and this one was found BY the harness, against my
# own expectation. `roll_window_not_applied` reverts ONLY the slice, leaving the count assert in
# place -- and it is NOT CAUGHT, correctly. Given `len(all_roll) == P2_ROLL_WINDOW` (which the
# assert guarantees for every input that gets past it), `all_roll[-P2_ROLL_WINDOW:]` IS
# `all_roll`: the slice is redundant BY CONSTRUCTION, so removing it cannot change any accepted
# input and no test could distinguish the two. That is a fact about the code, not a hole in the
# suite -- the distinction the `expect_caught` field exists for.
#
# The slice is kept anyway, deliberately: it is the belt to the assert's braces, and if anyone
# later relaxes the count check the window would otherwise silently stop being applied, which is
# the original defect returning. `roll_window_fix_fully_reverted` above reverts BOTH halves and
# IS caught -- that is the lesion that measures whether C-C is closed.
MUTATIONS["roll_window_not_applied"] = replace(
    MUTATIONS["roll_window_not_applied"], expect_caught=False
)


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def build_shadow(dest: Path, needs: list[str], source: str = "worktree") -> None:
    """Materialise the repo's tracked files into `dest`, then symlink untracked inputs tests need.

    `source="head"` takes the last commit via `git archive`. `source="worktree"` takes the tracked
    files AS THEY CURRENTLY ARE, including uncommitted edits.

    ⚠ Worktree is the DEFAULT, and deliberately so. The whole point of this harness is to re-verify a
    lesion against a test you have JUST FIXED, and at that moment the fix is uncommitted -- a
    HEAD-based shadow would silently test the OLD test and report a reassuring, meaningless verdict.
    That is the same class of defect the harness exists to find.
    """
    dest.mkdir(parents=True, exist_ok=True)
    if source == "head":
        tar_path = dest.parent / "head.tar"
        r = run(["git", "-C", str(REPO), "archive", "--format=tar", "--output", str(tar_path), "HEAD"])
        if r.returncode != 0:
            raise SystemExit(f"git archive failed: {r.stderr}")
        with tarfile.open(tar_path) as tf:
            tf.extractall(dest)
        tar_path.unlink()
    elif source == "worktree":
        r = run(["git", "-C", str(REPO), "ls-files", "-z"])
        if r.returncode != 0:
            raise SystemExit(f"git ls-files failed: {r.stderr}")
        rels = [p for p in r.stdout.split("\0") if p]
        if not rels:
            raise SystemExit("git ls-files returned nothing -- refusing to build an empty shadow")
        for rel in rels:
            src = REPO / rel
            if not src.exists():          # a staged deletion; skip rather than crash
                continue
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out)
    else:
        raise SystemExit(f"unknown shadow source {source!r}")
    for rel in needs:
        src = REPO / rel
        if not src.exists():
            raise SystemExit(f"shadow needs {rel!r} but {src} does not exist")
        link = dest / rel
        if link.exists() or link.is_symlink():
            shutil.rmtree(link) if link.is_dir() and not link.is_symlink() else link.unlink()
        link.symlink_to(src)


def pytest_env(shadow: Path) -> dict[str, str]:
    env = dict(os.environ)
    # SELF-CHECK 2's mechanism: put the shadow's src FIRST so `import taxembed` cannot reach the real one
    env["PYTHONPATH"] = str(shadow / "src")
    return env


def assert_shadow_imports(shadow: Path) -> None:
    """SELF-CHECK 2: the shadow's own `taxembed` must be what gets imported."""
    r = run(
        [sys.executable, "-c", "import taxembed; print(taxembed.__file__)"],
        cwd=str(shadow), env=pytest_env(shadow),
    )
    resolved = r.stdout.strip()
    if r.returncode != 0 or not resolved:
        raise SystemExit(f"could not import taxembed inside the shadow: {r.stderr}")
    if not resolved.startswith(str(shadow)):
        raise SystemExit(
            "SELF-CHECK 2 FAILED -- the shadow tree imports the REAL package, so any verdict would "
            f"be about unmutated code.\n  taxembed.__file__ = {resolved}\n  shadow = {shadow}"
        )


COUNT_RE = re.compile(r"(\d+) (passed|failed|skipped|error(?:s|ed)?)")


def run_tests(shadow: Path, tests: list[str]) -> tuple[dict[str, int], str]:
    r = run(
        [sys.executable, "-m", "pytest", *tests, "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=str(shadow), env=pytest_env(shadow),
    )
    tail = "\n".join(r.stdout.strip().splitlines()[-25:])
    counts = {kind: int(n) for n, kind in COUNT_RE.findall(r.stdout)}
    return counts, tail


def check(mut: Mutation, keep: bool, source: str = "worktree") -> bool:
    print(f"\n{'=' * 78}\nLESION: {mut.name}\n{'=' * 78}")
    print(f"file   : {mut.rel_path}")
    print(f"tests  : {', '.join(mut.tests)}")
    print(f"source : {source}")
    print(f"why    : {mut.why}\n")

    work = Path(tempfile.mkdtemp(prefix=f"p2lesion_{mut.name}_"))
    shadow = work / "tree"
    try:
        build_shadow(shadow, mut.needs, source)
        assert_shadow_imports(shadow)
        print("  self-check 2 OK: shadow imports the shadow's own taxembed")

        # ---- SELF-CHECK 4: the UNMUTATED shadow must pass -----------------------------------
        base_counts, base_tail = run_tests(shadow, mut.tests)
        if base_counts.get("skipped"):
            raise SystemExit(
                "SELF-CHECK 3 FAILED -- the target test SKIPPED in the shadow tree, so a later "
                f"'0 failures' would mean nothing.\n{base_tail}"
            )
        if base_counts.get("failed") or base_counts.get("error") or not base_counts.get("passed"):
            raise SystemExit(
                f"SELF-CHECK 4 FAILED -- the UNMUTATED shadow does not pass, so nothing can be "
                f"concluded from the mutant.\n{base_tail}"
            )
        print(f"  self-check 3 OK: 0 skipped")
        print(f"  self-check 4 OK: unmutated shadow passes ({base_counts.get('passed')} passed)")

        # ---- SELF-CHECK 1: the mutation must actually apply ---------------------------------
        target = shadow / mut.rel_path
        text = target.read_text()
        if mut.old not in text:
            raise SystemExit(
                "SELF-CHECK 1 FAILED -- the mutation matched NOTHING, so the run would have "
                f"'proved' the test survives a lesion never introduced.\n  looking for: {mut.old!r}"
            )
        occurrences = text.count(mut.old)
        target.write_text(text.replace(mut.old, mut.new))
        print(f"  self-check 1 OK: mutation applied ({occurrences} occurrence(s))")

        # ---- the actual question ------------------------------------------------------------
        mut_counts, mut_tail = run_tests(shadow, mut.tests)
        if mut_counts.get("skipped"):
            raise SystemExit(f"target test SKIPPED under mutation -- verdict void\n{mut_tail}")

        caught = bool(mut_counts.get("failed") or mut_counts.get("error"))
        print()
        if caught:
            print(f"  ✅ CAUGHT -- the test FAILS on the lesion. It can fail; it is evidence.")
            print(f"     {mut_counts}")
            # WHICH tests caught it is the load-bearing detail, not the count. A lesion planted
            # outside the code path under test can be "caught" by unrelated tests and still prove
            # nothing about the test you were asking about -- so name them.
            for line in mut_tail.splitlines():
                if line.startswith("FAILED") or line.startswith("ERROR"):
                    print(f"       {line}")
        else:
            print(f"  🛑 NOT CAUGHT -- the test PASSES with the defect present. IT CANNOT FAIL.")
            print(f"     {mut_counts}")
            print(f"\n{mut_tail}")
        return caught
    finally:
        if keep:
            print(f"\n  shadow kept at {shadow}")
        else:
            shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mutation", action="append", default=[], help="lesion name (repeatable)")
    ap.add_argument("--all", action="store_true", help="run every registered lesion")
    ap.add_argument("--list", action="store_true", help="list registered lesions and exit")
    ap.add_argument("--keep", action="store_true", help="keep the shadow tree for inspection")
    ap.add_argument("--source", choices=("worktree", "head"), default="worktree",
                    help="build the shadow from the current tracked files (default) or from HEAD. "
                         "Use 'head' to reproduce a defect as committed; 'worktree' to re-verify a "
                         "fix that is not committed yet.")
    ap.add_argument("--json", type=Path, help="write a machine-readable verdict record here")
    ap.add_argument("--tests", action="append", default=[],
                    help="override the lesion's own test target(s). Use `--tests tests` to ask the "
                         "DIFFERENT question 'does the whole suite catch this?' rather than 'does "
                         "the named test catch this?' -- a hole in one test is not a hole in the "
                         "suite, and the two are worth telling apart before adding coverage.")
    args = ap.parse_args()

    if args.list:
        for name, m in MUTATIONS.items():
            print(f"{name}\n    {m.rel_path}\n    {', '.join(m.tests)}\n")
        return 0

    names = list(MUTATIONS) if args.all else args.mutation
    if not names:
        ap.error("give --mutation NAME, or --all, or --list")
    unknown = [n for n in names if n not in MUTATIONS]
    if unknown:
        ap.error(f"unknown lesion(s): {unknown}; --list shows the registered ones")

    muts = []
    for n in names:
        m = MUTATIONS[n]
        if args.tests:
            m = replace(m, tests=list(args.tests))
        muts.append((n, m))
    results = {n: check(m, args.keep, args.source) for n, m in muts}

    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    bad = []
    for n, caught in results.items():
        expected = MUTATIONS[n].expect_caught
        ok = caught == expected
        note = "" if expected else "   (expected NOT caught -- path boundary, not a hole)"
        print(f"  {'CAUGHT     ' if caught else 'NOT CAUGHT '} {'ok  ' if ok else 'BAD '} {n}{note}")
        if not ok:
            bad.append(n)
    n_bad = len(bad)
    print(f"\n{len(results) - n_bad}/{len(results)} lesions gave their EXPECTED verdict.")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n")
        print(f"verdict record: {args.json}")
    if n_bad:
        print(f"🛑 {n_bad} lesion(s) did NOT match expectation: {', '.join(bad)}")
        print("   For an expect_caught=True lesion this means the test CANNOT FAIL -- a finding, not")
        print("   a harness error. For expect_caught=False it means a path boundary moved.")
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
