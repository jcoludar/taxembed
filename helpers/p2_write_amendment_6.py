#!/usr/bin/env python3
"""Write `p2_amendment_6_20260926` into results/p2_heldout_preregistration.json.

ADDITIVE ONLY. Refuses to run if the key already exists (an amendment is written once; a
correction to one is a NEW amendment, never an edit of the old -- the same append-only posture
`*_VERDICT.md` files have). Refuses if any prior block would change. Prints a diff summary of
exactly which top-level keys it added, and re-reads the file afterwards to confirm every prior
amendment is byte-identical to what was there before.

Run once. Re-running is a no-op that exits 1.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
PREREG = _REPO / "results" / "p2_heldout_preregistration.json"
KEY = "p2_amendment_6_20260926"

BLOCK = {
    "status": (
        "WRITTEN 2026-09-26, BEFORE any P2 array was submitted and BEFORE any P2 outcome data "
        "had been observed. THE QUEUE WAS EMPTY AND NO GPU HAD BEEN SPENT ON P2 WHEN THIS WAS "
        "WRITTEN (verified: `squeue --me` returned no rows). It implements a USER DESIGN "
        "DECISION taken 2026-09-25 -- the same posture as p2_amendment_2_20260924 and "
        "p2_amendment_4_20260924, which were also USER design decisions recorded before any "
        "run -- and is therefore behind a p2_verdict(..., amendment_6=True) flag, leaving every "
        "earlier reading reachable and unchanged. Additive only: the frozen block and every "
        "prior amendment are left completely intact."
    ),
    "the_user_decision": (
        "2026-09-25, USER: compare each arm against ITS OWN TREE'S degree prior (the scorer "
        "already writes both), and make the normalized_rank equivalence band RELATIVE instead "
        "of the absolute MRR-scale P2_EQUIV_FLOOR = 0.01."
    ),
    "the_defect": {
        "summary": (
            "GENERALISES was STRUCTURALLY UNREACHABLE under the reading production computes "
            "(amendment_1 + amendment_2 + amendment_4). amendment_1 moved the cross-tree "
            "comparison to normalized_rank, which removes the POOL-SIZE (fan-out) confound. It "
            "does not remove the TASK-DIFFICULTY confound: the real tree and the degree-matched "
            "control have different training-free floors, and the control's is far better. A "
            "real arm therefore had to beat, on raw normalized_rank, a number its control "
            "obtains for free."
        ),
        "measured_on_the_production_splits": {
            "source": "helpers/p2_review3_measure_production_splits.py",
            "clade": "metazoa_33208_clean, 498,246 nodes, 28,418 held-out nodes per seed",
            "first_measured": "2026-09-25 (final-review-wave3)",
            "independently_re_derived": (
                "2026-09-26, before this amendment was written -- the numbers below drive a "
                "pre-registration document, so they were not taken on the reviewer's word"
            ),
            "degree_prior_normalized_rank_real": [0.15381, 0.15317, 0.15320],
            "degree_prior_normalized_rank_degmatch": [0.04041, 0.03824, 0.03851],
            "ratio": (
                "the control's task is ~3.8-4.0x EASIER for a ranker that learned NOTHING"
            ),
            "leaky_upper_bound": (
                "A checkpoint that trained on the full closure -- it saw every held-out parent "
                "edge -- scores normalized_rank 0.04049 on the real tree "
                "(helpers/p2_review3_can_a_real_model_clear_the_control.py). That is an UPPER "
                "BOUND on any honest P2 arm, and it sits 10% BELOW the control's TRAINING-FREE "
                "floor of 0.04494. A trained control goes lower still. 130-230 GPU-hours would "
                "have bought a MEMORISES caused by the control's construction."
            ),
            "direction_of_bias": (
                "AGAINST our method -- a false MEMORISES, the exact mirror of the false "
                "GENERALISES this review chain has been chasing. Unflattering, and still a "
                "wrong published conclusion."
            ),
        },
    },
    "the_amendment": {
        "rule_1_own_prior_ratio": (
            "Every CROSS-TREE quantity is read as a RATIO to the arm's OWN tree's training-free "
            "degree prior: (arm_nr / arm_own_prior_nr) vs (control_nr / control_own_prior_nr). "
            "1.0 is exactly 'no better than what this tree hands out for free' on either side, "
            "so the comparison asks which model improved MORE on its own baseline, and is "
            "invariant to the difficulty gap by construction. The scorer already emits a "
            "degree_prior block per --baselines-key, so the control's own prior was already on "
            "disk; the reading simply never used it."
        ),
        "rule_2_relative_equivalence_band": (
            "The normalized_rank equivalence band becomes max(P2_EQUIV_REL_FRACTION * "
            "|control_mean|, P2_FLOOR_SD_MULTIPLE * pooled_sd) with P2_EQUIV_REL_FRACTION = "
            "0.05, replacing max(P2_EQUIV_FLOOR, ...). P2_EQUIV_FLOOR = 0.01 was calibrated on "
            "MRR (~0.8 in production, so ~1.2%); applied unchanged to normalized_rank (~0.04) "
            "it was ~25% of the compared quantity. Measured consequence, review scenario S3: "
            "two arms differing by 0.2% read equivalent_to_control=True on |mean_diff| = "
            "0.00008, i.e. an automatic MEMORISES."
        ),
        "rule_3_generalises_requires_distinguishability": (
            "GENERALISES now additionally requires `not equivalent_to_control`. FOUND WHILE "
            "IMPLEMENTING THIS AMENDMENT, pre-run, by helpers/p2_amendment6_reachability.py "
            "scenario R2: `below_control_all_seeds` is a strict max<min, so a mathematical TIE "
            "is settled by the last bit of floating point. On the review's EXPECTED case -- arm "
            "and control improving on their own priors by the SAME factor -- the engine returned "
            "GENERALISES while reporting |mean_diff| = 0.000000 and equivalent_to_control = True "
            "in the same dict. The MEMORISES branch does test that boolean, but it is an elif "
            "and never ran. A verdict that contradicts its own reported booleans is not a "
            "reading."
        ),
        "rule_4_mixed_prose_names_the_binding_clause": (
            "The MIXED verdict string is now derived from the clause that actually bound. FOUND "
            "pre-run by the same probe, scenario R3: the MIXED branch is reachable for three "
            "different reasons (a depth-stratum sign flip, the arm failing to beat its control "
            "on every seed, or normalized_rank not clearing 0.5) but its prose asserted the "
            "first unconditionally. On an arm uniformly WORSE than its control there is no flip "
            "anywhere and the published sentence was simply untrue. A verdict string is quoted "
            "into the manuscript."
        ),
        "scope": (
            "The amendment_1 branch only. The frozen amendment_1=False raw-MRR reading carries "
            "the same rule_3 structure and is DELIBERATELY NOT CHANGED: it exists solely so the "
            "original 2026-09-24 freeze stays byte-identically reproducible, and it is never the "
            "published reading. Recorded here rather than silently fixed."
        ),
    },
    "stated_limitations": {
        "strata_use_the_aggregate_prior": (
            "The per-depth-stratum sign test applies the same own-prior scaling, but the scorer "
            "emits degree_prior only as an AGGREGATE (scripts/score_p2_linkpred.py writes one "
            "block per --baselines-key, with no by_depth breakdown), so each side is divided by "
            "its tree's aggregate prior rather than a per-stratum one. This is a uniform "
            "positive rescaling of each side: it cannot reorder strata within an arm, and what "
            "it does is relax the cross-tree sign test by the measured difficulty ratio. A "
            "per-stratum prior would be sharper and is NOT claimed."
        ),
        "amendment_6_requires_amendment_1": (
            "p2_verdict RAISES on amendment_6=True with amendment_1=False rather than silently "
            "ignoring the flag -- a flag that does nothing is how a pre-registration stops "
            "describing the computation it names."
        ),
    },
    "corrects_amendment_4": (
        "p2_amendment_4_20260924's `the_replacement` states the degree-matched shuffle preserves "
        "'the chance floor and the degree prior ... by construction'. MEASURED, that is false "
        "for the degree prior: per-node fan-out IS preserved elementwise (0 of 498,246 differ, "
        "confirmed at metazoa scale), but the degree PRIOR's normalized_rank is 0.1538 on the "
        "real tree vs 0.0404 on the control. Against the RETIRED RandomDAG control the same "
        "mismatch was 0.1538 vs 0.2487. In |log-ratio| the mismatch is 1.34 under the live "
        "control vs 0.48 under RandomDAG: amendment_4 made the degree-prior gap 2.8x LARGER in "
        "magnitude while FLIPPING ITS SIGN. The flip is what closes C1 (a fan-out-only model can "
        "no longer clear the control) and is a genuine improvement; the growth is what made "
        "GENERALISES unreachable. amendment_4's `the_result_is_an_improvement_not_full_parity` "
        "block concedes the chance-floor residual (1.84x on mollusca; 1.39-1.44x measured at "
        "metazoa) but records nothing about the degree-prior gap, and that omission is what hid "
        "this finding until the final review."
    ),
    "corrects_amendment_3": (
        "p2_amendment_3_20260924's c3 block asserts that chance_mrr_mean 'inherits the [k=1] "
        "exclusion for free'. It does not: chance_mrr_mean averages H_k/k over ALL k INCLUDING "
        "k=1, while every metric it anchors excludes k=1. Measured on the production splits: "
        "the as-coded value is +10.5% (real) and +23.1% (degmatch) above the k>=2 value, and "
        "sibling_chance_mean +30.6%. The k=1-inclusive behaviour is PINNED BY A TEST and is "
        "conservative for our own margin, but it inflates the CONTROL's gate-(b) floor by ~23%, "
        "which is a live UNINFORMATIVE risk rather than a safe direction. Corrected here as a "
        "statement of record; the behaviour is left as coded and tested."
    ),
    "verification": {
        "reachability_probe": (
            "helpers/p2_amendment6_reachability.py -- 5 scenarios through the REAL p2_verdict on "
            "measured baselines. It asks BOTH questions, because only the pair is evidence: "
            "GENERALISES is now reachable for an arm that out-improves its control on the "
            "own-prior scale (R1, R5), and is still REFUSED when the two improve equally (R2), "
            "when the control improves more (R3), and when the arm learned nothing beyond "
            "fan-out (R4). 5/5."
        ),
        "tests": (
            "tests/eval/test_preregistration.py::TestP2Amendment6 -- 12 tests. Suite 335 -> 347."
        ),
        "lesions": (
            "helpers/p2_lesion_harness.py mutations amendment6_ratio_reverted_to_raw, "
            "amendment6_equivalence_clause_dropped, amendment6_band_reverted_to_absolute, "
            "amendment6_strata_not_scaled, amendment6_zero_prior_guard_removed -- 5/5 CAUGHT, "
            "each with the harness's self-check 4 confirming the UNMUTATED shadow passes first."
        ),
    },
}


def main() -> int:
    original_text = PREREG.read_text()
    prereg = json.loads(original_text)

    if KEY in prereg:
        print(f"REFUSED: {KEY} already exists. An amendment is written once; a correction to one "
              "is a NEW amendment, never an edit of the old.")
        return 1

    before = {k: json.dumps(v, sort_keys=True) for k, v in prereg.items()}

    # TEXTUAL insertion, not json.dumps of the whole document. A first attempt re-serialised the
    # file with indent=2 and git reported 62 insertions / 39 DELETIONS: the original wraps some
    # lists onto a single line and carries blank lines that a plain dumper does not reproduce.
    # The parsed content was identical, so the content-level check below passed while 39 lines of
    # the FROZEN block and prior amendments had silently moved. "Left completely intact" is a
    # claim about bytes in a pre-registration, so the new key is spliced in as text and every
    # prior byte is preserved exactly -- asserted below, not assumed.
    closing = original_text.rstrip()
    if not closing.endswith("}"):
        print("REFUSED: the file does not end with '}' -- not the shape this splice assumes.")
        return 1
    body = closing[:-1].rstrip()
    if not body.endswith("}"):
        print("REFUSED: cannot find the last top-level value to append after.")
        return 1
    rendered = json.dumps({KEY: BLOCK}, indent=2)
    # strip the wrapper braces of the one-key document and re-indent to top level
    inner = rendered[rendered.index("\n") + 1: rendered.rindex("\n")]
    new_text = body + ",\n" + inner + "\n}\n"
    PREREG.write_text(new_text)

    # Re-read from disk: content additive AND every prior byte unmoved.
    after_text = PREREG.read_text()
    after_doc = json.loads(after_text)
    after = {k: json.dumps(v, sort_keys=True) for k, v in after_doc.items()}
    changed = [k for k in before if before[k] != after.get(k)]
    added = [k for k in after if k not in before]
    prefix_preserved = after_text.startswith(body)

    print(f"added            : {added}")
    print(f"changed          : {changed or 'none'}")
    print(f"prior bytes kept : {prefix_preserved}")
    if changed or added != [KEY] or not prefix_preserved:
        print("FAILED: the write was not purely additive. Restore with "
              "`git checkout -- results/p2_heldout_preregistration.json`.")
        return 1
    print(f"OK: {KEY} written, {len(after)} top-level keys, every prior byte identical.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
