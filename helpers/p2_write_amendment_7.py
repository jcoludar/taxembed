#!/usr/bin/env python3
"""Write `p2_amendment_7_20260926` into results/p2_heldout_preregistration.json.

Same append-only, byte-preserving discipline as helpers/p2_write_amendment_6.py: refuses if the
key exists, splices textually rather than re-serialising, and asserts every prior byte survives.

WHY A SEVENTH AMENDMENT RATHER THAN AN EDIT TO THE SIXTH. amendment_6 was written earlier the
same day and contains two provenance errors and several claims the code no longer matches. It is
tempting to fix them in place -- nothing has run, the file is hours old. That is exactly the edit
an append-only record exists to refuse: amendment_6 is already committed (24214db) and the whole
value of the chain is that a reader can see what was believed WHEN, including what was believed
wrongly. amendment_6 corrects amendment_4 and amendment_3 without editing either; this follows
the same shape.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
PREREG = _REPO / "results" / "p2_heldout_preregistration.json"
KEY = "p2_amendment_7_20260926"

BLOCK = {
    "status": (
        "WRITTEN 2026-09-26, hours after p2_amendment_6_20260926 and still BEFORE any P2 array "
        "was submitted, before any P2 outcome data existed, and with the LRZ queue empty. It "
        "arises from a SECOND adversarial review, of amendment_6 itself. Additive only: "
        "amendment_6 and every earlier block are left byte-identical, and the corrections below "
        "supersede specific claims in them rather than editing them -- the same posture "
        "amendment_6 took towards amendment_4 and amendment_3."
    ),
    "corrects_amendment_6_provenance": {
        "the_error": (
            "p2_amendment_6_20260926.the_defect.measured_on_the_production_splits declares "
            "clade 'metazoa_33208_clean, 498,246 nodes, 28,418 held-out nodes per seed' and then "
            "files a `leaky_upper_bound` claim under it that is measured on MOLLUSCA. Both of "
            "its numbers come from helpers/p2_review3_can_a_real_model_clear_the_control.py, "
            "which hard-codes results/task9_runs_mollusca/echino_canonical_s0_epoch200.pth and "
            "the mollusca_6447_clean manifest/heldout."
        ),
        "what_is_true": (
            "The leaky checkpoint's normalized_rank 0.04049 and the 'control training-free floor "
            "0.04494' it is compared against are BOTH mollusca_6447_clean seed 0. The mollusca "
            "real-tree own-prior is 0.16724, not 0.15381. Against the METAZOA control prior "
            "recorded in the same block (0.04041) the leaky value 0.04049 sits 0.2% ABOVE, not "
            "'10% BELOW' -- the comparison as written splices two clades."
        ),
        "does_it_change_the_conclusion": (
            "No, and that is worth stating explicitly rather than leaving implied. The argument "
            "amendment_6 rests on is the DEGREE-PRIOR GAP between the real and degree-matched "
            "trees, which was measured on the production metazoa splits and independently "
            "re-derived on 2026-09-26 (0.15381/0.15317/0.15320 vs 0.04041/0.03824/0.03851). The "
            "leaky-ceiling figure is corroboration, not the basis. But a pre-registration that "
            "misattributes its own measurement provenance is the worst defect available in this "
            "artefact class, so it is corrected in the record."
        ),
        "also_affects": (
            "helpers/p2_amendment6_reachability.py computes leak_factor = LEAKY_NR / REAL_DP "
            "with LEAKY_NR mollusca (0.040492) and REAL_DP metazoa (0.15381). The mollusca "
            "own-prior is 0.16724, so the reported improvement factor 0.2633 overstates by "
            "~8.8%. It is used only to pick scenario R5's arm value, so it changes no verdict, "
            "but the probe's own comment should not be read as a measured quantity."
        ),
    },
    "corrects_amendment_6_test_counts": (
        "p2_amendment_6_20260926.verification.tests says 'TestP2Amendment6 -- 12 tests. Suite "
        "335 -> 347.' The class is 12 tests, but amendment_6 shipped 20 (TestP2Amendment6Artifact "
        "adds 8) and the suite it shipped in was 365, not 347. '335 -> 347' describes an "
        "intermediate state that was never committed."
    ),
    "generalises_no_longer_implies_out_ranking_the_control": {
        "the_disclosure": (
            "THE MOST IMPORTANT ENTRY IN THIS AMENDMENT. Under amendment_6 the cross-tree "
            "comparison is a RATIO to each tree's own training-free degree prior, so GENERALISES "
            "means 'the arm improved MORE on its own tree's prior than the control did on its "
            "own' -- it does NOT mean 'the arm ranked the true parent better than the control "
            "did'. Measured through the real engine: an arm whose control out-ranks it 2.25x on "
            "raw normalized_rank reads GENERALISES, and so does an arm scoring EXACTLY what its "
            "control scores. The GENERALISES window extends to a raw ratio of about 3.6x, which "
            "is the measured difficulty gap, so this is a wide region and not a corner case."
        ),
        "why_this_is_the_intended_reading_and_still_needs_saying": (
            "It IS the question the design intends: the control's task is ~3.8x easier for a "
            "training-free ranker, so equal raw scores mean the arm did more. That is precisely "
            "what amendment_6 exists to express. But the verdict's own prose ('the model "
            "predicts relations it never saw') reads to any ordinary reader as a claim that the "
            "arm beat its control outright, and a manuscript sentence built on it would be "
            "wrong in a way nobody could detect from the published verdict."
        ),
        "what_changed": (
            "The reading now always carries `raw_mean_diff_vs_control`, `arm_beats_control_raw`, "
            "`arm_values_raw_normalized_rank` and `control_values_raw_normalized_rank`, and "
            "scripts/apply_preregistration.py prints them beside the ratio values, with an "
            "explicit warning line when GENERALISES is returned while arm_beats_control_raw is "
            "False. NOTHING IS GATED ON THEM: whether the raw comparison should also have to "
            "pass is a design decision for the USER, and taking it silently here would be the "
            "same class of error as the one being corrected."
        ),
    },
    "per_seed_own_prior_denominators": (
        "CORRECTION to amendment_6 as implemented. It divided all three seeds by SEED 0's "
        "own-prior, because p2_group_stats' group-level `degree_prior` comes from seeds[0]. That "
        "was defensible before amendment_6 -- the function's docstring argues the group-level "
        "block is cosmetic because each seed's GATE already uses its own dict -- and amendment_6 "
        "falsified it by making that value the DENOMINATOR of the published statistic. Measured "
        "on the project's own per-seed priors the choice alone flips the verdict (MEMORISES with "
        "the seed-0 denominator, GENERALISES with per-seed ones): the control's cross-seed prior "
        "spread is ~5.4%, the same order as the 5% decision band it feeds. Each seed now divides "
        "by its own seed's prior; the seed-0 scalar is retained for display only."
    ),
    "per_stratum_own_prior_denominators": {
        "the_correction": (
            "amendment_6's stated_limitations.strata_use_the_aggregate_prior defended dividing "
            "every depth stratum by the AGGREGATE prior on the grounds that it 'cannot reorder "
            "strata within an arm'. True, and beside the point: sign_consistent is a CROSS-ARM "
            "test, so what matters is whether the aggregate difficulty ratio equals the "
            "per-stratum one. It does not."
        ),
        "measured": (
            "Production metazoa, seed 0, independently re-derived 2026-09-26 "
            "(helpers/p2_verify_stratum_prior_emission.py): per-stratum degree-prior "
            "normalized_rank real 0.17640 / 0.16927 / 0.14286 vs control 0.06229 / 0.04138 / "
            "0.03419, giving true difficulty ratios 2.8318 / 4.0910 / 4.1786 against an "
            "AGGREGATE of 3.8064. Stratum 11-15 is therefore OVER-ALLOWED by 1.344x, on 4,751 "
            "scored queries (9.5x P2_MIN_STRATUM_N) -- enough to read a genuine per-stratum tie "
            "as a comfortable win, in the GENERALISES direction."
        ),
        "the_fix": (
            "scripts/score_p2_linkpred.py now emits degree_prior.by_depth, stratified with the "
            "same bins and the same reducer a checkpoint's ranks get, and _p2_depth_means "
            "divides each stratum by that stratum's own prior. The reading records "
            "depth_strata.stratum_prior_basis, including used_per_stratum_priors, so a fallback "
            "to the aggregate on an older scorer output can never be silent."
        ),
    },
    "control_baselines_may_not_be_borrowed_from_another_tree": (
        "amendment_6's C-B work removed the real tree's 'baselines' from the MATCHED controls' "
        "fallback chain but left the identical defect in _p2_single_shared_control one "
        "definition above, where ELEVEN tests pinned it -- the same 'the fixtures encoded the "
        "defect' finding, one function over, unacted on. It also left 'baselines_randomdag' in "
        "the matched chain, which for a degmatch control is a THIRD TREE: with the degmatch keys "
        "absent and a stale RandomDAG block present, degmatch_vis00's degree prior resolved to "
        "0.2487 instead of ~0.0404, a 6.2x wrong amendment_6 denominator, silently. Both chains "
        "now end inside the control's own family and raise by name otherwise."
    ),
    "supersedes": (
        "This block supersedes, in p2_amendment_6_20260926: the `leaky_upper_bound` text's clade "
        "attribution and its '10% BELOW' comparison; the `verification.tests` counts; and the "
        "`stated_limitations.strata_use_the_aggregate_prior` defence, which is now moot because "
        "per-stratum priors are computed. Every other claim in amendment_6 was checked against "
        "the implementation in the same review and held, including its 5/5 lesion claim, its "
        "reachability 5/5, its '44 insertions and ZERO deletions', and its corrects_amendment_4 "
        "arithmetic (ln(0.1538/0.0404)=1.337 vs ln(0.2487/0.1538)=0.481, ratio 2.78 ~ '2.8x')."
    ),
    "verification": (
        "Suite 365 -> 377. 22/22 registered lesions give their expected verdict, including four "
        "new ones for the corrections above (shared_control_falls_back_to_the_real_tree, "
        "degmatch_control_falls_back_to_randomdag, "
        "amendment6_uses_seed_zero_prior_for_every_seed, strata_use_the_aggregate_prior). Two "
        "documented path boundaries where NOT CAUGHT is the correct verdict. The per-stratum "
        "numbers above were re-derived from the production splits by this session rather than "
        "taken from the review that reported them."
    ),
}


def main() -> int:
    original_text = PREREG.read_text()
    prereg = json.loads(original_text)

    if KEY in prereg:
        print(f"REFUSED: {KEY} already exists. An amendment is written once.")
        return 1

    before = {k: json.dumps(v, sort_keys=True) for k, v in prereg.items()}

    closing = original_text.rstrip()
    if not closing.endswith("}"):
        print("REFUSED: the file does not end with '}'.")
        return 1
    body = closing[:-1].rstrip()
    if not body.endswith("}"):
        print("REFUSED: cannot find the last top-level value to append after.")
        return 1
    rendered = json.dumps({KEY: BLOCK}, indent=2)
    inner = rendered[rendered.index("\n") + 1: rendered.rindex("\n")]
    PREREG.write_text(body + ",\n" + inner + "\n}\n")

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
        print("FAILED: not purely additive. Restore with "
              "`git checkout -- results/p2_heldout_preregistration.json`.")
        return 1
    print(f"OK: {KEY} written, {len(after)} top-level keys, every prior byte identical.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
