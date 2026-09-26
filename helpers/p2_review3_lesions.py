#!/usr/bin/env python3
"""REVIEW WAVE 3: additional lesions, registered as DATA on top of `helpers/p2_lesion_harness.py`.

Extends the existing registry without editing it (the harness's own file is left byte-identical).
Every mutation runs in the harness's shadow tree with all four of its self-checks.

Each lesion below targets a guard whose NAME claims to protect a specific defect, chosen by asking
"which single edit would put a WRONG NUMBER in the paper, and what would notice?" -- not by picking
tests that look thin.

  <python> helpers/p2_review3_lesions.py --list
  <python> helpers/p2_review3_lesions.py --all
  <python> helpers/p2_review3_lesions.py --mutation <name> --tests tests     # whole-suite question
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import p2_lesion_harness as H  # noqa: E402

PREREG = "src/taxembed/eval/preregistration.py"
CLI_E2E = ("tests/eval/test_score_p2_cli.py::"
           "test_degmatch_and_seed_tagged_baselines_flow_through_merge_and_amendment_4_end_to_end")

NEW = {
    # ---- the C4 boundary: is the ONE test that runs the PRODUCTION flag combination able to fail?
    "control_baselines_fallback_to_the_real_tree": H.Mutation(
        name="control_baselines_fallback_to_the_real_tree",
        rel_path=PREREG,
        old='        prefixes = [f"baselines_{control_name}", "baselines_randomdag", "baselines"]',
        new='        prefixes = ["baselines"]',
        tests=[CLI_E2E],
        why=(
            "C4 #3 RESTORED: every matched control resolves its floor from the REAL tree's "
            "`baselines` block instead of its own. This is the exact defect the final review "
            "called Critical -- 'the fallback silently gives each control the REAL tree's floor'. "
            "Measured on the production metazoa splits the two floors differ by 1.43x "
            "(chance_mrr_mean 0.2711 real vs 0.3229 degmatch), so the control's gate (b) would be "
            "tested against a floor 23% too low and `randomdag_sibling_chance_mean` would be "
            "PUBLISHED as the real tree's 0.1313 instead of the control's 0.1883."
        ),
    ),
    "amendment_1_ignored_in_the_reading": H.Mutation(
        name="amendment_1_ignored_in_the_reading",
        rel_path=PREREG,
        old="    if not amendment_1:",
        new="    if True:",
        tests=[CLI_E2E],
        why=(
            "The cross-tree comparison silently reverts to RAW MRR -- the comparison "
            "p2_amendment_1_20260924 rule_1 forbids and p2_amendment_4_20260924 explicitly "
            "re-affirms as still load-bearing (residual chance-floor gap 1.84x). Asked of the ONE "
            "test that runs the production flag combination (amendment_1+2+4), because that test's "
            "verdict assertion is `v['verdict'] in {the four possible verdicts}`."
        ),
    ),
    "matched_controls_cross_matched": H.Mutation(
        name="matched_controls_cross_matched",
        rel_path=PREREG,
        old='    "vis00": "degmatch_vis00",                    # RandomDAG RETIRED as P2\'s control -- the',
        new='    "vis00": "degmatch_vis50",                    # RandomDAG RETIRED as P2\'s control -- the',
        tests=[CLI_E2E],
        why=(
            "amendment_2 rule_1 / amendment_4 rule_1: 'vis00 is read ONLY against degmatch_vis00 "
            "... never crossed'. This crosses it. The vis50 side has a load-bearing test "
            "(test_cross_matching_the_degmatch_controls_changes_the_verdict); this asks whether the "
            "vis00 side does too."
        ),
    ),
    # ---- the three amendment_3 measurement fixes, each asked of its own named guard -------------
    "c1_degree_prior_clause_removed_normalized_rank_branch": H.Mutation(
        name="c1_degree_prior_clause_removed_normalized_rank_branch",
        rel_path=PREREG,
        old='    below_degree_prior = bool(g_nr.max() < degree_prior["normalized_rank"])',
        new="    below_degree_prior = True",
        tests=["tests/eval/test_preregistration.py::TestP2DegreePriorClause"],
        why=(
            "C1's clause deleted from the amendment_1=True branch -- the branch PRODUCTION reads. "
            "Without it a training-free degree prior reads GENERALISES, which is the finding the "
            "whole of amendment_3 exists for."
        ),
    ),
    "c2_gate_b_reverted_to_the_hits_at_1_chance_rate": H.Mutation(
        name="c2_gate_b_reverted_to_the_hits_at_1_chance_rate",
        rel_path=PREREG,
        old='    gate_b = bool(rv["per_run_value"] > chance_mrr_mean)',
        new='    gate_b = bool(rv["per_run_value"] > sibling_chance_mean)',
        tests=["tests/eval/test_preregistration.py::TestP2ChanceMRRGateAnchor"],
        why="C2 RESTORED: an MRR-valued gate compared to the hits@1 chance rate (2.07x too low).",
    ),
    "c3_pool_size_one_exclusion_removed": H.Mutation(
        name="c3_pool_size_one_exclusion_removed",
        rel_path="src/taxembed/eval/linkpred.py",
        old="    trivial = n_candidates <= 1.0",
        new="    trivial = np.zeros(len(n_candidates), dtype=bool)",
        tests=["tests/eval/test_linkpred.py"],
        why="C3 RESTORED: pool-size-1 free wins counted in every metric again.",
    ),
    # ---- the gate (a) anchor, and the strictness of the all-below-all rule ----------------------
    "gate_a_anchor_dropped": H.Mutation(
        name="gate_a_anchor_dropped",
        rel_path=PREREG,
        old='    rise = rv["milestone_mrr_max"] - chance_mrr_mean',
        new='    rise = rv["milestone_mrr_max"]',
        tests=["tests/eval/test_preregistration.py::TestP2GateAAnchorIsChanceMRR",
               "tests/eval/test_preregistration.py::TestP2ValidityGate"],
        why=(
            "Gate (a)'s anchor deleted entirely, so 'did this run learn anything' becomes 'is its "
            "best MRR big'. I3 changed this anchor from the run's own epoch-10 value to "
            "chance_mrr_mean; the frozen block still declares it as `max_t MRR(t) - MRR(init)`. "
            "If nothing catches an anchor of ZERO, nothing is really testing the anchor."
        ),
    ),
    "all_below_all_weakened_to_means": H.Mutation(
        name="all_below_all_weakened_to_means",
        rel_path=PREREG,
        old='    below_control_all_seeds = bool(g_nr.max() < c_nr.min())      # mirrors above_control_all_seeds',
        new='    below_control_all_seeds = bool(g_nr.mean() < c_nr.mean())',
        tests=["tests/eval/test_preregistration.py::TestP2Amendment1",
               "tests/eval/test_preregistration.py::TestP2Amendment4"],
        why=(
            "amendment_1 rule_3: 'no run of the arm may be at or above any run of the control'. "
            "Weakened to a comparison of MEANS, which admits an arm whose worst seed is worse than "
            "the control's best -- exactly what the all-above-all rule exists to refuse."
        ),
    ),
    # ---- does any test PIN the k=1-inclusive chance level? (the reverse direction) --------------
    "chance_mrr_mean_excludes_k1": H.Mutation(
        name="chance_mrr_mean_excludes_k1",
        rel_path="src/taxembed/eval/baselines.py",
        old="    max_k = int(n_candidates.max())",
        new=("    n_candidates = n_candidates[n_candidates >= 2]\n"
             "    if len(n_candidates) == 0:\n"
             "        return float('nan')\n"
             "    max_k = int(n_candidates.max())"),
        tests=["tests/eval/test_baselines.py"],
        why=(
            "THE REVERSE QUESTION. `linkpred_metrics` excludes k=1 from mrr/hits@1/normalized_rank "
            "(C3), but `chance_mrr_mean` averages H_k/k over ALL k INCLUDING k=1, where H_1/1 = "
            "1.0. p2_amendment_3's c3 block claims 'chance_mrr_mean/degree_prior_metrics inherit "
            "the exclusion for free (both reduce through linkpred_metrics)' -- chance_mrr_mean does "
            "not reduce through it. This lesion makes the two denominators AGREE. If a test FAILS, "
            "the mismatch is PINNED by the suite, i.e. the published chance level is documented-"
            "wrong rather than accidentally wrong. Measured impact on the production splits: real "
            "0.27107 as coded vs 0.24525 comparable (+10.5%); degmatch 0.32292 vs 0.26228 (+23.1%)."
        ),
    ),
}

H.MUTATIONS.update(NEW)

if __name__ == "__main__":
    raise SystemExit(H.main())
