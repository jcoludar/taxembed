#!/usr/bin/env python3
"""Append p3_amendment_3_20260929 to the P3 pre-registration, and write the v2 verdict.

The pre-registration's own rule: "revise only by adding a dated amendment block, never by editing
this one." This script adds a TOP-LEVEL KEY and rewrites nothing that is already there — it asserts
that every pre-existing key survives byte-identically before writing (Rule 1: Know, Check, then
write).

The verdict is NOT overwritten. Rule 5 makes verdict files append-only AT THE FILE LEVEL, so
results/p3_placement_result_20260929.json is left exactly as the record of what amendment 2
produced, and the superseding verdict goes to results/p3_placement_result_v2_20260929.json.

Written 2026-09-29.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings")
PREREG = ROOT / "results" / "p3_placement_preregistration.json"
V1 = ROOT / "results" / "p3_placement_result_20260929.json"
V2 = ROOT / "results" / "p3_placement_result_v2_20260929.json"
KEY = "p3_amendment_3_20260929"

AMENDMENT = {
    "status": (
        "Recorded AFTER the amendment-2 run, BECAUSE A CONTROL THE PRE-REGISTRATION NEVER "
        "CONTAINED was built and the primary did not survive it. The ANTICIPATES verdict of "
        "results/p3_placement_result_20260929.json is WITHDRAWN. Superseded by "
        "results/p3_placement_result_v2_20260929.json. The withdrawn numbers are kept, not deleted."
    ),
    "what_prompted_it": (
        "The closing qualification 3 of the 2026-09-29 session: the pool excludes p_old's branch "
        "but not branches adjacent to it, so if p_new is systematically tree-nearer to p_old than "
        "a pool draw, the embedding ranks it well for encoding the OLD tree, not for anticipating. "
        "The prescribed fix was to match pseudo-parents on TREE DISTANCE from p_old."
    ),
    "finding_0_the_prescribed_fix_is_vacuous": {
        "measurement": "helpers/_p3_confound_diagnostics.py; "
                       "results/p3_confound_diagnostics_20260929.json",
        "result": "path(p_old, q) is CONSTANT across the pool in 1,241 / 1,241 pools (100.00%), "
                  "and equals path(p_old, p_new) in 1,241 / 1,241.",
        "why": "The pool is depth-homogeneous (all q at depth(p_new)) and, after amendment 2, "
               "branch-homogeneous (LCA(p_old, q) == L for every q). So "
               "path(p_old,q) = (depth(p_old)-depth(L)) + (depth(p_new)-depth(L)), which has no "
               "term varying with q. Matching on tree distance could not have moved one candidate "
               "in one pool; it would have been a control incapable of failing.",
    },
    "finding_1_the_moved_taxon_is_inert": {
        "measurement": "helpers/_p3_confound_diagnostics.py, substitution control",
        "result": "Ranking the IDENTICAL pool from p_old instead of v gives 0.4444 "
                  "[0.4257, 0.4628] against the primary's 0.4400 [0.4214, 0.4588]; paired delta "
                  "(v - p_old) = -0.0044, CI95 [-0.0145, +0.0062], CONTAINS ZERO.",
        "consequence": "p_old alone delivers 0.0556 of the 0.0600 total effect over chance "
                       "(92.7%). P3 IS NOT A HELD-OUT-TAXON TEST: the moved taxon's own "
                       "coordinate contributes nothing measurable. Any sentence attributing the "
                       "signal to the reclassified taxon is false.",
    },
    "finding_2_a_one_line_baseline_beats_the_embedding": {
        "measurement": "helpers/_p3_combinatorial_baselines.py; "
                       "results/p3_combinatorial_baselines_20260929.json",
        "result": "On the identical pools: random 0.4937 [0.4761, 0.5110] (harness PASS); "
                  "subtree size, larger first 0.3146 [0.2982, 0.3313]; n_children 0.3188; "
                  "|taxid - taxid(p_old)| 0.5571 (mirror 0.4429); embedding from p_old 0.4444. "
                  "Paired (embedding - size) = +0.1298, CI95 [+0.1107, +0.1488].",
        "consequence": "'Pick the largest candidate branch' predicts the destination far better "
                       "than the embedding does. NCBI moves taxa into big, actively curated "
                       "groups; p_new's mean within-pool size rank is 0.3218 against a background "
                       "of 0.5013.",
    },
    "finding_3_the_signal_is_a_size_proxy_and_nothing_more": {
        "measurement": "helpers/_p3_size_residual_control_v3.py; "
                       "results/p3_size_residual_control_v3_20260929.json",
        "design": "Condition on subtree size rather than match on it (two matching designs failed "
                  "their own gates first; see superseded_attempts). For every candidate compute "
                  "within-pool normalized ranks r_s (size, larger first) and r_e (Poincare "
                  "distance from p_old); estimate E[r_e | r_s] by binned mean over NON-TARGET "
                  "candidates only (166,852), weighted 1/n_q so the curve is fit under the same "
                  "weighting the query-averaged statistic uses; test each target's residual "
                  "r_e(p_new) - curve(r_s(p_new)).",
        "gates": "G1 null residual for uniformly drawn pseudo-targets (20 per query) must contain "
                 "0: +0.0010 / +0.0011 / +0.0009, CIs contain 0 -- PASS at 10/20/40 bins. "
                 "G2 curve spread must exceed 0.05: 0.3067 / 0.3626 / 0.4068 -- PASS.",
        "result": "G3 target residual from p_old: -0.0031 [-0.0203, +0.0137] (10 bins), "
                  "+0.0019 [-0.0149, +0.0185] (20), +0.0076 [-0.0091, +0.0241] (40). "
                  "ALL THREE CONTAIN ZERO and the sign CHANGES across binnings. From v: "
                  "-0.0075 / -0.0025 / +0.0032, same picture.",
        "decomposition": "Of the 0.0556 raw effect over chance, subtree size alone accounts for "
                         "0.0525 / 0.0575 / 0.0631 at 10/20/40 bins -- 94% / 103% / 113%. "
                         "Nothing is left over.",
        "consequence": "The P3 placement signal is a LOSSY ENCODING OF SUBTREE SIZE. Conditional "
                       "on how large a candidate subtree is, the embedding carries no information "
                       "about where NCBI will move the taxon.",
    },
    "integrity_note_the_gate_that_reversed_the_verdict": (
        "The FIRST v3 run, with one pseudo-target per query, returned G3 = -0.0325 / -0.0340 / "
        "-0.0269 with CIs EXCLUDING zero -- i.e. BEYOND SIZE, a positive result. It would have "
        "been reported. G1 passed there only because its CI half-width (0.017) was half the "
        "effect it was certifying. Raising R_NULL from 1 to 20 for POWER made G1 FAIL "
        "(+0.0122 / +0.0092 / +0.0090, CIs excluding zero), exposing a WEIGHTING MISMATCH IN MY "
        "OWN ESTIMATOR: the curve was candidate-weighted while the statistic is query-weighted. "
        "Fitting the curve with 1/n_q weights fixed it, G1 now passes at +0.001, AND THE VERDICT "
        "FLIPPED FROM POSITIVE TO NULL. The correction moved the headline the UNWELCOME way, "
        "which is the evidence it was made for the right reason."
    ),
    "superseded_attempts": {
        "helpers/_p3_size_matched_control.py": "Fixed log2 band around the target's size. Its gate "
            "(size must neutralize inside the band) FAILED at every band -- 0.4746 [0.455, 0.495] "
            "even at +-0.5 log2. Random control was one draw per query and mis-read two of four "
            "bands; rank_by then shown unbiased by synthetic test (20,000 reps x 15 pool sizes, "
            "all 0.494-0.504). DEPRECATED.",
        "helpers/_p3_size_matched_control_v2.py": "k-nearest-in-size pools + matched-procedure "
            "null. H1 (random, R=25) PASSED 0.4998/0.5012/0.4982. H2 FAILED worse than v1: size "
            "read 0.2445/0.2575/0.2907. Cause: subtree size is heavy-tailed and p_new sits in the "
            "UPPER TAIL, so the k nearest in log-size are drawn asymmetrically FROM BELOW and the "
            "target stays among the largest in its own 'matched' pool. That also invalidates its "
            "gate (e), whose pseudo-target is drawn uniformly and is therefore typically small. "
            "Its k=4 / k=8 'BEYOND SIZE' readings (-0.0792, -0.0356) are WITHDRAWN. DEPRECATED.",
    },
    "what_is_NOT_changed": (
        "The primary metric, the amendment-2 pool, the verdict rules, the windows and the eligible "
        "population are all unchanged. Amendment 3 adds a control the pre-registration did not "
        "contain and reports that the primary does not survive it. No endpoint was swapped and no "
        "window was promoted."
    ),
    "verdict_after_amendment_3": "UNINFORMATIVE — the primary does not survive conditioning on "
                                 "subtree size, and it was never a held-out-taxon test.",
}

V2_VERDICT = {
    "supersedes": "results/p3_placement_result_20260929.json",
    "supersedes_verdict": "ANTICIPATES (primary 0.4400, CI95 [0.4214, 0.4588], four gates passing)",
    "why_superseded": "p3_amendment_3_20260929 — a control the pre-registration did not contain. "
                      "The primary is fully accounted for by candidate subtree size, and the "
                      "moved taxon's own coordinate contributes nothing.",
    "preregistration": "results/p3_placement_preregistration.json",
    "amendment": KEY,
    "old_date": "2026-07-01", "new_date": "2026-09-01", "training_date": "2026-06-09",
    "n_scored": 1241,
    "primary_unchanged": {"mean_normalized_rank": 0.4400282297778546,
                          "ci95": [0.42142890356597273, 0.45882674455218864]},
    "substitution_control_p_old_for_v": {
        "mean": 0.4444, "paired_delta_v_minus_pold": -0.0044,
        "ci95": [-0.0145, 0.0062], "reading": "contains zero; v is inert"},
    "size_conditioned_residual": {
        "bins_10": {"mean": -0.0031, "ci95": [-0.0203, 0.0137]},
        "bins_20": {"mean": 0.0019, "ci95": [-0.0149, 0.0185]},
        "bins_40": {"mean": 0.0076, "ci95": [-0.0091, 0.0241]},
        "reading": "all contain zero, sign changes across binnings"},
    "non_embedding_baseline_subtree_size": {
        "mean": 0.3146, "ci95": [0.2982, 0.3313],
        "reading": "BEATS the embedding outright; paired +0.1298 [+0.1107, +0.1488]"},
    "verdict": "UNINFORMATIVE",
    "one_line": "P3 placement does not separate the real taxonomy's geometry from candidate "
                "subtree size, and never tested the held-out taxon at all. It joins P2.",
}


def main() -> None:
    raw = PREREG.read_text()
    before = json.loads(raw)
    if KEY in before:
        raise SystemExit(f"{KEY} already present — refusing to rewrite an existing amendment.")

    after = dict(before)
    after[KEY] = AMENDMENT

    # Rule 1: CHECK. Every pre-existing key must survive byte-identically.
    for k, v in before.items():
        assert json.dumps(after[k], sort_keys=True) == json.dumps(v, sort_keys=True), k
    assert set(after) - set(before) == {KEY}
    print(f"checked: {len(before)} existing keys unchanged, adding only {KEY}")

    PREREG.write_text(json.dumps(after, indent=2) + "\n")
    print(f"amended: {PREREG}")

    assert V1.exists(), "the superseded verdict must remain on disk as the record"
    v1 = json.loads(V1.read_text())
    assert v1["verdict"] == "ANTICIPATES", "V1 is not what amendment 3 supersedes"
    print(f"verified untouched: {V1} (verdict {v1['verdict']})")

    if V2.exists():
        raise SystemExit(f"{V2} exists — verdicts are append-only at the file level (Rule 5).")
    V2.write_text(json.dumps(V2_VERDICT, indent=2) + "\n")
    print(f"written: {V2}")


if __name__ == "__main__":
    main()
