#!/usr/bin/env python3
"""§4.5 amendment 1 — recorded BEFORE any correlation was computed. Two design defects, mine.

The pair sample is fetched (results/timetree_pairs_20260929.json, ages only, no embedding read).
Inspecting the LABEL distribution -- which is all that has been observed -- exposes two defects in
the pre-registration I froze an hour earlier. Both are recorded here before a single Spearman is
computed, which is the only reason this is an amendment and not a rescue.

Appends a top-level key; asserts every pre-existing key survives byte-identically first.

Written 2026-09-29.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path("/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings")
PREREG = ROOT / "results" / "timetree_preregistration.json"
KEY = "timetree_amendment_1_20260929"

AMENDMENT = {
    "status": (
        "Recorded 2026-09-29 AFTER the pair sample was fetched but BEFORE any distance, rank or "
        "correlation was computed against any embedding. Only the LABEL distribution (divergence "
        "times and study counts) has been observed. Both changes below make the test HARDER or "
        "more informative; neither touches the primary endpoint or the verdict rules."
    ),
    "defect_1_gate_c_could_not_have_failed": {
        "what_i_wrote": "Gate (c), the interpolation guard: 'the primary is read on pairs with "
                        "all_total >= 1; pairs with all_total == 0 are reported as a SEPARATE "
                        "stratum and never pooled.'",
        "what_happened": "ALL 5,984 pairs carrying an age have all_total >= 1. The gate passed at "
                         "100.0 %. A threshold that every observation clears is not a guard -- it "
                         "is the exact defect this project keeps re-learning, and I wrote it into "
                         "a pre-registration hours after documenting it as Finding 4's method "
                         "lesson. See feedback_a_check_that_could_not_have_failed_is_not_evidence.",
        "the_actual_distribution": {"min": 1, "q25": 14, "median": 19, "q75": 37, "max": 75,
                                    "ge_2": 5933, "ge_5": 5784, "ge_10": 5488, "ge_25": 2642,
                                    "ge_50": 47, "n": 5984},
        "why_the_substantive_worry_is_nonetheless_answered": (
            "The concern was that TimeTree might interpolate an obscure pair's age from its own "
            "topology rather than from dated studies. With a MEDIAN of 19 supporting studies and "
            "91.7 % of pairs at >= 10, this sample is not interpolation-dominated. The worry was "
            "real; the threshold chosen to test it was simply too low to discriminate."
        ),
        "the_fix": (
            "Gate (c) is REPLACED by a reported ROBUSTNESS SURFACE over all_total rather than a "
            "single threshold: the primary and the NCBI-path-length comparator are both reported "
            "at all_total >= 1 (n=5,984), >= 10 (n=5,488) and >= 25 (n=2,642). A verdict that "
            "holds at one cut and reverses at another is reported as unstable, not as a result. "
            "The >= 50 cut (n=47) is excluded as underpowered. No threshold is chosen after seeing "
            "a correlation."
        ),
    },
    "defect_2_uniform_pairs_are_almost_all_ancient": {
        "what_i_declared": "Cross-stratum pairs are excluded because 'their divergence times are "
                           "ancient and near-constant, which would inflate every correlation "
                           "identically.'",
        "what_happened": "WITHIN-stratum uniform pairs are ALSO overwhelmingly ancient. Median "
                         "divergence is 318.8 Mya (Vertebrata) and 331.3 Mya (Insecta); the full "
                         "ranges are 0.6-562.8 and 5.1-431.3 Mya. A uniformly drawn pair from "
                         "18,404 vertebrates or 6,122 insects is usually cross-ORDER.",
        "why_it_matters": (
            "The reasoning I applied to cross-stratum pairs applies one level down and I did not "
            "carry it there -- feedback_a_guard_carried_across_a_boundary_is_a_new_guard. If "
            "almost every pair is a deep cross-order split, both the embedding and NCBI path "
            "length are being scored mainly on 'which order is this taxon in', where they agree by "
            "construction. That biases the comparison toward NO_EVIDENCE: it hides a real "
            "difference rather than manufacturing one."
        ),
        "the_fix": (
            "The PRIMARY is unchanged -- the pre-registered uniform within-stratum sample stands, "
            "and its result is the headline. ADDED as a pre-declared SECONDARY robustness surface: "
            "the primary and the comparator are also reported within bins of NCBI LCA depth "
            "(the depth of LCA(a,b) in the training tree), so the comparison has resolution in the "
            "range where relatedness actually varies. Bins with fewer than 300 pairs are reported "
            "as underpowered and carry no verdict weight. This is a SECONDARY: it cannot overturn "
            "the primary, and if primary and secondary disagree that disagreement is the finding."
        ),
    },
    "what_is_NOT_changed": (
        "The primary metric (Spearman of ANGULAR distance vs divergence time), the co-reported "
        "NCBI-path-length comparator, the verdict rules, the taxon-level cluster bootstrap, the "
        "init null gate, the shuffle control, the strata and the pair sample itself are all "
        "unchanged. No endpoint is swapped and no new primary is introduced."
    ),
    "data_quality_note": "16 of 6,000 pairs returned no age (13 HTTP 500, 3 no data row), 0.27 %. "
                         "Recorded; too few to bias anything.",
}


def main() -> None:
    raw = PREREG.read_text()
    before = json.loads(raw)
    if KEY in before:
        raise SystemExit(f"{KEY} already present — refusing to rewrite an existing amendment.")
    after = dict(before)
    after[KEY] = AMENDMENT
    for k, v in before.items():
        assert json.dumps(after[k], sort_keys=True) == json.dumps(v, sort_keys=True), k
    assert set(after) - set(before) == {KEY}
    print(f"checked: {len(before)} existing keys unchanged, adding only {KEY}")
    body = json.dumps(after, indent=2, ensure_ascii=False) + "\n"
    PREREG.write_text(body)
    print(f"amended: {PREREG}")
    print(f"  original frozen SHA256 : "
          f"a3e733c781914a80046a7e116f491951269422d46ff31bf88bd2929521eb7fba")
    print(f"  SHA256 after amendment : {hashlib.sha256(body.encode()).hexdigest()}")
    print("\nNo correlation has been computed. Only divergence times and study counts observed.")


if __name__ == "__main__":
    main()
