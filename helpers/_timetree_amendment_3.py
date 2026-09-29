#!/usr/bin/env python3
"""§4.5 amendment 3 — the effective sample size is ~115 MRCA nodes, not ~3,000 pairs.

A LIMITATION discovered while explaining a nan, recorded against the primary rather than changing
it. No endpoint, gate or verdict rule moves.

The nan: amendment 2's tercile secondary printed `angular +nan path +nan` for Vertebrata's LCA
depth-16 bin (n=524). Not a code defect -- `age` has exactly ONE distinct value in that bin, so its
ranks are constant, the Spearman denominator is 0 and the statistic is genuinely undefined. Checked
directly: 0 nan in the distances, 0 nan in the embedding, 524 distinct angular values, 1 distinct
age.

WHY, and it generalises well beyond that bin: TimeTree ages are NODE-LEVEL. Every pair whose MRCA is
the same TimeTree node receives the SAME precomputed_age. Measured on the fetched sample:

    Vertebrata   2,993 pairs -> 118 DISTINCT ages; largest age-class 859 pairs = 28.7 %
    Insecta      2,991 pairs -> 113 DISTINCT ages; largest age-class 881 pairs = 29.5 %
    (54 and 55 pairs respectively sit in an age-class of size 1)

⇒ The label carries ~115 distinct values, heavily tied, and roughly 29 % of pairs in each stratum
share a single value. The primary is therefore closer to "can the embedding ORDER ~115 clade-split
ages" than to "does it predict divergence time across 3,000 independent comparisons". That is a
coarser question than the pair count suggests, and the pair count must never be quoted as the
sample size.

WHAT SAVES THE CIs, PARTLY: the pre-registered uncertainty is a CLUSTER BOOTSTRAP OVER TAXA, not
over pairs, so resampling taxa does destroy and recreate whole age-classes. That was the right call
for a reason adjacent to this one. But the stricter clustering unit is arguably the MRCA NODE
itself, since pairs within a class share the label EXACTLY, and the reported CIs are not clustered
at that level. Treat the CIs as lower bounds on the true width.

NOTHING IS CHANGED. Primary, comparator, gates, strata, sample and verdict rules all stand; the
verdicts were read before this was discovered and are not revisited. This amendment exists so the
result cannot be read as stronger than its label supports.

Written 2026-09-29.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path("/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings")
PREREG = ROOT / "results" / "timetree_preregistration.json"
KEY = "timetree_amendment_3_20260929"

AMENDMENT = {
    "status": "A LIMITATION recorded after the primary was read. Changes no endpoint, gate, "
              "stratum or verdict rule. Exists so the result cannot be read as stronger than its "
              "label supports.",
    "discovered_via": "amendment 2's tercile secondary printed nan for Vertebrata LCA depth 16 "
                      "(n=524). Not a code defect: `age` has ONE distinct value there, so the "
                      "Spearman denominator is 0 and the statistic is undefined. Verified 0 nan in "
                      "distances, 0 nan in the embedding, 524 distinct angular values.",
    "the_finding": "TimeTree ages are NODE-LEVEL — every pair sharing an MRCA gets the same "
                   "precomputed_age.",
    "effective_sample_size": {
        "Vertebrata": {"pairs": 2993, "distinct_ages": 118, "largest_age_class": 859,
                       "largest_age_class_pct": 28.7, "singleton_classes": 54},
        "Insecta": {"pairs": 2991, "distinct_ages": 113, "largest_age_class": 881,
                    "largest_age_class_pct": 29.5, "singleton_classes": 55},
    },
    "consequence": (
        "The primary is closer to 'can the embedding ORDER ~115 clade-split ages' than to 'does it "
        "predict divergence time over 3,000 independent comparisons'. 🛑 THE PAIR COUNT MUST NEVER "
        "BE QUOTED AS THE SAMPLE SIZE."
    ),
    "effect_on_the_reported_CIs": (
        "The pre-registered bootstrap clusters over TAXA, so resampling destroys and recreates "
        "whole age-classes — the right call, for a reason adjacent to this one. But the stricter "
        "clustering unit is the MRCA NODE, since pairs within a class share the label EXACTLY, and "
        "the reported CIs are not clustered at that level. TREAT THE REPORTED CIs AS LOWER BOUNDS "
        "ON THE TRUE WIDTH."
    ),
    "does_it_change_the_verdicts": (
        "Not by itself. Insecta's TRACKS_CONVENTION rests on the comparator BEATING the primary on "
        "the identical label, so label coarseness applies to both arms equally and cannot explain "
        "the gap. Vertebrata's TRACKS_RELATEDNESS is the one to treat cautiously, and amendment "
        "2's secondary already does so: within LCA-depth terciles the angular advantage over path "
        "shrinks from +0.1035 pooled to +0.0221 (depth 17-33), and at depth 10-15 BOTH "
        "correlations are NEGATIVE (angular -0.3852, path -0.3151). Most of the pooled advantage is "
        "BETWEEN-bin, i.e. the embedding encoding LCA depth — a property of the tree, not of "
        "relatedness."
    ),
}


def main() -> None:
    before = json.loads(PREREG.read_text())
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
    print(f"amended: {PREREG}\n  SHA256 now: {hashlib.sha256(body.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
