#!/usr/bin/env python3
"""§4.5 amendment 2 — the LCA-depth bins I chose in amendment 1 were vacuous. Third time today.

Amendment 1 added an LCA-depth-binned SECONDARY surface to give the comparison resolution in the
range where relatedness actually varies. The bins were hard-coded (0-6, 7-9, 10-12, 13-40) and the
run put essentially EVERY pair in the last one:

    Vertebrata   0-6: 0    7-9: 0    10-12: 48    13-40: 2,945
    Insecta      0-6: 0    7-9: 0    10-12: 0     13-40: 2,991

So the secondary resolved nothing, and the defect amendment 1 existed to fix is still unfixed. The
bins were guessed from an assumption about NCBI depth rather than read off the data -- the same
mistake shape as the vacuous tree-distance control (Finding 4 §4.2) and the all_total >= 1 threshold
(amendment 1, defect 1). THREE self-inflicted non-discriminating guards in one day, all of the form
"a cut chosen without looking at the distribution it is meant to cut."

FIX: the secondary's bins are QUANTILES of the observed LCA-depth distribution, computed per
stratum (terciles, plus the distribution itself reported). A quantile bin cannot be empty by
construction, which is the property the fixed bins lacked.

⚠ THIS CHANGES ONLY THE SECONDARY. The primary, the comparator, the gates and the verdict rules are
untouched, and the secondary explicitly cannot overturn the primary. The primary verdicts were
already read before this amendment and are NOT revisited here:
    Vertebrata  angular +0.8946 [0.8798, 0.9085]  vs  NCBI path +0.7911 [0.7695, 0.8123]
    Insecta     angular +0.2963 [0.2473, 0.3457]  vs  NCBI path +0.4434 [0.3992, 0.4853]
Recorded so it is visible that the amendment could not have been chosen to move them.

Written 2026-09-29.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path("/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings")
PREREG = ROOT / "results" / "timetree_preregistration.json"
KEY = "timetree_amendment_2_20260929"

AMENDMENT = {
    "status": (
        "Recorded AFTER the primary was read, and it touches ONLY the secondary surface. The "
        "primary verdicts (Vertebrata TRACKS_RELATEDNESS on its own terms, Insecta "
        "TRACKS_CONVENTION) are quoted inside this amendment so it is visible that the change "
        "could not have been selected to move them."
    ),
    "what_failed": (
        "Amendment 1's LCA-depth SECONDARY used hard-coded bins (0-6, 7-9, 10-12, 13-40). The run "
        "placed 2,945 of 2,993 Vertebrata pairs and 2,991 of 2,991 Insecta pairs in the LAST bin; "
        "the first two were EMPTY in both strata. The secondary resolved nothing."
    ),
    "the_pattern_this_is_the_third_instance_of": (
        "A cut chosen without looking at the distribution it is meant to cut. (1) The prescribed "
        "tree-distance control for P3 could not move a candidate because tree distance was "
        "constant across the pool (Finding 4 §4.2). (2) Gate (c)'s all_total >= 1 was cleared by "
        "100 % of pairs (amendment 1, defect 1). (3) These bins. All three were written by me, all "
        "three were caught by running them, and none would have been caught by reading them. "
        "See feedback_a_check_that_could_not_have_failed_is_not_evidence."
    ),
    "the_fix": (
        "The secondary's bins are TERCILES of the observed per-stratum LCA-depth distribution, and "
        "that distribution is reported alongside. A quantile bin cannot be empty by construction -- "
        "which is exactly the property the fixed bins lacked. If the LCA depth distribution is too "
        "concentrated for terciles to separate (all boundaries equal), that is REPORTED as "
        "'secondary has no resolution on this tree' rather than papered over with another guess."
    ),
    "what_is_NOT_changed": (
        "Primary metric, co-reported NCBI-path-length comparator, gates a/b/d, the all_total "
        "surface, the strata, the pair sample, the taxon-level cluster bootstrap and every verdict "
        "rule. The secondary cannot overturn the primary; if they disagree, the disagreement is the "
        "finding."
    ),
    "primary_as_read_before_this_amendment": {
        "Vertebrata": {"angular": [0.8946, 0.8798, 0.9085], "ncbi_path": [0.7911, 0.7695, 0.8123],
                       "verdict": "TRACKS_RELATEDNESS"},
        "Insecta": {"angular": [0.2963, 0.2473, 0.3457], "ncbi_path": [0.4434, 0.3992, 0.4853],
                    "verdict": "TRACKS_CONVENTION"},
    },
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
    print(f"amended: {PREREG}")
    print(f"  SHA256 now: {hashlib.sha256(body.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
