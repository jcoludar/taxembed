#!/usr/bin/env python3
"""Spec §4.5 TimeTree — write and FREEZE the pre-registration. Computes no correlation.

Only COUNTS have been observed (results/timetree_feasibility_20260929.json: 43,287 of 50,574 TTOL
leaves carry an embedding coordinate; Vertebrata 18,404, Insecta 6,122) and one API response for a
single pair (9606/10090) inspected to learn the schema. No distance, no divergence time, and no
correlation has been computed against any embedding. This script records the design and prints its
SHA256 so the freeze is evidenced independently of a commit -- the pattern
results/p3_placement_preregistration.json used.

Refuses to overwrite an existing file (Rule 1, and a pre-registration you can silently rewrite is
not one).

Written 2026-09-29.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path("/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings")
OUT = ROOT / "results" / "timetree_preregistration.json"

PREREG = {
    "status": (
        "PRE-REGISTERED 2026-09-29, BEFORE any divergence time, embedded distance or correlation "
        "was computed. Only COUNTS have been observed (helpers/_timetree_feasibility.py) plus one "
        "API response inspected for its schema. Revise only by adding a dated amendment block; "
        "never edit this one."
    ),
    "task": "Spec §4.5 — external ground truth. Does the embedding's geometry track evolutionary "
            "relatedness, or NCBI's classification convention?",
    "why_this_is_the_remaining_instrument": (
        "Steelman (ii) -- 'embedded distance tracks NCBI convention rather than relatedness' -- "
        "cannot be refuted by any NCBI-internal test. P2 was vacuous by construction (Finding 3) "
        "and P3 measured candidate subtree size (Finding 4, amendment 3), so P3 was the LAST "
        "NCBI-internal candidate. TimeTree brings dated information from outside NCBI entirely. "
        "The spec's standing instruction is 'Schedule it, or delete the relatedness claim.'"
    ),
    "why_it_survives_finding_4": (
        "Finding 4 §4.7: outside-the-tree information must enter the ANSWER, not merely the "
        "QUESTION -- a label can be unseen and still be predictable from a cheap statistic of the "
        "training data, which is exactly how P3 came to measure subtree size. §4.5 names its cheap "
        "tree-internal competitor BY CONSTRUCTION: NCBI path length is a co-reported PRIMARY "
        "comparator, not a robustness check. If the embedding does not beat it, the steelman is "
        "CONFIRMED and that is the finding."
    ),
    "data": {
        "divergence_times_primary": "TimeTree pairwise API, http://www.timetree.org/api/pairwise/"
                                    "<taxid_a>/<taxid_b>; CSV with precomputed_age, "
                                    "precomputed_ci_low, precomputed_ci_high, all_total.",
        "key_join": "🎯 taxon_a_id / taxon_b_id ARE NCBI taxids (verified: 9606/10090 -> Homo "
                    "sapiens / Mus musculus, age 83.514 Mya, all_total 75). No name "
                    "reconciliation needed for the API route.",
        "divergence_times_bulk": "TimetreeOfLife2015.nwk (1.9 MB, one request), 50,574 leaves, for "
                                 "the at-scale robustness surface. Name->taxid via names.dmp.",
        "embedding": "release/taxembed-cellular-v1/cellular_embedding.safetensors, the SHIPPED "
                     "artifact. Index alignment verified: md5(taxid_to_index.tsv) == "
                     "md5(cellular_organisms_131567_clean.mapping.tsv) = "
                     "a8ef06b048e03613230a0a14908f515e, so release row i is closure node i.",
        "tree": "data/taxopy/cellular_organisms_131567_clean/"
                "taxonomy_edges_cellular_organisms_131567_clean_transitive.npz",
    },
    "population": {
        "eligible_taxa": "TTOL leaves resolving to an NCBI taxid that carries an embedding "
                         "coordinate: 43,287 measured (Vertebrata 18,404, Insecta 6,122).",
        "strata": ["Vertebrata (7742)", "Insecta (50557)"],
        "pair_sample": "N_PAIRS = 3,000 per stratum, drawn uniformly WITHOUT replacement from "
                       "within-stratum pairs, seed 0, drawn BEFORE any distance is computed. "
                       "Cross-stratum pairs are excluded: their divergence times are ancient and "
                       "near-constant, which would inflate every correlation identically.",
    },
    "metrics": {
        "primary": (
            "Spearman rho( ANGULAR distance, divergence time ) within each stratum. ANGULAR "
            "(radius-free, 1 - cosine of direction) is the PRIMARY distance, NOT Poincare."
        ),
        "why_angular_and_not_poincare": (
            "🛑 The radial prior is PLANTED: norm = target_radius(depth) at init, held there by the "
            "radial regularizer. Depth correlates with divergence time, so Poincare distance -- "
            "which reads the norm -- could track divergence time through GIVEN structure alone and "
            "report it as a learned result. This is the same trap Task 9's radius-free scorer and "
            "C3's initialization floor exist for. Angular distance reads no norm."
        ),
        "secondary_descriptive": "Spearman rho( Poincare distance, divergence time ). DESCRIPTIVE "
                                 "ONLY. It is expected to be HIGHER than the primary and that is "
                                 "not evidence of anything -- see why_angular_and_not_poincare.",
        "co_reported_primary_comparator": (
            "Spearman rho( NCBI path length, divergence time ) on the SAME pairs. Path length is "
            "depth[a] + depth[b] - 2*depth[LCA(a,b)] in the training tree."
        ),
        "uncertainty": (
            "🛑 CLUSTER BOOTSTRAP OVER TAXA, never over pairs. 936,860,541 pairs arise from 43,287 "
            "taxa and are massively non-independent; a pair-level bootstrap would report an "
            "absurdly tight CI. Resample TAXA with replacement and rebuild the pair set from the "
            "resampled taxa. n_boot = 1000, seed 0."
        ),
    },
    "gates": {
        "a_power": "n_scored pairs >= 1000 per stratum AND >= 300 distinct taxa per stratum.",
        "b_init_null_FLOOR_NOT_ZERO": (
            "Random directions at the TRAINED radii (the project's house null). ⚠ Its expected "
            "value is NOT 0 for the SECONDARY (Poincare) metric, because the planted radius "
            "survives randomisation and depth correlates with divergence time. For the PRIMARY "
            "(angular) metric it MUST be ~0, since a random direction carries no information. "
            "Gate: |rho_angular(init null)| <= 0.05. If it exceeds that, the angular distance is "
            "not radius-free in practice and the primary is VOID."
        ),
        "c_interpolation_guard": (
            "⚠ A Finding-4-shaped risk INSIDE §4.5: TimeTree's age for an obscure pair may be "
            "INTERPOLATED from TimeTree's own topology rather than derived from dated studies, and "
            "that topology is not independent of NCBI's. The API's all_total is the study count. "
            "Gate: the primary is read on pairs with all_total >= 1; pairs with all_total == 0 are "
            "reported as a SEPARATE stratum and never pooled into the primary. If fewer than 1000 "
            "pairs per stratum have all_total >= 1, the primary is UNDERPOWERED and reads "
            "UNINFORMATIVE rather than being pooled to reach n."
        ),
        "d_shuffle_control": (
            "Permute divergence times across pairs within a stratum, seed 0. MUST return rho ~ 0. "
            "Catches any sign or join error, the failure mode P3's ranking direction needed."
        ),
    },
    "verdict_rules": {
        "TRACKS_RELATEDNESS": "primary rho CI excludes the NCBI-path-length comparator's rho AND "
                              "is the stronger of the two, in BOTH strata.",
        "TRACKS_CONVENTION": "the NCBI-path-length comparator's rho is >= the primary's, CIs not "
                             "overlapping, in either stratum. ⇒ steelman (ii) CONFIRMED; per the "
                             "spec this is 'the honest finding and it is publishable', and the "
                             "relatedness claim comes out of the paper.",
        "NO_EVIDENCE": "CIs overlap; the embedding is not distinguishable from NCBI topology as a "
                       "predictor of divergence time.",
        "UNINFORMATIVE": "any gate fails.",
    },
    "declared_confounds": [
        "Transductive coverage: 1,961,538 of 2,986,118 present-day NCBI taxa have NO coordinate "
        "(~34% covered), and TTOL leaves that resolve to an embedded taxid are a biased, "
        "well-studied subset. This measures the embedding where it exists, not everywhere.",
        "TTOL is the 2015 tree while the embedding trained on the 2026-06-09 taxonomy; topology "
        "revisions between the two add noise, which is conservative for the primary.",
        "Divergence times are themselves estimates with CIs; precomputed_ci_low/high are recorded "
        "but the primary uses the point estimate, since Spearman is rank-based.",
        "Angular distance at the SAME depth is what Task 9 validated; §4.5 pairs span depths, so "
        "the angular metric here is used outside the regime where it was characterised. Stated, "
        "not resolved.",
    ],
    "what_would_make_this_vacuous": (
        "Applying Finding 4's own operational test to §4.5 before running it: name the cheapest "
        "statistic of the training data that could pass. It is NCBI path length -- and it is the "
        "co-reported comparator, so it cannot pass unnoticed. The second cheapest is depth alone, "
        "which the init null (gate b) measures. Both are instrumented BEFORE the primary is read."
    ),
    "pointers": {
        "spec": "docs/specs/2026-08-11-taxembed-overfitting-and-plm-showcase-design-v3.md §4.5",
        "finding_that_makes_this_the_last_option":
            "docs/FINDING_protocols_that_are_vacuous_on_taxonomy_trees.md §4.7",
        "feasibility": "results/timetree_feasibility_20260929.json",
        "fetch_probe": "results/timetree_fetch_probe_20260929.json",
        "provenance_of_the_headline_S_angle": "MANUSCRIPT_CORRECTIONS_PENDING.md C11",
        "session_log":
            "SpeciesEmbedding/docs/sessions/2026-09-29-taxembed-p3-tree-distance-control.md",
    },
}


def main() -> None:
    if OUT.exists():
        raise SystemExit(
            f"{OUT} already exists — a pre-registration is not rewritten. Add an amendment block.")
    body = json.dumps(PREREG, indent=2, ensure_ascii=False) + "\n"
    OUT.write_text(body)
    h = hashlib.sha256(body.encode()).hexdigest()
    print("=" * 92)
    print("§4.5 TimeTree pre-registration FROZEN")
    print("=" * 92)
    print(f"  written : {OUT}")
    print(f"  SHA256  : {h}")
    print(f"  bytes   : {len(body):,}")
    print("\n  Nothing has been computed against an embedding. The primary is the ANGULAR")
    print("  (radius-free) correlation; NCBI path length is a CO-REPORTED PRIMARY comparator,")
    print("  not a robustness check; the bootstrap resamples TAXA, not pairs.")
    print("=" * 92)


if __name__ == "__main__":
    main()
