"""Are the SHARED runs bit-identical across the Task 8 and Task 9 scoring jobs?

WHY THIS EXISTS (2026-09-24).

task9_canonical_s{0,1,2} is scored TWICE: once by task9_lrz_score_seeds.sh (as `canonical_s*`)
and again by task8_lrz_score_seeds.sh (as `unfixed_s*`). Different jobs, different output trees,
different arm labels, different src mounts (src vs src_task8) -- but the same checkpoints, the
same tree and the same seed 0, so the query set and clade level are deterministic and the numbers
MUST agree to the last bit.

This is a check that could have failed: a wrong mount, a truncated checkpoint, a stale src tree or
a different metazoa npz would all move it. [[feedback_agreement_is_evidence_only_if_it_could_have_disagreed]]

Compares every numeric field the pre-registration engine reads, per checkpoint, with EXACT
equality -- not np.isclose. Exits 1 on any mismatch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings")
TASK9 = REPO / "results" / "seeds_20260924_084146.json"
TASK8 = REPO / "results" / "task8_seeds_20260924_132043.json"

# the shared arm, under its two labels
PAIRS = [(f"canonical_s{s}{suf}", f"unfixed_s{s}{suf}") for s in (0, 1, 2) for suf in ("_ms", "_roll")]

# The REAL field names in score_recipe_checkpoints.py output. A first pass guessed
# these and only 3 of 10 existed -- one of which was `epoch`, the join key, which
# cannot fail. A comparison over fields that are absent silently compares nothing.
# `epoch` is deliberately EXCLUDED: joining on it and then asserting it is circular.
SCALAR_FIELDS = [
    "S_angle", "S_poincare", "S_angle_at_k", "S_angle_cluster_se", "S_poincare_cluster_se",
    "level_auc_mean", "depth_norm_r", "radial_dev_mean", "mean_mu0", "mean_mustar",
    "mean_s_angle", "hubness_angle", "n_clusters", "norm_min", "norm_max",
    "n_norm_below_0p05", "flag_poincare_above_angle",
]
# nested one level under `trainer`
TRAINER_FIELDS = ["loss", "depth_norm_corr", "hierarchy_pct", "knn_purity",
                  "class_sep_ratio", "reg_loss", "epoch"]


def runs_of(doc: dict) -> dict:
    for key in ("runs", "arms", "results"):
        if key in doc and isinstance(doc[key], dict):
            return doc[key]
    raise SystemExit(f"cannot find a runs mapping; top-level keys = {sorted(doc)}")


def checkpoints_of(run) -> list:
    if isinstance(run, list):
        return run
    for key in ("checkpoints", "points", "epochs"):
        if isinstance(run, dict) and key in run:
            return run[key]
    raise SystemExit(f"cannot find checkpoints; run keys = {sorted(run) if isinstance(run, dict) else type(run)}")


def main() -> int:
    for p in (TASK9, TASK8):
        if not p.exists():
            print(f"MISSING: {p}")
            return 1

    r9 = runs_of(json.loads(TASK9.read_text()))
    r8 = runs_of(json.loads(TASK8.read_text()))

    total_ck = 0
    total_cmp = 0
    mismatches: list[str] = []
    missing: set[str] = set()

    for name9, name8 in PAIRS:
        if name9 not in r9:
            mismatches.append(f"{name9} absent from the Task 9 file")
            continue
        if name8 not in r8:
            mismatches.append(f"{name8} absent from the Task 8 file")
            continue

        ck9 = checkpoints_of(r9[name9])
        ck8 = checkpoints_of(r8[name8])
        if len(ck9) != len(ck8):
            mismatches.append(f"{name9}/{name8}: {len(ck9)} vs {len(ck8)} checkpoints")
            continue

        by_ep9 = {c.get("epoch"): c for c in ck9}
        by_ep8 = {c.get("epoch"): c for c in ck8}
        if set(by_ep9) != set(by_ep8):
            mismatches.append(f"{name9}/{name8}: epoch sets differ")
            continue

        for ep in sorted(by_ep9):
            total_ck += 1
            a, b = by_ep9[ep], by_ep8[ep]
            for f in SCALAR_FIELDS:
                if f not in a or f not in b:
                    missing.add(f)
                    continue
                total_cmp += 1
                if a[f] != b[f]:          # EXACT, not approximate
                    mismatches.append(
                        f"{name9}/{name8} ep{ep} {f}: {a[f]!r} vs {b[f]!r}"
                    )
            ta, tb = a.get("trainer") or {}, b.get("trainer") or {}
            for f in TRAINER_FIELDS:
                if f not in ta or f not in tb:
                    missing.add(f"trainer.{f}")
                    continue
                total_cmp += 1
                if ta[f] != tb[f]:
                    mismatches.append(
                        f"{name9}/{name8} ep{ep} trainer.{f}: {ta[f]!r} vs {tb[f]!r}"
                    )

    print(f"shared-arm runs compared : {len(PAIRS)}")
    print(f"checkpoints compared     : {total_ck}")
    print(f"scalar comparisons       : {total_cmp}")
    print(f"fields per checkpoint    : {total_cmp / max(total_ck, 1):.1f}")
    if missing:
        print(f"fields NOT found (not compared): {sorted(missing)}")
    if total_cmp == 0:
        print("\n!! NOTHING WAS COMPARED. A check that compares nothing always passes.")
        return 1
    if mismatches:
        print(f"\n>>> MISMATCH ({len(mismatches)}):")
        for m in mismatches[:40]:
            print(f"    {m}")
        return 1
    print("\n>>> BIT-IDENTICAL across both jobs. The cross-check PASSES.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
