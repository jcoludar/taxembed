# Handoff — 2026-06-10 — TaxEmbed Application #2 (taxonomy QC, the LEAD app)

Branch: `feat/taxembed-eval-foundation` (poincare-embeddings repo). 9 commits this session, **not pushed**.
Full eval suite: **34 passed** (`tests/eval/`). Reading order: this file → `docs/SESSION_LOG.md` (2026-06-09
entries) → `docs/specs/2026-06-09-taxembed-paper-design.md` §9 (authoritative) → `docs/plans/2026-06-09-taxembed-app2-taxonomy-qc.md`.

## TL;DR — what happened
- **Plan 2 (App #2 taxonomy QC) BUILT end-to-end**, subagent-driven + TDD (Tasks 1–7). New pure cores
  `src/taxembed/eval/{anomaly,release_diff}.py`; CLIs `scripts/taxonomy_anomaly.py` (scorer) +
  `scripts/_anomaly_validation.py` (legs A/B/C).
- **GPU-ported the heavy kernel** (user pivot: heavy compute/data → LRZ). The numpy full-pool scorer was
  single-core O(P²) (26 min+, ~15 GB, no ETA on 877k). New `scripts/_anomaly_knn.py`: device-aware
  (`cuda|mps|cpu`) batched `torch.topk` kNN (all float32 — MPS has no float64; numpy kernel returns float32
  so ordering matches) + **vectorized matched-null** (per-node ~877k Python loop → one draw per depth×size
  bin). Cross-checked vs the numpy kernel exactly (`tests/eval/test_anomaly_knn.py`).
- **S0274 local gate PASSED**: scorer + leg A + leg C run clean end-to-end on echino_softmax (3864 pool)
  via MPS with the exact flag combo.
- **LRZ run IN FLIGHT**: see "Current LRZ state" below.

## Current LRZ state (VERIFY FIRST next session)
- **Anomaly job `5674199`** (eukaryota 877k: scorer + leg A + leg C) — RESUBMITTED 2026-06-10 after the
  first attempt (5673257) FAILED. **Check it first:** `ssh ai 'sacct -j 5674199 --format=JobID,State,Elapsed,End -X'`.
  - If COMPLETED → pull `…/taxembed_lrz/artifacts/tags/eukaryota_canonical/anomaly/{anomaly_summary.json,
    roc_by_displacement.json,enrichment.json}` (small JSONs) and READ them. **These are the go/no-go** (below).
  - If FAILED → read `…/taxembed_lrz/logs/anomaly_5674199.{err,out}`.
- **Cellular training `5673097`** — now **RUNNING** (~9h elapsed at handoff; ~23–27h est). When it lands,
  `cellular_canonical.pth` appears in `…/artifacts/tags/cellular_canonical/`. Re-run the SAME analysis on it:
  `ssh ai 'cd …/src && sbatch scripts/analyze_lrz_anomaly.sh cellular_canonical taxonomy_edges_cellular_organisms_131567_clean.mapping.tsv'`.
  **Re-derive every rank-dependent baseline on the cellular tree — do NOT carry a eukaryota constant** (spec §9C).

## The bug that bit + the fix (already applied, committed `e887891`)
Job 5673257 FAILED in 1 min: TaxoPy `load_taxonomy_with_depth` → `[Errno 30] Read-only file system:
/data/taxdump_current/nodes.dmp`. **TaxoPy writes inside `taxdb_dir`; `/data` is mounted ro in the container.**
Fix in `scripts/analyze_lrz_anomaly.sh`: copy `nodes/names/merged.dmp` to `$TMPDIR/taxdump_work` (rw) and point
`--data-dir` there for the scorer + leg A. Leg C reads `names.dmp` directly (ro is fine). The fixed script is
already scp'd to LRZ and 5674199 uses it.

## THE open scientific decision — leg B (the headline) — KI-8 (PROJECT_STATE.md item 8)
User greenlit a **30h retrain CONDITIONAL on the eukaryota leg A/C results**. The reasoning:
- Leg B (predictive NCBI release-diff) needs the embedding's training taxonomy to **predate** the
  reclassification window (`training ≤ old < new`, enforced by the CLI guard). Our eukaryota/cellular
  embeddings were trained on a **recent (~2026-06) taxdump** → no leakage-free window exists yet. The plan's
  own prose ("old = 3 yr ago, new = training dump") contradicts its guard and would be REFUSED.
- It is **NOT circular**: leg B uses ONE (old) embedding; the new taxdump supplies only the outcome label
  (NCBI's later human reclassification). But the mechanism is narrow — a topology-only embedding can only flag
  revisions **foreshadowed by tension in the old tree**; expect modest enrichment, and beware the score
  collapsing into confounds (depth/size/incertae-sedis).
- **GO/NO-GO (read from job 5674199):**
  - **Leg A** (`roc_by_displacement.json`): does `score_z` AUC **beat the trivial baselines** (clade_size,
    depth, degree, dist_to_parent_centroid) **at SMALL displacement classes (0,1)** — the sister-genus/family
    regime that reclassification actually is? That's the proof-of-mechanism.
  - **Leg C** (`enrichment.json`): `odds_ratio > 1` with Fisher `p < 0.05` (high-score nodes concentrate in
    incertae-sedis/environmental)?
  - **If both encouraging → commit the 30h retrain** on an OLD (~2022) archived taxdump → enables a true
    predictive 2022→2025 release-diff (the strong #2 claim). **If the score ≈ baselines → do NOT retrain**
    (it would just measure confounds); fall back to leading #2 on legs A+C and demoting B to supporting (spec §9F honesty).

## To run leg B once decided (retrain path)
1. Build a dataset from an archived ~2022 taxdump + train the canonical recipe (~30h, mirror
   `scripts/train_lrz_*canonical.sh`). 2. Fetch the OLD + a NEWER archived taxdump
   (`taxembed.utils.ensure_taxdump_archive`, already built). 3. Run the `releasediff` subcommand (already built,
   leakage-guarded; see the commented block at the tail of `scripts/analyze_lrz_anomaly.sh`) with
   `training_date ≤ old_date < new_date`.

## Files / commits this session (branch `feat/taxembed-eval-foundation`)
- `src/taxembed/eval/anomaly.py`, `release_diff.py`; `src/taxembed/utils/taxdump.py` (+`ensure_taxdump_archive`).
- `scripts/{taxonomy_anomaly,_anomaly_validation,_anomaly_knn,_extract_taxdump_dmps}.py`, `analyze_lrz_anomaly.sh`.
- `tests/eval/{test_anomaly,test_release_diff,test_anomaly_cli,test_anomaly_validation_cli,test_anomaly_knn}.py`.
- Local outputs: `artifacts/tags/echino_softmax/anomaly_smoke/` (the MPS validation); `data/taxdump_current/` (extracted dmps).
- LRZ staged: eval pkg + analysis scripts under `…/src/`, taxdump under `…/data/taxdump_current/`.
- 2 review-caught latent bugs fixed (`taxonomy` unbound on `--parent-from-mapping` w/o `--rank-from-mapping`, both CLIs).

## Verify-before-trust counters
- eval suite = 34 passed. · eukaryota ckpt on LRZ = `eukaryota_canonical.pth` ep200 (702 MB). · jobs:
  5674199 (anomaly, check state), 5673097 (cellular, RUNNING). · branch not pushed (sentinel discipline).
