# Handoff — 2026-06-11 — Cellular separation VERIFIED; App#2 anomaly OOM fixed + resubmitted

Branch: `feat/taxembed-eval-foundation` (poincare-embeddings repo). 2 new commits this session
(`cdae1ff` OOM fix, `240eada` docs) + this handoff — **not pushed**. Reading order: this file →
`docs/SESSION_LOG.md` (2026-06-11 entry) → `docs/PROJECT_STATE.md` §STATUS(2026-06-11) →
`docs/NEXT_SESSION_HANDOFF_2026-06-10_app2_anomaly.md` §"THE open scientific decision — leg B"
(the go/no-go logic below is unchanged and still authoritative).

## TL;DR — what happened
- **Cellular all-of-Life separation VERIFIED (the headline).** `cellular_canonical` (job 5673097,
  ep200) — the FULL cellular tree, **1,102,163 nodes / 3 domains**. Seeded separation phylum/class/
  order/family **3.67 / 5.97 / 7.17 / 7.69 ±0.02×** (EXCELLENT, ≈/above eukaryota), depth↔norm +0.954;
  kNN-purity@10 0.91–0.99, lift→**317×**. Domain ratio 1.16× is a top-rank-diversity artifact (domain
  **purity@1 0.979** ⇒ clean non-interpenetrating neighbourhoods). Recipe scales 4k→498k→877k→1.1M, no
  degradation. Artifacts: `artifacts/tags/cellular_canonical/{analysis_final_seeded,knn_purity}/`.
- **App#2 anomaly OOM FIXED + revalidated + resubmitted.** Job 5674199 re-FAILED 06-11 (CUDA OOM, not
  the taxdump-ro bug — that's fixed). Root cause: `observed_purity` materialised ~6 live (block×P) f32
  tensors; block=2048 × 864k pool → ~30 GiB on a 16 GiB V100. Fix (`cdae1ff`, TDD): `_safe_batch` caps
  the block to a ~4 GiB working set regardless of `--knn-batch` (P=1.1M → block ~120, ~5 GiB peak);
  batch-invariant ⇒ lossless. +4 tests (**38/38 eval pass**); sbatch hardened
  (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`); fixed scripts scp'd + verified on LRZ. **S0274
  local end-to-end gate PASSED** (`scripts/_smoke_anomaly_echino.py` — scorer+legA+legC on echino/MPS).

## CHECK FIRST next session (needs VPN → LRZ)
- **Anomaly job `5675791`** (eukaryota: scorer + leg A + leg C) — was **PENDING** at handoff.
  `ssh ai 'sacct -j 5675791 --format=JobID,State,Elapsed,End -X'`.
  - If COMPLETED → pull `…/taxembed_lrz/artifacts/tags/eukaryota_canonical/anomaly/{anomaly_summary.json,
    roc_by_displacement.json,enrichment.json}` and READ them — these are the go/no-go.
  - If FAILED → read `…/taxembed_lrz/logs/anomaly_5675791.{err,out}` (OOM should NOT recur; the cap is in).

## THE go/no-go (unchanged from the 2026-06-10 handoff — KI-8 / PROJECT_STATE item 8)
User greenlit a **30h retrain CONDITIONAL on the eukaryota leg A/C results**:
- **Leg A** (`roc_by_displacement.json`): does `score_z` AUC beat the trivial baselines (clade_size,
  depth, degree, dist_to_parent_centroid) **at SMALL displacement classes (0,1)** — the sister-genus/
  family regime reclassification actually is? That's proof-of-mechanism.
- **Leg C** (`enrichment.json`): `odds_ratio > 1` with Fisher `p < 0.05`?
- **Both encouraging → commit the 30h retrain** on an OLD (~2022) archived taxdump → enables a true
  predictive 2022→2025 release-diff (leg B, the strong #2 claim). **If ≈ baselines → do NOT retrain**;
  lead #2 on legs A+C and demote B to supporting (spec §9F honesty).

## Open items (none blocking the above)
- **Cellular anomaly run** (optional, 2nd job): `ssh ai 'sbatch …/src/scripts/analyze_lrz_anomaly.sh
  cellular_canonical taxonomy_edges_cellular_organisms_131567_clean.mapping.tsv'`. Re-derive every
  rank-dependent baseline on the cellular tree — do NOT carry a eukaryota constant (spec §9C).
- **Local `data/taxdump_current` is BROKEN** — `taxopy.TaxDb` *consumes/deletes* names.dmp+nodes.dmp
  from its taxdb_dir on load (this, not Dropbox, ate them). LRZ's original is pristine (ro mount +
  scratch-copy). To do local analysis: re-extract (`scripts/_extract_taxdump_dmps.py --tarball
  data/new_taxdump.tar.gz --out-dir <scratch>`) and **always point taxopy at a scratch copy**, never the
  canonical dir. The `_smoke_anomaly_echino.py` gate shows the correct WORK-copy-vs-pristine-names pattern.
- **Disk was 98% full** (the real constraint, not RAM — machine has 96 GB). Reclaimed ~14 GB this session
  (artifacts 18→4.1 GB; LRZ-backed milestones + smokes deleted, all finals kept). A TM local snapshot
  (2026-06-10) still pins the freed blocks as purgeable — auto-reclaims under pressure / ~24h, or
  `tmutil thinlocalsnapshots / 16000000000 4`.
- **Uncommitted 06-03 e1c work** (untracked specs/plans/scripts + M train_hierarchical/train_small/
  cli.main) — untriaged; SUPERSEDED by the breakthrough (E1c not needed at scale) but kept for the record.

## Verify-before-trust counters
- cellular: 3.67/5.97/7.17/7.69 ±0.02× sep, depth +0.954, purity@10 0.91–0.99 (lift→317×), domain
  purity@1 0.979. · OOM fix `cdae1ff`, eval **38/38**, S0274 gate PASSED. · job **5675791** (PENDING).
- LRZ `/data/taxdump_current` names/nodes = pristine. · local `data/taxdump_current` = BROKEN (taxopy-eaten).
- branch `feat/taxembed-eval-foundation`, 2 commits + handoff, **not pushed** (sentinel discipline).
