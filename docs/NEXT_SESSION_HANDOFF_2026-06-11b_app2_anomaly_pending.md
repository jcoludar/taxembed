# Handoff — 2026-06-11 (session b) — App#2 anomaly job PENDING; counters re-verified; gated on LRZ

Branch: `feat/taxembed-eval-foundation` (poincare-embeddings repo). **No new commits this session**
(verification + status check only) — `docs/SESSION_LOG.md` (2026-06-11 "(same day, later)" sub-entry) +
this file are the only deltas. **Supersedes `NEXT_SESSION_HANDOFF_2026-06-11_app2_anomaly_resubmitted.md`
for STATUS only** — that doc's **go/no-go logic is unchanged and still authoritative** (read it for the full
leg-A/C/B reasoning). Reading order: this file → SESSION_LOG 2026-06-11 "(same day, later)" →
`PROJECT_STATE.md` §STATUS(2026-06-11) → the `_resubmitted.md` handoff §"THE go/no-go".

## STATE — gated on the LRZ queue (~24h+)
- **Anomaly job `5675791`** (eukaryota: scorer + leg A + leg C) = **PENDING**. `squeue` reason `(Priority)`,
  partition `lrz-v100x2`, est. start **`2026-06-12T22:43:30`** (SLURM backfill estimate — slips; verify,
  don't trust), 4h walltime. **It has NOT run** → the go/no-go JSONs do not exist yet.
- The OOM fix (`cdae1ff`) is **already in the staged LRZ code** → **no resubmit needed**, just wait for the
  queue.

## CHECK FIRST next session (needs VPN → LRZ)
`ssh ai 'sacct -j 5675791 --format=JobID,State,Elapsed,End -X'`
- **COMPLETED** → pull `…/taxembed_lrz/artifacts/tags/eukaryota_canonical/anomaly/{anomaly_summary.json,
  roc_by_displacement.json,enrichment.json}` and READ them → apply the go/no-go below.
- **still PENDING** → `ssh ai 'squeue -j 5675791 …'` for a fresh ETA; hold (no action — fix is staged).
- **FAILED** → read `…/taxembed_lrz/logs/anomaly_5675791.{err,out}` (the `_safe_batch` OOM cap is in;
  should not recur — if it does, it's a new failure mode, root-cause before resubmitting).

## THE go/no-go (UNCHANGED — KI-8 / PROJECT_STATE item 8; full reasoning in the `_resubmitted.md` handoff)
- **Leg A** (`roc_by_displacement.json`): does `score_z` AUC **beat the trivial baselines** (clade_size,
  depth, degree, dist_to_parent_centroid) **at SMALL displacement classes (0,1)** — the sister-genus/family
  regime reclassification actually is? = proof-of-mechanism.
- **Leg C** (`enrichment.json`): `odds_ratio > 1` with Fisher `p < 0.05`?
- **Both encouraging → commit the 30h leg-B retrain** on an OLD (~2022) archived taxdump (enables a true
  predictive 2022→2025 release-diff, the strong #2 claim). **≈ baselines → do NOT retrain** (would only
  measure confounds); lead #2 on legs A+C, demote B to supporting (spec §9F honesty).
- **Both follow-ups are LRZ submits → require the user's "go"** (and per S0274, no LRZ submit without a
  local end-to-end first — for leg B that's the canonical-recipe smoke).

## Verify-before-trust (RE-CONFIRMED this session — all local, read-only)
- **eval 38/38** (`38 passed in 11.63s`, incl. the 4 new `tests/eval/test_anomaly_knn.py` cap tests).
- **kNN-purity exact** from `cellular_canonical/knn_purity/knn_purity.json`: domain purity@1 **0.979**,
  purity@10 0.985/0.987/0.985/0.979/0.907, lift ladder 1.46→5.72→7.89→56.9→**317.4×**.
- Commits `cdae1ff` (OOM fix) / `240eada` (docs) / `ccb270a` (06-11 handoff); branch
  `feat/taxembed-eval-foundation`, **not pushed** (sentinel discipline).
- **Caveat:** cellular seeded separation **3.67/5.97/7.17/7.69×** is PNGs-only on disk (analyzer prints to
  console; no numeric JSON) and the checkpoint was deleted in the disk reclaim → re-deriving the numbers
  needs an LRZ re-pull. Identical across SESSION_LOG / PROJECT_STATE / handoff — corroborated, not a
  discrepancy.
- Local `data/taxdump_current` = **BROKEN** (only `delnodes.dmp`; names/nodes taxopy-eaten). LRZ
  `/data/taxdump_current` = **pristine** (ro mount + scratch-copy pattern).

## Open / optional (none blocking the lead task)
- **Local taxdump scratch-restore:** `scripts/_extract_taxdump_dmps.py --tarball data/new_taxdump.tar.gz
  --out-dir <scratch>` (tarball 153 MB present; extractor present). **NOT needed for leg-A/C** — those
  compute on LRZ and we only pull+read the JSONs. Needed only for local anomaly work; always point taxopy at
  the scratch copy, never the canonical dir (that's what ate names/nodes).
- **Cellular anomaly run** (2nd job): `ssh ai 'sbatch …/src/scripts/analyze_lrz_anomaly.sh
  cellular_canonical taxonomy_edges_cellular_organisms_131567_clean.mapping.tsv'`. Re-derive **every**
  rank-dependent baseline on the cellular tree — do NOT carry a eukaryota constant (spec §9C). LRZ submit → user go.
- **Uncommitted 06-03 e1c work** (untracked specs/plans/scripts + modified train_hierarchical/train_small/
  cli.main) — untriaged; superseded by the breakthrough (E1c not needed at scale), kept for the record.
- **Disk** 95% / 48 GiB free; TM local snapshot (06-10) still pins freed blocks as purgeable
  (auto-reclaims under pressure / ~24h, or `tmutil thinlocalsnapshots / 16000000000 4`).

## One-liner next-session start prompt (paste-ready)
> Next session (TaxEmbed/App#2): read docs/NEXT_SESSION_HANDOFF_2026-06-11b_app2_anomaly_pending.md +
> SESSION_LOG 2026-06-11 "(same day, later)" + PROJECT_STATE §STATUS(2026-06-11); verify-before-trust (eval
> 38/38 · kNN-purity domain p@1 0.979 / lift→317× · job 5675791 was PENDING est-start 2026-06-12T22:43;
> local taxdump BROKEN, LRZ pristine). Lead task: `ssh ai 'sacct -j 5675791 -X'` → if COMPLETED, pull + read
> eukaryota anomaly {roc_by_displacement,enrichment,anomaly_summary}.json → leg-B go/no-go (leg A: score_z
> AUC beats baselines at displacement 0/1? leg C: odds_ratio>1 & Fisher p<0.05?). If still PENDING, re-check
> ETA and hold. Read-only on LRZ submits until I say go (S0274: no submit without a local end-to-end).

## Discipline (carried)
Verify before trust — eyeballed/re-run evidence above any label; **read-only on LRZ until the user says go**;
**no LRZ submit without a local end-to-end first** (S0274); don't poll/kill background jobs on
"header-only" stdout; record everything durable in SESSION_LOG / PROJECT_STATE, not just chat.
