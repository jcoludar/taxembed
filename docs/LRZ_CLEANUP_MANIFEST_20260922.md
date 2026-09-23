# LRZ cleanup manifest — TaxEmbed (USER request, 2026-09-22)

> *"make sure that once we have done the runs, we remember to clean up our project's files on LRZ so
> as not to clutter it"* — USER, 2026-09-22.

**Root:** `/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/` (called `$R`
below; paths are written out in full in the commands). The container `pr63ci-dss-0004` is
**shared**. On 2026-09-22 it held **840 / 1000 GB** before the Task 9 array (~65 GB) was submitted.
TaxEmbed was ~108 GB of that.

**Rule:** bank first, then delete. Copy locally anything a result depends on (the scored JSON/NPZ,
parsed logs), verify the copy, and only then delete. Nothing here is covered by Dropbox; an LRZ
delete is permanent. Re-run `ssh ai dssusrinfo all` after and record before/after below.

## A. Safe now (done runs, verified, nothing depends on them)

| path | size | why safe |
|---|---|---|
| `$R/artifacts/tags/smoke_task9_canonical/` | 1.5 G | smoke 5743270 passed 2026-08-11; log verified this session |
| `$R/artifacts/tags/smoke_task9_prior/` | 1.5 G | same |
| `$R/artifacts/tags/metazoa_smoke/` | 1 K | May smoke |

```
ssh ai rm -r /dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/artifacts/tags/smoke_task9_canonical /dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/artifacts/tags/smoke_task9_prior /dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/artifacts/tags/metazoa_smoke
```

## B. After the Task 9 results are banked locally

| path | size | condition |
|---|---|---|
| `$R/artifacts/tags/task9_{canonical,prior}_s{0,1,2}/` | ~65 G total | array 5802007 scored with `scripts/score_recipe_checkpoints.py`, JSON+NPZ copied to local `results/` |
| `$R/artifacts/task9_scoring/fig4_runs_SMOKE_*` | small | the smoke output, once the full scoring has run |
| `$R/logs/task9_*` (the `.err` files especially) | varies | after trajectories are parsed into a local file |

| `$R/artifacts/tags/task8_fixed_canonical_s{0,1,2}/` | ~20 G total | array 5802297 scored and banked locally |
| `$R/artifacts/tags/smoke_task8_fixed/` | small | smoke 5802277 PASSED 2026-09-22; delete any time |
| `$R/src_task8/` | ~5 M | after array 5802297 and its scoring are done (keep until then: it is mounted) |
| `$R/artifacts/task9_scoring/*` | small | once each JSON/NPZ is banked locally with md5 (fig4_runs_20260922_104857 and transplant_2x2_20260922_132857 already are) |

**Keep** the per-run `run.json` and the final `task9_*_s*.pth` until the USER decides whether the
seeded runs become a release artifact. They are small relative to the milestones.

## C. Ask the USER (older runs; not needed by any current result I know of)

| path | size |
|---|---|
| `$R/artifacts/tags/eukaryota_canonical/` | 18 G |
| `$R/artifacts/tags/metazoa_lower_lr_bigger_batch_repro/` | 11 G |
| `$R/artifacts/tags/metazoa_softmax/` | 2.6 G |
| `$R/artifacts/tags/metazoa_softmax_nocurric/` | 2.6 G |
| `$R/artifacts/tags/metazoa_v2_euclparam/` | 2.6 G |
| `$R/artifacts/tags/metazoa_v2_nocurric/` | 2.6 G |
| `$R/artifacts/tags/metazoa_v1/` | 1.3 G |
| the ~2 GB of old `.err` progress-bar logs in `$R/logs/` | ~2 G |

The ablation-era runs may back the manuscript's ablation table (spec v3 §6). Check before deleting.

## D. Keep (back a figure or a shipped artifact)

| path | size | why |
|---|---|---|
| `$R/artifacts/tags/cellular_canonical/` | 23 G | the shipped 1.1M model |
| `$R/artifacts/tags/metazoa_softmax_milestones/` | 11 G | Figure 4 prior trajectory; Task 9 scoring input |
| `$R/artifacts/tags/metazoa_lower_lr_bigger_batch/` | 11 G | Figure 4 canonical trajectory; Task 9 scoring input |
| `$R/data/`, `$R/src/`, the container `.sqsh` | ~1 G + image | needed by every job |

## E. Container-wide audit, 2026-09-23 (USER request)

⚠ **This manifest covers TaxEmbed only — ~168 GB of the 743 GB the account holds.** The USER asked
for a whole-container audit, so the other nine project directories are recorded here. Sizes
measured 2026-09-23; `du` on this filesystem is slow (~15 min for the large trees).

| dir | size | project / status | verdict |
|---|---|---|---|
| `taxembed_lrz` | **168 G** | TaxEmbed — live | KEEP; sections A-D above govern |
| `unknown_unknowns_lrz` | ~150-215 G | unknown_unknowns — **jobs running now** | KEEP; it has self-cleaned twice |
| `ant_venoms_lrz` | not measured (du >15 min) | ant_venoms — preprint out, but TE/repeats fetched for only **48 of 54** genomes | KEEP |
| `oe_autoeval_lrz` | 94 G → **83 G** | ProteEmbedExplorations sweep, complete; 14 reports local | `out/` 80 G still there |
| `plm_choice_lrz` | 83 G | plm_choice — `paper_revision`; compute done 2026-06-09 | `data/hf_cache` 80 G is the target |
| `syntile_lrz` | 17 G | syntile — **active**, ERC preprint, needs DSS rows | KEEP (self-capped at 100 G) |
| `tax_disentangle_lrz` | 1.8 G | tax_disentangle — results "complete and triple-checked" | deletable; back up untracked local panel/H5 first |
| `exabayes_lrz` | 229 M | **shared container**, not a project | 🛑 KEEP — **16 committed sbatch files** reference `exabayes-1.5.1.sqsh`, and no local copy exists |
| `fry_lab` | 23 M | Fry_lab — finished 2026-06-20, trees local | deletable |
| ~~`peptideminer_lrz`~~ | ~~99 G~~ | peptideminer_embeddings — Phase B done | ✅ **DELETED 2026-09-23** |

🧨 **The big directories are mostly CACHES, not results.** `plm_choice`'s real `artifacts/` is
**547 KB**; its 83 G is a HuggingFace download cache. `oe_autoeval`'s `reports/` is 5.2 MB. Size is
a poor guide to value here — measure the subdirectories before judging a directory.

🛑 **A dormant directory can break a live one.** `exabayes_lrz` holds the ExaBayes container used by
ant_venoms and Fry_lab; `taxembed_lrz` holds the PyTorch `.sqsh` that plm_choice used and
tax_disentangle mounts read-only. Check for cross-mounts before deleting anything.

## Log

| date | action | container before → after |
|---|---|---|
| 2026-09-22 | manifest written; nothing deleted yet | 840 / 1000 GB |
| 2026-09-22 pm | Task 8 queued (+~20 G expected); section B extended; nothing deleted | 840 / 1000 GB (arrays not started) |
| 2026-09-23 | container-wide audit (section E) | 920 / 1000 GB |
| 2026-09-23 | **`peptideminer_lrz` deleted (~99 G)**, USER-authorised. All four primary outputs verified **byte-identical** to `projects/peptideminer_embeddings/work/lrz_candidates/` first | 920 → (quota lags) |
| 2026-09-23 | **`oe_autoeval_lrz/{venv,pip_cache}` deleted (~11 G)**, USER-authorised. `venv` was already a broken symlink; `venv.freeze.txt` kept, so the env is reproducible. `out/`, `reports/`, `logs/` untouched | → 918 / 1000 GB; files 585,922 → 516,768 |
| 2026-09-23 | `plm_choice_lrz/{artifacts,logs}` (1.8 MB) banked to `projects/plm_choice/results/lrz_artifacts_20260923/` — the one thing never confirmed downloaded | — |
| 2026-09-23 | ✅ **`plm_choice_lrz/data/{hf_cache,torch_cache}` deleted (~80 G) — RUN BY THE USER**, after the harness permission classifier refused it to Claude. USER-authorised conditionally ("if not touched in the last week or two"); condition **verified met** — `find -newermt 2026-09-01` returned nothing, against a control at `-newermt 2026-06-01` that returned ~100 hits. `artifacts/` + `logs/` banked first | 918 → **858 / 1000 GB** |

**Net effect of the 2026-09-23 audit: 920 → 858 GB with ~20-30 GB of run output written meanwhile,
i.e. ~190 GB reclaimed. Free headroom 80 GB → 142 GB**, which clears the overnight risk that the
Task 8/9 arrays (~37 GB still to write) would hit a full container.

**Still available, not done:** `oe_autoeval_lrz/out` (80 G) · `plm_choice_lrz/data/venv` (781 M,
another dead env) · `tax_disentangle_lrz` (1.8 G) · `fry_lab` (23 M). `ant_venoms_lrz` was never
sized — `du` exceeded 15 min twice.
