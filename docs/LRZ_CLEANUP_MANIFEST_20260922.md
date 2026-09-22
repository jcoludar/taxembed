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

## Log

| date | action | container before → after |
|---|---|---|
| 2026-09-22 | manifest written; nothing deleted yet | 840 / 1000 GB |
| 2026-09-22 pm | Task 8 queued (+~20 G expected); section B extended; nothing deleted | 840 / 1000 GB (arrays not started) |
