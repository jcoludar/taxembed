# Project State — Poincaré Taxonomy Embeddings

Living snapshot of "what we know right now." Update this when the recipe, metrics, or
open issues change. Timestamped narrative lives in `SESSION_LOG.md`.

_Last updated: 2026-06-09 (ALL-LIFE scale EXCELLENT + reproducibility lock CLOSED)_

## ✅ STATUS (2026-06-03): METAZOA SCALE SOLVED — EXCELLENT achieved at 498k

**Experiment 1 (`metazoa_lower_lr_bigger_batch`, job 5664609) cleared the EXCELLENT bar at full
metazoa scale: depth↔norm +0.984, separation phylum 2.56× / class 3.86× / order 6.65× / family
10.31× (final ep200).** The 1.20× ceiling that this doc previously described as the architecture's
limit is **obsolete** — it was a training-dynamics (curriculum-transition collapse) problem, fixed by
**effective batch 2048 (bs256 × grad-accum 8) + n_negatives 300 + lr 0.001 + cosine warm-restart on
each curriculum-phase boundary** (default sampler; Job-C base recipe). See "Roadmap" for the trajectory
+ POST-BREAKTHROUGH next steps. **Any text below describing a "1.20× ceiling" or "echino recipe doesn't
scale" or the E1c/E2/E3 negative-sampling plan is SUPERSEDED** (kept for the record + the reusable
diagnostic tooling/learnings).

**✅ RESULT LOCKED (2026-06-04).** Two independent anti-Goodhart checks confirm the 6–10× is genuine
*angular* structure, not a radial/Poincaré-metric artifact:
- **Seeded separation — mean±std over 5 seeds** (`analysis_final_seeded/`, `analyze_hierarchy_hyperbolic.py
  --seed 0 --repeats 5`): phylum **2.60 ± 0.01** / class **3.85 ± 0.01** / order **6.68 ± 0.01** /
  family **10.30 ± 0.01×**; depth↔norm +0.984 (Pearson) / +0.998 (Spearman). The ±0.01 noise floor
  puts the magnitude far beyond subsampling slop, and reproduces the original unseeded sweep.
- **kNN-purity per rank** (`knn_purity/`, `knn_purity_hyperbolic.py`, Poincaré k-NN over the full 498k
  pool): purity@10 = 0.997 / 0.997 / 0.995 / 0.919 (phylum→family), with **lift-over-chance rising**
  2.1× → 3.2× → 17.9× → **216.6×**. A node's nearest neighbours are 92–99% same-clade — a local,
  radius-independent measure. A radial Goodhart artifact would leave family purity at chance (0.004);
  it sits at 0.92 (217× chance). ⇒ separation is real angular clustering. (depth↔norm flat at +0.984
  while separation climbs 1→10× already pointed here; kNN-purity nails it.)

## ✅ STATUS (2026-06-09): ALL-LIFE SCALE EXCELLENT + REPRODUCIBILITY LOCK CLOSED

The canonical recipe now clears EXCELLENT from a single phylum (echino 4k) all the way to **all of
Life (Eukaryota, 877k nodes)** — and the metazoa breakthrough reproduces independently. Two LRZ runs
pulled + analyzed 2026-06-09 (analysis under each tag's `analysis_final_seeded/`):

- **Reproducibility lock CLOSED — `metazoa_lower_lr_bigger_batch_repro` (job 5666348).** Same canonical
  config, fresh run. Seeded (5 seeds): phylum **2.59 ± 0.01** / class **3.84 ± 0.01** / order
  **6.69 ± 0.01** / family **10.27 ± 0.01×**; depth↔norm +0.984 / +0.998. Matches Experiment 1
  (2.60/3.85/6.68/10.30) to within the ±0.01 noise floor ⇒ the breakthrough is **deterministic, not a
  lucky seed**. (`--seed` still not wired into training; reproduced by re-running the exact config.)
- **All-Life — `eukaryota_canonical` (job 5666473), 877k nodes, 20h30m on V100.** Final ep200, seeded
  (5 seeds): phylum **3.12 ± 0.02** / class **5.09 ± 0.01** / order **7.13 ± 0.02** / family
  **8.23 ± 0.01×**; depth↔norm +0.978 / +0.999. **EXCELLENT at every rank** ("phylum" here ≈ kingdom).
  No architecture ceiling, no hard-negative-starvation collapse at the largest, deepest tree we have.
- **Stall-watch resolved (the ep80–120 worry did NOT materialize).** Milestone trajectory
  (`trajectory_readout.json`, family-rank): ep80 **2.34** → ep100 **3.75** → ep120 **3.54** →
  ep150 **3.68** → ep180 **6.56** → ep200 **8.23**. There is a **soft plateau in ep100–150** (the
  flagged window) but it is a *plateau, not a collapse* — nothing like the metazoa curriculum-transition
  failure that motivated the E1c/E2/E3 negative-sampling plan. The final dd≤all phase + cosine
  warm-restart then drives a hard late climb (separation **more than doubled** in the last 50 epochs).
  ⇒ reinforces **use `final` (ep200), NOT `best`** for these recipes, even more strongly than at metazoa.

**Implication:** the E1c hard-negative sampler (and E2 cones / E3 structural alternatives) is **NOT
needed to reach EXCELLENT at scale** — the default sampler + the four canonical levers suffice through
877k. The E1c tooling/diagnostics remain valuable for the record, but the headline question ("does PC
embedding represent taxonomy well at full scale?") is answered YES. Remaining work is presentational
(figures/write-up) + optional polish, not a capability gap. Space note: Exp-1 tag trimmed to final+best
locally (milestones recoverable from LRZ); ~9.5 GB reclaimed.

## Goal & bar

Taxonomic embeddings where **distance is monotone in relatedness at every level**
(sister species closest → genus → family → …) and **radius encodes depth** (root near
origin, leaves near boundary). A position should be taxonomically meaningful enough to
represent any species/clade. The bar is high: we want EXCELLENT separation
(≫1.5–2×), not just >1×.

## How to evaluate an embedding

```
.venv/bin/python scripts/analyze_hierarchy_hyperbolic.py \
  --checkpoint artifacts/tags/<tag>/<tag>_best.pth \
  --mapping data/taxopy/<dataset>/taxonomy_edges_<dataset>.mapping.tsv \
  --ranks <ranks for the clade> \
  -o artifacts/tags/<tag>/analysis/
```

- Reports **depth↔norm correlation** (radial structure; want ~0.9) and per-rank
  **separation ratio** = mean(inter-group dist) / mean(intra-group dist). Thresholds in
  the script: >2 EXCELLENT, >1.5 GOOD, >1.2 MODERATE, else POOR.
- **Ranks must suit the clade.** Whole-Metazoa: `phylum class order` (+ finer). A single
  phylum like Echinodermata: use `class order family` (phylum = the whole set → N/A).
- **GOTCHA — `--tag` mode fails locally.** It reads `training.paths.mapping` from
  `run.json`, which stores the *LRZ container* path `/data/...`. Always pass explicit
  `--checkpoint` + `--mapping` for local analysis.

## The recipe (what's load-bearing)

- **🏆 CANONICAL METAZOA-SCALE RECIPE (2026-06-04 — Experiment 1
  `metazoa_lower_lr_bigger_batch`, job 5664609; LOCKED, see STATUS banner).** Clears EXCELLENT on
  every rank at full 498k scale (phylum 2.60 / class 3.85 / order 6.68 / family 10.30×, depth↔norm
  +0.984). **Validated at metazoa 498k.** ✅ **Generalizes across scale with ONE knob** (2026-06-04): the only
  scale-dependent lever is the **effective batch** (= gradient-step count). Drop it to 256 on smaller clades
  (`--grad-accum-steps 1`) and mollusca 32k goes from POOR to EXCELLENT (order 3.62 / family 4.76×); keep/raise
  it on larger clades. Three scales now EXCELLENT — echino 4k, mollusca 32k, metazoa 498k — same recipe modulo
  eff-batch. See known-issue #7 + SESSION_LOG 2026-06-04.
  Four levers on top of the Job-C softmax base:
  **effective batch 2048 (`--batch-size 256 --grad-accum-steps 8`) · `--n-negatives 300` · `--lr
  0.001` · `--lr-schedule cosine_warmrestart --warm-restart-on-phase --lr-min-multiplier 0.01`**
  (cosine decays to 1% of base within each curriculum phase, then restarts at every dd-window
  boundary so base_lr returns when a harder window loads). Base carried from Job C: `--loss softmax
  --euclidean-param --curriculum --curriculum-phases auto --radial-nudge 0.05 --radial-schedule log
  --depth-scale-margin --margin-min 0.05 --margin-max 1.0 --epoch-fraction 0.3 --amp
  --early-stopping 999 --dim 100 --epochs 200 --save-every 10`. **Default negative sampler (tiered
  OFF).** Verbatim command: `artifacts/tags/metazoa_lower_lr_bigger_batch/run.json`; submit script:
  `scripts/train_lrz_metazoa_lower_lr_bigger_batch.sh`. **⚠ Use `final` (ep200), NOT `best`** — the
  loss/quality-"best" checkpoint is 2.43 / 3.35 / 4.20 / 4.53× (well below final on separation);
  separation keeps climbing for ~90 epochs after the loss plateaus, so the loss-"best" selector saves
  the wrong checkpoint. (Add separation to checkpoint selection, or just take ep200.)

- **(echino-scale / small-clade recipe — still valid below ~10k nodes)** BEST RECIPE (2026-05-29):
  `--loss softmax --euclidean-param` + radial machinery
  (radial-nudge 0.05, lambda-reg 0.1).** On echino this gives depth↔norm +0.96 and
  **order 3.01× / family 2.63× separation (EXCELLENT)** — vs 1.21× for the same setup with
  the margin-ranking loss. The softmax/NLL objective (push against ALL negatives, no hinge
  dead-zone) is what unlocks strong lateral clustering. NOTE: softmax WITHOUT the radial
  scaffolding collapses to ~1.0× — the scaffolding (depth frame) and softmax (angular
  structure) are complementary, both required. Class stays MODERATE (~1.46×; a class spans
  many depths so intra-class spread is inherently large). `--loss` lives in train_small.py
  (default `ranking`); NOT yet wired into the `taxembed` CLI.


- **`--euclidean-param` is REQUIRED.** Optimize in Euclidean tangent space, map to the
  ball via tanh. Without it (Adam on raw ball coords) you hit conformal-factor gradient
  collapse near the boundary and angular clustering dies. **Both metazoa_v1 and the
  first echino baselines omitted it — that's the main misconfiguration.**
- Documented best recipe (`echino_v9d`, r=+0.990, **1.68×**):
  `--euclidean-param --optimizer adam --radial-nudge 0.05 --tiered-negatives
  --class-weighted-loss --lambda-reg 0.1`, ~100 epochs.
- `--tiered-negatives` (sibling hard-negatives) HELPS only *with* the full recipe;
  used alone (no euclidean-param) it HURTS (pushes relatives apart).
- **Validated 2026-05-29:** `--euclidean-param` alone lifts depth↔norm +0.85→+0.96 and
  separation to 1.21× on the current cleaned echino (= historical echino_v4). But
  `--tiered-negatives --class-weighted-loss` did NOT help here and did NOT reproduce the
  documented echino_v9d **1.68×** — likely because v9d used a 7,833-node echino set vs
  our 3,965-node `_clean` build. Best current config = **`--euclidean-param` alone**.
  Lateral separation ceiling so far ~1.2× (MODERATE), below the EXCELLENT bar.
- Don't let **early-stopping fire mid-curriculum.** The stepwise curriculum keeps dd≤1
  until ~ep40; patience 25 + a `quality` metric that peaks early = saving a barely-trained
  checkpoint. Disable early-stop or set patience past the curriculum length.
- **Ranking loss at scale is worse than ranking loss small (2026-05-30 finding).** Per-phylum
  slice of `metazoa_v2_euclparam_best` (498k → echino subset of 3,965 nodes) gives
  separation **1.05-1.09×** across class/order/family — vs **1.21×** for the standalone
  echino_euclparam run on the same 4k nodes. Same data, same recipe, different training
  scope. **Rules out the "scale crowds lateral structure" hypothesis.** Likely mechanism:
  with 498k candidates, n_negatives=100 ranking-margin sampling is dominated by trivially-far
  random metazoans, so the gradient signal for within-phylum geometry collapses. Softmax
  pushes against ALL negatives (no random-easy-negative dilution) — exactly the lever we
  need. Implication: softmax isn't just "better" at scale, it's the **only remaining lever**
  for clearing the bar at metazoa scale. Slicing helper:
  `scripts/_slice_by_phylum_and_analyze.py`; per-phylum TSV at
  `artifacts/tags/metazoa_v2_euclparam/analysis/per_phylum_separation.tsv`.
- **🛑 dd≤9 → dd≤18 transition at ep 80 is the breaker (2026-06-02 finding, LOAD-BEARING; supersedes the open question in the 2026-05-31 bullet).**
  The 20-milestone trajectory from `metazoa_softmax_milestones` pinpoints collapse to the
  dd≤18 phase transition, NOT the dd≤all jump as previously assumed. Trajectory at 498k:
  - **dd≤1 (ep 1-39):** model warms up cleanly; peak depth↔norm +0.93 at ep 30; sep 1.00-1.10× (sub-task is too narrow for lateral structure).
  - **dd≤9 (ep 40-79):** 🟢 **PEAK QUALITY AT SCALE.** Depth +0.88, separation 1.14-1.20×
    sustained for 20 epochs (ep 50-70). This matches the historical `metazoa_softmax_best`
    numbers — that "best" was captured during this phase.
  - **dd≤18 jumps in at ep 80:** 🔴 immediate damage. Depth +0.88→+0.80 in 10 epochs;
    separation 1.20→1.07×. The model is unlearning the dd≤9 representation it just built.
  - **dd≤all (ep 120):** by the time this loads in, the model is already at +0.67 / 1.04× —
    the dd≤all phase is just locking in a wall the dd≤18 phase already crashed it into.
  Per-axis numbers in the Current metrics table. Mechanism: dd≤9 → dd≤18 doubles the
  transitive-pair density per node and explodes the number of meaningful negatives per
  example; n_negatives=100 partition becomes too sparse to give a clean gradient. The
  no-curriculum runs (A=softmax, B=ranking) BOTH start at the dd≤all wall and never escape
  — confirming the model can't bootstrap directly on the hard objective at this scale with
  the current recipe.
- **Echino-recipe ceiling at metazoa scale = 1.20×, not 3.01× (2026-06-02 finding).**
  The peak quality the current architecture can achieve on 498k metazoa is the dd≤9-phase
  plateau at ep 50-70: depth +0.88, separation 1.14-1.20× (POOR/borderline MODERATE). The
  echino 3.01× does NOT transfer to scale. Even if the dd≤18/dd≤all transitions were
  survived perfectly, there's no evidence within reach that the architecture can clear the
  EXCELLENT bar (≥1.5×) at 498k. Next experiments listed in Roadmap.

## Current metrics

| Run | euclidean-param | extras | depth↔norm | separation (best ranks) | verdict |
|-----|-----------------|--------|-----------|--------------------------|---------|
| metazoa_v1 (498k) | no | curriculum, es25 | +0.35 | ~1.0× all ranks | FAILED |
| metazoa_v2_euclparam BEST (498k, ep ~21, dd≤1) | **yes** | ranking | +0.878 | 1.05 / 1.08 / 1.09 / 1.05× | dd≤1 phase only — best.pth saved before curriculum advance |
| metazoa_v2_euclparam FINAL (498k, ep 200, dd≤all) | **yes** | ranking | +0.727 | 1.02 / 1.03 / 1.02 / 1.00× | degraded through curriculum — both depth + separation worse than best |
| metazoa_softmax BEST (498k, ep 39, dd≤1) | **yes** | **softmax** | +0.875 | **1.14 / 1.20 / 1.19 / 1.15×** | best result at scale; ~10% lift over v2 — but still on the dd≤1 easy task |
| metazoa_softmax FINAL (498k, ep 200, dd≤all) | **yes** | **softmax** | +0.667 | 1.03 / 1.04 / 1.04 / 1.03× | CATASTROPHIC — degraded harder than v2 final on every axis |
| **metazoa_softmax_milestones ep50-70** (498k, dd≤9 phase) | yes | softmax | **+0.875** | **1.14 / 1.20 / 1.19 / 1.15×** | 🟢 **PEAK QUALITY AT SCALE** — sustained 20-epoch plateau before dd≤18 jump |
| metazoa_softmax_milestones ep80 (498k, dd≤18 jumps in) | yes | softmax | +0.796 | 1.08 / 1.10 / 1.09 / 1.07× | 🔴 **COLLAPSE BEGINS** — depth drops 0.88→0.80 within the phase transition |
| metazoa_softmax_milestones ep120 (498k, dd≤all jumps in) | yes | softmax | +0.673 | 1.04 / 1.05 / 1.04 / 1.03× | catastrophic — already collapsed by the dd≤18 phase |
| metazoa_softmax_milestones FINAL ep200 | yes | softmax | +0.669 | 1.03 / 1.04 / 1.04 / 1.03× | identical to original metazoa_softmax FINAL — deterministic reproduction |
| metazoa_softmax_NOCURRIC best/final ep26/200 | yes | softmax, **no curric** | +0.669 | 1.03 / 1.04 / 1.04 / 1.03× | 🛑 **dropping curriculum does NOT help** — softmax stuck at dd≤all wall from ep 1 |
| metazoa_v2_NOCURRIC best/final ep38/200 | yes | ranking, **no curric** | +0.726 | 1.02 / 1.02 / 1.02 / 1.00× | floor case — ranking + no-curric = essentially nothing learned |
| **metazoa_lower_lr_bigger_batch ep80** (Exp1, dd≤18 loads) | yes | softmax, **effbatch 2048 + n_neg 300 + lr 0.001 + cosine-warm-restart** | **+0.984** | **1.94 / 2.46 / 2.08 / 1.61×** | 🟢 **SURVIVES the dd≤18 transition** (Job C died here at +0.796/1.10×) |
| **metazoa_lower_lr_bigger_batch FINAL ep200** (Exp1, job 5664609) | yes | " (default sampler) | **+0.984** | **2.56 / 3.86 / 6.65 / 10.31×** | 🎉 **SOLVED — EXCELLENT every rank at 498k; 1.20× ceiling shattered** (phylum/class/order/family) |
| **metazoa_lower_lr_bigger_batch_repro FINAL ep200** (job 5666348, 2026-06-09, 5-seed) | yes | " (default sampler) | **+0.984** | **2.59 / 3.84 / 6.69 / 10.27 ±0.01×** | ✅ **REPRODUCIBILITY LOCK CLOSED** — matches Exp1 within the ±0.01 noise floor; breakthrough is deterministic |
| **eukaryota_canonical FINAL ep200** (877k ALL-LIFE, job 5666473, 2026-06-09, 5-seed) | yes | " (default sampler) | **+0.978** | **3.12 / 5.09 / 7.13 / 8.23 ±0.01×** | 🌍 **ALL-LIFE EXCELLENT** ("phylum"≈kingdom); recipe scales 4k→877k. Trajectory: family 2.34(ep80)→3.75(100)→3.54(120)→3.68(150)→6.56(180)→8.23(200) — soft ep100–150 plateau, NO collapse |
| echino_v1 (4k) | no | — | +0.85 | 1.13/1.16/1.07× | weak |
| echino_v2_tiered (4k) | no | tiered | +0.85 | 1.03/1.03/0.97× | worse |
| echino_euclparam (4k) | **yes** | — | +0.96 | 1.21/1.21/1.11× | reproduces echino_v4 |
| echino_v9d_repro (4k) | **yes** | tiered + class-wt | +0.96 | 1.12/1.18/1.09× | extras didn't help |
| **echino_softmax (4k)** | **yes** | **softmax loss + scaffolding** | **+0.96** | **1.46 / 3.01 / 2.63×** | **EXCELLENT (order/family)** |
| echino_softmax_pure (4k) | yes | softmax, NO radial scaffolding | +0.81 | 1.00/0.99/0.97 | no clustering |
| echino_softmax_abctrl (4k, 2026-06-03 repro) | yes | softmax + scaffold, es-off, 200ep | +0.956 | 1.46 / 2.54 / 1.99× | reproduces winning regime (order EXCELLENT) — A/B baseline |
| echino_softmax_tiered (4k, 2026-06-03) | yes | softmax + scaffold + **tiered-neg** | +0.967 | 1.28 / 1.61 / 1.31× | **tiered DEGRADES every rank** — 50% same-grandparent hard tier cannibalizes within-clade clustering; radial untouched |
| _historical echino_v4_ | yes | — | +0.95 | 1.21× | (doc) |
| _historical echino_v9d_ | yes | tiered + class-wt | +0.99 | 1.68× | (doc, best) |

## Known issues / gotchas

1. `--euclidean-param` required (see recipe). Omitting it ≈ broken angular structure.
2. Early-stop-during-curriculum saves an under-trained "best" checkpoint.
3. `--tag` analysis mode broken locally (LRZ container paths in run.json).
4. `--tiered-negatives` without the full recipe degrades clustering.
5. Even the historical best is 1.68× ("GOOD"), not EXCELLENT — may not clear the bar;
   the archived Nickel–Kiela softmax loss
   (`docs/archive/legacy_facebook/hype/energy_function.py`) is an untried alternative.
6. **✅ RESOLVED 2026-06-04 (Experiment 1) — the recipe NOW scales.** The "does not scale" finding
   below held for the *old* recipe only (n_neg 100, lr 0.005, eff-batch 512, no warm-restart): the
   four-lever canonical recipe survives both curriculum transitions and clears EXCELLENT at 498k
   (LOCKED — see STATUS banner). The collapse was curriculum-transition dynamics, fixed by bigger
   batch + more negatives + warm-restart, NOT a capacity ceiling and NOT the negative sampler (Exp1
   used the default sampler). Historical finding kept for the record:
   **🛑 (HISTORICAL) The full recipe DOES NOT SCALE from 4k echino to 498k metazoa (2026-05-31 finding, REFINED 2026-06-02).**
   The 3-diagnostic sweep (A=softmax-no-curric, B=ranking-no-curric, C=softmax+milestones)
   completed 2026-06-02. **Verdict:** the dd≤9 → dd≤18 transition at ep 80 is where collapse
   begins — NOT the dd≤all jump as previously hypothesized. Peak quality at scale = ep 50-70
   plateau (1.14-1.20× sep, depth +0.88). Dropping curriculum did NOT rescue softmax (A=B=1.02-1.04×
   from ep 1 onward). See the dd≤9→dd≤18 bullet under "The recipe" + 6 new rows in the
   metrics table + `SESSION_LOG.md` 2026-06-02 entry. **Do not re-launch the same recipe at
   metazoa scale.** Next experiments in Roadmap target either truncating at dd≤9 or surviving
   the dd≤18 transition via larger batch / more negatives / LR schedule.
7. **✅ RESOLVED 2026-06-04 — generalization needs ONE scale-aware knob: effective batch (gradient-step count).**
   The canonical (metazoa) recipe did NOT transfer *unchanged* to the 32k mollusca clade — with eff-batch
   2048 it gave radial +0.99 but **POOR lateral separation (class 0.96 / order 0.98 / family 1.01× at ep100)**.
   Diagnosis (confirmed): at 32k nodes with epoch-fraction 0.3, eff-batch 2048 yields only ~41 grad-steps/epoch
   → far too few updates (the big-batch regime is calibrated to metazoa's pair volume). **The single-lever fix
   — `--grad-accum-steps 1` (eff-batch 256, 8× more steps), everything else canonical — works: mollusca final
   (ep200) = class 1.93 / order 3.62 / family 4.76× (GOOD/EXCELLENT), depth↔norm +0.990; kNN-purity@10
   0.97/0.95/0.91 (lift 1.5/7.0/78.8×) → genuinely angular.** That beats echino (order 3.01 / family 2.63) at
   the same ranks. **Generalization principle:** the recipe is *softmax + euclidean-param + curriculum + radial
   scaffold + an effective batch sized so the dataset yields enough gradient steps* (rule of thumb: keep
   steps/epoch in the same ballpark as the metazoa run — shrink eff-batch on smaller clades, keep/raise it on
   larger ones). NOT a curriculum-schedule issue (auto spreads correctly at n_epochs=200; the ep1/2/3/4 ramp
   was a `--epochs 2` smoke artifact). Artifacts: `artifacts/tags/mollusca_effbatch256/` (the WIN: analysis +
   knn_purity) and `artifacts/tags/mollusca_canonical_local/` (the eff-batch-2048 negative control). Three
   scales now EXCELLENT with this principle: echino 4k (3.01/2.63×), mollusca 32k (3.62/4.76×), metazoa 498k
   (6.65/10.31×). See SESSION_LOG 2026-06-04.

## Roadmap / next steps

_Roadmap rev 2026-06-03 — reframed after a 3-angle deep dive (code mechanics + full
results ledger + hyperbolic-embedding literature). Supersedes the 2026-06-02 "survive the
transition via batch/LR" framing._

**Status:**
- ✅ Echino-scale recipe nailed (`echino_softmax`: 3.01×/2.63× EXCELLENT) — see metrics table.
- ✅ Both original LRZ metazoa runs analyzed (5660105 ranking + 5660187 softmax) — both POOR.
- ✅ 3 diagnostics analyzed 2026-06-02 (A=5661756, B=5661758, C=5661777): collapse localized
  to **dd≤9 → dd≤18 transition at ep 80**. Sweep at `artifacts/tags/metazoa_diagnostics_summary.tsv`.
- ✅🎉 **Experiment 1 (`metazoa_lower_lr_bigger_batch`, job 5664609) — SOLVED metazoa scale (2026-06-03).**
  Completed clean (exit 0, 10h13m). Analyzed (22-checkpoint sweep): depth↔norm **+0.984 throughout**;
  separation climbs through BOTH curriculum transitions (ep80 dd≤18 = 1.6–2.5×, **NO collapse** — Job C
  died here) to final **phylum 2.56× / class 3.86× / order 6.65× / family 10.31× — EXCELLENT on every
  rank**, far past the ≫1.5–2× bar. The 1.20× metazoa ceiling is **shattered.** depth↔norm flat while
  separation 1.0→10× ⇒ the gain is purely lateral (genuine, NOT radial Goodhart). Recipe = the four
  levers: **eff. batch 2048 (bs256×ga8), n_negatives 300, lr 0.001, cosine warm-restart on each
  curriculum-phase boundary** (+ Job-C base: softmax + euclidean-param + curriculum auto + radial-nudge
  0.05 + log radial + depth-scale-margin + epoch-fraction 0.3 + AMP). Default sampler (tiered OFF).
  Sweep: `artifacts/tags/metazoa_lower_lr_bigger_batch_summary.tsv`. The prior session's batch/LR bet was right.

**🛑 THE NEGATIVE-SAMPLING DIAGNOSIS + E1c/E2/E3 PLAN BELOW ARE SUPERSEDED (2026-06-03).** Experiment 1
used the DEFAULT sampler and cleared EXCELLENT — so the sampler was NOT the binding constraint; the
curriculum-transition collapse was, and the warm-restart + bigger batch + more negatives fixed it. **E1c
(band sampler), E2 (cones), E3 (optimizer/manifold) all STAND DOWN — unnecessary for the goal.** Step-0's
within-grandparent starvation is structurally real but NOT binding. The reframed diagnosis + E0–E3 below
are kept for the record; the Phase-1 diagnostic tooling (`diagnose_negative_hardness.py`,
`probe_band_gradient.py`) + the "validate-before-build" learnings remain reusable. The Phase-2 band-sampler
build is CANCELLED.

**POST-BREAKTHROUGH — active next steps (2026-06-03):**
1. **✅ Lock the result — DONE (2026-06-04).** Seeded analyzer (`--seed 0 --repeats 5`) → phylum 2.60 /
   class 3.85 / order 6.68 / family 10.30× ± 0.01 each; kNN-purity (`knn_purity_hyperbolic.py`) →
   purity@10 0.997/0.997/0.995/0.919, lift-over-chance rising to 216.6× at family. **Confirmed angular,
   not a radial/Poincaré artifact.** Artifacts: `analysis_final_seeded/`, `knn_purity/{json,tsv}`.
   Remaining lock step (optional, gated on LRZ): one fixed-seed reproducibility re-run (item 2 below /
   "REPRODUCIBILITY" in the entry prompt) — `--seed` is not yet wired into training, so reproduce by
   re-running the exact config and confirming the same ballpark.
2. **Promote the recipe:** Experiment 1's config IS now the metazoa recipe — update "The recipe" section
   + treat it as the default for other large clades (arthropoda/mollusca/chordata) and full-Life later.
3. **Write-up:** the project's core goal (a taxonomy embedding clearing EXCELLENT at 498k scale) is
   ACHIEVED — move toward results/figures (the ep10→200 separation trajectory + the radial-flat/lateral-
   climb story is the key figure).
4. Note `best.pth` (2.43/3.35/4.20/4.53) < `final.pth` ep200 (2.56/3.86/6.65/10.31): the loss/quality-
   "best" is NOT the highest-separation checkpoint — **use final ep200** (or add separation to checkpoint
   selection).

**DIAGNOSIS (reframed 2026-06-03; ⚠️ SUPERSEDED — see banner above) — negative-sampling + objective:**
- The default negative sampler draws **same-depth** nodes. At echino scale a depth level is a
  few dozen mostly-related nodes → same-depth negatives are *already* within-clade hard
  negatives → softmax gets a strong angular gradient "for free" → 3.01×. At metazoa scale a
  depth level spans tens of thousands of nodes across foreign phyla → the same sampler feeds
  trivially-far negatives → the softmax denominator fills with e^(−d)≈0 terms → the within-clade
  (angular) gradient — exactly what the separation ratio measures — STARVES.
- This one mechanism explains all the evidence: (a) echino works / metazoa doesn't under the
  identical recipe; (b) the per-phylum slice is uniformly POOR regardless of phylum size
  (1.05–1.09× for echino *sliced from* the 498k model vs 1.21× standalone — global sampling
  pool, not per-clade capacity); (c) **why dd≤9→dd≤18 is the breaker** — widening the closure
  floods the candidate pool with *more far pairs*, pushing the sample further toward all-foreign.
- **The bar is reachable, and the collapse is NOT a capacity ceiling.** De Sa et al. (ICML 2018)
  prove required hyperbolic dim scales with the *log of max branching factor* (they hit WordNet
  MAP 0.989 in **2 dims**). dim-100 has ample angular volume for 498k. So EXCELLENT should be
  *recoverable* — but it needs a structural change to negatives/objective, not hyperparam nudging.
- **The gap:** hard / within-clade negatives have NEVER been tried at metazoa scale. The
  `--tiered-negatives` flag exists but was OFF in every metazoa run (incl. Experiment 1).

**Next experiments (sequenced; each gates the next; validate echino-first per RED-LINE):**

- **E0 — read Experiment 1 (5664609) when it lands.** Don't cancel (queued). Score at ep80.
  Even a pass only buys the dd≤all path; EXCELLENT still needs E1+.
- **E1 — make negatives within-clade-hard at scale (highest leverage). Re-sequenced after E1a.**
  - **E1a — DONE 2026-06-03 (echino A/B).** `scripts/_s_echino_tiered_ablation.py` → tags
    `echino_softmax_abctrl` (baseline) vs `echino_softmax_tiered`. Result: baseline reproduces
    the winning regime (order **2.54×** EXCELLENT, family 1.99× GOOD); **`--tiered-negatives`
    DEGRADES every rank** (order 2.54→1.61, family 1.99→1.31, class 1.46→1.28) with radial
    untouched. Mechanism: the 50% **same-grandparent** hard tier pushes apart co-clade members
    that should cluster. The existing tiered scheme is a blunt instrument.
  - **E1b — DOWNGRADED (do NOT do blind).** "Flip `--tiered-negatives` ON at metazoa" is a bad
    bet: the flag demonstrably hurts at echino. The scale-dependence cuts both ways (at metazoa
    the hard tier might *add* needed foreign-clade discrimination, but the same sibling-
    cannibalization still operates), so the net is genuinely ambiguous — not worth a V100 run.
  - **E1c — NOW THE PRIMARY NEGATIVE-SAMPLING PATH (new code → spec→review→plan).** The fix must
    add *informative* hardness WITHOUT pushing apart immediate siblings: **SANS-style k-hop /
    within-subtree negatives with a tuned hop-band that EXCLUDES the genus/immediate-sibling
    group**, and/or **soft self-adversarial weighting** (RotatE — down-weights easy negatives by
    current-model score rather than forcing structural ones; safer than hard tiers). Add a
    per-epoch diagnostic: fraction of negatives within-clade vs foreign + mean neg-distance.
    Validate echino-first (must NOT degrade the 2.54× baseline) before any metazoa run.
    **Spec + 4-reviewer fold (2026-06-03):** `docs/specs/2026-06-03-e1c-hard-negative-sampling-spec.md`
    (v2). Review corrections: band must be **ancestor-anchored LCA mixture** (negatives' distance is
    measured from the ancestor; "class"=phylum at scale, so the naive grandparent-exclusion is wrong);
    **Step 0 = validate the diagnosis (within-clade p_j starvation) on the default sampler at metazoa
    for ~0 GPU** before building anything; seeded gates + paired controls + kNN-purity (anti-Goodhart) +
    a multi-phylum middle gate; self-adversarial *weighting* DROPPED (redundant with softmax NLL, which
    already self-weights to hard negatives).
    **Phase-1 plan + 2nd fan review DONE; STEP 0 RUN (2026-06-03) → PREMISE CONFIRMED.**
    Plan: `docs/plans/2026-06-03-e1c-phase1-instrumentation.md`; diagnostic: `scripts/diagnose_negative_hardness.py`
    (+ `scripts/_negative_hardness.py`, `tests/test_negative_hardness.py`). **Result (checkpoint-free, the
    confound-free signal):** within-class(=phylum) fraction is SATURATED at scale (one phylum dominates →
    ~0.99 for every sampler incl. uniform; achievable≈1 → no headroom → metric mis-specified at scale,
    exactly the reviewer's "class=phylum" warning). The decisive metric is **within-grandparent (≈family)
    fraction**: default sampler **collapses 0.129 (echino) → 0.015 (arthropoda) → 0.011 (metazoa)** while
    tiered/achievable stays flat ~0.46–0.48 and uniform/chance ~0.0005 — so at metazoa the default sampler
    captures only **2.4% of achievable** within-family negatives (vs 27% at echino) = **~42× headroom**.
    This both confirms within-clade gradient STARVES at scale AND explains the echino A/B (echino isn't
    starved → tiered over-repels siblings → hurts; metazoa IS starved → within-clade negatives fill the
    gap). **→ E1c justified. NEXT: write the Phase 2 plan** (ancestor-anchored LCA-mixture band sampler,
    excluding immediate siblings, targeting the family granularity where default starves); fold in
    Experiment-1 (5664609) ep80 readout for E1c-vs-E2/E3 priority. (p_j corroboration deferred — PART B
    p_j was measured at the saturated phylum granularity; the fraction signal is already decisive.)
    **Phase 2 plan DRAFTED + 4-reviewer fold + PHASE 2.0 PROBE RUN (2026-06-03):**
    `docs/plans/2026-06-03-e1c-phase2-band-sampler.md`; probe `scripts/probe_band_gradient.py`.
    Plan review raised 3 BLOCKERs incl. "band anchored to descendant but loss measures from the
    ANCESTOR → band negs get ~0 gradient." **Phase 2.0 empirically REFUTED that BLOCKER:** on the
    metazoa ep60 PEAK, deep-ancestor (da≥6) negatives — default d(anc,neg)=4.27 (FARTHER than the
    positive 3.65, i.e. already-beaten/easy) vs **band 3.64–3.87 (≈ positive distance → genuinely
    hard → real gradient)**. Both descendant- AND ancestor-anchored bands deliver hard negatives;
    the review's "far in tree → far in embedding" intuition was wrong (a trained hyperbolic embedding
    clusters relatives, so a descendant's cousins sit NEAR the ancestor). **→ band-sampler BUILD is
    justified; use descendant-anchored sibling-excluded band (matches Step-0, simplest, hardest).**
    Remaining real risk (unresolved, needs training): harder negatives ≠ guaranteed leaf-level
    separation lift (train↔eval geometry gap) — settled only by the gated echino-first grid +
    arthropoda/metazoa runs. Other review fixes (kNN-purity scalar bug, w_near<w_far guard, analyzer
    min_size/return contract, ≥3 training seeds, LRZ-out + hard STOP, commit hygiene, regression
    checkpoints, inline CLI) still apply. NEXT: rewrite plan → v2.2 (apply fixes, unblock), then build.
- **E2 — entailment cones (the code is already in the repo, unused).**
  `docs/archive/legacy_facebook/hype/energy_function.py` has a working `EntailmentConeEnergyFunction`
  (Ganea ICML 2018). Bakes the asymmetric ancestor⊃descendant order into geometry — the exact
  radius=depth + monotone-relatedness dual objective. Recipe: pretrain distance/softmax curriculum
  → rescale ×0.7 → fine-tune cones. Reported +9–14 F1 over distance loss on WordNet hypernymy.
- **E3 — structural alternatives (only if E1/E2 stall, and only the one the failure points to):**
  - **Boundary/optimizer:** drop `--euclidean-param`+tanh+Adam for geoopt **RiemannianAdam**
    (Bécigneul-Ganea ICLR 2019: RAdam-on-manifold ≥ RSGD ≥ Adam-on-tangent on this WordNet task),
    or the **Lorentz model** (Nickel-Kiela ICML 2018) for boundary numerical stability at depth 37.
  - **Anti-forgetting / tree-as-teacher:** the metazoa failure IS catastrophic forgetting (dd≤9
    structure unlearned at dd≤18). An **EMA-teacher distillation** term (mean-teacher; pull student
    toward a frozen end-of-dd≤9 snapshot) or a **cophenetic-correlation loss** (HypStructure,
    NeurIPS 2024 — regress embedding distances to known tree distances; *negative-free*, sidesteps
    the sampling pain) are the well-posed versions of the data2vec/self-distillation idea. (Vanilla
    data2vec/DINO/BYOL don't map directly — no input modality to mask, no natural node "view".)
  - **Product-of-balls / mixed-curvature** (Gu ICLR 2019) — last resort for inter-clade smearing.

**DO NOT:** raise `--dim` (theory says 100 is ample); jump to product-manifold first (heavy,
not yet indicated). **Process:** echino-first gate (~12 min/run on MPS, `--gpu 0`) before any
LRZ submit; never sbatch on offline-only validation (RED-LINE).

**After each experiment:** update the metrics table + this Roadmap + a fresh SESSION_LOG entry.

## Data & artifacts

- Datasets: `data/taxopy/{echinodermata_7586_clean, arthropoda_6656_clean,
  mollusca_6447_clean, metazoa_33208_clean}/` (transitive `.npz` + `.mapping.tsv`).
- Checkpoints/results: `artifacts/tags/<tag>/`.
- LRZ: `/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/`
  (ssh alias `ai`); pytorch 2.4 sqsh, `MKL_THREADING_LAYER=GNU`.
