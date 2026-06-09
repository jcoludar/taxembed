# Session Log — Poincaré Taxonomy Embeddings

Append-only, point-in-time record of working sessions. **Newest entry at top.**
Each entry: date, what we set out to do, what we found/changed, and what's next.
Living state (current best recipe, metrics, open issues) lives in `PROJECT_STATE.md`,
not here — this log is the timestamped trail.

---

## 2026-06-09 — All-Life (877k) lands EXCELLENT + metazoa reproducibility lock CLOSED

**Set out to:** check the outstanding LRZ runs, pull + analyze the two that the 2026-06-04 handoff was
waiting on (eukaryota all-Life + metazoa repro), staying mindful of local disk (started at 96% / 44 GB free).

**LRZ status:** queue empty — all `taxembed_*` jobs finished. The two target runs both COMPLETED:
`metazoa_metazoa_repro` (5666348, 10h09m, ended 2026-06-05) and `eukaryota_canonical` (5666473, 20h30m,
877k nodes, ended 2026-06-05). (Full account sweep also showed the plm_choice `oe_*`/`esmc_*` jobs done;
one `oe_ESM2-3B` OOM at 5665521 then succeeded on rerun 5666300 — not ours.)

**Space discipline:** did NOT mirror the tags (eukaryota 18 GB + repro 11 GB = 29 GB would have hit ~99%).
Pulled only what analysis needs — final ckpt + ep80/100/120/150/180 milestones for eukaryota, final-only
for repro (~4.4 GB). Then trimmed the locked Exp-1 tag (`metazoa_lower_lr_bigger_batch`) to final+best
(milestones all still on LRZ = recovery path), reclaiming ~9.5 GB (10 GB → 768 MB; macOS holds the freed
blocks as APFS purgeable). Helpers: `scripts/_eukaryota_trajectory_readout.py`, `scripts/_trim_exp1_checkpoints.py`.

**Results (seeded analyzer, `analysis_final_seeded/` under each tag):**
1. **Reproducibility lock CLOSED — repro (5666348).** Same canonical config, fresh run → phylum 2.59 /
   class 3.84 / order 6.69 / **family 10.27 ±0.01×**; depth↔norm +0.984/+0.998. Matches Experiment 1
   (2.60/3.85/6.68/10.30) inside the ±0.01 noise floor ⇒ the metazoa breakthrough is **deterministic**,
   not a lucky seed.
2. **ALL-LIFE EXCELLENT — eukaryota (5666473), 877k nodes.** Final ep200 → phylum 3.12 / class 5.09 /
   order 7.13 / **family 8.23 ±0.01×**; depth↔norm +0.978/+0.999. EXCELLENT at every rank ("phylum"≈
   kingdom at this scale). The canonical recipe (eff-batch 2048 + n_neg 300 + lr 0.001 + cosine
   warm-restart, **default sampler**) holds at the largest, deepest tree we have — no ceiling.
3. **Stall-watch resolved.** Milestone trajectory (family rank): ep80 2.34 → ep100 3.75 → ep120 3.54 →
   ep150 3.68 → ep180 6.56 → ep200 8.23. The flagged ep80–120 window shows a **soft plateau (ep100–150),
   not the metazoa-style curriculum-transition collapse**. The final dd≤all phase + warm-restart then
   drives a hard late climb (separation more than doubled in the last 50 ep) ⇒ **use `final`, not `best`**.

**Implication for the roadmap:** the E1c hard-negative sampler / E2 cones / E3 structural alternatives are
**NOT required to reach EXCELLENT at full scale** — the default sampler + four canonical levers suffice
through 877k. Those plans stay on the record (+ reusable diagnostics) but the core capability question is
answered. **PC embeddings represent taxonomy well across 4k → 877k (all of Life), reproducibly.** Remaining
work is presentational (figures, write-up) + optional polish, not a capability gap.

**Next:** results/figures pass (the ep10→200 separation-trajectory story now has a 4-scale ladder: echino
4k / mollusca 32k / metazoa 498k / eukaryota 877k). Optional: kNN-purity anti-Goodhart check on eukaryota
final (pull is local-only now); arthropoda 325k canonical (LRZ) if a 5th ladder rung is wanted.

### (same day, later) — eukaryota anti-Goodhart CLOSED + TRUE all-Life (cellular) built, smoked, launched

- **Eukaryota kNN-purity DONE (asterisk closed).** `knn_purity/` (k=10, 3000 q/rank, 3 reps): phylum
  0.9945 (4.1×) / class 0.9912 (5.8×) / order 0.9850 (43.9×) / **family 0.9124 (274.5× chance)**. NN are
  91–99% same-clade ⇒ all-Life separation is genuine angular structure, not a radial artifact (family lift
  even exceeds metazoa's 216×). Eukaryota result now has the SAME two-check rigor as metazoa.
- **Test suite:** 64 passed (loss / training-pairs / negative-hardness math covered).
- **TRUE all-Life dataset built — `cellular_organisms_131567_clean`.** `taxembed build 131567 --clean`:
  2.625M raw → **1,102,163 clean nodes** (58% noise stripped), **21.4M pairs**, depth 40. Scope =
  Bacteria + Archaea + Eukaryota; **Viruses excluded** deliberately (polyphyletic, artificial root, no
  shared ancestry with cellular life — embedding them would assert false common origin). Verified (Rule 10,
  `scripts/_verify_cellular_clean.py`): nodes==mapping; domains Bacteria 216k / Archaea 7k / Eukaryota 878k
  (Euk matches standalone build = consistency check); residual name-noise 0.023% (internal env-sample /
  informal-bacterium containers w/ real children — topology only, names don't affect training).
- **Recipe UNCHANGED.** 21.4M pairs is only 1.13× eukaryota's 18.9M (raw-vs-clean: the earlier 56M/3×
  scare was the RAW clade count) ⇒ epoch_fraction 0.3 + eff-batch 2048 stays put; est. ~23–27h fits the
  48h walltime. Probe: `scripts/_probe_all_life_size.py`.
- **RED-LINE honored (S0274):** exact flag combo ran end-to-end locally on echino (40 ep, all curriculum
  auto-phases + warm-restarts + AMP + grad-accum + euclidean-param + softmax), depth↔norm +0.980, clean
  exit, BEFORE any sbatch. Echoed command byte-identical to the LRZ script.
- **LAUNCHED — `taxembed_cellular_canonical`, job 5673097** (PENDING, lrz-v100x2, 48h). Script
  `scripts/train_lrz_cellular_canonical.sh`. On landing: pull final + ep80–200 milestones (local-only;
  watch the ep100–150 plateau→climb shape), seeded separation + kNN-purity, ranks superkingdom/phylum/
  class/order/family. Use `final`, not `best`.

### (same day, later still) — paper scoped + Plan 1 executed + Application #1 RESULT
- **Brainstormed the paper** (Bioinformatics tool venue): method + 5-scale validation ladder + 3 apps
  (#1 fidelity, #2 taxonomy QC = LEAD, #3 sampling-bias) + #4 bridge as Outlook. Spec:
  `docs/specs/2026-06-09-taxembed-paper-design.md` (§9 = 4-reviewer fan + recon fold).
- **4-reviewer fan + recon** reshaped it: lead with artifact+scale (not "hyperbolic taxonomy embedding");
  #2 release-diff QC is the real "so what"; #1 demoted to a validation subsection (Macaulay 2023 / De Sa
  2018 precede the finding — defend on scale+service); #3 must benchmark vs Faith's PD; radial-only null +
  taxon-bootstrap CIs mandatory; LCA/old-taxdump-canonicalize/UniProt-join are the real (non-thin) work;
  ship an installable artifact + DOI'd embedding. #4 recon: ~100–200k ProtT5 proteins / ~200–300 taxa
  offline; UniProt has per-entry embeddings for all → #3 coverage fetchable; #4 ~80% data (ant-venom
  corpus best prototype) — prototype-able, its own effort.
- **Plan 1 (eval foundation + #1) WRITTEN + EXECUTED** (subagent, TDD). Branch
  `feat/taxembed-eval-foundation`; new `src/taxembed/eval/{treedist,nulls,pairs,bootstrap,fidelity}.py`
  (binary-lifting LCA, radial-only/shuffled/random nulls, taxon-bootstrap CI, distortion + kNN-retrieval
  + within-clade rank corr) + `scripts/cophenetic_fidelity.py`; **14/14 eval tests pass**, 6 commits.
- **Application #1 RESULT (eukaryota 877k):** kNN-retrieval precision@10 = **0.636** [0.627–0.644] vs
  **radial-only null 0.026** ⇒ **delta 0.609 (~24× null)**; distortion median 1.14. Fidelity is genuine
  ANGULAR structure, not a radial artifact. (`artifacts/tags/eukaryota_canonical/cophenetic_fidelity/`.)
- **Plans 2 & 3 SCOPED** (8 TDD tasks each, on disk): `docs/plans/2026-06-09-taxembed-app2-taxonomy-qc.md`
  and `docs/plans/2026-06-09-taxembed-app3-sampling-bias.md`.

---

## 2026-06-04 — Locked the metazoa breakthrough (kNN-purity + seeded separation); recipe promoted

**Set out to:** execute the POST-BREAKTHROUGH "lock the result" step (local, ~0 GPU) so the 6–10×
metazoa separation is defensible for the writeup, then promote Experiment 1's config to the canonical
recipe.

**Did (all local, MPS-free CPU numpy):**
1. **kNN-purity anti-Goodhart check — new tool `scripts/knn_purity_hyperbolic.py`.** Per-rank purity of
   each node's k nearest neighbours under full Poincaré distance, over the entire 498k pool. Batched
   matmul-expanded distance (only neighbour *order* matters → float32 matmul; a float64 cross-check vs
   `_negative_hardness.numpy_poincare_distance` guards it — top-10 order agrees). Imports the analyzer's
   label machinery so groups are identical to the separation metric. Seeded query subsample + repeats.
   **Result (final ckpt):** purity@10 = 0.997 / 0.997 / 0.995 / 0.919 (phylum→family); chance = 0.466 /
   0.314 / 0.056 / 0.0042; **lift rises 2.1× → 3.2× → 17.9× → 216.6×.** Nearest neighbours are 92–99%
   same-clade — local + radius-independent. A radial Goodhart artifact would sit at chance; family is
   217× chance. ⇒ **separation is genuine angular structure.** Saved `knn_purity/{json,tsv}`.
2. **Seeded reproducibility on the analyzer — `analyze_hierarchy_hyperbolic.py` gains `--seed` +
   `--repeats`** (mean±std over seeded group/pair subsamples; single-run output shape preserved so
   `_sweep_diagnostic_analyses.py`'s parser still matches). Final ckpt, 5 seeds: phylum **2.60±0.01** /
   class **3.85±0.01** / order **6.68±0.01** / family **10.30±0.01×**; depth↔norm +0.984 / +0.998.
   ±0.01 noise floor → magnitude is far beyond sampling slop; reproduces the original unseeded sweep.
   Echino sanity (3,965 nodes) reproduces documented `echino_softmax` (class 1.45 / order 3.03 /
   family 2.63×). Saved `analysis_final_seeded/`.
3. **Promoted the recipe in `PROJECT_STATE.md`:** new STATUS "RESULT LOCKED" block; "The recipe" now
   leads with the 🏆 canonical metazoa/large-clade four-lever recipe (eff-batch 2048 · n_neg 300 · lr
   0.001 · cosine warm-restart-on-phase) with the echino recipe demoted to small-clade; known-issue #6
   ("does not scale") marked RESOLVED (historical kept); Roadmap lock-item marked DONE. Use `final`
   (ep200), NOT `best` (separation climbs ~90 ep past the loss plateau).

4. **Generalization probe — mollusca (32k), local MPS → NEGATIVE / non-transfer (important finding).**
   Ran the EXACT canonical recipe on `mollusca_6447_clean` (32,017 nodes, tag `mollusca_canonical_local`,
   200 ep, save-every 25). The exact-flag-combo smoke ran clean end-to-end first (Rule-10 gate). Auto
   curriculum spread correctly to ep1/40/80/120 (the earlier ep1/2/3/4 ramp was a `--epochs 2` smoke
   artifact, NOT a real schedule bug — `auto_curriculum_phases` keys off n_epochs). **Result: radial frame
   is perfect (depth↔norm +0.99) but lateral separation does NOT form** — real analyzer (seeded, 3 reps):
   ep100 = class 0.96 / order 0.98 / family 1.01×; ep150 (30 ep into dd≤all) = 1.08 / 1.10 / 1.10× — a
   crawl from 1.0, still POOR, nowhere near metazoa's 2.6–10× or even echino's 3×. **⇒ the canonical
   recipe does NOT transfer unchanged to a smaller clade.** Diagnosis: optimization-budget starvation —
   at 32k nodes, 278k pairs × epoch-fraction 0.3 ÷ eff-batch 2048 = **~41 grad-steps/epoch** (~8k total
   over 200 ep), far fewer than at 498k; the big-batch/low-lr regime is calibrated to metazoa's pair
   volume. The "recipe" is therefore *softmax + euclidean-param + curriculum + radial scaffold + an
   optimization budget matched to the pair volume* — the four metazoa levers fixed the curriculum-collapse,
   they do NOT auto-calibrate step-count for a new scale. **This corrected the PROJECT_STATE overclaim**
   (it had promoted the recipe as "default for all large clades" before the test; now labelled
   metazoa-scale + caveated; known-issue #7 added). Artifacts kept as the negative control:
   `artifacts/tags/mollusca_canonical_local/`; checks in `/tmp/mollusca_ep{100,150}_check/`.
   **Next-step hypothesis to test (single lever):** restore eff-batch 256 (`--grad-accum-steps 1`) →
   8× more grad-steps/epoch, everything else canonical; if lateral separation then forms, generalization
   = "scale eff-batch to the dataset", and that's the fix.
5. **✅ FIX CONFIRMED — generalization demonstrated (tag `mollusca_effbatch256`).** Reran mollusca with the
   single lever (`--grad-accum-steps 1`, eff-batch 256), everything else canonical. **ep200 = class 1.93 /
   order 3.62 / family 4.76× (GOOD/EXCELLENT)**, depth↔norm +0.990; kNN-purity@10 0.97/0.95/0.91 (lift
   1.5/7.0/78.8×) ⇒ genuinely angular. (ep125 already 1.42/1.67/1.66 and climbing — and the inline Sep proxy
   wildly undersells: it read ~1.05 while the real ratio was 1.4–4.8×, so trust the analyzer, not the inline
   column.) This BEATS echino (3.01/2.63) at order/family. **Generalization principle confirmed:** softmax +
   euclidean-param + curriculum + radial scaffold + eff-batch sized for enough gradient-steps. Three scales
   now EXCELLENT (echino 4k, mollusca 32k, metazoa 498k). PROJECT_STATE recipe + known-issue #7 updated to
   RESOLVED. Negative control kept (`mollusca_canonical_local`, eff-batch 2048 → POOR).

6. **UMAP figures (user request) — upgraded viz `scripts/figure_umap_taxa.py`.** The old
   `visualize_multi_groups.py` coloured by tree-depth (Metazoa's tree-children are Eumetazoa/Porifera, so
   it can't show phyla). New tool: rank-based colouring (reuses the analyzer's `get_ancestor_at_rank`),
   `--node-rank` (one point per coarse taxon — the key to seeing phylum REGIONS instead of pure species
   micro-clusters), `--restrict-to-taxid`/`--exclude-taxids`, balanced sampling. Finding: the embedding makes
   thousands of taxonomically-PURE micro-clusters (family 10× dominates), so phylum regions only resolve when
   you plot one point per order. Figures in `paper/figures/`: `metazoa_umap_phylum_orders` (clean phylum
   separation), `metazoa_umap_minor_phyla` (giants excluded → small phyla separate), `arthropoda_umap_class`
   (zoom), `metazoa_umap_phylum` (species-level purity). Committed 32c2b49.
7. **Eukaryota (all-Life) prep — built, verified, uploaded; ready to sbatch.** `taxembed build 2759 --clean`:
   1,968,485 raw → **877,584 clean nodes** (55.4% noise stripped), **18.9M pairs**, max depth 39 (~1.76× metazoa).
   Verified (Rule 10, `scripts/_verify_eukaryota_clean.py`): npz n_nodes == mapping rows; residual noise
   **0.004%** (38 legit internal containers); full kingdom balance (Opisthokonta 598k / Viridiplantae 251k /
   Fungi 99k / SAR 17k / Rhodophyta / Amoebozoa / Discoba / Haptista); eyeball clean (no `sp.`/environmental;
   named strains kept, consistent with metazoa build). **Build LOCALLY, upload — don't taxdump on LRZ**
   (compute nodes offline; local build lets us verify the cleaning before GPU; npz is only 7MB). Uploaded
   npz+mapping (~20MB) + `scripts/train_lrz_eukaryota_canonical.sh` to LRZ /data (verified on disk).
   Recipe = metazoa canonical (eff-batch 2048) — appropriate since eukaryota > metazoa (big-batch is for
   LARGE clades; the mollusca shrink-batch fix was for small ones). **WATCH ep80–120 for a separation stall**
   = within-clade hard-negative starvation re-emerging at this scale/depth → enable hard-neg sampler or
   n_neg→500 (NOT an architecture ceiling). Dataset regenerable via the build cmd (not committed).

**Verdict:** Task-1 "lock the result" COMPLETE — metazoa-scale EXCELLENT defensible. Task-4 generalization
COMPLETE — recipe generalizes across 4k→498k with one scale-aware knob (effective batch); mollusca 32k now
EXCELLENT (order 3.62 / family 4.76×). The "finished" bar (result locked + generalizes to ≥1 clade) is MET.
UMAP figures delivered. Eukaryota (877k nodes) built+verified+uploaded, ready to launch.

**Next:** Eukaryota FIRED — `sbatch` job **5666473** (PENDING, queued behind repro 5666348 + plm_choice
jobs). On landing: pull + analyze (seeded separation + kNN-purity, ranks phylum/class/order/family — phylum
here = kingdom-ish), read ep80–120 for the stall signal. Repro **5666348** RUNNING (~1h38m/10h) — when it
lands, confirm it reproduces ~2.6/3.8/6.7/10× to close the reproducibility lock.

**Next:** (a) test the eff-batch-256 mollusca fix (local, ~½h) — if it lifts, the recipe generalizes with
one scale-aware lever. (b) LRZ reproducibility re-run of the exact Exp1 config (new tag
`metazoa_lower_lr_bigger_batch_repro`, script `scripts/train_lrz_metazoa_repro.sh`) — RED-LINE: echino
end-to-end first; session can't ssh, so user runs `!`-prefixed scp/sbatch. (c) arthropoda (325k, LRZ-only;
`scripts/train_lrz_arthropoda_canonical.sh`, dataset upload caveat). Figure scaffolding DONE
(`paper/figures/metazoa_separation_trajectory.{png,pdf}` + README). Housekeeping commit pending user approval.

---

## 2026-06-03 — Deep-dive + roadmap reframe: it's a negative-sampling problem; echino A/B launched

═══════════════════════════ EOS DEBRIEF (2026-06-03) ═══════════════════════════
_Consolidated end-of-session handoff. Detailed trail is the rest of this entry, below._

⚡⚡ **UPDATE (2026-06-03, later — Experiment 1 landed and CHANGES EVERYTHING):** job 5664609
COMPLETED (exit 0, 10h13m); artifacts pulled; 22-checkpoint sweep analyzed
(`metazoa_lower_lr_bigger_batch_summary.tsv`). **Experiment 1 SOLVED metazoa scale** — depth↔norm
**+0.984 throughout**, separation climbs through BOTH curriculum transitions (ep80 dd≤18 = 1.6–2.5×,
NO collapse; Job C died here) to final **phylum 2.56 / class 3.86 / order 6.65 / family 10.31× —
EXCELLENT every rank**. The 1.20× ceiling is shattered; the gain is purely lateral (depth↔norm flat
while sep 1→10×). **Experiment 1 used the DEFAULT sampler** → the negative sampler was NOT the binding
constraint (the curriculum-transition collapse was; the four levers — eff. batch 2048, n_neg 300, lr
0.001, cosine warm-restart — fixed it). **⇒ The entire E1c/E2/E3 plan below STANDS DOWN; the Phase-2
band-sampler build is CANCELLED.** The negative-sampling diagnosis (Step 0) was structurally real but
NOT binding. What held up: keeping E1c gated on the Exp1 readout (zero GPU burned on it) + "validate
before build" (never built the now-unneeded sampler). Active next steps now in PROJECT_STATE Roadmap
"POST-BREAKTHROUGH": lock the result (seeded analyzer + kNN-purity / reproduce), promote the recipe,
write up. The handoff below (which framed the session as PAUSED pending this readout) is now historical.

**One-line:** Reframed the metazoa-scale separation failure as a **negative-sampling** problem
(not curriculum dynamics), validated it end-to-end (Step 0 CONFIRMED the premise; Phase 2.0 probe
confirmed the band-sampler mechanism and *refuted* a review BLOCKER), and produced a build-ready
design. **PAUSED pending Experiment-1 (job 5664609) ep80 readout on LRZ.**

**The arc:** deep-dive (3 agents) → diagnosis reframe → echino A/B (tiered degrades echino; baseline
reproduces order 2.54×) → spec v2 (4 reviews) → Phase-1 plan (4 reviews) → built diagnostic + ran
**Step 0** → within-grandparent default-negative fraction collapses **0.129 (echino) → 0.011
(metazoa)**, ~42× headroom = **PREMISE CONFIRMED** → Phase-2 plan (4 reviews, 3 BLOCKERs) → **Phase
2.0 probe** → anchoring BLOCKER **REFUTED** (band negs ARE hard at scale) = **BUILD GO**.

**Decision state:**
- **E0 / Experiment 1 (5664609):** RUNNING on LRZ (last user check: 8h24m, gpu-002, lrz-v100x, 48h
  walltime). **This is the gating event.** Read at the ep80 dd≤18 milestone: depth↔norm ≥+0.85 AND
  sep ≥1.10×.
- **Negative-sampling premise:** CONFIRMED (Step 0).
- **E1c band sampler:** mechanism VALIDATED (Phase 2.0). Design decided = **descendant-anchored,
  sibling-excluded band** (matches Step-0 within-gp metric; simplest; empirically hardest). BUILD
  justified, NOT yet built.
- **Open risk (only training settles):** harder negatives ≠ guaranteed *leaf-level* separation lift
  (train-geometry vs eval-geometry gap).

**Artifacts:**
- **Committed** (submodule `feat/lrz-readiness-prep`): `31b1ccc` (`_negative_hardness.py` + tests),
  `0ff54b2` (`diagnose_negative_hardness.py`). Both passed spec+quality review.
- **Working-tree, UNCOMMITTED (user to review/commit):** `docs/PROJECT_STATE.md`, `docs/SESSION_LOG.md`,
  `docs/specs/2026-06-03-e1c-hard-negative-sampling-spec.md`, `docs/plans/2026-06-03-e1c-phase1-instrumentation.md`,
  `docs/plans/2026-06-03-e1c-phase2-band-sampler.md`, `scripts/probe_band_gradient.py`.
- **KEEP** (real artifacts): tags `echino_softmax_abctrl` / `echino_softmax_tiered` (the A/B); the 3
  diagnostic scripts.
- **CLEANUP-OWED (rm blocked by permission engine — user `rm`):** `scripts/_inspect_dataset_readonly.py`,
  `scripts/_p4_check_nnodes.py`, `scripts/_inspect_checkpoint_command.py`, and the `smoke_abtest_delete/` tag dir.

**PICKUP CHECKLIST (next session — gated on LRZ):**
1. **Check Experiment 1:** `ssh ai 'sacct -j 5664609 --format=JobID,State,ExitCode,Elapsed,End -X'`.
   If COMPLETED → pull `metazoa_lower_lr_bigger_batch` artifacts + run `_sweep_diagnostic_analyses.py`;
   read the ep80 criterion. If still R → come back.
2. **Decide E1c-vs-E2/E3 priority** using the Exp1 ep80 result (does the batch/LR/warm-restart lever
   survive dd≤18? — independent of E1c).
3. **Rewrite Phase-2 plan → v2.2:** apply the still-valid review fixes (kNN-purity scalar bug;
   `w_near<w_far` guard; analyzer `min_size=10` + `(sep,qual)` return contract; ≥3 training seeds for
   the gate; ALL-LRZ-out + hard STOP >30min; commit via `git commit -F`; full-suite + bit-identical-
   default regression checkpoints; inline CLI). **DROP the v2.1 ancestor-anchoring correction** — Phase
   2.0 refuted its premise; descendant-anchored is fine and simplest.
4. **Build (subagent-driven):** seeds → analyzer rigor + kNN-purity → ancestor/lineage index → band
   sampler → CLI → tests; then echino-grid **G0** (don't-break-the-working-case) + arthropoda **G0.5**
   (the lift test; local ≤30min or master-gated; **no blind LRZ**).
5. **Only if G0.5 lifts** → master-launch the metazoa run (echino-end-to-end first per RED-LINE,
   paired control, read ep80–110).

**Durable learnings:**
- Within-class(=phylum) metric SATURATES at scale (one phylum dominates) → use within-grandparent/finer.
- **Tree-distance ≠ embedding-distance:** validate a negative's "hardness" on a real checkpoint, not tree topology (this refuted a plausible BLOCKER).
- `mean p_j/neg ≈ 1/n_neg` is non-discriminating → use `mean d(anc,neg)` + `neg-mass (1−p_pos)`.
- "Validate the mechanism before building" earned its keep TWICE today (caught a Step-0 metric mis-spec; refuted a Phase-2 BLOCKER) — keep doing it.
═══════════════════════════════════════════════════════════════════════════════

**Set out to:** with Experiment 1 (5664609) still pending on LRZ, do a full ins-and-outs deep
dive of the project and decide the best path forward; make a sensible roadmap and execute what
makes sense now.

**Method:** fanned 3 read-only agents — (1) training-code mechanics, (2) full data/results
ledger + raw diagnostic numbers, (3) hyperbolic-embedding literature + the archived Nickel-Kiela
code. Cross-read `metazoa_diagnostics_summary.tsv` + `per_phylum_separation.tsv` directly.

**Key reframe (now the leading diagnosis — see PROJECT_STATE Roadmap rev 2026-06-03):** the
metazoa collapse is a **negative-sampling + objective** problem, NOT a curriculum-dynamics one.
The default sampler draws **same-depth** negatives; at echino scale same-depth ≈ within-clade
(hard) so softmax gets strong angular gradient → 3.01×; at 498k scale same-depth spans foreign
phyla → trivially-far negatives → within-clade (angular) gradient starves → ≤1.20×. This single
mechanism explains: echino-works/metazoa-doesn't under the identical recipe; the uniformly-POOR
per-phylum slice (1.05–1.09× sliced-from-498k vs 1.21× standalone, size-independent); and why
**dd≤9→dd≤18 is the breaker** (widening the closure floods the pool with more far pairs).
Literature corroborates: the "uniform negatives → vanishing gradient as pool grows" failure is
named/documented; SANS (EMNLP 2020) = k-hop/within-subtree negatives is the targeted fix;
self-adversarial weighting (RotatE) complements; **De Sa et al. ICML 2018 prove dim scales with
log(branching factor)** → dim-100 is ample → the bar is *reachable*, the collapse is NOT a
capacity ceiling. **The gap: hard negatives were NEVER tried at metazoa scale** (`--tiered-negatives`
OFF in every metazoa run incl. Experiment 1). Bonus find: the repo already ships an unused
`EntailmentConeEnergyFunction` (Ganea ICML 2018) in `docs/archive/legacy_facebook/hype/`.

**Side question (data2vec / teacher-student):** vanilla data2vec/DINO/BYOL don't map (no input
modality to mask, no natural taxonomy-node "view"). The valid kernel: EMA-teacher anti-forgetting
across the curriculum (targets the exact dd≤9→dd≤18 forgetting) and a cophenetic-correlation
"tree-as-teacher" loss (HypStructure NeurIPS 2024, negative-free → sidesteps the sampling pain).
Both filed under E3 in the roadmap.

**Executed now (echino-first, RED-LINE-compliant):** launched a controlled echino A/B of the
winning recipe ± `--tiered-negatives` — `scripts/_s_echino_tiered_ablation.py` → tags
`echino_softmax_abctrl` (baseline) and `echino_softmax_tiered`. Identical command both runs (dim
100, 200 ep, softmax + euclidean-param + radial-nudge 0.05 + λ-reg 0.1, `--early-stopping 0`,
`--gpu 0` MPS). (Caught + fixed mid-run: default es=15 fires ~ep49 on the loss plateau → set
es=0 to match the historical 200-ep regime.)

**RESULT (2026-06-03):** baseline reproduced the winning regime — depth↔norm +0.956, **class
1.46 / order 2.54 / family 1.99×** (order EXCELLENT, family GOOD; cf. historical 1.46/3.01/2.63).
**`--tiered-negatives` DEGRADED every rank** — +0.967 / **1.28 / 1.61 / 1.31×** — radial
untouched. The prediction ("~neutral on echino") was WRONG in direction: tiered is actively
*harmful* because its 50% same-grandparent hard tier pushes apart co-clade members that should
cluster. Course-correction: E1b (flip `--tiered-negatives` at metazoa) downgraded — bad bet
given it hurts at the only cheaply-measurable scale. E1c (SANS k-hop EXCLUDING immediate
siblings + soft self-adversarial weighting) becomes the primary negative-sampling path. The
echino-first gate did its job — saved a likely-wasted V100 run. Metrics table + Roadmap updated.

**Changed:**
- `docs/PROJECT_STATE.md`: Roadmap section fully rewritten (rev 2026-06-03) — diagnosis +
  sequenced E0–E3 plan + DO-NOTs.
- `scripts/_s_echino_tiered_ablation.py` (NEW): the echino A/B runner.
- `scripts/_inspect_checkpoint_command.py` (NEW): read-only checkpoint-metadata dumper.

**Cleanup owed (rm blocked by permission engine):** `scripts/_inspect_dataset_readonly.py`
(subagent throwaway) + the `smoke_abtest_delete` tag dir (10-epoch smoke).

### PICKUP — NEXT
1. ✅ E1a done (see RESULT above) — tiered degrades echino; E1b downgraded.
2. ✅ E0 / Experiment 1 (5664609) confirmed **RUNNING** (user-checked: 8h24m, gpu-002, lrz-v100x).
   Read vs ep80 dd≤18 criterion (depth↔norm ≥+0.85, sep ≥1.10×) when it reaches/passes ep80.
   Re-check: `! ssh ai 'squeue -j 5664609'` / `sacct -j 5664609 ...`.
3. **E1c spec DONE + 4-reviewer fold DONE (2026-06-03):**
   `docs/specs/2026-06-03-e1c-hard-negative-sampling-spec.md` (v1 + findings ledger + v2 fold).
   Reviews caught a real correctness bug (band must be **ancestor-anchored**, not descendant) + two
   experimental BLOCKERs (unseeded noisy gates; echino fp32 vs metazoa AMP precision mismatch) +
   "class=phylum at scale". v2: Step 0 = validate diagnosis on default sampler at metazoa (~0 GPU)
   BEFORE building; ancestor-anchored LCA mixture sampler w/ guardrails; seeded gates + paired controls
   + kNN-purity + multi-phylum middle gate; dropped softmax-temp + RotatE weights from v1.
   **PLAN + 2nd fan review DONE:** `docs/plans/2026-06-03-e1c-phase1-instrumentation.md` (Phase 1 =
   diagnostic + Step-0 premise validation; Phase 2 sampler deferred until Step 0 confirms). 4 plan
   reviewers caught: BLOCKER — the diagnostic fed raw euclidean-param `z` (norm>1) into the Poincaré
   distance → corrupt p_j (fix = map `tanh(‖z‖/2)·z/‖z‖`); confounded comparator — metazoa `_best.pth`
   is COLLAPSED, use the ep60 PEAK milestone; missing chance baseline — fraction falls with N under
   uniform sampling regardless (fix = scale-invariant **enrichment ratio** default/uniform vs tiered
   upper-bound). v2 fold appended: checkpoint-free enrichment ratio = PRIMARY signal, p_j corroboration-
   only. **PHASE 1 EXECUTED (subagent-driven, 2026-06-03) — STEP 0 → PREMISE CONFIRMED.** Committed:
   `scripts/_negative_hardness.py` + `tests/test_negative_hardness.py` (31b1ccc, 5/5 pass) and
   `scripts/diagnose_negative_hardness.py` (0ff54b2); both passed spec+quality review. Step-0 sweep
   (4 datasets × default/uniform/tiered + p_j at ep30/60/90/120):
   - within-class(=phylum) fraction SATURATED at scale (~0.99 for every sampler; achievable≈1 → no
     headroom). Metric mis-specified at scale — exactly the reviewer's class=phylum warning. p_j (PART B)
     was at this saturated granularity too → uninformative.
   - **DECISIVE METRIC = within-grandparent (≈family) fraction:** default **0.129 (echino) → 0.015
     (arthropoda) → 0.011 (metazoa)**; tiered flat ~0.46–0.48; uniform/chance ~0.0005. Default captures
     **27% of achievable at echino but only 2.4% at metazoa → ~42× headroom.** Confirms within-clade
     gradient starves at scale AND explains the echino A/B (echino not starved → tiered over-repels;
     metazoa starved → within-clade negs fill the gap).
   **→ E1c JUSTIFIED.** Phase 2 plan DRAFTED + 4-reviewer fold (2026-06-03):
   `docs/plans/2026-06-03-e1c-phase2-band-sampler.md`. **STATUS: BLOCKED — not execution-ready.**
   Reviews caught 2 design BLOCKERs + 1 code BLOCKER: (1) the drafted band was anchored on the
   DESCENDANT, but the loss measures negatives from the ANCESTOR (reverted spec-v2's top folded fix
   — band negatives would get ~0 gradient); (2) no `p_j`-from-ancestor diagnostic ships → results
   uninterpretable; (3) kNN-purity broadcast bug (analyzer `poincare_distance` returns a scalar).
   Plus MAJORs: training(ancestor↔neg)-vs-eval(leaf↔leaf) geometry mismatch unvalidated; no
   `w_near<w_far` guard; default mixture degenerates to one rank; analyzer refactor changes the
   `min_size=10` threshold + breaks the `(sep,qual)` contract; 2σ over analyzer-repeats ≠
   training-seed variance; arthropoda-LRZ floated inside a subagent task (RED-LINE risk); commit
   steps compound-bash. v2.1 fold appended: **corrected ANCESTOR-anchored band** (sibling clades of
   the anchor, `subtree(far_anc)∖subtree(excl_anc)` at depth dd) + a **Phase 2.0 gate** = extend the
   existing diagnostic to measure `p_j`-mass-from-ancestor on band negatives BEFORE building the
   sampler (validate-the-mechanism-first, mirrors Step 0).
   **PHASE 2.0 PROBE RUN (2026-06-03, `scripts/probe_band_gradient.py`) → BUILD GO; anchoring BLOCKER
   REFUTED.** On the metazoa ep60 PEAK, deep-ancestor (da≥6) negatives: default d(anc,neg)=4.27
   (FARTHER than the positive 3.65 → already-beaten/easy → little gradient) vs band 3.64 (desc-anchored)
   / 3.87 (anc-anchored) (≈ positive distance → genuinely hard → real gradient). BOTH anchorings give
   hard negatives; the review's "descendant-cousins far from ancestor → no gradient" intuition was
   WRONG (tree-distance ≠ embedding-distance — a trained hyperbolic embedding clusters relatives, so
   descendant-cousins sit near the ancestor). Echino (shallow) couldn't test this; metazoa did.
   **→ Use descendant-anchored sibling-excluded band (matches Step-0, simplest, hardest). Build justified.**
   The probe also showed mean-p_j/neg ≈ 1/n_neg is a non-discriminating metric; mean d(anc,neg) +
   neg-mass(1−p_pos) are the right ones. UNRESOLVED (needs training): harder negs ≠ guaranteed
   leaf-level separation lift (train↔eval geometry) — only the gated echino-grid + arthropoda/metazoa
   settle it. **NEXT: rewrite Phase 2 plan → v2.2 (apply the still-valid review fixes: kNN-purity
   scalar bug, w_near<w_far guard, analyzer min_size/return contract, ≥3 training seeds, ALL-LRZ-out +
   hard STOP, commit-via-`git commit -F`, regression checkpoints, inline CLI), unblock, then build.**
   Fold Experiment-1 (5664609) ep80 readout for E1c-vs-E2/E3 priority.
4. Cleanup owed (rm blocked): `scripts/_inspect_dataset_readonly.py`, `scripts/_p4_check_nnodes.py`,
   `scripts/_inspect_checkpoint_command.py`, `smoke_abtest_delete/` tag.

---

## 2026-06-02 (PM) — Experiment 1 patch landed + submitted (job 5664609) after one false start

**Set out to:** implement the LR-schedule patch from this morning's pickup checklist (cosine + warm-restart on each curriculum phase boundary), wire it through the CLI, sanity-check, write the Experiment 1 sbatch, submit.

**Patch (`train_small.py` + `src/taxembed/cli/main.py`):**

Added three orthogonal flags:
- `--lr-schedule {const, cosine, cosine_warmrestart}` (default `const`, preserves existing behavior)
- `--warm-restart-on-phase` (required with `cosine_warmrestart`; the only restart trigger supported)
- `--lr-min-multiplier` (default 0.01 = floor at 1% of base_lr)

New helper `_compute_scheduled_lr(epoch, n_epochs, base_lr, schedule, lr_min_mult, phase_boundaries)`. In `train_with_visualization`, the per-epoch LR-set is wired AFTER the burn-in handling and BEFORE the curriculum-phase switching, so a phase boundary at epoch `e_start` both (a) loads the new dd window and (b) gets base_lr back via cosine restart.

Validation: `cosine_warmrestart` requires `--curriculum` + `--warm-restart-on-phase` (both checked at argparse-validation time before any setup work).

Math verified offline by `scripts/_smoke_lr_schedule.py`: under Experiment 1 conditions (n_epochs=200, base_lr=0.001, auto curriculum phases [(1,1),(40,9),(80,18),(120,None)]), the trajectory at the dd≤18 leverage point is **ep79=1.15e-5 → ep80=1.00e-3** — full reset on the same epoch the curriculum loads dd≤18. That's exactly the lever the morning's diagnostic verdict asked for.

**First sbatch was wrong — cancelled before queue start:**

Submitted job 5664584 on `_smoke_lr_schedule.py` (math-only) + `sbatch --test-only` (header-only) validation. Neither exercised the actual training code path: the per-epoch LR-set inside the epoch loop, the optimizer.param_groups mutation, the AMP × grad-accum × curriculum-phase-switch interactions. User pushback was unambiguous; cancelled before the queue let it start. New standing rule saved as feedback memory `feedback_no_lrz_submit_without_local_end_to_end.md`: no LRZ sbatch on offline-only validation. Echino end-to-end (~3 min on MPS) first, then sbatch.

**Local end-to-end smoke (the real one, now required):**

```
taxembed train --file data/taxopy/echinodermata_7586_clean/...transitive.npz \
  -as smoke_lr_schedule_echino --dim 50 --epochs 30 --batch-size 256 \
  --n-negatives 100 --lr 0.001 --grad-accum-steps 4 --curriculum \
  --curriculum-phases auto --lr-schedule cosine_warmrestart \
  --warm-restart-on-phase --lr-min-multiplier 0.01 \
  --euclidean-param --loss softmax --epoch-fraction 0.3 [+radial flags]
```

30 epochs ran clean. Loss 4.624 → 4.620 (best @ ep 4, before the first phase boundary at ep 6 — expected). Best/rolling-5/milestone-10-20-30 checkpoints all written. `run.json` correctly persists `lr_schedule: cosine_warmrestart`, `warm_restart_on_phase: true`, `lr_min_multiplier: 0.01` + the new flags in the saved command. Config line at startup correctly prints `LR schedule: cosine_warmrestart → floor 1.00e-05 (warm-restart on each curriculum phase boundary)`.

The smoke exercises THE SAME code path that will run on LRZ — only difference is dataset scale + device (MPS vs CUDA + AMP scaler, which doesn't touch `param_groups['lr']`). High confidence the LRZ run won't hit a patch-level bug.

**Resubmitted: job 5664609** (`taxembed_metazoa_lower_lr_bigg`), PD/Priority on `lrz-v100x2`, 48h walltime, expected queue clear hours-to-1-day per the morning session's priors. Same Job C recipe (softmax + euclidean-param + curriculum auto + radial-nudge + log radial schedule + depth-scale margin + epoch_fraction 0.3 + save-every 10), four levers changed:

| Lever | Job C | Experiment 1 |
|---|---:|---:|
| batch-size | 128 | 256 |
| grad-accum-steps | 4 | 8 |
| n-negatives | 100 | 300 |
| lr | 0.005 | 0.001 |
| lr-schedule | const | cosine_warmrestart + warm-restart-on-phase |

**Changed:**
- `train_small.py`: `_compute_scheduled_lr` helper + 3 argparse flags + validation + per-epoch LR set in `train_with_visualization` after burn-in.
- `src/taxembed/cli/main.py`: 3 flags forwarded to `train_cmd` + persisted in `run.json` metadata.
- `scripts/_smoke_lr_schedule.py` (NEW): math-only unit smoke for the schedule helper.
- `scripts/train_lrz_metazoa_lower_lr_bigger_batch.sh` (NEW): Experiment 1 sbatch in actual LRZ container shape (squashfs `pytorch-2.4.0-cuda12.1.sqsh`, `--container-mounts` for src/data/artifacts, `MKL_THREADING_LAYER=GNU`, `pip install -e .` at startup). This is the canonical Experiment-1 launcher; the original `scripts/train_lrz.sh` `.venv`-based template doesn't match the live LRZ layout.
- LRZ-side files synced: `src/train_small.py`, `src/src/taxembed/cli/main.py`, `metazoa_lower_lr_bigger_batch.sbatch` (at project root, matching the existing one-sbatch-per-experiment convention).
- Memory: `feedback_no_lrz_submit_without_local_end_to_end.md` + MEMORY.md index entry.

### 🎯 PICKUP CHECKLIST — NEXT SESSION

1. **Check job 5664609 state:**
   ```
   ssh ai 'sacct -j 5664609 --format=JobID,JobName,State,ExitCode,Elapsed,End -X'
   ssh ai 'squeue -j 5664609'
   ```
   If still PD: come back. If R: read live progress via `watch_train.py` or `wait_job.py` on LRZ. If COMPLETED: pull artifacts.

2. **Pull artifacts when COMPLETED:**
   ```
   scp -pr ai:/dss/.../taxembed_lrz/artifacts/tags/metazoa_lower_lr_bigger_batch \
       artifacts/tags/
   ```
   Expect: 20 milestone checkpoints (ep 10/20/.../200), best, final, run.json, + the rolling 5.

3. **Run the sweep analyzer over Experiment 1's milestone trajectory:**
   ```
   .venv/bin/python scripts/_sweep_diagnostic_analyses.py
   ```
   May need a one-line tweak to add `metazoa_lower_lr_bigger_batch` to its tag list — check first. Output: per-checkpoint depth↔norm + per-rank separation, like the Job C diagnostic table this morning.

4. **Read against the dd≤18 success criterion** (from this morning's pickup):
   - depth↔norm ≥+0.85 at ep 80 (the dd≤18 transition)
   - sep ≥1.10× at ep 80
   - Bonus signal: does the trajectory show LR-restart effects — loss bumps at ep 40/80/120, then re-decay (vs Job C's flat collapse)?

5. **Branch:**
   - **Success** → queue full dd≤all run (drop early-stopping back on, no other changes). EXCELLENT (≫1.5×) probably still requires architecture work — see Experiment 3 candidates.
   - **Failure** → architectural decision becomes urgent. Three Experiment 3 candidates:
     - (a) `--dim 200` (cheap to try; run `taxembed dim --tag metazoa_softmax_milestones` first to see if angular packing predicts headroom)
     - (b) `--optimizer radam` + hyperbolic-aware negative sampling (NOT in code yet; design+implement first)
     - (c) Product manifold (one ball per top-level class). Largest change, last resort.

6. **Backstop in parallel (regardless of Exp 1 outcome):** queue Experiment 2 (`--curriculum-phases 1:1,1:9 --epochs 100`, dd≤9 truncate). Locks in the 1.14-1.20× peak as a reproducible non-failing baseline. Cheap, ~9h on V100.

7. **Update logs:**
   - `PROJECT_STATE.md` "Current metrics" gets Experiment 1 row(s) when artifacts land.
   - `PROJECT_STATE.md` "Roadmap" gets the next-experiment direction.
   - New SESSION_LOG entry with the verdict.

**Open questions tracked in memory + log:**
- Does dd≤9 quality survive the dd≤18 transition under structural levers? (Experiment 1 will answer.)
- Is 1.20× the architectural ceiling at 498k nodes, or just the recipe ceiling? (Experiment 3 will answer if 1 fails.)
- Does dim=100 have enough angular volume for 498k? (`taxembed dim --tag` is a cheap proxy.)

---

## 2026-06-02 — diagnostic sweep verdict: dd≤9 → dd≤18 transition is the breaker

**Set out to:** pull the 3 LRZ diagnostics (A, B, C; A+B finished after last session close, C had finished earlier), run the consolidated sweep, decide which next experiment to launch using the decision matrix from `PROJECT_STATE.md`.

**Found:**

All 3 diagnostics ran clean (A=27h16m, B=26h49m, C=17h28m, all exit 0). Pulled best + final from A/B + best/final + 20 milestones from C; ran `scripts/_sweep_diagnostic_analyses.py` over 26 checkpoints. Summary at `artifacts/tags/metazoa_diagnostics_summary.tsv`.

**A and B end POOR as expected — but the C milestone trajectory is the real story.** Removing curriculum did not rescue softmax (A=1.03×, depth +0.67); did not rescue ranking (B=1.02×, depth +0.73). Both stuck at the dd≤all wall from epoch 1.

**C's 20-milestone trajectory at 498k metazoa:**

| Phase | Milestone | Depth↔norm | Best sep | What's happening |
|---|---|---:|---:|---|
| dd≤1 | ep 10 | +0.853 | 1.00× | initialization |
| dd≤1 | ep 20-30 | +0.91 → +0.93 | 1.02 → 1.10× | mastering trivial pairs, peak depth |
| **dd≤9** | ep 40 | +0.845 | 1.10× | depth dips, sep climbs |
| **dd≤9** | **ep 50-70** | **+0.875** | **1.14-1.20×** | 🟢 **PEAK QUALITY** — sustained 20-epoch plateau |
| **dd≤18** | ep 80 | +0.796 | 1.10× | 🔴 collapse begins — depth crashes 0.88→0.80 |
| **dd≤18** | ep 90-110 | +0.78 | 1.06× | continued regression |
| **dd≤all** | ep 120 | +0.673 | 1.04× | catastrophic — locked in |
| **dd≤all** | ep 130-200 | +0.668 | 1.03× | terminal wall, no recovery |

**The diagnostic question is now answered with precision:**

1. **The collapse mechanism is the dd≤9 → dd≤18 transition at ep 80**, NOT the dd≤all jump at ep 120 as the 2026-05-31 reading hypothesized. By the time dd≤all loads in, the model is already at +0.67 / 1.04× — dd≤all is just locking in damage already done by dd≤18.
2. **The dd≤9 phase IS solvable at scale.** Softmax + euclidean-param + radial scaffolding sustains 1.14-1.20× sep at depth +0.88 for 20 epochs (ep 50-70). This matches the original `metazoa_softmax_best` — that "best" was captured here.
3. **No-curriculum is strictly worse than curriculum-peak.** A (softmax-no-curric) never even bootstraps to dd≤9 quality (1.03× from ep 1 onward). Softmax NEEDS the curriculum warm-start; the loss landscape at dd≤all directly is too rough at 498k with n_negatives=100.
4. **The architecture ceiling at 498k is ~1.20×.** That's POOR/borderline MODERATE. Echino's 3.01× does NOT transfer.

**Likely mechanism for the dd≤18 breaker:** dd≤9 → dd≤18 roughly doubles transitive pair density per node + explodes meaningful-negatives population per anchor. n_negatives=100 partition becomes too sparse → noisy gradient → model wanders off the dd≤9 manifold faster than it can re-converge on the larger objective.

**Decision matrix outcome vs priors (from `PROJECT_STATE.md` 2026-05-31):**
- ~70% "structural fix needed" — **directionally CORRECT**, but the fix should target the dd≤9→dd≤18 transition specifically (bigger batch, more negatives, LR cosine warm-restart on phase boundaries).
- ~25% "softmax no-curric converges to softmax-best" — **REFUTED**. A is worse than curric-peak.
- ~5% "both A+B converge to ≥1.5×" — **REFUTED**.

**Changed:**
- `PROJECT_STATE.md`: added 7 new rows in "Current metrics" (A, B, C peak/collapse-onset/post-collapse/final + no-curric A/B). Replaced the 2026-05-31 "open question" load-bearing bullet with the dd≤9→dd≤18 verdict. Added the 1.20× ceiling bullet. Refined Known-issue #6. Replaced the "active diagnostics + decision matrix" Roadmap section with the next-experiments ranking (dd≤18-survival, dd≤9-truncate, architectural alternatives).
- `artifacts/tags/metazoa_diagnostics_summary.tsv`: 26-row TSV with per-checkpoint depth + per-rank separation.
- Per-checkpoint plots at `artifacts/tags/<tag>/analysis_<label>/` for all 26 analyses.

**Side cleanup this session (disk was at 97% full):**
- Deleted 9 obsolete `.pth` files (metazoa_v1, metazoa_v2_euclparam, metazoa_softmax, smoke_arthro_e1) — 2.0 GB, analysis subdirs preserved.
- Moved `/Users/.../SpeciesEmbedding/_archive/` (19.9 GB) to `Dropbox/Science/SpeciesEmbedding/_archive/` + symlinked. User marked online-only in Dropbox; eviction frees ~20 GB locally.
- Outstanding task to definitively cull `_archive/` logged to project memory (`project_archive_definitive_cleanup_pending.md`).

### 🎯 PICKUP CHECKLIST — NEXT SESSION

1. **Decide which next experiment to launch (rank by leverage):**
   - **Experiment 1 (dd≤18 survival):** `--batch-size 256 --grad-accum-steps 8 --n-negatives 300 --lr 0.001` with cosine + warm-restart on each curriculum phase boundary. Tag: `metazoa_lower_lr_bigger_batch`. Highest leverage — if dd≤18 transition is survived, the dd≤all path opens up and EXCELLENT is in reach.
   - **Experiment 2 (dd≤9 truncate, backstop):** `--curriculum-phases manual` capped at dd≤9, 100 epochs. Locks in the known 1.14-1.20× peak as a reproducible non-failing baseline.
   - **Experiment 3 (architectural):** only if Experiment 1 doesn't reach EXCELLENT. Riemannian optimizer, hyperbolic-aware negative sampling, dim 200.

2. **Implement what Experiment 1 needs:**
   - LR cosine + warm-restart support in `train_small.py` (not yet there per a quick grep — sanity-check before launching). The `taxembed.cli.main` argparse pass-through needs a `--lr-schedule {const,cosine,cosine_warmrestart}` flag and a `--warm-restart-on-phase` flag.
   - If we want grad-accum > current `--grad-accum-steps 4`, verify upper bound on what fits in V100 16GB at batch 256 (V100 ran fine at 128 + grad-accum 4; 256 × grad-accum 8 = same accumulated batch, just smaller per-step → should fit).

3. **After Experiment 1 lands:** read against the dd≤18 success criterion (depth↔norm ≥+0.85 at ep 80, sep ≥1.10×). If yes → keep going to dd≤all. If no → diagnose which lever was insufficient.

4. **Architectural decision pending:** the 1.20× ceiling at metazoa scale may be hard. After Experiment 1 settles, consider whether the project's stated bar (EXCELLENT ≫1.5-2×) is reachable with the current architecture or if Experiment 3 is needed first.

---

## 2026-06-01 → 02 — diagnostic mid-training observation: at-scale objective looks like the wall, not curriculum

**Status of the 3 diagnostics submitted at the end of the 2026-05-30→31 session:**

Queue cleared **much faster than the projected 2026-06-02 02:03** start — all three started 2026-06-01 ~07:00 on three different GPUs (gpu-002 ranking, gpu-003 milestones, gpu-005 no-curric softmax). True parallel, no further queue contention.

**Last live read this session (2026-06-01, ~8h elapsed = ~1/3 to 2/3 through):**

| Job | Epoch | Loss | Best loss @ epoch | Reading |
|---|---|---|---|---|
| A (5661756, no-curric softmax) | 66 / 200 | 4.404 | **4.404 @ ep 26** | Stuck at SAME 4.40 plateau the original failed softmax hit at ep 100+ |
| B (5661758, no-curric ranking) | 65 / 200 | 0.392 | **0.392 @ ep 38** | Same plateau as v2_euclparam's final state |
| C (5661777, softmax + milestones) | 135 / 200 | 4.405 | **2.575 @ ep 39** | Reproducing original softmax IDENTICALLY: best @ ep 39 = 2.575 (orig was 2.572, within noise), then climb to 4.40 plateau. Curriculum schedule deterministic; failure not a fluke. |

**Early signal (mid-training, NOT a verdict — to be confirmed when runs finish):** the "curriculum transitions are the breaker" hypothesis is **in serious trouble**:
- A removed curriculum entirely and hit the same loss plateau as the curriculum-collapsed run.
- B removed curriculum entirely on the ranking recipe and also hit the same plateau v2 settled at.
- C reproduces the original failure deterministically.

If A and B end POOR at 200 epochs, the at-scale objective itself is the wall (likely n_negatives=100 too low for 498k candidates → noisy gradient, or LR=0.005 too high for the converged regime). The path forward then is **structural** — bigger batch, lower LR, more negatives, different optimizer — not curriculum or loss-function tweaks. The decision-matrix branch in `PROJECT_STATE.md` "Roadmap" handles this.

**Also confirmed mid-session:** C has already produced milestones for ep 10/20/.../100 (10 of 20 expected). The `--save-every 10` flag works as designed. The full milestone trajectory will be available after run completion.

**Classifier outage at session close:** the Anthropic Bash safety classifier went down at ~end of session — `ssh ai` calls all returned the temporarily-unavailable error. Tried 5+ times across ~15 min. Could not:
- Confirm final job state
- Pull artifacts
- Run sweep analysis

**Prepared during the outage:** `scripts/_sweep_diagnostic_analyses.py` — drop-in runner that, when classifier recovers and artifacts are pulled, processes all 3 runs (best + final for A/B + best + final + all 20 milestones for C), writes a summary TSV at `artifacts/tags/metazoa_diagnostics_summary.tsv`, and prints a paste-ready markdown table. Designed to be one-shot.

### 🎯 PICKUP CHECKLIST — RE-RUN AT THE TOP OF THE NEXT SESSION

When you start the next session (likely once the classifier is back / harness restarted):

1. **Confirm jobs completed:**
   ```
   ssh ai 'sacct -j 5661756,5661758,5661777 --format=JobID,JobName,State,ExitCode,Elapsed,End -X'
   ```
   Expected: all three COMPLETED with exit 0 (they were 8-9h in last we checked, with 17-18h typical runtime — should be done by 2026-06-02 morning).

2. **Pull artifacts to local** (~10 GB total for C; A+B ~800 MB each):
   ```
   # A — best + final + run.json
   scp -p ai:/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/artifacts/tags/metazoa_softmax_nocurric/'*' \
       /Users/.../TaxPointCare/poincare-embeddings/artifacts/tags/metazoa_softmax_nocurric/
   # B — same
   scp -p ai:/dss/.../metazoa_v2_nocurric/'*' .../metazoa_v2_nocurric/
   # C — pulls all 20 milestones too (use scp -r tag dir for convenience)
   scp -pr ai:/dss/.../metazoa_softmax_milestones .../artifacts/tags/
   ```

3. **Run the sweep analyzer (one command, ~5 min × ~25 checkpoints = ~2 hours total):**
   ```
   /Users/.../TaxPointCare/poincare-embeddings/.venv/bin/python \
     /Users/.../TaxPointCare/poincare-embeddings/scripts/_sweep_diagnostic_analyses.py
   ```
   Output: `artifacts/tags/metazoa_diagnostics_summary.tsv` + paste-ready markdown table on stdout.

4. **Apply the decision matrix** in `PROJECT_STATE.md` "Roadmap" — given the mid-training observation above, the probabilistic verdict (before-results) is:
   - **Most likely (~70%):** both A and B end POOR at ~1.0-1.1× separation. The at-scale objective is the wall. Next experiment is structural: bigger batch (try 512 via grad-accum 16) and/or lower LR (5e-4) and/or more negatives (500).
   - **Less likely (~25%):** A converges to something resembling softmax-best (~1.14-1.20×) but B stays POOR. Means: dropping curriculum is enough for softmax. Recipe direction = pure softmax without curriculum.
   - **Tail (~5%):** A and B both converge to MODERATE+ (1.5×+). Means: curriculum transitions WERE the breaker but the recipe is otherwise fine. Reintroduce curriculum with LR scheduling.

5. **For C's milestones:** loop the analyzer over `_milestone_epoch{10,20,...,200}.pth`. The trajectory should show separation+depth↔norm by epoch — pinpointing exactly where the dd≤1→9 transition (ep 40) and subsequent phases hurt the model. Even if A/B confirm the at-scale-wall hypothesis, C's per-epoch data is publication-grade evidence for the failure mechanism.

6. **Update logs:** PROJECT_STATE.md "Current metrics" gets the new rows; "Roadmap" gets the verdict + new experiment direction; SESSION_LOG.md gets a fresh dated entry; memory file gets a "State as of 2026-06-02" appended section + updated description frontmatter.

**Reference points to compare new results against** (unchanged from last session):
- v2 ranking best (dd≤1, ep 21): depth +0.878 / 1.05-1.09× — floor.
- softmax best (dd≤1, ep 39): depth +0.875 / 1.14-1.20× — current ceiling at scale.
- Both finals (curriculum-collapsed): depth +0.67-0.73 / ~1.00-1.04× — failure signature.
- Echino softmax: 1.46/3.01/2.63× — the bar.

**Helpers (now in `scripts/`, all preserved):**
- `_sweep_diagnostic_analyses.py` — NEW this session, the one-shot runner above.
- `_slice_by_phylum_and_analyze.py` — per-phylum slicer (from previous session).
- `_tally_rank_distribution.py`, `_check_local_deps.py` — utilities.

---



**Goal:** evaluate the two LRZ runs launched at S0229 (job 5660105 v2_euclparam + 5660187
softmax) and decide whether softmax clears the EXCELLENT bar at 500k metazoa scale.

**Status of jobs (all completed cleanly, exit 0):**
- `metazoa_v2_euclparam` (5660105): COMPLETED 2026-05-30 09:54, 17h03m wall.
- `metazoa_softmax` (5660187): COMPLETED 2026-05-30 20:59, 17h18m wall.
- pLM choice orphan_extract (5660122-25 for ankh_large, esmc_600m, esm3, prost_t5):
  all 4 wrote `11444/11444 → orphan_bromberg/embs/<plm>.h5`. The orphan-benchmark embeddings
  dir is now complete for the 9 local-pipeline pLMs (excluding the biocentral-side ones).

**Analysis matrix (this session's main work):**

| Run | Phase saved | depth↔norm | phylum | class | order | family |
|---|---|---:|---:|---:|---:|---:|
| v2 best (ep ~21) | dd≤1 only | +0.878 | 1.05× | 1.08× | 1.09× | 1.05× |
| v2 final (ep 200) | dd≤all | +0.727 | 1.02× | 1.03× | 1.02× | 1.00× |
| softmax best (ep 39) | dd≤1 only | +0.875 | **1.14×** | **1.20×** | **1.19×** | **1.15×** |
| softmax final (ep 200) | dd≤all | +0.667 | 1.03× | 1.04× | 1.04× | 1.03× |

**Findings — the load-bearing four:**
1. **Curriculum schedule (from log):** `ep 1: dd≤1 | ep 40: dd≤9 | ep 80: dd≤18 | ep 120: dd≤all`.
   **Both** runs saved `best.pth` during dd≤1 — i.e., trained on direct parent-child edges ONLY,
   never on transitive closure. Best-by-loss isn't best-by-objective; it's best on the easy
   sub-problem the curriculum starts with.
2. **Both runs catastrophically degrade after the curriculum advances.** Depth↔norm drops
   +0.88 → +0.73 (v2) / +0.88 → +0.67 (softmax). Separation collapses to 1.00-1.04× across
   every rank in both. The recipe is destroying its own dd≤1-phase achievement during the
   hard phases. Likely mechanism: with 498k candidates and 11.8M transitive pairs by dd≤all,
   batch_size=128 + LR=0.005 + Adam can't track the loss-landscape shifts at curriculum
   transitions, and the model wanders.
3. **Echino successfully converged through curriculum because 4k nodes → ~40k transitive
   pairs is tractable.** Metazoa at 11.8M pairs thrashes. The echino-validated recipe was
   validated at a scale where curriculum was free.
4. **Softmax DOES have a real signal at scale — but only on the easy phase.** Softmax-best
   beats v2-best by ~10% on every rank (1.14-1.20× vs 1.05-1.09×). That's directionally
   consistent with the 2.5× echino lift but tiny in magnitude — and on a strictly easier
   sub-task. **We do not yet know whether softmax can clear the bar at metazoa scale**
   because we never successfully trained the harder objective.

**Side investigations done this session:**
- **"Are there too many `Metazoa sp.`-style placeholder nodes?"** Helper `_tally_rank_distribution.py`
  scanned all 498k. Only 0.17% of nodes match any noise pattern (4 with " sp.", 225
  "unclassified", 0 "uncultured"/"endosymbiont"). 100% of species-rank nodes have a phylum
  ancestor; 99.84% have family. The `_clean` preprocessing scrubbed it. NOT a contributor
  to the gap.
- **"Is it scale density crowding lateral structure?"** Helper `scripts/_slice_by_phylum_and_analyze.py`
  sliced v2_best by phylum (echino 3,965, mollusca 32k, chordata 93k, arthropoda 325k).
  Echino-slice-of-v2 gave 1.05-1.09× vs standalone echino_euclparam's 1.21× on the same 4k
  nodes. **Rules out scale-crowding** — same nodes, same recipe, different training scope.
  Result is the loss-signal-dilution at scale, since-superseded by the curriculum-collapse
  finding (which is a separate, larger effect).

**Parked from yesterday (now done after VPN returned):**
- v2 final.pth pulled and analyzed → confirms it ALSO degrades from best (not just softmax).
  The collapse is a recipe property, not a softmax-specific failure.

**Updated:** `PROJECT_STATE.md` — added 4 run rows + the load-bearing bullet
"CURRICULUM PROGRESSION COLLAPSES BOTH RUNS AT 498K". Future runs should read it before
launching another 17h experiment on the same recipe.

**Next (all three SUBMITTED in parallel, projected start 2026-06-02 02:03):**
1. **5661756 — `metazoa_softmax_nocurric`** — drop `--curriculum`; train 200 ep on dd≤all
   from step 1. Tests whether curriculum *transitions* are the breaker vs the final
   objective itself. sbatch: `taxembed_lrz/metazoa_softmax_nocurric.sbatch`.
2. **5661758 — `metazoa_v2_nocurric`** — same drop-curriculum change applied to v2 ranking
   recipe. Loss-control for #1. sbatch: `taxembed_lrz/metazoa_v2_nocurric.sbatch`.
3. **5661777 — `metazoa_softmax_milestones`** — full curriculum softmax + `--save-every 10`
   (new flag added to `train_small.py` + `taxembed.cli.main`; smoke-tested locally on echino
   before push). Gives 20 milestone checkpoints through the dd≤1 → 9 → 18 → all
   transitions, so post-mortem can identify exactly where collapse begins. sbatch:
   `taxembed_lrz/metazoa_softmax_milestones.sbatch`.

**Decision matrix when results land** (~2.5 days from submission):
- (1) converges, (2) collapses → softmax is robust without curriculum, ranking isn't.
- Both (1)+(2) converge → curriculum was the issue; reintroduce with LR scheduling.
- Both collapse → at-scale gradient budget is the wall; need structural change (bigger
  batch, lower LR, different optimizer like Riemannian) rather than parameter tweaks.
- (3) gives epoch-level resolution regardless of (1)/(2) outcomes — points at the specific
  curriculum transition that breaks the model.

**Harnessing changes this session:**
- `train_small.py`: added `--save-every N` arg + `save_every` param to
  `train_with_visualization` + milestone save block in training loop. Default 0 = old
  behavior; backwards-compatible (A+B will run normally even though they share the
  modified file). LRZ src updated via `scp` to `taxembed_lrz/src/train_small.py`.
- `src/taxembed/cli/main.py`: pass-through for `--save-every`. LRZ src updated via `scp`
  to `taxembed_lrz/src/src/taxembed/cli/main.py` (the double-`src/` path is correct).
- `scripts/_slice_by_phylum_and_analyze.py`: new helper for per-phylum slicing analyses.
- `scripts/_tally_rank_distribution.py`, `scripts/_check_local_deps.py`: preserved from /tmp.

### 🎯 PICKUP CHECKLIST FOR THE NEXT SESSION

When the next session starts (earliest useful time = 2026-06-02 evening, ~17h after the
2026-06-02 02:03 projected start):

1. **Check job status:**
   ```
   ssh ai 'sacct -j 5661756,5661758,5661777 --format=JobID,JobName,State,ExitCode,Elapsed,End -X'
   ssh ai 'squeue -u ge94xik2'
   ```
   If still PENDING, sleep until later. If RUNNING, wait. If COMPLETED with exit 0,
   continue.

2. **Pull artifacts to local** (398 MB per `.pth` × multiple files; scp via `ai` alias works):
   - `metazoa_softmax_nocurric/{best,_,milestone_epoch*}.pth` from
     `/dss/.../taxembed_lrz/artifacts/tags/metazoa_softmax_nocurric/`
   - Same for `metazoa_v2_nocurric/` and `metazoa_softmax_milestones/`
   - Plus each tag's `run.json`
   - Local destination: `TaxPointCare/poincare-embeddings/artifacts/tags/<tag>/`
   - Note: A+B don't have `--save-every` set so only best + last 5 epoch + final exist.
     C will have 20 milestones (ep 10, 20, ..., 200) + best + last 5 + final.

3. **Run hyperbolic analysis on each** (~5 min each, uses poincare repo's own .venv
   which has taxopy):
   ```
   /Users/.../TaxPointCare/poincare-embeddings/.venv/bin/python \
     /Users/.../TaxPointCare/poincare-embeddings/scripts/analyze_hierarchy_hyperbolic.py \
     --checkpoint <tag>_best.pth \
     --mapping data/taxopy/metazoa_33208_clean/taxonomy_edges_metazoa_33208_clean.mapping.tsv \
     --ranks phylum class order family \
     -o artifacts/tags/<tag>/analysis_best/
   ```
   And same for `<tag>.pth` (final). For C: also run on each milestone to build the
   per-epoch trajectory — cheap loop.

4. **Read against the decision matrix in `PROJECT_STATE.md` "Roadmap":**
   - Score depth↔norm and per-rank separation per tag.
   - Compare to v2 best (1.05-1.09×) and softmax best (1.14-1.20×) reference points.
   - Apply the 4-branch decision matrix to pick what to launch next.

5. **Update logs:** add 4 new rows to PROJECT_STATE.md "Current metrics" (one per
   no-curric ranking + softmax best/final + one for C aggregated), update the load-bearing
   bullet under "The recipe" with the new finding, and append a fresh dated entry to
   `SESSION_LOG.md`.

**Key reference points to compare against** (don't re-derive from scratch — pre-existing
metazoa-scale anchors):
- v2 ranking best (dd≤1 only): depth +0.878 / 1.05-1.09× — *floor* for what these
  diagnostics should beat to mean anything.
- softmax best (dd≤1 only): depth +0.875 / 1.14-1.20× — *current ceiling* at scale.
- Both finals (curriculum-collapsed): depth +0.67-0.73 / ~1.00-1.04× — *failure
  signature* to recognize.
- Echino softmax: 1.46×/3.01×/2.63× — the bar.

**Pitfalls to avoid:**
- Don't trust best.pth blindly — at 498k scale "best" so far has always been an
  early-curriculum easy-phase artifact. Compare best vs final vs (for C) per-milestone.
- The `--tag` mode of `analyze_hierarchy_hyperbolic.py` FAILS locally because run.json
  stores LRZ container paths. Always pass `--checkpoint` + `--mapping` explicitly.
- The training data uses BSD `grep` semantics on macOS — `command grep` works without
  the `-G` flag bug; pipe chains and `&&` chains are hook-blocked anyway, prefer Read.
- A+B were submitted with the modified `train_small.py` + `cli/main.py` on LRZ but
  default `save_every=0` → they behave identically to the original recipe. If you see
  unexpected milestone files in A or B output dirs, something's wrong.

---



---



**Goal:** evaluate the just-finished `metazoa_v1` embedding (498,246 nodes, dim 100,
trained on LRZ) against the project's real bar: distances monotone in taxonomic
relatedness at every level (sister species closest, then genus, family, …), with
radius encoding depth.

**Findings**
- **metazoa_v1 FAILED.** Group separation ≈1.0× at *every* rank (phylum→genus);
  even congeneric species are no closer than random insects. Depth↔norm only +0.35
  (weak), mean norm 0.92 (everything jammed at the boundary). Analysis:
  `scripts/analyze_hierarchy_hyperbolic.py` (run with explicit `--checkpoint` +
  `--mapping`; `--tag` mode fails locally because `run.json` stores LRZ container
  paths `/data/...`). Plots in `artifacts/tags/metazoa_v1/`.
- **Two contributing causes identified:**
  1. *Early-stop trap.* Curriculum is stepwise (dd≤1 until ep40, then 9/18/all at
     ep80/120). Early-stopping patience 25 fired at ep45; the "best" checkpoint is
     **ep20 — trained on dd≤1 (parent→child) edges only.** Selecting on the
     composite `quality` metric, which peaks during the trivial early curriculum,
     guarantees saving a barely-trained model. (ep45 checkpoint: still flat,
     norms inflating → longer patience alone insufficient.)
  2. *Missing `--euclidean-param`.* `run.json` shows `euclidean_param: false`, i.e.
     Adam was run directly on Poincaré-ball coordinates (the mismatched mode that
     suffers conformal-factor gradient collapse). The documented working recipe
     REQUIRES `--euclidean-param` (see below). **This is likely the dominant cause.**
- **echino baselines (local, clean 3,965-node clade) to split tuning-vs-method:**
  - `echino_v1` (no euclidean-param): depth↔norm **+0.85 GOOD**, separation
    class/order/family **1.13 / 1.16 / 1.07×** (weak).
  - `echino_v2_tiered` (+ `--tiered-negatives`, still no euclidean-param):
    separation *dropped* to 1.03 / 1.03 / 0.97× — sibling hard-negatives push
    close relatives apart and HURT clustering when used without the rest of the recipe.
- **Archaeology (git history + `docs/JOURNEY.md` + README):** the historical
  "clean" echino runs used the SAME margin-ranking + radial-nudge architecture —
  NOT a different loss. Documented best:
  - `echino_v4`: r=+0.950, **1.21×** — "Euclidean Adam + radial nudge"
  - `echino_v9d`: r=+0.990, **1.68×** — "+ tiered negs + class-weighted loss" (best)
  - (`echino_v3` was the RiemannianAdam *failure*, not a clean run.)
  The `--euclidean-param` flag (optimize in tangent space, map to ball via tanh)
  was added deliberately to fix RiemannianAdam's gradient collapse. **My echino_v1/v2
  AND metazoa_v1 all omitted it** → that's the misconfiguration.
  The true Nickel–Kiela softmax loss exists only in archived legacy code
  (`docs/archive/legacy_facebook/hype/energy_function.py`), unused by current paths —
  a fallback experiment, not the source of the clean echino result.

**Corrected-run results (local):**
- `echino_euclparam` (+ `--euclidean-param`): depth↔norm **+0.96** (was +0.85),
  separation 1.21/1.21/1.11× — reproduces historical echino_v4 (r0.95, 1.21×).
  **`--euclidean-param` CONFIRMED as the key fix.**
- `echino_v9d_repro` (+ tiered + class-weighted): depth +0.96, separation 1.12/1.18/1.09×
  — extras did NOT help and did NOT reproduce the documented echino_v9d 1.68×. Likely the
  historical v9d used a 7,833-node echino set; our `_clean` build is 3,965 nodes.
- **Takeaway:** radial axis solved (r0.96). Lateral separation tops out ~1.21× (MODERATE)
  on current data, below the EXCELLENT bar. metazoa_v1's collapse is substantially the
  missing `--euclidean-param`.

**BREAKTHROUGH (same session) — softmax loss clears the bar.** Implemented the Nickel–Kiela
softmax/NLL objective as `--loss softmax` in train_small.py (`softmax_loss` +
unit-tested `softmax_loss_from_dists` in train_hierarchical.py). echino results:
- `echino_softmax` (softmax + euclidean-param + radial scaffolding): depth↔norm +0.96,
  separation **class 1.46 / order 3.01 / family 2.63×** — order & family EXCELLENT (≫2×),
  vs 1.21× for the identical setup with the ranking loss. Intra-order distance collapsed
  to 1.16 (vs 3.49 inter) — tight clusters.
- `echino_softmax_pure` (softmax, radial-nudge=0, lambda-reg=0): ~1.0× — no clustering.
  → radial scaffolding is NECESSARY; softmax + scaffolding are complementary.
- **Conclusion: the LOSS was the bottleneck.** New best recipe = `--loss softmax
  --euclidean-param` + radial machinery. This is the recipe to carry to metazoa.

**Metazoa runs launched on LRZ (fan-reviewed first — loss code / SLURM hygiene / config,
all clean bar a stale-src sync that was fixed + verified):**
- `metazoa_v2_euclparam` (job 5660105, RUNNING): ranking + euclidean-param + early-stop off
  — the radial-fix-at-scale baseline.
- `metazoa_softmax` (job 5660187, queued): the winning recipe (softmax + euclidean-param +
  scaffolding) at 500k — differs from v2 only in `--loss`. sbatch: `metazoa_softmax.sbatch`.
  Synced 3 changed files (train_small.py, train_hierarchical.py, src/taxembed/cli/main.py)
  to LRZ src/ before submit.

**Next**
1. ✅ Corrected echino runs done — euclidean-param fixes radial; softmax fixes clustering.
2. Decide whether 1.68× clears the high bar; if not, consider the archived softmax
   loss or other improvements.
3. Re-run **metazoa with `--euclidean-param`** (the proven recipe) at 500k scale on
   LRZ — that's the real payoff. Also fix the early-stop/curriculum mismatch.

**Harnessing:** this session also stood up `SESSION_LOG.md` + `PROJECT_STATE.md`
(the repo previously had no session continuity docs).
