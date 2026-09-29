# Manuscript corrections pending — TaxEmbed objective-integrity review

**Purpose.** One accumulator for every manuscript change the objective-integrity work implies, so the
draft is edited **once**, in a single pass, rather than piecemeal across sessions. Nothing here has
been applied to `manuscript.v3_draft.md` yet.

**Target file:** `manuscript/manuscript.v3_draft.md` (SpeciesEmbedding repo, `main`).
**Rebuild after applying:** `SRC=<abs>/manuscript.v3_draft.md bash scripts/build.sh docx`
(⚠ `build.sh` defaults to the older `manuscript.md` — the SRC override is required).

**House style that applies to every edit below** — zero em dashes · strengths lead, one clean
limit-clause where it belongs, zero preemptive defending · no "X rather than Y" defensive
constructions · "prior (Nickel-Kiela) approach" is the fixed term for the failing baseline ·
"canonical recipe" for ours · quote numbers at the source's precision, never re-round.

**Status key:** `UNCONDITIONAL` = holds regardless of any experiment outcome, apply on sight.
`PENDING <task>` = wording depends on a result not yet in hand.

---

## C1 — Methods: the negative sampler is not the objective the paper describes

**Status:** UNCONDITIONAL
**Evidence:** `results/negative_sampler_audit.json` · `src/taxembed/eval/sampler_audit.py` ·
commit `755cad8`

The Methods describe a clean Nickel & Kiela softmax. N&K sample negatives from `{v' : (u,v') ∉ D}`,
so ancestry exclusion is part of the canonical objective. Our sampler draws from
`_depth_to_nodes[depth(descendant)]` (`train_hierarchical.py:398`; fast path `:400-403`, bare
`np.random.randint` — with replacement, no self-exclusion, no ancestry check) and scores the draw
against the **ancestor** (`softmax_loss`, `:730-732`).

Two deviations, in opposite directions, and the sentence should carry both:

1. **Harder than N&K:** the draw is depth-matched, i.e. deliberate hard negatives. The file's own
   header calls this "cousins at same depth" (`train_hierarchical.py:9`).
2. **Missing guard:** the "not under the anchor" condition is absent, so a draw that is genuinely a
   descendant of the anchor is scored as a negative.

Measured exactly over the full closure (21,399,053 pairs, 1,102,163 nodes, no sampling, no seed):

| quantity | value |
|---|---|
| overall false-negative rate | **47.375240%** |
| by anchor depth | 1.0 @0 · 0.9236 @1 · 0.6636 @2 · 0.5712 @5 |
| pairs with zero valid negatives | **3,026,809 (14.1446%)** |
| root-anchored pairs | 1,102,162 (5.1505%) |

The zero-pool figure is why the obvious fix is not a one-liner and deserves its own clause: for one
pair in seven the valid negative pool is empty once descendants are excluded.

**Note for whoever writes this:** the in-subtree draws are not "slightly wrong hard negatives". Every
`(a, n)` with `n` under `a` is enumerated as a **positive** elsewhere in the closure, so the same pair
is pulled together in one batch and pushed apart in another. And when `n` and `d` are both under `a`
at the same depth they are symmetric with respect to `a`, so the softmax question has no correct
answer. A true cousin makes it answerable.

---

## C2 — The seeded-reproducibility claim — RESOLVED, claim is now true

**Status:** RESOLVED 2026-08-11 by landing plan v2 Task 5. The claim can stand, with a caveat.
**Evidence:** `results/seed_reproducibility.json` · `scripts/verify_seed_reproducibility.py` ·
`tests/test_negative_sampling.py`

**Was:** `seed` occurred zero times in `train_hierarchical.py`, `train_small.py` or
`src/taxembed/cli/main.py`, so `docs/specs/2026-06-09-taxembed-paper-design.md:24`'s "Reproducible
(seeded; repro run matches to ±0.01)" was unsupported by the code. The negative sampler uses the
**legacy global** numpy RNG, so `np.random.default_rng` would not have fixed it.

**Now:** `seed_everything` seeds `random`, legacy global `np.random`, and global `torch` (plus CUDA
when present); `--seed` is wired through `train_small.py` (the real entrypoint) and the CLI, and is
recorded in `run.json`.

Verified by moving a number, not by reading the flag — `--help` alone would look identical if the
seed were ignored. Echinodermata, 2 epochs, both directions:

| run | best loss |
|---|---|
| seed 0, run a | 3.932757 |
| seed 0, run b | 3.932757 (identical) |
| seed 1 | 3.932451 (differs) |

**Caveat the Methods must carry:** this is CPU reproducibility. With `--amp` and CUDA scatter
nondeterminism the shipped GPU recipe gives near-reproducibility, not bitwise equality — so quote a
tolerance, not exactness.

⚠ The shipped 1.1M artifact was trained **before** this landed, so it is not itself reproducible from
a seed. The claim applies to runs from here on; do not retroactively attach it to the released model.

---

## C3 — Figure 4's y-axis has an initialization floor

**Status:** UNCONDITIONAL for the caption; the *headline claim* is PENDING Task 9
**Evidence:** `results/radial_init_floor.json` · `src/taxembed/eval/radial.py` · commit `fc29676`

`_initialize_by_depth` (`train_hierarchical.py:65-96`, esp. `:82`) sets every node's norm to
`target_radius(depth)` at step 0; `radial_regularizer` (`:748-778`) penalizes drift from the same
target. So depth-norm correlation is not an outcome of training.

Measured over the real closure (1,102,163 nodes, max_depth 40):

| schedule | init floor |
|---|---|
| log | **0.957151** |
| linear | **1.000000** |

Released model, trained, same closure-derived depths: **0.957244**. Training moves the correlation by
**~1e-4**.

The caption cannot stand as written. Minimum fix: print the floor beside the value. This also closes
spec v3 §6's open "0.957 vs 0.952" item — the discrepancy is a depth-source difference; on any
consistent source the correlation is fixed at initialization.

⚠ Figure 4's actual *claim* is canonical-vs-prior recipe, and that contrast has only ever been
measured on this planted axis. Whether the claim survives is **Task 9**; do not finalize this
figure's caption or Results sentence until Task 9 reports. Its three readings are pre-registered in
plan v2 Task 9 Step 4.

---

## C4 — Any "vs radial-only null" result needs restating

**Status:** UNCONDITIONAL that it must change; the replacement number is PENDING Task 3

`radial_only_null` is degenerate: it returns the shallow-norm shell (mean retrieved depth 2.95 vs
pool 20.94; 0.00% deep), so it can never predict a deep majority class. That is the source of the
"480x below chance" figure (0.00089 vs frequency-matched 0.4326), and it contaminates the
0.5455-vs-0.0091 lateral-generalization evidence. The direction of that result is very likely right;
the *size* needs a valid null.

Related, already known: multiplicative distortion is non-discriminative at cellular scale (model
median 1.1424 vs radial-only null 1.1374 — the null wins). Report precision@10, not distortion.

---

## C5 — The bridge sits below the majority-class baseline

**Status:** UNCONDITIONAL

Class-rank ladder on the 24k panel, n=24,653, denominator reconciled two independent ways:

raw ProtT5 kNN **0.8368** > majority-"Mammalia" **0.6448** > bridge **0.6029** > frequency-matched
chance 0.4326.

`read_eval.py` never computes the majority baseline. Whatever the bridge section ends up claiming, it
cannot imply the bridge beats a constant answer at this rank on this panel.

---

## C6 — Prior art now exists for NCBI taxonomy embedding

**Status:** UNCONDITIONAL

**v-PuNNs** (arXiv:2508.01010; v1 2025-08-01, v2 2026-01-06; preprint, no venue). Mammalia only, no
train/test split, and the same unguarded −0.96 norm-vs-depth claim. Its unvalidated status is
precisely the gap TaxEmbed can claim, so this is a strengthening citation, not a threat.

Also standing: do **not** cite N&K's Euclidean column (Bansal & Benton 2021 showed it was crippled),
but note their unit-norm-ball explanation is explicitly their own speculation.

---

## C7 — kNN purity under Poincaré distance rewards destroying the planted radius

**Status:** UNCONDITIONAL. Found 2026-09-22 by a planted-truth test.
**Evidence:** `results/task9_scorer_dose_response_mollusca.json` · `scripts/validate_task9_scorer.py`

On a trained mollusca checkpoint:
- Directions were held fixed.
- Each node's radius was blended toward a random, depth-independent value, which drove depth-norm r
  from 0.990 to −0.009.
- The trainer's top-level kNN purity **rose** from 0.715 (±0.076) to 0.969 (±0.005). Sep rose
  slightly, 1.022 → 1.036.
- The radius-free S_angle stayed **bit-identical** (0.8218), because the angular structure never
  changed.

So any kNN-purity or separation figure computed with Poincaré distance across depths is confounded
with the radial schedule. It favours embeddings whose radius has drifted away from depth, and
Figure 4's prior arm is exactly such an embedding. Before the manuscript quotes kNN purity
(e.g. family purity 0.907) or separation ratios as evidence of learned structure, check each against
S_angle or a same-depth / radius-rectified variant. The Task 9 local runs' "prior kNN 99.6% vs
canonical 73.4%" is partly this artefact.

**Confirmed on the actual Figure 4 runs** (metazoa, ep200, `results/transplant_2x2_20260922_132857.json`,
3 seeds × 2,000-node samples of the trainer's own metrics):

| condition | trainer kNN% | trainer Sep | purity 4 levels down | S_angle |
|---|---|---|---|---|
| init (random directions) | 0.512 | 0.744 | 0.007 | −0.002 |
| prior, raw | 0.947 | 0.903 | 0.575 | 0.621 |
| prior directions, planted radii | 0.644 | 0.747 | 0.221 | 0.621 |
| canonical, raw (= planted radii) | 0.977 | 0.764 | 0.955 | 0.973 |
| canonical directions + prior radii | 0.994 | 0.921 | 0.973 | 0.973 |

- **Sep, the trainer's top-level separation ratio, measures radius almost entirely.** Canonical's
  0.764 barely clears random directions (0.744) and rises to 0.921 on the prior's radii. Do not
  quote it as learned structure.
- **About +0.30 of the prior's raw kNN% is radial drift.**
- With radii equalised, canonical directions dominate on the paper's own metrics (kNN 0.977 vs
  0.644), which points to Task 9 reading 1. This is n=1, reported not read.
- ⚠ The headline per-rank separation ratios in PROJECT_STATE come from
  `analyze_hierarchy_hyperbolic.py`, a different metric. They have not been transplant-tested yet;
  do so before quoting them.

---

## C8 — P2's held-out protocol: why the obvious alternatives are not what we did

**Status:** UNCONDITIONAL
**Evidence:** `docs/FINDING_protocols_that_are_vacuous_on_taxonomy_trees.md` ·
`helpers/p2_check_closure_is_a_tree.py` · `helpers/p2_randomdag_changes_the_chance_floor.py` ·
`results/p2_heldout_preregistration.json` (`p2_amendment_1_20260924`)

Two protocols borrowed from other papers' benchmarks were tested against our NCBI closures before
either was written into the P2 evaluation, and both degenerate:

1. Ganea et al. 2018's split — keep the transitive reduction always in training, hold out non-basic
   edges — hands the mandatory Vendrov trivial baseline **100.00%** of every held-out edge, because
   all six clade closures on disk are strict trees (single parent per node), and on a tree the
   reduction determines the whole closure exactly (0/0 symmetric difference on every clade checked).
   WordNet is a DAG (537 basic edges beyond a tree's count = multiple inheritance), which is why the
   same protocol is a real prediction task there.
2. The RandomDAG memorisation control (GRAM, Choi et al. KDD'17) preserves each node's depth and the
   closure's total pair count exactly, but not fan-out — mean chance-floor ratio randomised/real =
   **5.403** on mollusca. Comparing arms on raw MRR would have let the control win regardless of what
   either model learned, producing a false "the model memorises tree shape" verdict.

🛑 **SUPERSEDED 2026-09-29 — DO NOT PASTE THE PARAGRAPH BELOW.** It prescribes a sentence asserting a
working held-out evaluation, and the P2 array has since been run and read: the leaf parent-edge
holdout adopted as the fix is **itself vacuous** for this model class (Finding 3 of the same note).
The superseded prescription is kept, not deleted, because the reasoning above it still stands.

> *(superseded)* **What the manuscript should say (final form only — do not narrate the rejected
> alternative):** the held-out evaluation uses a **leaf parent-edge holdout** over the interquartile
> depth band (`depth ∈ [11, 28]`, 501,037 eligible leaves at cellular_organisms scale), and
> cross-tree comparisons against the RandomDAG control use **chance-normalised `normalized_rank`**,
> not raw MRR. Cite the leaf-holdout precedent (TaxoExpan, Arborist, Octet — spec §3.4) as the
> methodological grounding, not Ganea's split. The full reasoning — including why a leaf restriction
> specifically is required (the internal-node leak) and why RandomDAG needs chance normalisation —
> lives in `docs/FINDING_protocols_that_are_vacuous_on_taxonomy_trees.md`, not in the manuscript
> itself.

**What the manuscript should say INSTEAD (2026-09-29, one sentence, final form only):** held-out link
prediction cannot test generalisation for a transductive, feature-free taxonomy embedding, because
removing a leaf's parent edge removes the only training row that distinguishes its parent from that
parent's siblings. Cite `docs/FINDING_protocols_that_are_vacuous_on_taxonomy_trees.md` §3 for why.

**Additional evidence (2026-09-29):** `results/p2_verdict_20260929.json` (verdict `UNINFORMATIVE`,
all 12 runs) · `helpers/_p2_peak_real_vs_control.py` · `helpers/_p2_curriculum_vs_collapse.py` ·
`helpers/_p2_cosine_vs_poincare.py` · submodule `e04c478`.

**Do not amend or re-run P2.** Read at each arm's own best epoch — a post-hoc reading, admissible
only in the negative direction — the real tree does not beat its degree-matched control: mean delta
**−0.0008** against pooled within-arm seed SD **0.00626**/**0.01056**, sign-changing across seeds.
No schedule fix recovers a verdict, so GPU time spent here buys nothing.

⚠ **Consequence for the paper's claims.** P2 was the designated answer to Burkhard's
*"TaxEmbed: overfitting?"* (the headline S_angle 0.9726 is measured **in-sample**; a memorising model
scores the same). That question now splits:
- *Is the headline planted at initialization?* Answerable now, no GPU — see **C3** (depth-norm init
  floor 0.957151 vs trained 0.957244) and the Task 9 Figure 4 re-plot onto S_angle, whose null is 0
  by construction (canonical 0.9726 vs init null −0.0015).
- *Does it generalise or memorise?* **Not answerable by any node holdout on this model class.** It
  needs an inductive, feature-bearing probe — spec §P4, the pLM showcase, which is Burkhard's own
  suggestion 1. Until that runs it is a **stated limitation**, not a result.
  🛑 **UPDATED 2026-09-29 — see C10.** The P3 placement arm was built as the out-of-sample answer and
  read `ANTICIPATES`; it has since been **withdrawn** — it measured candidate subtree size and never
  tested the held-out taxon. So this bullet now holds more strongly than when written: **no NCBI
  data of any vintage can answer it.** The remaining instruments are §4.5 TimeTree and §P4, and C10
  records that P4 is currently losing to its own majority-class baseline.

⚠ **Also from this run, for C-anything that quotes a chance floor:** `chance_mrr_mean` (0.2711)
disagrees with the empirical untrained MRR (0.2457) by ~10 %, while `normalized_rank` sits on its 0.5
floor exactly. Do not quote `chance_mrr_mean` as a floor until it is re-derived.

---

## C10 — the P3 placement arm measured subtree size; `ANTICIPATES` is withdrawn

**Status:** UNCONDITIONAL — nothing about P3 may be written into the manuscript.
**Evidence:** `results/p3_placement_preregistration.json` key **`p3_amendment_3_20260929`** (read
this first; it is self-contained) · verdict `results/p3_placement_result_v2_20260929.json`
(**UNINFORMATIVE**) · `docs/FINDING_protocols_that_are_vacuous_on_taxonomy_trees.md` **§4** ·
`helpers/_p3_confound_diagnostics.py` · `_p3_combinatorial_baselines.py` ·
`_p3_size_residual_control_v3.py`

C8 closed by saying the generalisation question *"needs an inductive, feature-bearing probe"*. In the
interim the **P3 placement arm** was built and read as `ANTICIPATES` (primary 0.4400, CI95
[0.4214, 0.4588], n = 1,241, four gates passing) — the first apparent out-of-sample evidence. **It
does not survive, for three independent reasons, and the verdict is withdrawn.**

1. **It was never a held-out-taxon test.** Ranking the identical pool from `p_old` instead of from
   the moved taxon `v` gives 0.4444; paired delta **−0.0044, CI95 [−0.0145, +0.0062]**, containing
   zero. `p_old` alone delivers **92.7 %** of the effect. The moved taxon's own coordinate
   contributes nothing measurable.
2. **A one-line tree statistic beats it outright.** "Pick the largest candidate branch" scores
   **0.3146** [0.2982, 0.3313] against the embedding's 0.4444; paired **+0.1298** [+0.1107, +0.1488].
   NCBI moves taxa into big, actively curated groups.
3. **Conditioned on subtree size, nothing remains.** Target residual against a calibration curve
   `E[r_e | r_s]`: **−0.0031 / +0.0019 / +0.0076** at 10/20/40 bins, all CIs containing zero, sign
   changing across binnings, both harness gates passing.
   Subtree size accounts for **94 % / 103 % / 113 %** of the raw effect.

⛔ **Do not quote `results/p3_placement_result_20260929.json`.** It is the superseded `ANTICIPATES`
record, kept unedited on purpose (Rule 5: verdicts are append-only at the file level).

**What the manuscript should say (final form only, if P3 is mentioned at all):** nothing. P3 produced
no result. If a reviewer asks whether later NCBI releases were used as external validation, the
honest answer is that they were, and the test was confounded by candidate subtree size — a statistic
requiring no embedding.

⚠ **The general lesson, and it constrains what may be attempted next.** Outside-the-tree information
is **necessary but not sufficient** (Finding 4 §4.7): a label can be unseen and still be predictable
from a cheap statistic of the training data. **Before any remaining protocol is run, name the
cheapest statistic that could pass it, compute it on the same data, and report it beside the
embedding.**
- **§4.5 TimeTree already satisfies this by construction** — it specifies Spearman(embedded distance,
  divergence time) *beside* Spearman(NCBI path length, divergence time). It is also the only
  remaining instrument that can touch steelman (ii), *embedded distance tracks NCBI convention rather
  than relatedness*, which no NCBI-internal test can refute — **and P3 was the last NCBI-internal
  candidate.** The spec's standing instruction is *"Schedule it, or delete the relatedness claim."*
- **§P4 is already failing this test** — C5 records the bridge at 0.6029 against a majority-class
  baseline of 0.6448, and P4.2's three asymmetries all favour the embedding arm, so fixing them moves
  it down. That comparison is not a formality; it is the same test P3 failed.

⚠ **Recommendation, recorded so it is a decision and not a drift: do not spend GPU on the P3 strong
variant** (train-on-T, score-on-T+1 across release pairs). It runs the same statistic on more pairs;
the size confound applies identically and the size-conditioned residual is already zero on this pair.
More pairs buys a distribution of nulls for queue wall-clock. Snapshots are on disk if this is
revisited.

---

## Still to be added

- Task 9 verdict — Figure 4 re-plot or caption rewrite (the single most likely change to what the
  manuscript says). **2026-09-22, provisional (n=1 historical pair, REPORTED not read):** on the
  radius-free S_angle (`src/taxembed/eval/angular.py`, `results/fig4_runs_20260922_104857.json`),
  prior peaks at 0.938 (ep40) and collapses in steps at each curriculum transition to 0.62.
  Canonical climbs to 0.973. So the collapse is **angular, not only radial**, which points to
  reading 1. That would mean re-plotting Fig 4 on S_angle, with the radius floor in the caption.
  The verdict waits for seeded array 5802007 under `preregistration_v2_20260922`.
- C4's valid null now exists: S_angle's closed-form null is the initialization state. On metazoa
  it scores −0.0015 (cluster se 0.0033), against the degenerate radial-only null.
- Task 8 verdict — whether the sampler fix materially moves the numbers. If yes, Methods must say the
  shipped artifact was trained with a defective objective and a retrain is scoped; if no, it is a
  robustness result worth reporting.
- Two cites owed to `REFERENCES.md` from the July round: `vankempen2024foldseek`,
  `heinzinger2024prostt5`.
- Data-availability: Zenodo DOI (must include the parent/depth **edgelist** — the HF release lacks it
  and the bridge cannot run without it).
- **C9 candidate (2026-09-29, from the UU cross-check; USER: "mark for it") — the `--clean` filter
  misses the dominant PROKARYOTE placeholder form.** `taxopy_clade.py:26-33` catches `sp.`, `cf./aff./nr.`,
  `environmental`, `uncultured`, `unidentified`, `hybrid`. It does **not** catch
  `"<clade> bacterium|archaeon <id>"` (e.g. *Verrucomicrobiia bacterium DG1235*, *Thermoplasmatales
  archaeon BRNA1*), whose NCBI placement is just the clade the submitter typed. Measured on the UU
  all-cellular panel (5,962 proteomes, same new_taxdump release 2026-06-09): **3,144 of 4,820
  prokaryotes** have this form vs 801 `sp.`-form; NCBI-vs-GTDB order agreement is 81.6 % for properly
  named taxa, 76.0 % for `sp.`, **69.3 % for `bacterium/archaeon`** (after stripping "Candidatus ").
  So the unfiltered form is the least reliable one. Evidence:
  `SpeciesEmbedding/projects/unknown_unknowns/helpers/{taxembed_noise_crosscheck,ncbi_vs_gtdb_order_agreement}.py`.
  ✅ **MEASURED 2026-09-29 on TaxEmbed's own embedded set** — `helpers/_c9_placeholder_census.py`,
  `results/c9_placeholder_census_20260929.json`, 1,025,217 embedded taxa located in the 2026-07-01
  tree:

  | domain | embedded | placeholder, UNFILTERED | already caught by the filter |
  |---|---:|---:|---:|
  | Bacteria | 200,712 | **88,108 (43.9 %)** | 201 (0.1 %) |
  | Archaea | 6,667 | **3,209 (48.1 %)** | 25 (0.4 %) |
  | Eukaryota | 817,837 | **0 (0.0 %)** | 234 (0.0 %) |

  **44.0 % of embedded prokaryotes (91,317 of 207,379) carry the unfiltered form; that is 8.91 % of
  the whole embedding.** The current filter reaches **226** prokaryote taxa — adding the pattern
  would multiply its reach by **405×**. Eukaryota is exactly 0, confirming the form is purely
  prokaryotic and that metazoan panels are unaffected. **C9's premise is confirmed and it is
  material.**

  🛑 **DECISION 2026-09-29: DO NOT RETRAIN for this paper. Disclose, and measure by MASKING AT
  EVALUATION TIME instead.** Reasons, in order of weight:
  1. **A retrain does not touch the paper's actual problem.** Finding 4 / C10 established that
     TaxEmbed has no out-of-sample generalisation evidence at all. S_angle 0.9726 is in-sample and a
     memorising model scores the same. A cleaner training set moves that number slightly and leaves
     Burkhard's question exactly as unanswered.
  2. **The question a retrain would answer can be answered without one.** Mask the 91,317 placeholder
     taxa out of *scoring* and re-read S_angle on the shipped artifact — minutes of CPU, no GPU, no
     re-plotting. If the headline barely moves, the whole issue is one Methods sentence. If it moves,
     that is itself the finding, and it is reportable without retraining. **⛔ Owed: this run.**
  3. ⚠ **The direction of a retrain is unfavourable to credibility.** Dropping 44 % of prokaryote
     leaves removes low-information nodes sitting in big flat fans, which would most likely *raise*
     S_angle. Reporting a better in-sample number bought by deleting the tautological cases is the
     same species of error as P3's — improving a metric by changing what is measured. A referee asks
     that question immediately.
  4. **Cost and calendar.** `scripts/train_lrz.sh` requests `--time=48:00:00` on a depleted
     fairshare, and a retrain invalidates every cellular number — all figures, C1–C10, the P2/P3
     work. ERC StG is 14 Oct 2026.

  **What Methods must say instead (no retrain):** the noise filter does not exclude the
  `"<clade> bacterium|archaeon <id>"` form; such taxa are 43.9 % of Bacteria and 48.1 % of Archaea in
  the embedded set (8.91 % overall) and their NCBI placement reflects the clade named by the
  submitter rather than an independent placement. Report the masked-subset S_angle beside the full
  one. ⚠ Carry C2's caveat separately: the shipped artifact predates the seeding fix and is **not**
  seed-reproducible — that is also a disclosure, not a retrain trigger.

  **Named triggers that WOULD justify a retrain** (recorded so a future session does not re-litigate
  it): a reviewer demanding a seed-reproducible released artifact · the masking run showing the
  headline is substantially carried by placeholder taxa **and** the paper's central claim resting on
  that number · §4.5 TimeTree returning "tracks NCBI convention, not relatedness" **and** the
  placeholder nodes being the suspected cause — in which case the retrain is a hypothesis test, not
  a cleanup.
- **P2 verdict READ 2026-09-29: `UNINFORMATIVE`** (`results/p2_verdict_20260929.json`, all four
  amendment flags, 9 score JSONs md5-verified against LRZ job `5815567`). All 12 runs fail gate (b),
  the floor, and all six `*vis00*` runs also fail gate (a), learning (MRR falls over training). No
  direction may be read. ⚠ Descriptively, per-run MRR (0.18–0.23) sits **below** the arm's own
  `chance_mrr_mean` (~0.27 real, ~0.32 degmatch). That needs a diagnosis (metric? candidate pool?
  held-out node embeddings?) before any P2 sentence is written. Not rescued post hoc.
