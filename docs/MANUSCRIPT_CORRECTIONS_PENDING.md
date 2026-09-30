# Manuscript corrections pending — TaxEmbed objective-integrity review

**Purpose.** One accumulator for every manuscript change the objective-integrity work implies, so the
draft is edited **once**, in a single pass, rather than piecemeal across sessions.

## STATUS 2026-09-30 — C1–C12 APPLIED in `manuscript/manuscript.v5_draft.md`; C13 and C14 added below

The live draft is `manuscript/manuscript.v5_draft.md` in the SpeciesEmbedding repo (v3 and v4 kept,
untouched). Every entry below is applied there in its "final form only" wording, or resolved as noted:

| entry | state in v5 |
|---|---|
| C1 sampler | Methods "Objective and parametrization": depth-matched draw, missing guard, 47.4 %, 14.1 % empty pools, guarded retrain did not converge |
| C2 seeding | Methods "The recipe": CPU-exact, GPU near-reproducible, released artifact predates seeding |
| C3 init floor | Results §2 + Table 1: depth-norm printed beside its initialization value (+0.957 / +0.957) |
| C4 radial-only null | not quoted anywhere; the null is the initialization state (−0.004 cellular) |
| C5 bridge below majority | bridge section deleted; logged as the CLEAN follow-up quest |
| C6 v-PuNNs prior art | cited in the Introduction |
| C7 kNN purity confounded | per-rank purity/separation demoted to Supplementary Table S1 as descriptive, with the caveat |
| C8 / P2 vacuous | held-out link prediction not reported; no generalisation claim in the paper |
| C9 placeholders | Methods "Data": 91,942 rows (8.3 %), one denominator; masking moves S_angle by +0.0002 |
| C10 P3 withdrawn | nothing about P3 in the paper |
| C11 Metazoa vs cellular | 0.973 is labelled Metazoa (Table 1, Results §1); the released model's 0.965 is the headline |
| C12 TimeTree | Results §4 with all three qualifications; no relatedness claim |

**C13 (2026-09-30) — 76,766 rows (7.0 %) are SUPERSEDED taxids carried as leaves beside their
replacement.** The v4 sentence "prunes stale or merged taxids" was false; v5 Methods "Data" discloses
the composition (1,025,397 current taxids; aliases share their replacement's parent; alias→replacement
cosine 0.998 = alias→random sibling). Evidence: `results/closure_taxid_drift_20260930.json`
(`helpers/_closure_taxid_drift_20260930.py`). Also on the HF model card.

**C14 (2026-09-30) — the Euclidean-vs-hyperbolic dimension sweep EXISTS** (the 2026-09-30 assessment
said it did not; it had searched only this submodule):
`SpeciesEmbedding/projects/tax_disentangle/results/app_hyperbolic_necessity_20260725.json`
(Echinodermata, bare objective, d = 2–100). In v5 as Supplementary Table S3 + one Discussion sentence.
The open experiment is the same curve under the full recipe at ≥10^5 taxa.

**Owed numbers now recorded** on the training closure, one denominator:
`results/owed_numbers_20260930.json` (per-domain rows, edges, depths, query cost).

*(The section below is the historical accumulator; "Target file" and "Rebuild" lines refer to v3 and
are superseded by `manuscript/manuscript.v5_NOTES.md`.)*

**Target file (historical):** `manuscript/manuscript.v3_draft.md` (SpeciesEmbedding repo, `main`).
**Rebuild after applying (historical):** `SRC=<abs>/manuscript.v3_draft.md bash scripts/build.sh docx`
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

## C11 — the headline S_angle is a METAZOA number; the released artifact is the CELLULAR model

**Status:** UNCONDITIONAL that it must be checked against the manuscript before submission.
**Evidence:** `helpers/_sangle_provenance_audit.py` · `results/sangle_provenance_audit_20260929.json`
· `results/c9_masked_sangle_20260929.json` · `results/fig4_runs_20260922_104857.json`

Found while chasing what looked like a 0.0073 discrepancy. It was not a discrepancy.

**The headline's source.** `results/fig4_runs_20260922_104857.json`, which carries the 0.9726-class
numbers, records:

```
closure  /data/taxonomy_edges_metazoa_33208_clean_transitive.npz
n_nodes  498,246        max_depth 37
runs     prior, canonical, prior_roll, canonical_roll
```

**Metazoa**, 498k nodes, scored on Task-9 **recipe-contrast checkpoints**. The shipped artifact is
`release/taxembed-cellular-v1/cellular_embedding.safetensors` — the **cellular** model, 1,102,163
nodes. A different model on a different tree.

**The audit over every `results/*.json`:**

| closure | files |
|---|---:|
| METAZOA (498,246) | 5 |
| mollusca (32,017) | 4 |
| **CELLULAR, carrying an S_angle** | **0, before 2026-09-29** |

`negative_sampler_audit.json` and `radial_init_floor.json` are on the cellular tree but contain no
S_angle. ⇒ **No S_angle had ever been computed for the released model.**

**It has now.** `results/c9_masked_sangle_20260929.json`, cellular closure, shipped artifact,
production settings (n=10,000 queries, k=10, seed=0): **S_angle = 0.9653, cluster se 0.0069**, with
the init null at **−0.0035** on the same tree and queries. Index alignment is verified, not assumed:
`md5(release/taxembed-cellular-v1/taxid_to_index.tsv)` **equals**
`md5(…cellular_organisms_131567_clean.mapping.tsv)` = `a8ef06b048e03613230a0a14908f515e`, so release
row *i* is closure node *i*.

⛔ **What must happen before submission — a manuscript check I cannot do from the repo.** Read what
the paper actually claims 0.9726 is:
- **If the paper scopes it to Metazoa**, it is correct as written; the fix is only that C3/C8's
  phrase *"the headline S_angle"* invites conflation, and the released model's own number (0.9653)
  should be reported alongside it, since that is the artifact readers download.
- **If the paper presents 0.9726 as the released model's score**, it is a **misattribution** and must
  be corrected to 0.9653 (cellular), with the Metazoa number labelled as such.

⚠ **Related, and the reason this went unnoticed:** most `results/*.json` record `closure`, but the
ones that do not (`transplant_2x2_*`, `task9_scorer_dose_response_*`) cannot be provenance-checked at
all, and my own masking artifact initially lacked the field too — fixed by recording `closure`,
`embedding`, `clade`, the mapping md5 and `is_shipped_release_artifact`. **A score without its
closure and its checkpoint is not a comparable number**
([[feedback_two_numbers_are_comparable_only_if_their_definitions_are]]). Every S_angle written from
here on should carry those keys.

---

## C12 — §4.5 TimeTree RUN: the relatedness claim does not survive. Delete it or restrict it.

**Status:** UNCONDITIONAL. The spec's standing instruction was *"Schedule it, or delete the
relatedness claim from the paper."* It is now scheduled, run, and read.
**Evidence:** `results/timetree_preregistration.json` (frozen SHA256 `a3e733c7…`, amendments 1–3) ·
`results/timetree_pairs_20260929.json` · `results/timetree_result_20260929.json` ·
`helpers/_timetree_{fetch_probe,feasibility,freeze_preregistration,fetch_pairs,score,amendment_1,amendment_2,amendment_3}.py`

6,000 pre-registered pairs (3,000 per stratum, uniform within-stratum, seed 0, drawn before any
distance); 5,984 carry a TimeTree age. **Primary = Spearman(ANGULAR distance, divergence time)**,
with **Spearman(NCBI path length, divergence time)** as the co-reported primary comparator.

| stratum | **primary, angular** | **comparator, NCBI path** | verdict |
|---|---|---|---|
| Vertebrata (n=2,993) | **+0.8946** [0.8798, 0.9085] | +0.7911 [0.7695, 0.8123] | TRACKS_RELATEDNESS |
| Insecta (n=2,991) | **+0.2963** [0.2473, 0.3457] | **+0.4434** [0.3992, 0.4853] | **TRACKS_CONVENTION** |

**Gates b and d pass in both strata** — angular init null +0.0307 / +0.0126 (≤0.05 required);
shuffle control −0.0092 / +0.0041.

🎯 **The design choice that mattered, vindicated by measurement.** Making the primary *angular*
rather than Poincaré was not fastidiousness: the **Poincaré init null is −0.2296 (Vertebrata) and
−0.3712 (Insecta)** — strongly nonzero, because the planted radius alone correlates with divergence
time. Had Poincaré been the primary, a large part of the "signal" would have been the radial prior
we put there ourselves. The angular init null is ≈0, as required.

🛑 **THE VERDICT BY THE PRE-REGISTERED RULES IS `TRACKS_CONVENTION`.** The rule is asymmetric and
deliberately so: `TRACKS_RELATEDNESS` requires the primary to beat the comparator in **both** strata;
`TRACKS_CONVENTION` fires on **either**. Insecta fires it. Per the spec, this is *"the honest finding
and it is publishable."*

**Three things make the Vertebrata result weaker than its headline, all pre-declared or recorded:**
1. **Insecta collapses on better-supported pairs.** At `all_total ≥ 10` (n=2,639) the angular
   correlation falls to **+0.0425** while the comparator holds at +0.2161. On the insect pairs with
   the most study support, the embedding carries **almost no** divergence-time signal.
2. **Most of the Vertebrata advantage is between-bin, not within** (amendment 2's tercile secondary).
   Within LCA-depth terciles the angular advantage shrinks from **+0.1035 pooled to +0.0221**
   (depth 17–33), and at depth 10–15 **both** correlations are *negative* (angular −0.3852, path
   −0.3151). The pooled advantage largely reflects the embedding encoding **LCA depth** — a property
   of the tree, not of relatedness. Simpson's-paradox-shaped, and it points the unwelcome way.
3. **The effective sample size is ~115 MRCA nodes, not ~3,000 pairs** (amendment 3). TimeTree ages
   are node-level: 2,993 Vertebrata pairs carry **118 distinct ages**, and one age-class is **28.7 %**
   of them. ⛔ **Never quote the pair count as the sample size.** The taxon-level bootstrap partly
   absorbs this; the stricter clustering unit is the MRCA node, so **treat the reported CIs as lower
   bounds on the true width.**

**What the manuscript should say (final form only):** external divergence-time validation was run
against TimeTree. In vertebrates the embedding's angular geometry correlates with divergence time
somewhat better than NCBI path length does; **in insects NCBI path length wins**, and on the
best-supported insect pairs the embedding's correlation is indistinguishable from zero. Stratifying
by LCA depth removes most of the vertebrate advantage. ⇒ **The general claim that embedded distance
tracks evolutionary relatedness rather than NCBI convention is NOT supported and must be deleted**,
or restricted to vertebrates with items 1–3 stated. Steelman (ii) stands.

⚠ **This is now the answer to Burkhard's *"TaxEmbed: overfitting?"*, and it is a negative one.**
P2 was vacuous (Finding 3), P3 measured subtree size (Finding 4), and §4.5 — the last instrument,
and the only one drawing on data outside NCBI — returns TRACKS_CONVENTION. **The honest position is
that TaxEmbed's geometry is an efficient encoding of the NCBI topology it was trained on, with no
demonstrated generalisation beyond it.** That is publishable as stated; it is not a relatedness
claim.

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
  3. ~~⚠ The direction of a retrain is unfavourable: dropping 44 % of prokaryote leaves removes
     low-information nodes in big flat fans and would most likely *raise* S_angle.~~
     🛑 **WITHDRAWN — this argument was WRONG, and the masking run below refutes it.** Masking moves
     S_angle by **+0.0002**. The speculation was never needed; reason 2 said to measure it, and
     measuring it replaced the guess with a number that supports the same decision far better.
     Recorded rather than deleted, because a withdrawn argument of mine is exactly the kind of thing
     that otherwise gets re-invented.
  4. **Cost and calendar.** `scripts/train_lrz.sh` requests `--time=48:00:00` on a depleted
     fairshare, and a retrain invalidates every cellular number — all figures, C1–C10, the P2/P3
     work. ERC StG is 14 Oct 2026.

  ### C9 masking run — DONE 2026-09-29. The headline is NOT carried by placeholder taxa.

  `helpers/_c9_masked_sangle.py` → `results/c9_masked_sangle_20260929.json`. Production settings
  (n=10,000 queries, k=10, seed=0) on the shipped artifact and the closure it trained on. 91,305 of
  91,317 placeholders are leaves (99.99 %), so pruning cannot orphan a subtree. **29 seconds, one
  core, no GPU.**

  | | S_angle | cluster se | n |
  |---|---:|---:|---:|
  | all queries | +0.9653 | 0.0069 | 10,000 |
  | real queries, **full** pool | +0.9657 | 0.0068 | 9,156 |
  | real queries, **masked** pool | **+0.9659** | 0.0063 | 9,156 |
  | placeholder queries | +0.9497 | 0.0076 | 844 |

  **ΔS (masked − full, same 9,156 queries) = +0.0002.** Per band: shallow +0.9392 → +0.9420
  (+0.0028), mid and deep **identical** to four decimals. Even in the shallow band — where pruning
  takes the depth-4 pool from 49,560 members to ~573 — the number does not move.

  **GATE G1 PASSED:** the init null (random directions at planted radii) reads −0.0035 on the full
  tree and −0.0022 on the pruned tree. Both ≈ 0, so the pruning did not break the tree, the depths
  or the per-query bounds — which is what makes the masked condition trustworthy.

  ⚠ **Two corrections to my own reading, both recorded rather than quietly fixed:**
  - The headline "placeholder − real = −0.0160" is a **band-composition artifact**. *Every*
    placeholder query is shallow (they live at depths 4–9), and shallow scores lower for everyone.
    Band-matched, placeholder queries score **+0.0105 HIGHER** than real shallow queries
    (0.9497 vs 0.9392) — the opposite sign, though within ~1.4 cluster SEs and not significant.
    Comparing a shallow-only subset against an all-band average is exactly
    [[feedback_two_numbers_are_comparable_only_if_their_definitions_are]].
  - Reason 3 above predicted masking would *raise* S_angle. It does not. Withdrawn above.

  ~~⚠ Open discrepancy: this run's all-query S_angle is 0.9653 against the manuscript headline
  0.9726, a 0.0073 gap, not yet explained — query seed, k, or the `radii` argument.~~
  🛑 **RESOLVED SAME DAY, and it was not a discrepancy — it was a CATEGORY ERROR of mine. See C11.**
  The headline's source is on the **METAZOA** closure (498,246 nodes); this run is on **CELLULAR**
  (1,102,163). Different trees, different models. There was never a gap to reconcile, and I
  proposed three mechanisms for it before checking whether the two numbers measured the same thing
  — [[feedback_two_numbers_are_comparable_only_if_their_definitions_are]], which is indexed and
  which I had quoted twice earlier in this same session.

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
