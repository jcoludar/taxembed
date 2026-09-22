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
