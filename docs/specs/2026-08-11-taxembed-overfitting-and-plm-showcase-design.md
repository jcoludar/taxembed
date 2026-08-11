# TaxEmbed: overfitting evidence + the pLM showcase — design

**Date:** 2026-08-11
**Status:** DRAFT — for adversarial review before any implementation
**Origin:** Burkhard raised overfitting as an issue (without having read the paper). The goal is
to answer him **factually**, not rhetorically.

---

## 0. The three suggestions being logged

Verbatim, as relayed:

> - **TaxEmbed: pLM** — train to predict ID of taxa from protein, predict the ID of our embedding
>   from protein — if embedding is better, then we're better on those 24k. (RedundReduce 90%? Mark
>   mmseqs2 clusters — pair to pair to pair, from every family maximally 3, in the sampling do again
>   and again — 100 times.) **Start without sampling.**
> - **TaxEmbed: overfitting?**
> - **ToxFam** — if that helps instead of taxonomy ID?

Ordering constraint from the user: **ToxFam last** — "we have all the stuff".

Scoping constraint from the user: the overfitting question is **about training the taxonomy
embedding itself**. The pLM work is downstream application/showcase, not the overfitting evidence.
This spec therefore splits into Track A (overfitting, primary) and Track B/C (showcase).

---

## 1. What was actually measured (2026-08-11)

All numbers below are either read from committed JSON or computed by a read-only census
(`scratchpad/panel_redundancy_census.py`). Marked accordingly. **Nothing here is estimated.**

### 1.1 The training objective is purely vertical

`build_transitive_closure.py:118-161` emits **only (ancestor, descendant) pairs**;
`src/taxembed/utils/training_pairs.py` stores exactly that schema. The canonical cellular run
(`scripts/train_lrz_cellular_canonical.sh`) trains on 21,399,053 such pairs over 1,102,163 nodes
with `--n-negatives 300`.

Consequence: **sibling/cousin (lateral) pairs are never positive training examples.** They receive
signal only as sampled negatives — i.e. binary repulsion, *not* their true cophenetic distance.
See §5.1 for why this is a "weakly supervised", not "unsupervised", claim.

### 1.2 There is no train/test split anywhere in the embedding pipeline

`train_hierarchical.py` contains no validation path — no holdout, no link-prediction, no
MAP/mean-rank evaluation. `--early-stopping 999` disables stopping; the canonical run takes `final`
at epoch 200. Every reported embedding metric is computed on nodes and pairs the model trained on.

### 1.3 Capacity ratio

1,102,163 nodes × 100 dims = **110,216,300 free coordinates** against **21,399,053** supervised
pairs — ≈5.15 parameters per constraint. This is the number a reviewer reaches for first.

Counter-evidence already in hand: `helpers/app_hyperbolic_necessity_20260725.py` (echinodermata,
3,965 taxa) shows hyperbolic retrieval precision@10 saturating at its ceiling by **d≈8**. A model
whose performance came from memorization would need capacity growing with N, not saturating at 8
dimensions. Currently a single point; §3.3 turns it into a curve.

### 1.4 The 24k panel (computed census)

`src/taxembed/bridge/data/panels/sp_metazoa_panel.tsv`:

| Property | Value |
|---|---|
| proteins | 24,779 |
| unique taxa | 2,166 |
| tax_class / tax_order / tax_phylum | 43 / 204 / 14 |
| Pfam families | 169 |
| median proteins per taxon | 1 (max 3,871) |
| taxa with exactly 1 protein | 1,094 |

**Redundancy at 90% identity** (from `sp_metazoa_cluster_ids.tsv`):

| Property | Value |
|---|---|
| distinct 90%-id clusters | 14,384 |
| proteins in singleton clusters | 10,423 (42.1%) |
| largest cluster | 179 sequences |
| one-per-cluster reduction | 24,779 → 14,384 (**42.0% discarded**) |

Redundancy reduction at 90% is therefore **load-bearing on this panel**, not a no-op. (A prior note
recorded "99% singletons" — that applies to the *all-life* panel, a different object.)

**Family sizes:** median 86, largest 1,651 (PF00001). A cap of 3 retains 507 proteins per draw
(2.0%); cap 10 → 6.8%; cap 50 → 34.1%.

**Class imbalance:** Mammalia is **64.2%** of the panel (15,896 proteins).

### 1.5 The baseline ladder — READ on the 24k panel

From `data/results/read_sp_metazoa_pfam.json` (class rank, LOSO, n=24,653 scored, 2,135 species,
43 classes), plus the majority baseline computed on the same scored denominator:

| Route | Class accuracy | Source |
|---|---|---|
| raw ProtT5 kNN (LOSO) | **0.837** | JSON `leave_species_and_clade.rawknn_loso_class_acc` |
| always answer "Mammalia" | **0.645** | computed (15,896/24,653) |
| bridge → TaxEmbed coordinate | **0.603** | JSON `class_rank_acc` |
| frequency-matched chance | 0.433 | JSON `matched_candidate_set` |
| depth-only null | 0.0009 | JSON `depth_null.class_rank_acc` |

**The bridge sits below the majority-class baseline.** `read_eval.py` reports frequency-matched
chance but never computes the majority baseline, which is the harder and more standard null. This is
a reporting gap in the harness, not a modelling failure — and it is the first thing a hostile reader
will find.

Two structural notes on the same JSON:

- `leave_clade_out.weighted_class_acc = 0.0` for both the bridge and raw kNN. This is
  **structurally degenerate**: leave-one-clade-out at class rank asks for a label that by
  construction is absent from the training set, so it can only ever score zero. It currently reads
  as catastrophic failure and means nothing. §4.4 re-scores it with a graded metric.
- `leave_species_and_clade.leak_drop = 0.837` — raw-kNN's LOSO advantage is entirely same-clade
  nearest-neighbour leakage. The existing harness already surfaces this honestly.

### 1.6 What is NOT in question

The depth-only null at 0.0009 establishes that the bridge retrieval is not riding the radial
supervision. Separately, the embedding is transductive by construction — every named taxon in the
2026-06-09 taxdump has a coordinate — so there is no train/test boundary for information to leak
across in the classical sense.

---

## 2. What "overfitting" does and does not mean here

Three distinct claims travel under one word. Separating them is most of the work.

**(a) Memorizing the tree — largely a category error.** The taxonomy is a deterministic, complete,
noise-free structure, and the deliverable is a coordinate for every named taxon. Reconstructing it
exactly *is* the product. There is no held-out "true taxonomy" the model should generalize to. A
lookup table with 5 params per constraint is not automatically a defect.

**(b) Failing to generalize beyond the supervised signal — the real, testable question.** The
objective only ever saw vertical pairs. Whether the geometry encodes anything it was not told is an
empirical question with a cheap answer (§3.1), and it is the honest reading of Burkhard's concern.

**(c) Overfitting the evaluation — selection leakage / Goodhart.** The canonical recipe was chosen
by watching depth-norm r and per-rank separation; Fig 4 *is* that curve. The same metric selects and
reports. `--radial-nudge` directly supervises radius←depth, so reporting depth-norm r = 0.952 partly
measures whether the supervision took. There is no `--seed` flag, so the Fig-4 canonical-vs-prior
contrast rests on n=1 per arm. These are addressed in §3.4-3.5.

---

## 3. Track A — overfitting in TaxEmbed training (PRIMARY)

### 3.1 Vertical/lateral fidelity split — THE headline test

**Claim under test:** if the geometry merely memorized its training pairs, fidelity on lateral
(non-ancestral) pairs should collapse relative to vertical pairs. If the two are comparable, the
embedding generalized past its supervision.

**Protocol.** Sample evaluation pairs stratified by true tree distance (reuse
`taxembed/eval/pairs.py:sample_pairs_stratified`). Partition each pair by whether it is an
ancestor-descendant pair (was a positive training example) or lateral (was not). Compute, for each
partition separately:

- multiplicative distortion (`eval/fidelity.py:multiplicative_distortion`) — median, p95
- cophenetic Spearman (`eval/fidelity.py:within_clade_rank_corr`)
- kNN retrieval precision@10 (`eval/fidelity.py:knn_retrieval_precision`)

against exact tree distance from `eval/treedist.py:TreeDistance.path_length`, with taxon-block
bootstrap CIs (`eval/bootstrap.py:taxon_bootstrap_ci`).

**Report:** the vertical-minus-lateral gap per metric, with CI. Pre-register the reading:
a lateral score within the vertical score's CI, or a gap small relative to the null-model gap,
supports generalization; lateral collapsing toward the shuffled-label null supports memorization.

**Cost:** local CPU, hours. **No retraining.** Uses the shipped `cellular_canonical` checkpoint.

**Why it matters:** this is the only test on the list that answers Burkhard using the artifact he is
actually asking about, at full 1.1M scale, without new compute.

### 3.2 Edge-holdout link prediction (the Nickel & Kiela protocol)

**Gap:** N&K's own evaluation is reconstruction/link-prediction with mean rank and MAP over held-out
edges. This pipeline dropped it entirely. Its absence is the most conspicuous methodological hole.

**Protocol.** Hold out 10% of *direct* parent-child pairs. The held node remains connected through
its grandparent-and-above pairs, so it still receives a coordinate — this is an **edge** holdout, not
a node holdout, and is compatible with the transductive design. Retrain. For each held-out (child,
parent), rank the true parent among all candidate nodes by Poincaré distance. Report mean rank and
MAP against (i) the same model with the edge retained, (ii) the radial-only null, (iii) random.

**Scale:** echinodermata (3,965 taxa) first — the rig from `app_hyperbolic_necessity_20260725.py`.
If it behaves, one clade larger.

**Threat to validity:** the transitive closure means a held-out direct edge is partially implied by
the retained ancestor pairs (if A→B→C is held at B→C, then A→C still constrains C). This makes the
test *easier* than a true link prediction and must be stated. Mitigation: report alongside a variant
that holds out the node's entire ancestor column (a true node holdout, which will have no
coordinate — scored only as "does the rest of the tree stay intact", i.e. a stability check).

### 3.3 Capacity saturation curve

Extend the existing d-sweep readout (d ∈ {2,4,8,16,32,64,100}) to at least one clade substantially
larger than echinodermata's 3,965 taxa. If the saturation dimension stays ~8 as N grows by an order
of magnitude, memorization is ruled out on capacity grounds. If it grows with N, that is a finding.

**Cost:** GPU, small. Reuses the existing sweep harness (which already carries the eps/grad-clip fix
for the euclidean arm's sqrt(0) singularity).

### 3.4 Seed variance on the recipe contrast

`--seed` is **not currently a CLI flag** — it must be added to `taxembed.cli.main` and threaded to
the torch/numpy RNGs. Then 3 seeds per arm (canonical, prior-approach) at echinodermata scale, to
put an interval on the Fig-4 contrast that currently rests on n=1 vs n=1.

### 3.5 Radial null against the headline numbers

`eval/nulls.py:radial_only_null` exists and its docstring calls it "the key Goodhart guard", but it
is not run against the headline depth-norm / separation figures. Run it, report the headline net of
the radial supervision that `--radial-nudge` provides. Local CPU.

---

## 4. Track B — the pLM showcase (Burkhard's suggestion #1)

Downstream application, not overfitting evidence. Runs on its own track.

### 4.1 What already exists

- `bridge/read_eval.py:run_loso` — ridge ProtT5→TaxEmbed coordinate → nearest node → read lineage.
  **This is "predict the ID of our embedding from protein".**
- `bridge/read_eval.py:comparator_flat` — ridge onto a **one-hot genus target** instead of
  hyperbolic positions. **This is "train to predict ID of taxa from protein".**
- `bridge/read_eval.py:comparator_logistic` — per-rank logistic classifier on PCA features.
- `bridge/clusters.py:mmseqs_cluster(min_seq_id=0.9)` — **RedundReduce 90% / mark mmseqs clusters.**
- `bridge/splits.py` — `grouped_holdout` (no cluster spans train/test), `leave_one_group_out`,
  `leave_species_out`, `per_fold_sign_test`.

Roughly 80% of the requested experiment is built. What is missing is the **scoring**.

### 4.2 Scoring layer (prerequisite)

New `src/taxembed/eval/baselines.py`:

- `majority_class_baseline(labels)` — accuracy of always predicting the modal class.
- `macro_balanced_accuracy(true, pred)` — mean per-class recall.
- `graded_tree_score(pred_idx, true_idx, treedist)` — cophenetic path-length distribution
  (median, mean, fraction within k edges) via `TreeDistance.path_length`.

Wire all three into `run_sp_read`'s JSON beside the existing `class_rank_acc`. Separate module
because `read_eval.py` is already ~900 lines, these are reused by the PLA2 path, the SP path and
ToxFam, and they are pure functions testable without the 1.4 GB H5.

### 4.3 The matched head-to-head

Arm A (predict taxon ID): classifier ProtT5→PCA-64→class label.
Arm B (predict the embedding): ridge ProtT5→PCA-64→TaxEmbed tangent coordinate→nearest node→label.

Both arms scored on an identical readout — pooled accuracy, macro accuracy, graded tree distance —
with taxon-bootstrap CIs, at matched rank granularity. (`comparator_flat` currently uses a genus
one-hot while the primary readout is class rank; that mismatch must be fixed for the comparison to
be fair.)

**Split:** cluster-grouped holdout on `cluster_id` at 90% identity. LOSO retained as secondary for
continuity with existing JSONs.

**Pre-registered primary endpoint** (committed before the run): macro-averaged class accuracy,
Arm B vs Arm A, cluster-grouped, CIs separated. Secondary: median cophenetic distance — the metric
on which a coordinate *can* beat a one-hot and an exact-match accuracy cannot show it.

### 4.4 Redundancy + resampling protocol

Per Burkhard, and in his order:

1. **Start without sampling** — one representative per 90% cluster (24,779 → 14,384), plain run.
2. **Then the resampling variant:** cap **3 per family**, draw **100 times**, report the across-draw
   distribution of every metric in §4.2. 507 proteins per draw is ample for a class-accuracy
   estimate; the point is the distribution across draws, not the size of one.
3. Sensitivity: cap curve C ∈ {3, 10, 50, ∞} as a secondary table.

**Technical note to carry into the analysis:** capping per *family* controls family composition but
does **not** fix the taxonomic imbalance — drawing 3 at random from a family that is 64% mammalian
still yields ~64% mammals. The two controls are orthogonal: family-cap-3 × 100 handles family
redundancy; **macro-averaged accuracy** (§4.2) handles the Mammalia prior; the majority-class
baseline is reported alongside so the 0.603-vs-0.645 gap is visible rather than implied.

Also re-score the degenerate leave-one-clade-out (§1.5) with `graded_tree_score` so it reports "how
far off was the placement" instead of a structurally guaranteed 0.000.

New module: `src/taxembed/eval/resample.py`.

---

## 5. Known threats to validity (state these; do not hide them)

### 5.1 "Lateral pairs are unsupervised" is too strong

With `--n-negatives 300` over 21.4M positives, lateral pairs are sampled as negatives at enormous
volume. They therefore receive **binary repulsion** signal. The precise claim is that their **true
graded cophenetic distance was never supervised** — the model was told "not an ancestor pair", never
"seven edges apart". §3.1's framing must say exactly this. See
`docs/specs/2026-06-03-e1c-hard-negative-sampling-spec.md` for how negatives are drawn; if the
sampler is distance-aware, the claim weakens further and the spec must be revised.

### 5.2 The transitive closure leaks into the edge holdout

See §3.2. A held-out direct edge remains partially implied by retained ancestor pairs. The test is
therefore easier than true link prediction and must be reported as such.

### 5.3 Selection leakage is not fully closable retroactively

The recipe was already chosen while watching the metrics now being reported. Adding seeds and nulls
tightens the estimate but cannot un-see the selection. The honest move is to state it, not to claim
a clean separation that does not exist. A genuinely clean version requires a clade held out of all
development, reported once.

### 5.4 The ablation table remains confounded

Previously parked and unchanged: parametrization and loss are confounded; the curriculum is only
removed from an already-broken baseline; effective batch differs in four hyperparameters. Full
isolation needs a clean leave-one-out grid at 498k plus a true no-curriculum run.

---

## 6. Track C — ToxFam (last)

Config paths already exist: `TOXFAM_FASTA`, `TOXFAM_LABELS`, `TOXFAM_RESIDUE_H5`
(`bridge/config.py:44-46`). Same harness as Track B with the label axis swapped from Pfam to toxin
family, answering "does ToxFam help instead of taxonomy ID?". Blocked on nothing but Track B landing.

---

## 7. Sequencing

| # | Item | Compute | Blocks on |
|---|---|---|---|
| 1 | §3.1 vertical/lateral split | local CPU, hours | — |
| 2 | §3.5 radial null vs headline | local CPU, hours | — |
| 3 | §4.2 scoring layer | local CPU | — |
| 4 | §4.3 matched head-to-head | local CPU (needs 1.4 GB H5, already pulled) | 3 |
| 5 | §4.4 resampling (100 draws) | local CPU | 3, 4 |
| 6 | §3.4 `--seed` flag + 3×2 seed runs | small GPU | flag work |
| 7 | §3.2 edge-holdout link prediction | small GPU | — |
| 8 | §3.3 capacity curve, larger clade | small GPU | — |
| 9 | Track C ToxFam | local CPU | 5 |

Items 1-2 answer Burkhard's actual question and need no new data or compute.

## 8. Explicitly out of scope

- **No retrain of the 1.1M cellular model.** Nothing here requires it.
- **No manuscript edits** until numbers exist (user's decision, 2026-08-11).
- Not deleting the degenerate leave-clade-out test — re-scoring it instead.

## 9. Testing

- §4.2 metrics are pure functions → unit tests with hand-computed fixtures, following the existing
  `tests/eval/` pattern.
- §4.4 resampler → tests that caps hold, draws differ across seeds, one-per-cluster is respected.
- Harness integration stays locally untestable (needs the large H5), matching the posture already
  documented in `read_eval.py:_main_sp_read`.
- Per Rule 16: any GPU item gets `bash -n` + `py_compile` + a smoke run on a tiny input before
  submission.
