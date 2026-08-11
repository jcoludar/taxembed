# TaxEmbed: overfitting evidence + the pLM showcase — design v2

**Date:** 2026-08-11
**Supersedes:** `2026-08-11-taxembed-overfitting-and-plm-showcase-design.md` (v1, kept intact —
all three review passes cite it by section)
**Status:** DRAFT — revised against 3 independent reviews (hostile methodological, independent
verification, literature grounding)
**Origin:** Burkhard raised overfitting as an issue. Answer him **factually**.

---

## 0. The three suggestions being logged

Verbatim, as relayed:

> - **TaxEmbed: pLM** — train to predict ID of taxa from protein, predict the ID of our embedding
>   from protein — if embedding is better, then we're better on those 24k. (RedundReduce 90%? Mark
>   mmseqs2 clusters — pair to pair to pair, from every family maximally 3, in the sampling do again
>   and again — 100 times.) **Start without sampling.**
> - **TaxEmbed: overfitting?**
> - **ToxFam** — if that helps instead of taxonomy ID?

Constraints from the user: **ToxFam last**; the overfitting question is about **training the
taxonomy embedding**, with the pLM work as downstream showcase.

**v2's central structural change:** those two are not separable. The literature's strongest
anti-memorization design is a **clade/node holdout**, which is only scoreable if some side channel
can place the held-out node. TaxEmbed's side channel is exactly the ProtT5 bridge. Burkhard's pLM
suggestion and his overfitting question are **the same experiment**.

---

## 1. What v1 got wrong (recorded so the error is not repeated)

v1's headline test — split fidelity into vertical (supervised) vs lateral (unsupervised) pairs —
**cannot fail, and is unimplementable as specified.** Three independent objections:

1. **LCA additivity.** For a lateral pair (x, y) with LCA L, tree distance is exactly
   d(x,L) + d(y,L) — and the transitive closure makes **both** (L,x) and (L,y) positive training
   pairs. Hyperbolic space is chosen *because* tree metrics embed there with near-additive distance
   through a common ancestor. "Vertical good ⟹ lateral good" is close to a theorem about the
   geometry, not a finding about the model.
2. **Disjoint support.** Vertical pairs satisfy `depth_gap == d_tree` exactly; lateral pairs satisfy
   `depth_gap ≤ d_tree − 2`. At any fixed tree distance the partitions occupy disjoint depth-gap
   support, so stratifying on distance cannot remove the confound. Since radius is planted from
   depth (§2.3), the radial-only null *alone* separates them: median embedded distance 3.4709
   (vertical) vs 1.3430 (lateral) at d_tree = 2.
3. **Metrics can't be partitioned.** `knn_retrieval_precision` and `within_clade_rank_corr` take
   (Q,C) query×candidate matrices, not pair lists. `multiplicative_distortion` renormalizes by its
   own sample median per call, deleting exactly the offset the test looks for.

v1's other retracted claims: the "lateral pairs are unsupervised" framing (§2.1 below), the d≈8
capacity argument (§6.4), and two wrong citations (§9.3).

---

## 2. Verified facts

Everything here was independently re-derived. Sources marked. Nothing estimated.

### 2.1 The training objective has a false-negative defect

**This is the most consequential finding in the whole review round, and it is about the model, not
the evaluation.**

Negatives are drawn from the pool of nodes at the **descendant's** depth
(`train_hierarchical.py:398`), via a bare `np.random.randint` over that pool with **no ancestry
check and no self-exclusion** on the fast path (`:400-403`). The loss then scores each negative as
distance from the **ancestor** (`softmax_loss`, `:730-732`). Any node at the descendant's depth that
is also a descendant of the same ancestor is therefore a **false negative** — a valid positive pair
being actively pushed apart.

Confirmed by direct code read. At the extreme it is deductive: for a root-anchored pair every node
at the descendant's depth is a descendant of the root, so **100%** of its negatives are false.

Measured rate (independent verification pass; two methods — the repo's binary-lifting LCA and an
Euler-tour containment check — agreeing to six digits):

| Anchor depth | False-negative rate |
|---|---|
| 0 | 100.00% |
| 1 | 91.79% |
| 2 | 65.48% |
| 5 | 55.49% |
| ≥10 | ~35-45% |
| **overall** | **47.07%** |

5.15% of the 21.4M pairs are root-anchored; 52.19% have anchor depth ≤ 10. Negative-sample
composition: 52.93% lateral / 47.07% vertical. Because negatives sit at the descendant's depth,
those false negatives are at **exactly the positive's tree distance from the anchor** — the softmax
is asked to rank the positive strictly nearer than ~141 of 300 equidistant, equally valid candidates.

**⚠ NOT YET INDEPENDENTLY RECOMPUTED BY THE AUTHOR.** The closure `.npz` is not on this machine
(only `taxonomy_edges_cellular_organisms_131567_clean.mapping.tsv`). The mechanism is verified; the
47.07% is not yet ours. **Task A0 pins it down before any of it is quoted.**

Also verified: the sampler is depth-stratified, not uniform (99.82% of negatives at
`depth == depth(descendant)`, 0.00% at the ancestor's depth). The LCA-band sampler from the E1c v2
spec was **never implemented** — no `--neg-sampling` flag exists; only `--tiered-negatives`, and the
canonical run has `tiered_negatives: false`.

**Consequence for the Methods section:** the paper describes a clean Nickel-Kiela softmax. The code
is not that.

**Consequence for the capacity framing:** v1 quoted 5.15 parameters per constraint counting
positives only. Counting negative comparisons it is 0.057. v1 used the alarming number to steelman
the worry and the "lateral is unsupervised" assumption to rebut it. Both cannot hold. Report the
negative volume alongside: `--epoch-fraction 0.3` × 21.4M × 300 negatives ≈ **1.93e9 graded lateral
constraints per epoch**, 200 epochs.

### 2.2 No train/test split, and nothing is seeded

`train_hierarchical.py` (full 1,074-line read): no holdout, no link prediction, no MAP, no mean
rank. Early stopping compares **training** loss (`:926`); `--early-stopping 999` makes it
unreachable.

The string `seed` occurs **zero times** in `train_hierarchical.py`, `train_small.py`, and
`cli/main.py`; `run.json` has no seed field. Unseeded: direction init (`torch.randn`, `:85`), every
negative draw (legacy global numpy RNG — so `default_rng` would not fix it), the per-epoch shuffle
and `epoch_fraction` subsample (`:592-623`), plus `--amp`/CUDA nondeterminism.

**⚠ The paper design doc claims "Reproducible (seeded; repro run matches to ±0.01)"
(`2026-06-09-taxembed-paper-design.md:24`). That claim is not supported by the code.**

### 2.3 Radius is initialized at target, then regularized to it

`_initialize_by_depth` sets every node's norm to `target_radius(depth)` at step 0 with a random
direction (`train_hierarchical.py:65-96`, esp. `:82`, `:90-93`); `radial_regularizer` (`:748-778`)
then L2-penalizes deviation from the same target. depth-norm r = 0.952 is **not an outcome** — it is
planted, watered, and reported as harvest. Any headline radial number needs the **initialization
floor** reported beside it.

### 2.4 Closure integrity — clean

On the shipped cellular artifact: `depth_diff` min 1 / max 40; **zero** rows with dd≤0, **zero**
self-loops, **zero** duplicate pairs, **zero** bidirectional pairs (strictly antisymmetric); 100% of
a 500k sample satisfies ancestor-containment under an independent Euler-tour check. 21,399,053 pairs
over 1,102,163 nodes; dim 100, `n_negatives` 300, `loss: softmax`, `epoch_fraction` 0.3
(`artifacts/tags/cellular_canonical/run.json`). Max depth 40.

Latent caveat: `ensure_complete_coverage` (`build_transitive_closure.py:279-281`) *can* emit
self-pairs for root-adjacent nodes — not exercised in this dataset, so "ancestor-descendant only" is
a property of the artifact, not a guarantee of the code.

### 2.5 The 24k panel — every number verified exact

24,779 proteins / 2,166 taxa / 43 class / 204 order / 14 phylum / 169 Pfam families; median 1
protein per taxon (max 3,871); 1,094 single-protein taxa. Family median 86, largest 1,651
(PF00001); no family has fewer than 50 members, so cap-3 gives exactly 3×169 = 507 (2.05%), cap-10
1,690 (6.82%), cap-50 8,450 (34.10%). Mammalia 15,896 = 64.15%.

90%-identity clustering: **14,384 clusters**, 10,423 singletons (42.06%), largest 179, one-per-
cluster discards 41.95%.

⚠ **Provenance correction (v1 was wrong):** 14,384 is the distinct count of the **panel's own
`cluster_id` column**, not of `sp_metazoa_cluster_ids.tsv` — that file has 30,497 rows → 18,118
clusters, and reconciles only after restricting to the 24,779 panel accessions (cluster-id agreement
on shared accessions = 1.000). §7.3 must therefore state "one *panel member* per cluster, chosen by
a stated rule" — the mmseqs representative may not be in the panel.

### 2.6 The baseline ladder — verified apples-to-apples

From `bridge/data/results/read_sp_metazoa_pfam.json`, class rank, LOSO, n = 24,653 scored:

| Route | Class accuracy |
|---|---|
| raw ProtT5 kNN | **0.8368** |
| always answer "Mammalia" | **0.6448** |
| bridge → TaxEmbed coordinate | **0.6029** |
| frequency-matched chance | 0.4326 |
| depth-only null | 0.00089 |

**The bridge sits below the majority-class baseline**, and the comparison is legitimate: the
126-row gap between 24,779 and 24,653 is *exactly* the rows with null `tax_class`, **none of them
Mammalia**; independently, the JSON's own `leave_clade_out.folds` sum to 24,653 with Mammalia
`n_held` = 15,896. `class_rank_acc` is a pooled protein-level accuracy over that same denominator.
`read_eval.py` never computes a majority baseline (verified: zero hits for `majority`/`macro`).

### 2.7 The lateral-generalization evidence already exists

Only **0.15%** of a query's true top-10 tree neighbours are ancestral (0.0006% of query×candidate
cells) — so **precision@10 is already an essentially pure lateral metric**. The committed
`cophenetic_fidelity.json` reports, at full 1.1M scale:

- model precision@10 = **0.5455 [0.5354, 0.5546]**
- radial-only null = **0.0091**

A 0.536 delta on structure that was never a positive training pair. **This is the substance of the
answer to Burkhard, and it is already computed.**

⚠ Do **not** report multiplicative distortion: on the same artifact the model's median is 1.1424
against the radial-only null's 1.1374 — **the null wins**. The metric is non-discriminative at
cellular scale.

---

## 3. What the literature says (grounding pass)

⚠ **All citations below come from the literature review and must be verified at source before any
of them enters a manuscript.**

### 3.1 The deductive-closure trap is a named, published objection

Holding out an edge from a transitively closed graph does not hold anything out — the edge remains
derivable from the survivors. Explicitly stated in: **Zhapa-Camacho & Hoehndorf, NeSy 2023**
(arXiv:2303.16519); **Box2EL** (Jackermeier, Chen, Horrocks, WWW '24) — *"we find that there is
significant leakage (overlap) between their testing and training sets"*, and they refused to quote
the affected baseline; **Mashkova, Zhapa-Camacho & Hoehndorf 2024** (arXiv:2405.04868); **Liu,
Cuenca Grau, Horrocks & Kostylev, KR 2023**.

**The mandatory baseline:** Vendrov et al., ICLR 2016 define a no-learning rule that *"classifies
hypernym pairs as positive if they are in the transitive closure of the union of edges in the
training and validation sets"* — it scores **88.2%** against their learned order-embeddings' 90.6%.
Without this baseline a link-prediction number is uninterpretable.

v1 filed closure leakage as a §5.2 caveat. It is a design requirement.

### 3.2 The closest methodological sibling: HiG2Vec

**Kim, Kim & Sohn, *Bioinformatics* 37(18):2971-2980, 2021** — Poincaré embedding of the Gene
Ontology. Three transferable designs:

- **Level-stratified held-out link prediction**: 90/10 split with test links restricted to
  **levels 6-11 (first-to-third quartile of hierarchy level)**, explicitly to avoid trivial
  root/leaf edges. Test sets 3,845 BP / 473 MF / 274 CC.
- **A temporal, leakage-free task**: predict relations *newly added* between 2019-02-27 and
  2020-10-09 — 4,521 new GO-GO relations; 68.72% (GO-GO), 74.75% (GO-gene).
- The only ontology-embedding paper to state in print that fully-observed reconstruction measures
  *"how well the information is captured in the embeddings, not how well the embeddings work on a
  data leakage-free task."*

**NCBI taxdump is versioned. The temporal task is directly available to us and costs one download.**

### 3.3 Clade/node holdout is the design that defeats memorization

The held-out entity has no embedding to memorize, so the objection cannot apply. Scoreable whenever
side information places the node:

- **BioCLIP** (Stevens et al., CVPR 2024, Oral / Best Student Paper): TreeOfLife-10M, 10.4M images,
  454,103 taxa, 108 phyla; **400 rare species completely removed** from training; zero-shot top-1
  **37.8%** vs CLIP 26.6%.
- **DeepGOZero** (*Bioinformatics* 38(Suppl 1), ISMB 2022): **16 GO classes held out entirely**,
  predicted from axiomatic definitions; class-centric AUC 0.745; proteins grouped at >50% identity
  then split 81/9/10.
- **TEPI** (arXiv:2401.13219): node2vec over the **72,378-species bacterial NCBI Taxonomy subset**;
  58 seen / 35 unseen species.
- Taxonomy-expansion family: **TaxoExpan** (WWW 2020, 10-20% of leaves), **Arborist** (WWW 2020,
  15%, Pinterest production), **STEAM/QuanTaxo** (20%), **Octet** (KDD 2020, Amazon, 64/16/20).

**Metrics to adopt instead of a bespoke `graded_tree_score`:** **Wu & Palmer** =
`2·depth(LCA) / (depth_pred + depth_true)` (TaxoExpan, STEAM) and **SPDist** = undirected
shortest-path distance between predicted and true parent (Arborist justifies it as a
human-verification-cost measure).

### 3.4 The memorization control the field accepts: RandomDAG

**GRAM** (Choi, Bahadori, Song, Stewart & Sun, KDD '17) ablates by assigning each leaf **five
randomly chosen ancestors** — architecture and parameter count held *exactly* fixed, ancestry
semantics destroyed. RandomDAG consistently underperforms (heart-failure AUC 0.8226 vs 0.8447).

GRAM also supplies the right reporting shape: accuracy **bucketed into quintiles by training
frequency**. The structural prior helps most on the rarest labels (Sutter Acc@5: 0.0150 vs 0.0080,
~1.9×) and *loses* on the commonest (0.4903 vs 0.4959). **That crossover is the expected shape for
TaxEmbed on a 64% Mammalia panel, and should be pre-registered as the prediction.**

### 3.5 Do not cite Nickel & Kiela's Euclidean column

**Bansal & Benton, "Comparing Euclidean and Hyperbolic Embeddings on the WordNet Nouns Hypernymy
Graph," Insights from Negative Results in NLP, 2021** (2021.insights-1.8): *"Counter to what they
report, we find that Euclidean embeddings are able to represent this tree at least as well as
Poincaré embeddings, when allowed at least 50 dimensions."* Cause: N&K constrained Euclidean
embeddings to a unit 2-norm ball. Removing that took d=50 reconstruction MAP from **14 → 88.9**; at
d=200 Euclidean (**92.2**) beats Poincaré (89.3).

This **corroborates** the project's existing decision (2026-07-25) to soften hyperbolic to an
architectural choice rather than a pillar, and supplies the citation for it. Two actions: never cite
N&K's Euclidean numbers; and **check our own Euclidean arm for norm constraints** before reporting
any d-sweep — §6.4 already carries an eps/grad-clip fix for a *different* Euclidean failure mode.

N&K's dimension-robustness result (Poincaré flat across d) is unaffected and remains citable.

### 3.6 Prior art on NCBI taxonomy embedding exists, and is weaker than ours

**v-PuNNs** (N'guessan, arXiv:2508.01010, preprint, no venue): NCBI **Mammalia** only, 12,205 taxa,
depth 15, ~2.08M params, **no train/test split of any kind**, leaf-path reconstruction LeafAcc
95.8%. Reports Spearman ρ = −0.96 between norm and depth — essentially our 0.952, and **equally
unguarded against the radial-supervision confound**. Weak comparator, but it exists; its unvalidated
status is precisely the gap TaxEmbed can claim by doing §6 properly.

### 3.7 You cannot inherit a number, only a protocol

The "WordNet noun hypernymy closure" is not one graph: Vendrov 82,192/838,073; Athiwaratkun
82,115/837,888; N&K 2017 82,115/**743,241**; N&K 2018 82,115/**769,130**; Ganea 82,114/661,127. The
two N&K papers disagree by 25,889 edges on a dataset both call the same thing, unexplained. Metrics
are also incomparable across branches (balanced binary accuracy vs filtered MR/MAP vs threshold-
tuned F1).

---

## 4. What "overfitting" means here — revised

**(a) Memorizing the tree.** Largely a category error *for the coordinate-lookup deliverable*, but
v1 overstated this. It is a real failure for the deployment modes the paper actually sells: a
relatedness proxy, a bridge target space, and — the paper design's lead application — **release-diff
taxonomy QC**, which is explicitly a prediction task about taxa whose placement changes. §6.3
addresses it directly.

**Steelman to be tested, not assumed away:** the geometry has fit the 2026-06-09 taxdump's
curatorial idiosyncrasies (rank inflation, singleton containers, the 0.023% residual
"environmental samples" nodes the sbatch header admits) such that (i) a new or moved taxon cannot be
placed by any inductive map, (ii) embedded distance tracks NCBI convention rather than relatedness,
and (iii) the bridge inherits both — which would be why projecting ProtT5 through it makes taxonomy
prediction *worse* (0.603) than leaving ProtT5 alone (0.837).

**(b) Failing to generalize beyond the supervised signal.** Testable; §2.7 already carries strong
evidence, and §6.2/§6.3/§6.5 harden it.

**(c) Overfitting the evaluation.** Selection leakage (recipe chosen while watching the reported
metrics), the planted radial axis (§2.3), and n=1 unseeded runs. §6.6-6.8.

---

## 5. Sequencing (literature-revised priority)

Ordered by decisiveness, not by cost. v1 ordered by cost and would have produced a fast, confident,
wrong answer.

| # | Item | § | Compute |
|---|---|---|---|
| **A0** | Quantify + fix the false-negative sampler | 6.1 | local, then GPU re-train |
| **A1** | Trivial transitive-closure baseline (Vendrov) | 6.2 | local, hours |
| **A2** | RandomDAG ancestor randomization (GRAM) | 6.2 | small GPU |
| **A3** | Temporal / release-diff holdout (HiG2Vec) | 6.3 | local + one taxdump download |
| **A4** | Clade holdout scored through the pLM bridge | 6.5 | local (needs Track B) |
| **A5** | Level-stratified link prediction (levels 6-11) | 6.2 | small GPU |
| **A6** | Reframe the existing precision@10 lateral evidence | 6.7 | local, hours |
| **A7** | Radial initialization floor | 6.6 | local |
| **A8** | `--seed` flag + seeded replicates | 6.8 | small GPU |
| **A9** | Diagnose/replace the misspecified depth-null | 6.9 | local |
| **B1-B4** | Scoring layer, matched head-to-head, resampling | 7 | local CPU |
| **C** | ToxFam | 8 | local CPU |

A3 is the highest value-per-hour item on the list: it is the only test with a **real, non-derivable
train/test boundary**, it has a published precedent in a sibling Poincaré/biology paper, and NCBI's
versioning makes it nearly free.

---

## 6. Track A — overfitting in TaxEmbed training

### 6.1 A0 — quantify and fix the false-negative sampler (blocks interpretation of everything else)

1. Rebuild or retrieve the cellular closure `.npz`; recompute the false-negative rate ourselves,
   overall and by anchor depth. Do not quote 47.07% until this is ours.
2. Instrument per-pair negative exposure (the E1c v2 "Step 0" diagnostic, specced and never built).
3. Fix: reject negatives that are descendants of the anchor (ancestry check via the existing
   binary-lifting LCA), and add self-exclusion to the fast path.
4. Retrain at echinodermata and one mid-size clade, fixed vs unfixed, same seed. Report the delta on
   every downstream metric.

**Decision this forces:** if the fix materially changes the numbers, the shipped 1.1M artifact was
trained with a defective objective and the Methods must say so. If it does not, that is a robustness
result worth reporting. Either way the paper's Methods description needs correcting to match
`train_hierarchical.py`.

### 6.2 A1/A2/A5 — link prediction done to the field's standard

- **A1 trivial baseline first.** Implement Vendrov's rule: classify a held-out pair positive iff it
  is in the transitive closure of the retained edges. Report it beside every learned number. If the
  learned model does not clear it by a stated margin, say so.
- **A5 level-stratified holdout.** 90/10, test links restricted to the **first-to-third quartile of
  node level** (HiG2Vec's levels 6-11 analogue; our max depth is 40, so compute the quartiles from
  our own depth distribution — do not transplant 6-11). Report filtered mean rank and MAP.
- **A2 RandomDAG control.** Retrain with each taxon's ancestor column replaced by a random ancestor
  set of the same size — parameters, pair count and architecture held exactly fixed. If lateral
  fidelity survives that, we are measuring geometry, not taxonomy.
- Report leakage as a **covariate**, not a caveat: stratify mean rank / MAP by the number of
  retained ancestor pairs and by the true parent's sibling count.

### 6.3 A3 — temporal (release-diff) holdout ★

The one place a genuine train/test boundary exists. The embedding is pinned to the 2026-06-09
taxdump; NCBI has published later releases.

Protocol: download a current taxdump; diff against 2026-06-09 to obtain (i) newly added taxa,
(ii) taxa whose parent changed, (iii) merged/deleted taxa. Then:

- **Placement:** for moved taxa, does the 2026-06-09 embedding already place them nearer their
  *new* parent than their old one, more often than chance? (Direct evidence that the geometry
  encodes relatedness rather than the pinned convention.)
- **QC prediction:** does embedding-derived anomaly score on the 2026-06-09 snapshot predict which
  taxa NCBI subsequently moved? This is the paper's own lead application, finally scored on held-out
  future data.
- **Coverage:** report the fraction of new taxa with no coordinate — the transductive limit,
  measured rather than asserted.

Score with **Wu & Palmer** and **SPDist** (§3.3).

### 6.4 Capacity — retracted as stated, re-specified

v1's "precision@10 saturates by d≈8, therefore not memorization" does not follow: precision@10 has a
ceiling (~0.41 on that run), a lookup table for 3,965 taxa also saturates well below 100 dims, and
the **euclidean arm reaches the same 0.409 ceiling at d=100** — so saturation is not
geometry-specific. Worse, the sweep it rests on used `n_neg=50`, no curriculum and no radial nudge,
and has **negative** depth-norm correlation (−0.48 to −0.70) — a materially different recipe from
the canonical model it was defending.

Re-specification: re-run the sweep under the **canonical** recipe, seeded, on two clade sizes; check
the Euclidean arm for norm constraints (§3.5) before reporting; interpret saturation only against
the RandomDAG control (§6.2), never on its own.

### 6.5 A4 — clade holdout through the bridge (couples Tracks A and B)

Hold out a clade's coordinates entirely; place its members from ProtT5 alone via the bridge; score
the placement against the true tree with Wu & Palmer and SPDist. This is the BioCLIP / DeepGOZero /
TEPI design, and it is the one family of test that structurally cannot be answered with "you
memorized it". Requires Track B's scoring layer, hence the coupling.

### 6.6 A7 — radial initialization floor

Report depth-norm r for the **initialization** as the floor and the trained model's r as a delta
over it. Without that floor the 0.952 is not reportable.

### 6.7 A6 — reframe the existing lateral evidence

Do not build v1's §3.1. Instead: take the committed precision@10 = 0.5455 [0.5354, 0.5546] vs
radial-only null 0.0091 (§2.7), add a **shuffled-label null** and a **vertical comparator**, and
state plainly that precision@10 is a near-pure lateral metric (0.15% ancestral). Drop multiplicative
distortion (the null beats the model). Use a **vertex bootstrap** for any pair-level statistic —
`taxon_bootstrap_ci` assumes one value per taxon and is the wrong unit for pair statistics; it stays
valid for per-query kNN precision and for Track B's per-species reductions.

### 6.8 A8 — seeding

Add `--seed`, thread it to torch, the *legacy* numpy global RNG (the sampler uses
`np.random.randint`/`choice`), the shuffle, and the `epoch_fraction` subsample; record it in
`run.json`. Note AMP/CUDA nondeterminism will remain. Then 3 seeds per arm on the recipe contrast.
**Also correct the paper design doc's reproducibility claim (§2.2).**

### 6.9 A9 — the depth-null is misspecified

0.00089 is ~480× *below* the same harness's frequency-matched chance (0.4326). A direction-
randomizing null should land near chance; far below means systematic anti-correlation — an artifact
of `retrieve_nearest` geometry, not "what radius alone achieves". The obvious mechanism (collapse
onto one small-norm node) was tested and did **not** reproduce (~2× concentration, not 480×). Cause
undiagnosed.

Do not use this null until it is diagnosed. Report alongside any null: number of distinct reference
nodes retrieved, modal share, and the class distribution of retrievals. Require null ≥
frequency-matched chance; otherwise replace it (permute directions *within* depth stratum, or a 1-d
classifier on predicted radius alone).

### 6.10 Selection leakage — schedule it, don't just admit it

v1 correctly said selection leakage is not retroactively closable, then failed to schedule the only
fix. **Schedule it:** designate a clade held out of all development and report it once, at the end.

---

## 7. Track B — the pLM showcase

### 7.1 What already exists

`run_loso` (`read_eval.py:233`) = predict-the-embedding; `comparator_flat` (`:326`) = predict-the-ID
(genus one-hot); `comparator_logistic` (`:270`); `mmseqs_cluster(min_seq_id=0.9)`
(`clusters.py:32-39`); all four `splits.py` functions. ~80% built; the gap is **scoring**.

### 7.2 B1 — scoring layer

New `src/taxembed/eval/baselines.py`: `majority_class_baseline`, `macro_balanced_accuracy`,
`wu_palmer(pred, true, treedist)`, `spdist(pred, true, treedist)` (§3.3 — replacing v1's bespoke
`graded_tree_score`). Wire into `run_sp_read`'s JSON. Pure functions, unit-testable without the
1.4 GB H5, reused by the PLA2 path, SP path, A4 and ToxFam.

Add **frequency-quintile bucketing** (GRAM, §3.4): report every metric bucketed by training
frequency, and pre-register the predicted crossover — structural prior wins on rare classes, loses
on Mammalia.

### 7.3 B2 — the matched head-to-head, asymmetries removed

Three real asymmetries in the current code, all favouring the embedding arm:

1. `select_alpha` minimizes tangent-space MSE on `log0(positions)` — **Arm B's objective** — and the
   same scalar is passed to `comparator_flat` (`:791`), whose target is a one-hot on a different
   scale.
2. On the SP path alpha is not selected at all: `Bridge.align(..., alpha=1.0)` is hardcoded (`:466`).
3. Arm A picks among train-fold **genera** and inherits the lineage of an *arbitrary representative*
   train taxid (`:352-357`). That is an architecture mismatch, not merely the rank-granularity
   mismatch v1 described.

Fix: Arm A = ridge on one-hot **class** labels, identical PCA, **alpha-selection procedure run
independently per arm**, identical decoding (argmax over the 43 classes). Target space is the only
permitted difference. Pre-register that.

**Also (v1 missed this): run the joint species-and-clade holdout for the bridge arm.**
`_sp_leave_species_and_clade` takes only `(reps, ann)` (`:640`) — no positions — so the leak
diagnostic is structurally raw-kNN-only. The bridge's 0.603 is produced by the *same*
nearest-train-protein mechanism (`:498-499`) as raw-kNN's 0.837 (`:655-656`), so under the same
joint holdout it would also be exactly 0. "raw kNN beats us but that's all leakage" is not
supportable while the artifact is exempt from the test. Report both, or drop `leak_drop` as a
standalone claim.

Split: cluster-grouped holdout on `cluster_id` (one **panel member** per cluster — §2.5), LOSO
secondary.

### 7.4 B3 — redundancy and resampling

Per Burkhard, in his order: (1) plain one-per-cluster run (24,779 → 14,384); (2) cap **3 per
family**, **100 draws**; (3) cap curve C ∈ {3, 10, 50, ∞} as sensitivity.

Two corrections to v1:

- **Macro accuracy cannot be the primary endpoint at cap-3.** Measured over 100 cap-3 draws: each
  draw holds only **14-26 of the 43 classes** (median 21), 21 classes have median count 0, **38 of
  43 fall below the harness's own `min_fold_n = 30`**, per-class n is min 1 / median 4. Macro over a
  class set that changes every draw is not a fixed estimand. **Primary at cap-3 = pooled accuracy;
  macro is primary only on the one-per-cluster run (n = 14,384), over a pre-declared class list
  meeting a minimum-n floor.**
- **The 100 draws are not a CI.** Mean pairwise protein overlap across draws is 3.50%, but all
  draws share one panel, one taxon set, one family set and one exclusion rule. The spread is a
  **panel-conditional stability measure**; label it "across-draw spread (panel-conditional)", keep
  the taxon bootstrap as the only inferential interval, and never combine them.

Verified and retained from v1: family capping genuinely does not touch the taxonomic prior —
Mammalia is 64.53% ± 1.94% across cap-3 draws vs 64.48% in the panel. Family-cap-3 × 100 controls
family redundancy; macro accuracy and the majority baseline handle the Mammalia prior. Both.

Also note cap-3 handicaps Arm A structurally (a discrete 25-43-way classifier on ~400 rows degrades
far faster than a ridge onto a 100-d continuous target with a structured prior). Cap per
(family × class) if the taxonomic control is wanted, and state the handicap either way.

Re-score the degenerate leave-clade-out (and `leak_drop`) with Wu & Palmer / SPDist so they report
"how far off" instead of a structurally guaranteed 0.000.

### 7.5 B4 — multiple comparisons

Roughly 50-100 reported comparisons are implied (2 arms × several metrics × 2 splits × a cap curve ×
6 ranks). Pre-register **one** primary endpoint at **one** rank on **one** split; everything else is
secondary/exploratory and routed through the existing `holm_adjust` (`read_eval.py:724`). Declare a
target effect size, and adopt the posture `read_eval`'s own `family_balanced` block already uses:
underpowered ⇒ reported UNINFORMATIVE, never read as a pass.

---

## 8. Track C — ToxFam (last)

`TOXFAM_FASTA` / `TOXFAM_LABELS` / `TOXFAM_RESIDUE_H5` already exist at `bridge/config.py:44-46`.
Same harness as Track B, label axis swapped from Pfam to toxin family.

---

## 9. Threats, corrections, and what could not be checked

### 9.1 Retained threats
Closure leakage (§3.1) — now a design requirement, not a caveat. Selection leakage (§6.10) — now
scheduled. Ablation-table confounds (parametrization/loss confounded; curriculum removed only from
an already-broken baseline; effective batch differs in four hyperparameters) — unchanged, needs a
clean LOO grid at 498k.

### 9.2 New threat: no external ground truth
Every Track A test except A3 scores the embedding against the same NCBI taxonomy it trained on. A3
partially escapes (future NCBI is still NCBI). A fully external target — TimeTree divergence times,
GTDB, or genome-wide ANI — remains unaddressed and is the strongest available answer if the budget
allows. **Flagged as an open decision, not scheduled.**

### 9.3 v1 citation errors, corrected
- 14,384 clusters: from the panel's `cluster_id` column, not `sp_metazoa_cluster_ids.tsv` (§2.5).
- `helpers/app_hyperbolic_necessity_20260725.py`: **not in this repo** — `poincare-embeddings/
  helpers/` is empty. It lives in `.worktrees/taxonomy-bridge/projects/tax_disentangle/helpers/`.
  Any §6.4 work must reach it there or vendor it.
- §1.1's closure citation pointed at the legacy root script; the canonical dataset came from
  `taxembed build 131567 --clean` → `builders/taxopy_clade.py:241-302`. Substance identical.
- "trains on 21,399,053 pairs" — `--epoch-fraction 0.3` means ~6.4M depth-stratified pairs per epoch.

### 9.4 Not checked
`read_eval.py` end-to-end (needs the 1.4 GB H5 + locked metazoa checkpoint); the actual
`cellular_canonical.pth` (absent locally — the released safetensors was used); the
`--tiered-negatives` path (unused by the canonical run, read-only audit); Figure 4's artifacts (not
found in `paper/` or `docs/`); whether the 0.957-vs-0.952 depth-norm gap is fully explained by depth
source (`nodes.dmp` absent locally, 14-node difference unreconciled).

## 10. Testing

Metrics in §7.2 are pure functions → unit tests with hand-computed fixtures, following the existing
`tests/eval/` pattern (11 modules, including `test_fidelity.py`, `test_nulls.py`, `test_pairs.py`,
`test_bootstrap.py`, `test_treedist.py`). Resampler → tests that caps hold, draws differ by seed,
one-per-cluster is respected. A0's ancestry check → tests against the binary-lifting LCA on a
hand-built tree, including the root-anchor case where the false-negative rate must be 100% before
the fix and 0% after. Per Rule 16, every GPU item gets `bash -n` + `py_compile` + a smoke run before
submission.
