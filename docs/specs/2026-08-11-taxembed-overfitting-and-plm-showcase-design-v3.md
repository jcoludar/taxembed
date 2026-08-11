# TaxEmbed: overfitting evidence + the pLM showcase — design v3 (CONSOLIDATED)

**Date:** 2026-08-11
**Supersedes:** v1 and v2 (both kept — the four review passes cite them by section)
**Status:** Consolidated against 4 independent reviews (hostile methodology, fact/statistics
verification, literature grounding, executability) + author re-derivation of every load-bearing
number.
**Origin:** Burkhard raised overfitting. Answer him **factually**.

---

## 0. The three suggestions being logged

> - **TaxEmbed: pLM** — train to predict ID of taxa from protein, predict the ID of our embedding
>   from protein — if embedding is better, then we're better on those 24k. (RedundReduce 90%? Mark
>   mmseqs2 clusters — pair to pair to pair, from every family maximally 3, in the sampling do again
>   and again — 100 times.) **Start without sampling.**
> - **TaxEmbed: overfitting?**
> - **ToxFam** — if that helps instead of taxonomy ID?

Constraints: ToxFam last; overfitting is about **training the taxonomy embedding**; pLM work is
downstream showcase.

**Structural finding:** these are not separable. The design that defeats a memorization objection is
a **clade holdout**, scoreable only via a side channel that places the held-out node. TaxEmbed's
side channel is the ProtT5 bridge. But see §4.4 — the *strong* form needs a taxonomy retrain, not
just an eval-time coordinate mask.

---

## 1. Verified facts

Every number below was re-derived by the author. `helpers/fn_rate_exact.py` and
`helpers/panel_redundancy_census.py` are the reproducers.

### 1.1 The training objective has a false-negative defect — 47.3752%, exact

Negatives are drawn from `_depth_to_nodes[depth(descendant)]` (`train_hierarchical.py:398`; fast
path `:400-403` uses `np.random.randint` — **with replacement, no self-exclusion, no ancestry
check**), then scored as distance from the **ancestor** (`softmax_loss`, `:730-732`). Any node at
the descendant's depth that also descends from the anchor is a **false negative**: a valid positive
being pushed apart.

Closed form — no sampling: `FN(a, dd) = C[a, dd] / N[dd]`, where `C[a,dd]` = descendants of `a` at
depth `dd` (read off the complete closure) and `N[dd]` = nodes at depth `dd`.

| anchor depth | pairs | share | FN rate | pairs with ZERO valid negatives |
|---|---|---|---|---|
| 0 | 1,102,162 | 5.15% | **100.0000%** | 1,102,162 |
| 1 | 1,102,159 | 5.15% | 92.3640% | 746,067 |
| 2 | 1,102,113 | 5.15% | 66.3580% | 290,613 |
| 3 | 1,052,529 | 4.92% | 60.0921% | 290,613 |
| 5 | 1,041,614 | 4.87% | 57.1238% | 290,613 |
| ≥10 | 11,084,067 | 51.80% | 33.3610% | 12,912 |
| **overall** | **21,399,053** | 100% | **47.3752%** | **3,026,809 (14.1446%)** |

Root-anchored 5.1505%; anchor depth ≤10 = 52.1909%. Nodes per depth 0-10:
`[1, 3, 46, 49584, 1990, 8925, 18866, 20605, 60509, 44289, 43982]` — note depth 4 has only 1,990
nodes, which is why shallow anchors saturate their pool.

⚠ v2's 47.07% was a sampled estimate, biased low at **every** depth. Superseded.

**Methods consequence:** the paper describes a clean Nickel-Kiela softmax. N&K sample negatives from
`{v' : (u,v') ∉ D}` — ancestry exclusion is part of the canonical objective. Our code omits it.
Fixing it **restores** N&K rather than departing from it, and that is how the Methods should say it.

### 1.2 No train/test split; nothing is seeded

No holdout, link prediction, MAP or mean rank anywhere in `train_hierarchical.py` (full read). Early
stopping compares **training** loss (`:926`); `--early-stopping 999` disables it. The string `seed`
occurs **zero times** in `train_hierarchical.py`, `train_small.py`, `cli/main.py`; `run.json` has no
seed. Unseeded: direction init (`torch.randn`, `:85`), every negative draw (**legacy** global numpy
RNG), the per-epoch shuffle and `epoch_fraction` subsample (`:592-623`), plus AMP/CUDA.

⚠ `2026-06-09-taxembed-paper-design.md:24` claims "Reproducible (seeded; repro run matches to
±0.01)". **Not supported by the code.**

### 1.3 Radius is initialized at target, then regularized to it

`_initialize_by_depth` sets every node's norm to `target_radius(depth)` at step 0 with a random
direction (`:65-96`, esp. `:82`, `:90-93`); `radial_regularizer` (`:748-778`) penalizes deviation
from the same target. **depth-norm r = 0.952 is not an outcome.** Report the initialization floor
beside it or do not report it.

### 1.4 The depth-null is degenerate — diagnosed

`radial_only_null` retrieval returns the **shallow-norm shell**: mean retrieved depth **2.95**
against a pool mean of 20.94; 99.5% of retrievals at depth ≤10, **0.00% deep**. Mammalia is deep, so
the null structurally never predicts the majority class. That is the 480×-below-chance
(0.00089 vs frequency-matched 0.4326) — **intrinsic hyperbolic geometry, not a bug.** An earlier
check that "did not reproduce" measured concentration of *identity*; the collapse is in *norm*.

**Consequence:** any result stated as "model vs radial-only null" is measured against a null that
cannot compete. That includes the precision@10 evidence in §1.6. A replacement null is required
before either is reportable — permute directions **within depth stratum**, or use a shuffled-label
null. Standing sanity check for every null: report distinct references retrieved, modal share, and
the **depth distribution of retrievals**; require null ≥ frequency-matched chance.

### 1.5 Closure integrity — clean

21,399,053 pairs / 1,102,163 nodes / max depth 40. Zero dd≤0, zero self-loops, zero duplicates,
strictly antisymmetric; 100% ancestor-containment on a 500k sample. `dim 100`, `n_negatives 300`,
`loss softmax`, `epoch_fraction 0.3`, `tiered_negatives false`
(`artifacts/tags/cellular_canonical/run.json`).

Closure npz **is local**: `data/taxopy/cellular_organisms_131567_clean/…_transitive.npz` (8.6 MB).
Sibling closures for echinodermata, mollusca, metazoa, arthropoda, eukaryota are alongside it.
(v2 wrongly said this was absent.)

Docstring bug: `:605-606` says the epoch sampler draws "equally from each depth_diff level"; `:616`
is proportional. Code is right, docstring is wrong — it will mis-model epoch composition for anyone
reading it.

### 1.6 The lateral evidence — real, but currently measured against a broken null

Only **0.15%** of a query's true top-10 tree neighbours are ancestral, so precision@10 is an
essentially pure **lateral** metric — i.e. it scores structure that was never a positive training
pair. `cophenetic_fidelity.json`: model **0.5455 [0.5354, 0.5546]** vs radial-only null **0.0091**.

⚠ Two caveats v2 got wrong: this is a **3,000 × 2,000 subsample**, not "full 1.1M scale"
(`cophenetic_fidelity.py:76-77`); and the null is the degenerate one (§1.4). The *finding* may well
survive a proper null — the point is that it has not yet been tested against one.

⚠ Do **not** report multiplicative distortion: model median 1.1424 vs radial-only null 1.1374 —
the null wins. Non-discriminative at cellular scale.

### 1.7 The 24k panel and the baseline ladder

24,779 proteins / 2,166 taxa / 43 class / 204 order / 14 phylum / 169 Pfam. Median 1 protein per
taxon (max 3,871); 1,094 single-protein taxa. Family median 86, largest 1,651 (PF00001); no family
below 50 members, so cap-3 = exactly 507 (2.05%), cap-10 1,690 (6.82%), cap-50 8,450 (34.10%).
Mammalia 15,896 = 64.15%.

90%-identity: **14,384 clusters** (from the panel's own `cluster_id` column — *not* from
`sp_metazoa_cluster_ids.tsv`, which has 30,497 rows → 18,118 clusters and reconciles only after
restriction to panel accessions). 10,423 singletons (42.06%), largest 179, one-per-cluster discards
41.95%.

Class rank, LOSO, n = 24,653 scored:

| Route | Class accuracy |
|---|---|
| raw ProtT5 kNN | **0.8368** |
| always answer "Mammalia" | **0.6448** |
| bridge → TaxEmbed coordinate | **0.6029** |
| frequency-matched chance | 0.4326 |
| depth-only null (degenerate, §1.4) | 0.00089 |

Denominator verified two ways: the 126-row gap to 24,779 is exactly the null-`tax_class` rows, none
Mammalia; and the JSON's own fold counts sum to 24,653 with Mammalia `n_held` = 15,896.
`class_rank_acc` is pooled protein-level over that denominator. **The bridge sits below the
majority-class baseline**, and `read_eval.py` never computes that baseline.

### 1.8 The leak diagnostic is applied only to the competitor

`_sp_leave_species_and_clade` takes only `(reps, ann)` (`read_eval.py:640`) — no positions — so the
joint species-and-clade holdout is structurally **raw-kNN-only**. The bridge is exempt from the test
used to discount its competitor. Since `pred[te] = classes[tr][nn]` can only emit a class present in
the train fold, the bridge under the same holdout would also be exactly 0. Either run it for both
arms, or drop `leak_drop` as a standalone claim. (Note: `:498-499` and `:655-656` are *not* the same
mechanism — Poincaré retrieval over placed positions vs raw ProtT5 cosine — but the conclusion holds
regardless, via the train-fold-class constraint.)

---

## 2. What "overfitting" means here

**(a) Memorizing the tree.** A category error for the *coordinate-lookup* deliverable, but a real
failure for what the paper sells: a relatedness proxy, a bridge target space, and — the paper's lead
application — **release-diff taxonomy QC**, explicitly a prediction task about taxa whose placement
changes.

**Steelman to be tested, not assumed away:** the geometry fit the 2026-06-09 taxdump's curatorial
idiosyncrasies such that (i) new/moved taxa cannot be placed by any inductive map, (ii) embedded
distance tracks NCBI convention rather than relatedness, and (iii) the bridge inherits both — which
would explain why projecting ProtT5 through it is *worse* (0.603) than leaving it alone (0.837).
**(ii) cannot be refuted by any NCBI-internal test** — see §4.5.

**(b) Failing to generalize past the supervised signal.** §1.6 is suggestive but needs a real null.

**(c) Overfitting the evaluation.** Selection leakage (recipe chosen while watching the reported
metrics — not retroactively closable; the only fix is a clade held out of all development, reported
once), the planted radial axis (§1.3), n=1 unseeded runs (§1.2).

---

## 3. Literature grounding — VERIFIED citations only

All checked at source. **Corrections from the first survey are marked ⚠.**

### 3.1 The protocol to copy is Ganea's, not Nickel & Kiela's

**N&K 2017** (NeurIPS, arXiv:1705.08039): transitive closure of WordNet nouns, 82,115 nouns /
743,241 relations; edges made **undirected** for Poincaré and Euclidean. Reconstruction = embed
fully observed data and reconstruct. Link prediction = randomly hold out observed links; validation
and test "**do not include links involving root or leaf nodes** as these links would either be
trivial or impossible to predict reliably." Ranking: for each (u,v) rank d(u,v) among
`{d(u,v′) | (u,v′) ∉ D}`; report **mean rank** and **MAP**; 10 negatives per positive.

⚠ **The held-out fraction is never stated**, there is no supplementary, and the folds were never
released. **N&K 2017's link-prediction split is not reproducible as published.**
⚠ **N&K 2018** (ICML, Lorentz; 82,115 / 769,130 / depth 19) reports **reconstruction only** on
WordNet — no link-prediction experiment. Its numbers are not comparable to 2017's LP rows.

**Ganea et al. 2018 is the only fully specified, leakage-aware split, and is what we should copy:**
compute the **transitive reduction** ("basic" edges), **always keep those in training**, split the
remaining 578,477 non-basic edges into validation 5% / test 5% / train, and **vary how much of the
closure is visible** (0%, 10%, 25%, 50%). That last knob directly measures the leakage we were
hand-wringing about, rather than caveating it.

⚠ You cannot inherit a number, only a protocol. The "WordNet noun closure" differs across papers:
Vendrov 82,192/838,073 · Athiwaratkun 82,115/837,888 · N&K'17 82,115/**743,241** · N&K'18
82,115/**769,130** · Ganea 82,114/661,127. The two N&K papers differ by exactly **25,889** edges on
the same named dataset, unexplained.

### 3.2 The mandatory trivial baseline

**Vendrov et al., ICLR 2016** (arXiv:1511.06361, §3.4, Table 1): a no-learning rule classifying
pairs positive iff they are "in the transitive closure of the union of edges in the training and
validation sets" scores **88.2%** vs order-embeddings' **90.6%** on 4,000 withheld edges. Report it
beside every learned link-prediction number.

### 3.3 The closure-leakage literature — narrower than first reported

⚠ **The first survey over-reached.** Only one of the four papers argues the thesis:

- **Mashkova, Zhapa-Camacho & Hoehndorf 2024** (arXiv:2405.04868; NeSy 2024, LNCS 14979, 331-354) —
  **genuine support**: explicitly reveals evaluation-dataset bias, entailed statements used as
  negatives, deductive closure not utilized. **Cite this one for the leakage thesis.**
- **Zhapa-Camacho & Hoehndorf, NeSy 2023** (arXiv:2303.16519; CEUR-WS Vol-3432, 85-102) — never uses
  the word "leakage". It *implements* a closure-aware control (test = axioms in `O⊢` but not
  `O⊢_reduced`; filtered metrics) without claiming existing benchmarks leak.
- **Box2EL** (Jackermeier, Chen, Horrocks, WWW '24, DOI 10.1145/3589334.3645648) — the leakage quote
  is **footnote 5**, about **one** prior paper (EmEL++), and is not characterized as
  deductive-closure leakage.
- **Liu, Cuenca Grau, Horrocks & Kostylev, KR 2023** (pp. 461-471) — objects that random splitting
  fails to capture the *causality of inference patterns*; a benchmark-design argument, not
  entailment contamination.

### 3.4 Clade/node holdout precedents

- **BioCLIP** (Stevens et al., CVPR 2024 — Oral, and one of two Best Student Papers): TreeOfLife-10M,
  10.4M images, 454,103 taxa, 108 phyla; **400 rare species removed from training**. ⚠ **Published
  zero-shot numbers are 38.0 vs CLIP 31.8** (Rare Species column); mean **39.4 vs 21.9**. The
  37.8/26.6 figures widely quoted online are **superseded arXiv v1**, as is the "17% to 20%"
  improvement claim (published: 16% to 17%).
- **DeepGOZero** (Kulmanov & Hoehndorf, *Bioinformatics* 38(Suppl_1):i238-i245, ISMB 2022): **16 GO
  classes held out entirely** (annotations removed before true-path propagation; criteria =
  equivalent-class axiom and ≥100 annotations), predicted from axiomatic definitions; **average AUC
  0.745** across all 16 classes spanning MF/BP/CC; proteins grouped at >50% identity, then **81/9/10
  over groups**.
- **TEPI** (Aakur et al., arXiv:2401.13219, 2024-01-24, IEEE JBHI 28(4):2385-2396): node2vec
  (Grover & Leskovec) over **72,378 bacterial species from NCBI Taxonomy** as the embedding search
  space; classified set is 93 species, **58 seen / 35 unseen** (Table III).
- Taxonomy expansion: **TaxoExpan** (WWW 2020) — ⚠ **20%** of leaf concepts on both MAG datasets
  (MAG-CS is 10% val + 10% test), not "10-20%". **Arborist** (WWW 2020) — 15% of leaf nodes,
  Pinterest production taxonomy. **STEAM** — ⚠ **KDD 2020, not WWW 2020**; 20% of *terms*, connected
  top-down seed. **QuanTaxo** — ⚠ **no venue** (arXiv only); samples 20% of leaf nodes with **no**
  top-down growth (the top-down seed is STEAM's). **Octet** (KDD 2020) — 64/16/20 over leaf nodes;
  3 of 4 taxonomies Amazon, 4th Yelp.

### 3.5 Metrics

**Wu & Palmer** — use `2·depth(LCA)/(depth_pred + depth_true)`. ⚠ Algebraically equivalent to the
1994 original but **not its literal form** (Wu & Palmer 1994 state
`ConSim = 2·N3/(N1+N2+2·N3)` in path-node counts), so do not cite the 1994 paper as the source of
the depth formula. TaxoExpan uses it on SemEval only; STEAM on all three datasets.
**SPDist** (Arborist §5.1) — undirected shortest-path distance between predicted and true parent,
justified as a human-verification-cost measure, with a documented disconnected-component fallback.
⚠ Arborist's "15% of leaf nodes" is verbatim twice, but Table 2's Pinterest counts (7,919 train /
2,873 test of 10,792) imply 26.6% of all nodes; the paper does not explain this.

### 3.6 The memorization control: RandomDAG

**GRAM** (Choi, Bahadori, Song, Stewart & Sun, KDD '17, arXiv:1611.07012): RandomDAG assigns each
leaf **five randomly chosen ancestors**; ⚠ parameter counts are "comparable"/"similar", **not
identical**. Heart-failure AUC **0.8226 (RandomDAG) vs 0.8447 (GRAM)** — Table 2(c), **100%
training-data column only**.

⚠ **The rare-code quintile claim is badly mis-cited and must not be repeated.** 0.0150 and 0.4903
are **GRAM+** (GloVe-initialized), not GRAM; and the baseline switches mid-comparison (0.0080 = RNN,
0.4959 = RNN+). Consistently paired, GRAM+ vs RNN = 0.0150/0.0080 and 0.4903/0.4951. **Plain GRAM
scores 0.0042 in the rarest quintile — worst of all seven models, below RandomDAG.** Any "structural
prior helps rare labels" framing rests entirely on GRAM+.

Retained: the **reporting shape** (bucket accuracy by training frequency) is sound and worth
adopting for our Mammalia-heavy panel. What we must *not* do is pre-register GRAM's crossover as an
expected result — the evidence for it is weaker than reported.

### 3.7 Do not cite N&K's Euclidean column

**Bansal & Benton 2021** (aclanthology 2021.insights-1.8), verbatim: *"Counter to what they report,
we find that Euclidean embeddings are able to represent this tree at least as well as Poincaré
embeddings, when allowed at least 50 dimensions."* Their reproduction: d=50 MAP **14 → 88.9**
(MR 1,281 → 1.8); at d=200 Euclidean **92.2** vs Poincaré **89.3** (both their reproductions; N&K's
published Poincaré at d=200 was 87).

⚠ **The unit-2-norm-ball cause is their speculation, not a finding** — they write "We *speculate*
that the authors may have normalized the Euclidean embeddings…" and "we *defer to the authors of the
original study for confirmation*." State it as a supported hypothesis.

This corroborates the July 2026 decision to soften hyperbolic to an architectural choice and gives
it a citation. **Action:** audit our own Euclidean arm for norm constraints before reporting any
d-sweep.

### 3.8 Prior art on NCBI taxonomy embedding

**v-PuNNs** (N'guessan, arXiv:2508.01010 — **v1 2025-08-01, v2 2026-01-05, preprint, no venue**):
p-adic/ultrametric (van der Put networks), not hyperbolic. NCBI **Mammalia**, Spearman ρ = **−0.96**
between norm and depth, **no train/test split**; also WordNet nouns (52,427 leaves) and GO molecular
function. Sub-claims still unverified: 12,205 taxa, depth 15, ~2.08M params, LeafAcc 95.8%.

Its unguarded depth-norm claim is the same exposure as ours (§1.3), and its lack of any held-out
validation is the gap TaxEmbed can claim by doing this properly.

---

## 4. The four plans

v2 was one document trying to be four. Split:

### P1 — Objective integrity (gates P2)

**P1.1 Quantify — DONE** (§1.1). Move `helpers/fn_rate_exact.py` into the repo as the reproducer.

**P1.2 Instrument.** `scripts/diagnose_negative_hardness.py` **is** the E1c Step-0 diagnostic
(v2 wrongly said it was never built). Its `label_negatives` (`:105`) labels negatives against the
**descendant**; add an ancestry-of-**anchor** labeler.

**P1.3 Redesign the fix.** ⚠ v2's "reject descendant-negatives" is **wrong** — it empties the pool
for 3,026,809 pairs (14.14%; 100% of root-anchored, 67.69% of depth-1), and an empty pool makes
`softmax_loss_from_dists` cross-entropy over a single logit ≡ 0, i.e. **zero gradient**, deleting 14%
of the signal exactly at the shallow anchors that set global structure. Correct design:
1. **Drop root-anchored pairs** (1,102,162; 5.15%) — unsupervisable by a contrastive loss, and
   `d(root,x)` is already supervised by `radial_regularizer`.
2. When the same-depth non-descendant pool is short, **relax the depth stratification, not the
   ancestry constraint.**
3. **Mask rather than resample**, and **log realized negative count per example** so the residual
   depth confound stays visible.
Acknowledge two effects v2 missed: rejection rate falls 100%→33% with anchor depth, so effective
negative count becomes a monotone function of depth (softmax loss scale depends on denominator size,
compounding with `depth_weight = sqrt(depth+1)` at `:737-738`); and rejection inside a same-depth
pool massively oversamples the few survivors at shallow anchors.

**P1.4 Retrain** echinodermata + one mid-size clade, fixed vs unfixed, same seed; report the delta on
every downstream metric.

**P1.5 Seeding** (`--seed` → torch, **legacy** numpy global RNG, shuffle, `epoch_fraction`; record in
`run.json`) and **P1.6 radial initialization floor**.

**Decision this forces:** if the fix moves the numbers, the shipped 1.1M artifact was trained with a
defective objective and Methods must say so. If not, that is a robustness result. Either way the
Methods description needs correcting.

### P2 — Held-out evaluation (downstream of P1)

**P2.1 Ganea-style split** (§3.1): keep the transitive reduction always in training; hold out 5%/5%
from non-basic edges; sweep closure visibility 0/10/25/50%. **P2.2 Vendrov trivial baseline**
(§3.2) reported beside every number. **P2.3 Level-stratified holdout** — our own quartiles are
**Q1 = 11, Q3 = 28** (not HiG2Vec's 6-11); 593,576 eligible nodes; a 10% holdout is 59,357 test
links. **P2.4 RandomDAG control** (§3.6) — randomize the **closure**, holding pair count fixed.
Report MR/MAP stratified by retained-ancestor count and by true-parent sibling count.

### P3 — Temporal QC (INDEPENDENT of P1; can run in parallel)

Scores the shipped artifact against future NCBI. **~70% already built** and v2 did not know it:
`src/taxembed/eval/release_diff.py` (full merged/delnodes canonicalization; `reclassified_taxa`
currently *excludes* merges/deletions), `scripts/_anomaly_validation.py:168-172` (leg-B CLI with
`--new-merged/--new-delnodes`), `utils/taxdump.py:100-132` `ensure_taxdump_archive` (offline-safe
download), `scripts/_extract_taxdump_dmps.py`, `tests/eval/test_release_diff.py`, and the invocation
already commented into `scripts/analyze_lrz_anomaly.sh:113-116`.

Data confirmed available: NCBI keeps **monthly dated snapshots** at `taxdump_archive/`
(`new_taxdump_YYYY-MM-01.zip`) back to August 2014. Since our pin there are **2026-07-01** and
**2026-08-01**, plus a live daily. Our pinned `data/new_taxdump.tar.gz` is dated Jun 9 14:37 — one
minute before the closure was built at 14:38.

**To build:** the new-taxa arm, the placement arm (is a moved taxon already nearer its *new* parent
than its old one, above chance?), coverage reporting (fraction of new taxa with no coordinate — the
transductive limit, measured), and Wu&Palmer/SPDist scoring.

**Strong variant:** the monthly archive back to 2014 allows the same test across **many** release
pairs — train on an old snapshot, score on the next — turning n=1 into a distribution. Costs
retraining, so clade-scale. HiG2Vec's temporal task was a single release pair.

### P4 — pLM showcase

**P4.1 Scoring layer** — new `src/taxembed/eval/baselines.py`: `majority_class_baseline`,
`macro_balanced_accuracy`, `wu_palmer`, `spdist`. Pure functions, unit-testable without the 1.4 GB
H5. Add frequency-bucketed reporting (§3.6) **without** pre-registering GRAM's crossover.

**P4.2 Matched head-to-head** — three verified asymmetries, all favouring the embedding arm:
`select_alpha` minimizes tangent MSE on `log0(positions)` (Arm B's objective) and hands the same
scalar to `comparator_flat` (`:791`); the SP path hardcodes `alpha=1.0` (`:466`); Arm A picks among
train-fold **genera** and inherits an *arbitrary representative's* lineage (`:352-357`). Fix: Arm A =
ridge on one-hot **class** labels, identical PCA, alpha selected **independently per arm**, identical
decoding over the 43 classes. Target space the only permitted difference. **Also run the joint
species-and-clade holdout for the bridge arm** (§1.8).

**P4.3 Redundancy + resampling** — plain one-per-cluster run (24,779 → 14,384) first, per Burkhard;
then cap-3 × 100 draws. ⚠ **Macro accuracy cannot be primary at cap-3**: each draw holds only 14-26
of 43 classes (median 21), 21 classes have median count 0, 38 of 43 fall below
`config.py:116 SP_READ_MIN_FOLD_N = 30`. **Primary at cap-3 = pooled accuracy; macro is primary only
on the one-per-cluster run**, over a pre-declared class list meeting a minimum-n floor. ⚠ The 100
draws are a **panel-conditional stability measure, not a CI** (mean pairwise overlap 3.50%, but one
panel, one taxon set, one exclusion rule); label as "across-draw spread (panel-conditional)" and keep
the taxon bootstrap as the only inferential interval. Cap curve reduced to **{3, ∞}**.

Verified and retained: family capping does not touch the taxonomic prior (Mammalia 64.53% ± 1.94%
across cap-3 draws vs 64.48% in the panel). Note cap-3 structurally handicaps Arm A.

**P4.4 Multiple comparisons** — pre-register **one** primary endpoint at one rank on one split;
route the secondary family through the existing `holm_adjust` (`read_eval.py:724`, currently applied
to a single p-value). Adopt the harness's own posture: underpowered ⇒ UNINFORMATIVE, never a pass.

### 4.5 External ground truth — SCHEDULED, not optional

Steelman (ii) — embedded distance tracks NCBI convention rather than relatedness — **cannot be
refuted by any NCBI-internal test, P3 included.** Cheap sufficient version: TimeTree divergence times
for a few hundred well-sampled vertebrates and insects; report Spearman(embedded distance,
divergence time) beside Spearman(NCBI path length, divergence time). Days, not weeks. If the
embedding tracks NCBI better than divergence, that is the honest finding and it is publishable.
GTDB/ANI are more work for the same inferential role.

**Schedule it, or delete the relatedness claim from the paper.**

---

## 5. Cuts

- **Capacity / d-sweep re-specification — CUT.** v1's "saturates at d≈8 ⟹ not memorization" does not
  follow (precision@10 has a ceiling; the euclidean arm reaches the same 0.409 at d=100; the sweep
  used `n_neg=50`, no curriculum, no radial nudge and has *negative* depth-norm correlation). The
  conclusion was already conceded in July 2026 and §3.7 supplies the citation. High cost, no reader
  asking.
- **ToxFam — re-filed to its own plan.** A third label axis on a harness that has not passed its own
  asymmetry audit or run end-to-end locally.
- **A4 clade-holdout-through-bridge — demoted.** ⚠ Removing coordinates at eval time is **not**
  removing them from training; the geometry was fit with those nodes shaping the space. The weak form
  tests bridge induction, not memorization. The BioCLIP-equivalent needs a taxonomy retrain without
  the clade — fold into P2 if budget allows, do not claim the strong result from the weak test.
- **Vertical/lateral split as headline — CUT** (v1 §3.1; see v2 §1 for the three-part refutation).

## 6. Open items

- Selection leakage: not retroactively closable. The only fix is a clade held out of all development,
  reported once. **Schedule it or state it as a limitation.**
- Ablation-table confounds (parametrization/loss confounded; curriculum removed only from an
  already-broken baseline; effective batch differs in four hyperparameters) — unchanged; needs a
  clean LOO grid at 498k.
- `helpers/app_hyperbolic_necessity_20260725.py` is **not in this repo** — it lives in
  `.worktrees/taxonomy-bridge/projects/tax_disentangle/helpers/`. Vendor it if needed.
- Not checked: `read_eval.py` end-to-end (needs the 1.4 GB H5 + locked metazoa checkpoint); the
  actual `cellular_canonical.pth`; the `--tiered-negatives` path; Figure 4 artifacts; the 0.957 vs
  0.952 depth-norm reconciliation (14-node difference, `nodes.dmp` absent locally).

## 7. Manuscript-facing consequences (text parts)

Independent of how the experiments turn out:

1. **Methods** describe a clean Nickel-Kiela softmax; the sampler omits ancestry exclusion (§1.1).
2. **Reproducibility claim** — "seeded; repro run matches to ±0.01" is unsupported (§1.2).
3. **depth-norm r = 0.952** needs its initialization floor (§1.3).
4. Any **"vs radial-only null"** result needs re-statement against a valid null (§1.4).
5. **Euclidean/hyperbolic framing** gains a citation (Bansal & Benton) and must not cite N&K's
   Euclidean column (§3.7).
6. **Prior art** now exists for NCBI taxonomy embedding (v-PuNNs, §3.8) and should be cited.
7. If the relatedness claim stays, §4.5 must run.

## 8. Testing

Pure functions (P4.1) → unit tests with hand-computed fixtures, per the existing `tests/eval/`
pattern (11 modules). Resampler → caps hold, draws differ by seed, one-per-cluster respected.
P1.3's ancestry logic → tests against the binary-lifting LCA on a hand-built tree, including the
**zero-pool** case (⚠ v2's proposed test — "root-anchor FN must be 100% before and 0% after" — is
unsatisfiable, since post-fix the root pool is empty; test the drop rule instead). Per Rule 16, every
GPU item gets `bash -n` + `py_compile` + a smoke run before submission.
