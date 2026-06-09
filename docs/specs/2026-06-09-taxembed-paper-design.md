# TaxEmbed paper — design / analysis spec

_Date: 2026-06-09 · Status: DRAFT for review · Venue target: Bioinformatics (tool/methods)_

## 1. One-line pitch

A reproducible recipe to embed the **whole cellular tree of life (~1.1M taxa)** in a 100-D
Poincaré ball — radius = taxonomic depth, distance = relatedness — and three applications the
*continuous geometry* enables that a discrete tree does not: a fast differentiable relatedness
proxy, automated taxonomy QC, and sampling-bias quantification.

## 2. The "so what" (framing discipline)

Every claim must answer the skeptic's "you embedded a tree you already had — why not use the tree?"
The value is never representation fidelity per se; it is **what the geometry buys that `nodes.dmp`
cannot**: O(1) differentiable distances, a global continuous consistency metric, and a volumetric
notion of coverage. Keep this test in front of every figure.

## 3. Contributions (claim list)

1. **Method** — softmax + euclidean-param + curriculum + radial scaffold + **scale-aware effective
   batch** clears EXCELLENT separation from 4k to 1.1M nodes. The curriculum-transition collapse and
   its fix (eff-batch 2048 · n_neg 300 · lr 0.001 · cosine warm-restart-on-phase) is the load-bearing
   methods result. Reproducible (seeded; repro run matches to ±0.01).
2. **Validation** — a 4–5-scale ladder (echino 4k · mollusca 32k · metazoa 498k · eukaryota 877k ·
   cellular 1.1M) with three orthogonal metrics: depth↔norm (radial), seeded separation (lateral),
   kNN-purity (anti-Goodhart; family lift 217–275× chance ⇒ genuine angular structure).
3. **Application #1 — geometry-as-a-service** (relatedness proxy).
4. **Application #2 — taxonomy QC / anomaly detection.**
5. **Application #3 — sampling-bias quantification.**
6. **Outlook** — the molecular→taxonomy bridge (#4), scoped, not built in this paper.

## 4. What we already have (no new compute)

- Recipe + locked metazoa result; reproducibility lock closed (job 5666348).
- Eukaryota 877k EXCELLENT (job 5666473): phylum 3.12 / class 5.09 / order 7.13 / family 8.23×;
  depth↔norm +0.978/+0.999; kNN-purity family 0.912 (274.5× chance).
- Cellular 1.1M (job 5673097) PENDING — the headline scale; numbers regenerated on landing.
- Tooling: `analyze_hierarchy_hyperbolic.py` (seeded separation), `knn_purity_hyperbolic.py`,
  `_negative_hardness.py` (exact Poincaré distance, float64-checked).

**Develop + validate all analyses on eukaryota 877k NOW; regenerate headline numbers on cellular
1.1M when 5673097 lands. Use `final` (ep200), not `best`.**

## 5. Applications — operationalized

### #1 Geometry-as-a-service
- **Core new result — cophenetic fidelity:** Pearson/Spearman of embedded Poincaré distance vs true
  tree distance (path length / LCA depth) over a seeded multi-million pair sample. Density hexbin +
  per-rank breakdown. This is the quantitative backbone licensing "relatedness proxy" (separation +
  purity show clusters are clean; this shows the *metric* is faithful).
- **Speed benchmark:** wall-time for N pairwise relatedness queries — embedding vector op vs
  LCA-on-tree traversal — across N up to full scale. Show the O(1) / vectorizable advantage.
- **New code:** `scripts/cophenetic_fidelity.py` (reuses the analyzer's tree-distance + the float64
  Poincaré distance). Thin.
- **Risk:** tree "distance" definition (topological vs depth-weighted) — **pre-register** the primary
  (with biological rationale), report both, relegate the other to supp. Do NOT choose post-hoc by which
  correlates better (p-hacking-adjacent).
- **MEASURED (eukaryota 877k ep200, 2026-06-09):** kNN-retrieval precision@10 = **0.636** [CI 0.627–0.644]
  vs **radial-only null 0.026** (≈chance) ⇒ **delta 0.609, model beats the depth-only null ~24×**;
  distortion median 1.14. The fidelity is genuine *angular* structure, NOT a radial artifact (clears
  reviewer B3). Built via Plan 1 (`taxembed.eval.*` + `scripts/cophenetic_fidelity.py`, 14 tests).

### #2 Taxonomy QC / anomaly detection
- **Score:** per-node geometric-vs-lineage disagreement = the kNN-purity check inverted — a taxon
  whose k nearest neighbours are predominantly a *different* family/order than its assigned lineage.
  Rank all nodes by an impurity / displacement score.
- **Validation (3 legs, all locked in):**
  1. **Synthetic ROC** — relocate a random sample of nodes to wrong clades, confirm the score
     recovers them (AUC); calibrates discriminative power cleanly.
  2. **NCBI-release diff (predictive, strongest)** — compute the score on an **older** taxdump
     release; test whether high-score taxa are enriched for ones NCBI **later** reclassified
     (parent changed) in a newer release. Requires fetching + aligning ≥2 taxdump versions.
  3. **Incertae-sedis / environmental enrichment** — supporting evidence that flagged nodes
     concentrate in known-uncertain regions.
- **New code:** `scripts/taxonomy_anomaly.py` (score + ranking), `scripts/_anomaly_validation.py`
  (synthetic ROC + release-diff + enrichment).
- **Risk:** release-diff taxon matching across versions (merged.dmp handles ID merges); confounds
  (reclassification ≠ embedding error) — frame as enrichment, not perfect recovery.

### #3 Sampling-bias quantification
- **Metric:** local density in the ball (k-NN radius or kernel density per region) → coverage score
  for any taxon subset vs the full tree, reported per major clade.
- **Headline dataset:** **UniProt/proteome coverage** — which taxa have proteome/UniProt
  representation — mapped onto the embedding. Quantify "fraction of <clade> volume covered." Ties to
  `unknown_unknowns` / pLM-data-bias; sets up #4.
- **New code:** `scripts/coverage_density.py`; a small fetch/join for the UniProt taxon list.
- **Risk:** density metric must be radius-corrected (depth confound — deeper = denser by
  construction); normalize against the all-taxa density at the same radius.

## 6. #4 outlook + assessment gate (after the triad)

The molecular→taxonomy bridge: an encoder mapping sequence / pLM-embedding → a point in the
hyperbolic ball, enabling inductive placement of unknown taxa (MAGs, eDNA) + contamination/HGT
detection. **Not built in this paper.** After the triad, run a reconnaissance pass over sibling
projects (`species_sampling`, `unknown_unknowns`, `plm_choice`, the protein-embedding H5s) to decide
**"already have enough to prototype" vs "needs a new framework."** Outcome documented as a follow-up
spec; the Bioinformatics paper's Outlook section states the bridge as motivated future work.

## 7. Out of scope (YAGNI)

- Viruses (polyphyletic, artificial root — excluded from the dataset by design).
- A full downstream-ML benchmark for #1 (would be needed for an ML venue, not Bioinformatics).
- Building the #4 encoder (separate effort, gated on the reconnaissance).
- Re-opening the negative-sampler / cones / structural alternatives (E1c/E2/E3) — superseded; the
  default sampler suffices at scale. Mention only as "not required."

## 8. Open questions

- Exact tree-distance definition for cophenetic fidelity (topological vs depth-weighted) — decide
  during #1.
- Which older NCBI taxdump release for the #2 diff (pick a gap large enough for real reclassification
  signal, e.g. 2–3 years).
- Figure budget / main vs supp split.

---

## 9. Review fold (2026-06-09) — 4-reviewer fan + reconnaissance, changes ADOPTED

Four independent reviewers (Bioinformatics-venue, statistical-rigor, scope/feasibility, novelty/prior-art)
+ an embeddings-ecosystem recon. All findings below are adopted into the plan; this section supersedes
§3/§5 ordering where they conflict.

### 9A. Repositioning (venue + novelty consensus)
- **Lead with artifact + scale, NOT "hyperbolic embedding of taxonomy"** (reads as Nickel & Kiela 2017
  redux). Defensible novelty spine: (1) **first continuous, metric, downstream-usable Poincaré embedding
  of the complete cellular tree of life (~1.1M NCBI taxa)** as a citable artifact; (2) the **scale-aware
  eff-batch + curriculum-collapse methods fix** that makes EXCELLENT separation reachable at that scale;
  (3) **predictive QC via NCBI taxdump release-diff** (no precedent found).
- **Reorder applications:** **#2 (taxonomy QC, release-diff) = LEAD** — it's the real answer to "why not
  use the tree?". **#1 (cophenetic fidelity) DEMOTED to a validation subsection**, not a headline —
  hyperbolic↔tree distance fidelity is established (Macaulay 2023 PLoS CB; De Sa 2018) for phylogenies;
  our only defensible angle is *scale* (1.1M, discrete NCBI tree) + *service*, never the finding itself.
  **#3 (sampling-bias) kept**, but MUST benchmark against Faith's PD (show agreement on small subsets,
  then claim speed/differentiability/scale).
- **Cite prior art head-on + add a comparison table** on axes {taxonomy source · scale (taxa) · space ·
  continuous-metric-output · applications}: Nickel & Kiela 2017/2018; Le et al. 2019 "Every child should
  have parents" (hyperbolic taxonomy-error refinement — closest #2 prior art, on WordNet); Macaulay 2023
  + De Sa 2018 (fidelity); Faith's PD 1992 + Picante (coverage); BIOSCAN-1M (arXiv 2508.16744, insects
  only, Lorentz 768-D supervised); HiG2Vec (GO); LifeMap / Walrus (NCBI viz-only). Demote the
  "taxon-vectors-for-ML" angle (node2vec/tax2vec own it) to a feature, not a contribution.

### 9B. Statistical rigor — cross-cutting requirements (heaviest fold)
- **Baselines/nulls are MANDATORY, not optional.** The radial-only (depth-only) null is the key guard:
  depth↔norm = 0.999, so a model that ignores all angular structure may already reproduce most distance
  correlation / coverage — the same Goodhart trap kNN-purity caught. Every headline number reports the
  **delta over the radial-only null** (+ shuffled-label + random-embedding refs).
- **Pairs are non-independent → effective N ≈ #taxa, not #pairs.** Use **taxon-level block bootstrap /
  jackknife** for all CIs; DROP pair-count p-values (they're meaninglessly tiny). FDR-control (BH) any
  per-node significance in #2 (~1.1M nodes).
- **#1:** replace global Pearson with **local rank fidelity** — within-clade Spearman/Kendall-τ +
  **multiplicative distortion** (d_emb/d_tree, the standard metric-embedding measure) + **kNN retrieval
  precision@k** (do the k nearest in the ball recover the k nearest on the tree). Global Pearson is
  secondary context only (dominated by trivially-far cross-kingdom pairs). **Stratified pair sampling by
  LCA-depth** (uniform sampling is swamped by easy far pairs). Speed claim is **O(D=100), "vectorizable,"
  not O(1)**; the tree baseline MUST be a fair fast LCA (binary-lifting / Euler+RMQ, batched), not a
  naive root-walk; report preprocessing + memory for both.
- **#2:** score = purity **relative to its size-conditioned expectation** (observed − `chance_purity`, or
  z vs a depth/clade-size-matched random-angle null) — raw kNN-impurity just rediscovers rare/small
  clades. Must beat **trivial baselines** (rank by clade size, depth, degree, distance-to-parent-
  centroid). **Synthetic ROC stratified by phylogenetic displacement** (sister-genus→cross-kingdom curve,
  not one gameable AUC). **Release-diff:** report **odds ratio + permutation/Fisher CI** vs a background
  **matched on depth + clade-size + study-effort** (e.g. #descendants/#sequences); training taxdump must
  **strictly predate** the scored "old" release (no leakage); **exclude pure ID-merges/rank-only changes**
  (canonicalize via merged.dmp + delnodes.dmp first); report actual n_flagged/n_reclassified/OR.
- **#3:** define coverage **on the tree partition** (covered lineages / branch fraction per clade), NOT
  geometric ball-volume (hyperbolic volume explodes with radius → meaningless denominator). Use the
  embedding for continuous interpolation/viz only. If geometric density kept: **kNN-radius density**
  (self-normalizing, dimension-robust) conditioned on **clade AND depth**, vs a **uniform-coverage null**.
  Add a **study-effort covariate** (proteome coverage correlates with model-organism status) so the novel
  signal = under-coverage *beyond* known sequencing priorities.

### 9C. Effort honesty (feasibility) — re-label "thin"
- **NOT thin wrappers (the real work):** (i) **LCA / tree-distance routine** for #1 — no such function
  exists yet; binary-lifting on the parent array (~1 day), seedable from the transitive npz ancestor sets;
  the heaviest new code. (ii) **#2b old-taxdump fetch + canonicalization** — fetch from NCBI
  `taxdump_archive`; handle merged.dmp + delnodes.dmp + rank-label drift; drop current taxids absent from
  the old release; pre-register "reclassification" = direct-parent change after canonicalization. (iii)
  **#3 UniProt proteome join** — UniProt proteome taxid list → roll up sub-leaf strain taxids to nearest
  embedded ancestor.
- **Genuinely thin (high reuse):** embedded Poincaré distance (`_batch_distances` in
  `knn_purity_hyperbolic.py`), the anomaly score (refactor of the kNN inner loop), incertae-sedis/env
  enrichment (reuse `audit_taxonomy_noise.py`'s classifier), `chance_purity`, `get_ancestor_at_rank`.
- **Build the tree-distance/LCA side NOW** against `cellular_organisms_131567_clean_transitive.npz`
  (already on disk) — decoupled from the pending checkpoint 5673097.
- **Local-path gotcha:** `run.json` holds LRZ container paths (`/data`, `/app`); tag-resolution defaults
  to `_best`. Always invoke analyses with explicit `--checkpoint <…>.pth --mapping <…>` and the **`final`
  (ep200)** checkpoint, never `best`.
- **Cellular rank-label trap:** re-derive EVERY baseline (chance-purity, density bins, distance-bin edges)
  from the dataset at hand; "phylum" semantics differ across bacteria/eukaryota and cellular adds
  superkingdom/domain ranks. Never hardcode a eukaryota-derived constant into a cellular figure.

### 9D. Artifact (publishability gate for the tool venue)
Ship: an installable package (pip/conda) + GitHub + frozen tagged release; the **pretrained 1.1M
embedding downloadable with a DOI** (Zenodo/figshare); and **one runnable biologist vignette** ("given a
species list / MAG set, run X → get QC flags / coverage report") reproducible from the README. Pick the
**research/methods-article track** (needs the baselines above) but still ship the artifact.

### 9E. #4 reconnaissance (the gate is largely resolved — it's prototype-able, not framework-blocked)
- Ecosystem holds ~**100–200K proteins across ~200–300 taxa** (mostly Hymenoptera/Formicidae + Squamata),
  predominantly **ProtT5 (1024-D)** H5/parquet, Dropbox-synced **offline**. No "One Embeddings" brand;
  closest = `projects/embeddings_to_phylogenies/src/one_embedding/` (+ a 411MB APH 299-protein set as the
  worked example with species/family structure).
- **#3 data:** no central proteome-coverage table yet, but **UniProt publishes per-entry embeddings for
  all of UniProtKB + a reference-proteome taxon list** (user-confirmed) → fetch via UniProt REST
  (`proteome:*`) and join to the embedded taxid set (~1–2 h). Not a blocker.
- **#4 readiness ~80% data / 20% curation:** best prototype corpus = ant venoms (largest N + family labels
  + 50+ genomes). Gap = a unified embedding × (taxid, family) matrix; the scattered family-silo H5s need
  harmonizing + taxid assignment. ~3–5 day prototype (sequence/pLM → hyperbolic point, contrastive,
  held-out-species validation). **Stays as Outlook in this paper**; keep #3 self-contained so reviewers
  don't demand the bridge as part of the triad.

### 9F. Honesty / limitations section to ADD to the paper
100-D is not 2-D-visualizable (no dishonest disk plot); training is **topology-only** (no branch
lengths / molecular signal) → the QC score detects *internal taxonomy inconsistency*, not biological
truth; coverage "bias" is partly real sequencing-priority biology. State all three up front.
