# Spec — E1c: rank-band hard-negative sampling for metazoa-scale separation

_Status: DRAFT v1 (2026-06-03), pre-review. Owner: master. Follows the design-work pattern
(spec → fan reviews → fold → plan → fan reviews → fold). Roadmap context: PROJECT_STATE.md
"Roadmap rev 2026-06-03", item **E1c**._

## 1. Context & problem

**Goal of the project:** Poincaré-ball taxonomy embeddings where per-rank **separation ratio**
= mean(inter-group dist)/mean(intra-group dist) is ≫1.5–2× ("EXCELLENT") at every rank
(class/order/family…), with radius encoding depth. At echino scale (3,965 nodes) the winning
recipe (`--loss softmax --euclidean-param` + radial scaffolding) hits order **2.54–3.01×**.
At metazoa scale (498,246 nodes, dim 100) it ceilings at **~1.20×** (POOR) and collapses at the
dd≤9→dd≤18 curriculum transition.

**Diagnosis (PROJECT_STATE rev 2026-06-03):** a negative-sampling problem. The default sampler
draws **same-depth** negatives (`_sample_negatives_default_vectorized`, train_hierarchical.py:384).
At echino a depth level is a few dozen mostly-related nodes → same-depth ≈ within-clade hard
negatives → strong angular gradient. At 498k a depth level spans tens of thousands of foreign-clade
nodes → trivially-far negatives → within-clade angular gradient starves.

**E1a result (echino A/B, 2026-06-03) — refines the diagnosis:** turning ON the existing
`--tiered-negatives` flag **degraded every rank** (order 2.54→1.61×, family 1.99→1.31×), radial
untouched. Mechanism: its 50% hard tier samples **same-grandparent** nodes
(`_gp_depth_to_nodes`, train_hierarchical.py:455) — i.e. immediate siblings/cousins the model should
*cluster* — and the loss pushes them apart, destroying within-clade tightness. So "more hardness"
via the existing scheme is counterproductive.

**Corrected understanding from code grounding (load-bearing — this REORDERS the plan):** the
softmax/NLL loss (`softmax_loss_from_dists`, train_hierarchical.py:705-715) ALREADY self-weights
toward hard negatives. Its gradient w.r.t. each negative distance is `∂L/∂d_neg_j = −p_j` where
`p_j = e^{−d_j}/Z` is that negative's softmax probability — so near (hard) negatives dominate the
gradient and far (easy) negatives contribute ≈0 **automatically**. Consequences:
- **RotatE-style self-adversarial *weighting* is largely redundant here** — softmax already does
  adaptive hard-negative weighting over the sampled set. (It would matter for the *ranking* loss,
  which averages negatives uniformly, train_hierarchical.py:688 — but ranking is not the winning
  recipe.) → demote self-adversarial weighting from "primary lever" (as the roadmap first framed it)
  to a minor, optional knob.
- The binding constraint is therefore **which negatives are in the sample**: softmax can only
  sharpen against hard negatives that the sampler actually drew. At scale the sampler draws none
  from the right band. → **the sampler is the primary lever.**

## 2. The target band ("informative middle")

The two failure modes bracket a sweet spot:
- metazoa default: negatives too **easy** (same-depth = mostly foreign clade) → no gradient.
- echino + tiered: negatives too **hard** (same-grandparent = siblings) → over-repels the clade.

For clean separation **at rank R**, the informative negative for an anchor is a member of a
**different group at rank R but the same group at the rank just above R** — a *cousin at level R*,
not a *sibling within the anchor's own rank-R group*. Pushing siblings apart destroys intra-group
tightness (the denominator of the separation ratio); pushing foreign-clade nodes apart is free but
uninformative. The sampler must hit the band in between, and that band is **rank-relative**.

Formalization by lowest-common-ancestor (LCA) depth. For anchor node `a` (the *descendant* of the
training pair) at depth `d_a`, and a candidate negative `n`, let `L = depth(LCA(a, n))`. Larger `L`
= closer relatives. Define a band by two integers measured as levels above the anchor:
- `w_near` (exclusion): reject negatives with `L > d_a − w_near` (too close — siblings). `w_near=1`
  excludes same-parent; `w_near=2` excludes same-grandparent (the tier that broke echino).
- `w_far` (inclusion ceiling): require `L ≥ d_a − w_far` (same broad clade; not foreign). Larger
  `w_far` widens the clade scope.

The negative pool for `a` = {n : d_a − w_far ≤ L ≤ d_a − w_near} (optionally also same-depth as `a`,
matching the existing depth-stratified scheme). Tiered's failure = `w_near = 0` (siblings allowed).

## 3. Design — two implementation tiers, simplest first

### 3a. MVP (reuses existing indices; minimal new code) — "sibling-excluded class-band"

The existing tiered sampler already has the pieces:
- **medium tier** samples `_class_depth_to_nodes[(class, depth)]` (train_hierarchical.py:495) =
  same top-level class, same depth — i.e. same broad clade.
- **hard tier** samples `_gp_depth_to_nodes[(gp, depth)]` = same grandparent (the harmful one).

MVP sampler = **same-class, same-depth, MINUS same-grandparent**, filling the rest from same-depth
easy negatives:
1. Candidate pool = `_class_depth_to_nodes[(class(a), depth(a))]`.
2. Reject any candidate sharing `a`'s grandparent (`_node_gp_arr[n] == _node_gp_arr[a]`) → excludes
   `w_near≈2`.
3. Top up to `n_negatives` from `_depth_to_nodes[depth(a)]` (easy) if the band pool is small.

This is `w_near=2`, `w_far=` (depth of class ancestor) using only data already built in
`_build_depth_index` (train_hierarchical.py:262-335). New code is a single sampler method +
a flag. **Cheap, echino-first testable immediately.**

### 3b. Principled (new index) — tunable LCA-band sampler

Generalize to arbitrary `(w_near, w_far)`:
- Precompute each node's **ancestor-at-level-k** chain (extend `_build_depth_index`).
- For band `[w_near, w_far]`, negative pool for `a` =
  `nodes_sharing_ancestor_at(d_a − w_far)` **minus** `nodes_sharing_ancestor_at(d_a − w_near + 1)`
  (set difference of two ancestor-pools), restricted to same depth as `a`.
- Cheaper realization: sample from the `w_far` pool, **reject** candidates whose LCA depth with `a`
  exceeds `d_a − w_near` (rejection sampling; low reject rate if pools are large).

`w_near`, `w_far` become CLI flags (e.g. `--neg-band-near 2 --neg-band-far 4`). 3a is the special
case `w_near=2, w_far=class`.

### 3c. Minor optional knob — softmax temperature (NOT RotatE weights)

Expose `--softmax-temp α` scaling the logits in `softmax_loss_from_dists`
(`cross_entropy(α·logits, target)`). `α>1` sharpens the softmax onto the single nearest negative;
`α<1` spreads gradient across more negatives. This is the *only* loss-side weighting lever worth
adding, because the softmax denominator already provides self-adversarial behavior; detached
RotatE-style per-negative weights are redundant and are **explicitly out of scope** (see §7).

### 3d. Diagnostic instrument (required, ships with 3a)

Per-epoch (or every N batches), log over a sampled minibatch of negatives:
- **within-clade fraction**: share of negatives sharing the anchor's class / order ancestor.
- **mean negative Poincaré distance** and the **mean softmax weight `p_j`** mass on within-clade
  negatives (confirms the band actually receives gradient).
This makes the hypothesis measurable: a working sampler should raise within-clade `p_j` mass at
metazoa scale relative to the default sampler, and separation should track it.

## 4. Success criteria & gates

- **G0 — echino non-regression (HARD gate, RED-LINE echino-first):** 3a must NOT degrade the
  `echino_softmax_abctrl` baseline (order **2.54×**, family 1.99×, class 1.46×, depth↔norm +0.956).
  Target: ≥ baseline on order & family; depth↔norm ≥ +0.94. If it degrades like tiered did, the band
  is still too hard — widen `w_near` and re-gate. No metazoa run until G0 passes.
- **G1 — metazoa dd≤9 lift:** at the ep50–70 plateau, beat the current 1.14–1.20× sep at
  matched depth↔norm (≥+0.85). Read via the milestone sweep (`_sweep_diagnostic_analyses.py`).
- **G2 — survive dd≤18:** at ep80 (dd≤18 loads), depth↔norm ≥+0.85 AND sep ≥1.10× (the criterion
  Experiment 1 is also judged on) — i.e., the sampler keeps within-clade gradient alive through the
  transition that currently collapses.
- **Stretch:** any rank reaches GOOD (≥1.5×) at metazoa scale — the first time the bar is cleared
  at 498k.

## 5. Implementation plan (change points — for the plan stage, listed for grounding)

- `train_hierarchical.py`: new sampler method `_sample_negatives_band_vectorized` (3a first);
  extend `_build_depth_index` (:262) with ancestor-at-level chains for 3b. Negatives are produced
  CPU-side in the dataloader and consumed unchanged by the loss (call site train_small.py:738), so
  **no loss/grad code changes for the sampler** — clean isolation.
- `softmax_loss_from_dists` (train_hierarchical.py:705): optional `temp` arg for 3c.
- `train_small.py`: wire the sampler-selection + diagnostic; flags `--neg-sampling {default,tiered,band}`,
  `--neg-band-near/--neg-band-far`, `--softmax-temp`.
- `src/taxembed/cli/main.py` (:701-754): forward new flags + persist in `run.json`.
- Tests: extend the sampler unit tests; assert sibling-exclusion (no same-grandparent negatives
  in band mode) and band membership.

## 6. Experimental protocol

1. Implement 3a + diagnostic. Echino-first: run `echino_softmax_band` (200 ep, es=0, MPS, identical
   to the A/B baseline otherwise). Gate **G0**.
2. If G0 passes → spec/queue ONE metazoa LRZ run `metazoa_softmax_band` (winning recipe + band
   sampler), save-every 10, read milestones for **G1/G2**. Echino-end-to-end first (RED-LINE).
3. If G0 fails → widen `w_near` (3→…) on echino until non-regressing; only then metazoa.
4. 3b/3c are follow-ups only if 3a lifts metazoa but stalls below GOOD.

## 7. Non-goals / out of scope

- RotatE detached self-adversarial per-negative weights (redundant with softmax NLL — see §1).
- Entailment cones (E2), Riemannian optimizer / Lorentz (E3), product manifold (E3) — separate specs.
- Raising `--dim` (theory: 100 is ample). Changing the curriculum schedule. Touching the radial
  scaffolding (radial axis is solved).
- Any metazoa/LRZ submission before the echino G0 gate passes.

## 8. Open questions (for reviewers)

- Q1: Is the rank-relative band right, given separation is measured at *multiple* ranks at once? A
  single `(w_near,w_far)` may favor one rank. Should the band be **mixed** (a spread of LCA depths
  per batch) so all ranks get cousins? 
- Q2: Does sibling-exclusion at `w_near=2` accidentally starve *family/genus*-level separation
  (where the things to separate ARE close relatives)? Or does same-depth-different-grandparent still
  include enough family-level cousins?
- Q3: Perf of 3b's per-anchor set-difference / rejection at 498k — acceptable inside the dataloader,
  or does it need precomputation?
- Q4: Is the diagnostic's "within-clade `p_j` mass" the right leading indicator, or should we track
  something closer to the separation ratio directly during training?
- Q5: Confirm the softmax-self-weighting claim (§1) — is there any regime (e.g. AMP, depth-weighting
  `sqrt(dd+1)`, class-weighting) where easy negatives still leak non-trivial gradient?

---

# Fan review — findings ledger (4 reviewers, 2026-06-03)

Reviewers R1 (ML/gradient correctness), R2 (code-integration), R3 (experimental design), R4
(scope/risk). Severity in brackets; disposition = how it folds into v2. Convergent findings (raised
by ≥2 reviewers) are flagged ⚑.

**Correctness (design is wrong as written — must fix before any build):**
- ⚑ **[BLOCKER/MAJOR] Band anchored to the WRONG node.** R1: negatives are sampled relative to the
  *descendant* (gp/class of `desc_idx`), but both losses compute negative distance from the
  **ancestor** (`d(anc_emb, neg_emb)`, train_hierarchical.py:731). For dd≤9/dd≤18 pairs — the exact
  metazoa regime that collapses — the ancestor is 9–18 levels shallower, so "same-grandparent-of-
  descendant" is not a hard negative for the loss that runs. → **ACCEPT**: v2 defines the band by
  LCA depth **relative to the ancestor**.
- ⚑ **[BLOCKER] Gates compare unseeded, noisy, order-dependent point estimates.** R3: the analyzer
  has no seed, subsamples groups/pairs, and the inter-group loop only compares each group to the next
  ≤9 in dict order (analyze_hierarchy_hyperbolic.py:315) — the separation ratio is a random variable
  with unmeasured variance; G1/G2 margins (0.06×; 1.10 threshold) sit inside noise. → **ACCEPT**: add
  `--seed`+`--repeats` to training and analyzer; gate on mean paired-difference > 2σ.
- **[BLOCKER] Precision-regime mismatch.** R3: echino G0 runs fp32/MPS (no `--amp`); the metazoa
  sbatch uses `--amp`/CUDA, where boundary `arccosh`/`e^{−d}` underflow at depth 37 differs. A G0 pass
  may not transfer. → **ACCEPT**: run the metazoa band run **without `--amp`** to match the gate (or
  add an AMP-on echino variant); precision is a pinned, gated variable.
- ⚑ **[MAJOR] "class" = depth-1 = PHYLUM at metazoa**, not taxonomic class. R3: MVP `w_far=class`
  spans an entire phylum (tens of thousands of nodes) at scale = the foreign-clade pool the spec
  diagnoses as the failure; echino (one phylum) can't exercise it. → **ACCEPT**: band measured in LCA
  *levels above the ancestor*, not "the class node"; add a multi-phylum middle gate (see protocol).

**Silent-failure guardrails (can pass gates while doing nothing):**
- ⚑ **[MAJOR] Easy-negative top-up silently restores the failure mode.** R1+R4: when the band pool is
  small (deep nodes at scale), §3a step 3 fills from the *default* same-depth pool — quiet reversion
  to the diagnosed-bad sampler, with no flag in the metric. → **ACCEPT**: log **band-fill-rate**
  per-depth, gate on a floor (≥50% at the ranks that matter), sample the band **with replacement** to
  hold band mass constant.
- **[MAJOR] `-1` sentinel collisions.** R2: `_node_gp_arr`/`_node_class_arr` use `-1` for missing
  gp/class; a naive `gp(n)==gp(a)` reject conflates all `-1` into a phantom group and over-excludes.
  Existing tiers guard `valid_gp = gps>=0` (:445). → **ACCEPT**: replicate the guards; self-exclude
  descendant **and** ancestor from negatives (R1 NIT, R2 MINOR).

**Diagnostic is not yet a trustworthy indicator:**
- ⚑ **[MAJOR] "within-clade p_j mass" can move without separation moving.** R1+R3: tiered RAISED
  hardness and DROPPED separation (E1a) — so high within-clade p_j is consistent with both success and
  the documented failure. → **ACCEPT**: pair p_j mass with a **within-minibatch separation proxy**
  (intra vs near-cousin Poincaré-distance ratio on already-embedded negatives) and **validate the
  indicator** post-hoc on the echino grid (must correlate with the analyzer's ratio before trusting it
  at metazoa). Stratify by dd bucket (R1).
- **[MAJOR] Diagnostic's "order ancestor" isn't free.** R2+R4: no order-level ancestor index exists
  (only class + grandparent). → **ACCEPT**: v1 diagnostic = class + grandparent + band-fill + proxy;
  order-level deferred.

**Goodhart on the metric:**
- **[MINOR] sep_ratio can rise via radial inflation** (Poincaré distance dominated by the `(1−‖u‖²)`
  denominator), and depth↔norm ≥+0.85 is satisfied by the *same* radial spreading — the two gates
  aren't independent; the angular signal (kNN purity) is computed but ungated. R1. → **ACCEPT**: add
  **kNN purity** to G1/G2; require it to move with sep_ratio.

**Experimental controls (R3 — protocol not decision-grade without these):**
- **[MAJOR] One metazoa run, no paired control** → can't attribute lift (training is unseeded).
  **ACCEPT**: paired same-seed default-vs-band control at both scales.
- **[MAJOR] G0 baseline is loss-best** (understates separation; decoupled from the separation peak).
  **ACCEPT**: compare full milestone trajectories / peak-separation epoch, not `_best.pth`.
- **[MAJOR] Multi-rank confound** (Q1/Q2 real) → **ACCEPT**: echino `(w_near,w_far)`/mixture **grid**
  with full per-rank vectors before metazoa; G0 = **no rank regresses**. Promote the per-batch
  **mixture over w_near** (Q1) to the v1 default, not a deferred option (R1).
- **[MINOR] G2 single-epoch read can false-pass** (collapse is a process ep80→120) → gate across the
  **ep80–110 window**.
- **[MINOR] echino can't exercise w_far that matters** (max_depth 14 vs 37) → multi-phylum middle gate
  (arthropoda+mollusca+echino), RED-LINE-compatible (echino-first ≠ echino-only).

**Scope cuts (R4 — v1 over-built):**
- **[MAJOR] Cut `--softmax-temp` / 3c from v1** (redundant with softmax self-weighting; entangled with
  the radial-distance scale per R1; breaks the "sampler needs no loss/grad change" cleanliness). →
  **ACCEPT** → §7 non-goal.
- **[MAJOR] Defer `--neg-band-near/far` tunables**; v1 band uses a fixed sensible **mixture** default.
  v1 flag surface = `--neg-sampling {default,tiered,band}` + `--seed`. → **ACCEPT**.
- ⚑ **[MINOR] Cheapest decisive experiment, currently implicit:** ship the **diagnostic alone** and run
  it on the **existing default sampler at metazoa** (offline/CPU on existing artifacts) to confirm the
  within-clade p_j mass is *actually* starved — falsifies the whole premise for ~0 GPU before any
  sampler is built. R4. → **ACCEPT** as v2 Step 0.
- **[MINOR] G0 must run through the exact `taxembed train --neg-sampling band` CLI argv** the LRZ
  sbatch uses (not the bespoke ablation script) and assert `neg_sampling: band` in `run.json` — closes
  the 2026-06-02 false-start class (validated the helper, not the real argv path). R4. → **ACCEPT**.

**Implementation facts (verified OK / corrected):**
- **[OK]** §1 softmax self-weighting (`∂L/∂d_neg=−p_j`) is **correct**, survives depth/class/AMP
  (per-example scalars don't change within-example weighting). R1+R3. Soften §1 wording: self-
  adversarial is redundant *up to the positive/negative temperature coupling* (R1 MINOR).
- **[OK]** MVP indices (`_class_depth_to_nodes`, `_node_gp_arr`, `_depth_to_nodes`) exist with the
  right dtypes; loss-isolation verified (drop-in sampler needs zero loss/grad changes). R2.
- **[MAJOR→cheaper] §3b needs precomputed-set-difference membership (`np.isin`), NOT runtime LCA**
  (no LCA primitive exists; runtime is 5–20× slower). Ancestor-at-level is **latent in the transitive
  closure** — build it via one groupby over `self.pairs`, not a parent-chain walk. R2. → **ACCEPT**.
- **[MAJOR] No existing test constructs the dataloader** — "extend sampler tests" is net-new infra
  (synthetic-tree fixture; assert sibling-/self-exclusion, positive band-membership, `-1` handling,
  band-fill). R2. → **ACCEPT**.
- **[NIT]** Stale file paths/line refs (files are repo-root, not `src/`; cite build-sites; dispatch is
  train_hierarchical.py:638). R2+R4. → fixed in the change-list at plan stage.

**Confirmed non-issues:** E1c is independent of Experiment 1 (no shared flags/checkpoint/queue);
file-safety clean (only NEW tags/files); the EXCELLENT bar is honestly framed as "Stretch."

---

# Spec v2 — folded design (supersedes §2–§6 above)

The reviews converted the "cheap flag-flip MVP" into a small but real build. The naive 3a (drop the
hard tier) is **abandoned** — it's descendant-anchored and `class=phylum` at scale, both wrong.

**Step 0 — validate the diagnosis for ~0 GPU (do FIRST).** Ship only the diagnostic instrument and run
it (offline/CPU) over the **default** sampler on a metazoa dataloader pass / existing checkpoint.
Confirm within-clade softmax-`p_j` mass is in fact starved at 498k. If it isn't, the premise is wrong
and we stop here. RED-LINE-safe (no LRZ).

**v2 sampler (band, ancestor-anchored, mixture).** For each training pair `(ancestor, descendant)`:
- Band defined by **LCA depth relative to the ANCESTOR** (the node distances are measured from), in
  *levels above the ancestor*: include negatives whose LCA-with-ancestor depth ∈ [d_anc − w_far,
  d_anc − w_near]; **exclude** the immediate sub-clade (w_near) and foreign clades (beyond w_far).
- **Per-batch mixture over w_near ∈ {2,3,4}** so class/order/family all receive cousin gradient
  (resolves the single-band multi-rank trade-off).
- Built from **precomputed ancestor-at-level membership** (one groupby over the transitive-closure
  `self.pairs`; `np.isin` set-difference — no runtime LCA). Guards: `valid` ancestor-level keys only
  (no `-1` phantom group); self-exclude ancestor & descendant; sample **with replacement** to hold
  band mass; **band-fill-rate** logged per depth.
- Fixed sensible default mixture in v1; `--neg-band-near/far` deferred.

**v2 diagnostic (ships with Step 0).** Per epoch, stratified by dd bucket: within-class & within-
grandparent negative fraction; mean negative Poincaré distance; **softmax-`p_j` mass on within-clade
negatives**; **band-fill-rate**; and a **within-minibatch separation proxy** (intra vs near-cousin
distance ratio on the already-embedded negatives). Validate that the proxy/`p_j`-mass *correlates*
with the analyzer separation ratio across the echino grid before using it as a metazoa leading
indicator.

**v2 gates.**
- **G0 (echino, HARD):** echino `(w_near,w_far)`-mixture **grid**, each cell run as a **paired
  same-seed** band-vs-default A/B through the **real `taxembed train --neg-sampling band` CLI argv**
  (assert `neg_sampling: band` in run.json), milestone trajectories saved, analyzer run with
  `--repeats` for mean±2σ. Pass = **no rank regresses** below the `echino_softmax_abctrl` baseline
  (order 2.54×, family 1.99×, class 1.46×) at peak-separation epoch, depth↔norm ≥+0.94, **and kNN
  purity not down**.
- **G0.5 (multi-phylum middle gate):** repeat the winning grid cell on a merged arthropoda+mollusca
  (+echino) set so the band's foreign-clade behavior at max_depth>14 is exercised before any V100.
- **G1/G2 (metazoa):** ONE band run **+ a paired same-seed default control**, same code rev, no
  `--amp` (precision parity), save-every 10. G1 = beat 1.14–1.20× at matched depth↔norm on the
  **paired** delta (>2σ). G2 = depth↔norm ≥+0.85 **and** sep ≥1.10× **and kNN purity held** across the
  **whole ep80–110 window**. Stretch = any rank ≥1.5×.

**v2 scope (v1 = the minimum correct build).** IN: Step-0 diagnostic; ancestor-anchored mixture band
sampler (precomputed membership) with guardrails; `--neg-sampling {default,tiered,band}` + `--seed`;
analyzer `--seed/--repeats`; kNN-purity in gates; new `tests/test_negative_sampling.py`. OUT (v1):
`--softmax-temp`/3c, `--neg-band-near/far` tunables, order-level diagnostic, RotatE weights, cones/
optimizer/manifold (E2/E3). **No metazoa/LRZ run before G0 + G0.5 pass through the real CLI path.**

_Next: this folds into the plan stage (writing-plans → second fan review on the plan), per the
design-work pattern._
