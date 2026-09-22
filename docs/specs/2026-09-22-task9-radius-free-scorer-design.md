# Task 9 scorer: a learned-structure metric the planted radius cannot reach

**Date:** 2026-09-22 · **Serves:** plan v2 Task 9 (`docs/plans/2026-08-11-taxembed-objective-integrity-v2.md`)
and correction C4 (valid null) in `docs/MANUSCRIPT_CORRECTIONS_PENDING.md`.

## 1. Why a new scorer

Task 9 asks whether Figure 4's claim (canonical recipe beats the prior approach) holds on
**learned** structure. Every metric used so far mixes in the **planted** radius:

- depth-norm r is fixed at initialization (log floor 0.957151; a run that learned nothing scored
  +0.980).
- The trainer's in-loop `kNN%` / `Sep` (`train_small.py:280-349`) and the headline separation ratios
  (`scripts/analyze_hierarchy_hyperbolic.py:254-328`) use **Poincaré distance between nodes of any
  depth**. Poincaré distance depends on both norms. So when the prior arm's radii drift (depth-norm
  falls to 0.669 in Fig 4), its separation and purity drop **whether or not its directions are any
  worse**. Those metrics cannot tell reading 1 from reading 2.

Needed: a score that (a) **cannot see the radius at all**, (b) has a **null that is exact, not
sampled**, and (c) has a ceiling, so a value can be read as "fraction of achievable structure".

## 2. The metric — same-depth angular LCA score

Notation: depth `δ(v)`, parent array, Euler intervals `[tin, tout)` (`taxembed.eval.subtree`),
Poincaré embedding `x_v`, direction `u_v = x_v / ‖x_v‖`.

For a query node `q` at depth `δ`:

1. **Pool** `P(q) = {v : δ(v) = δ, v ≠ q}`, i.e. only nodes at the query's own depth.
2. **Rank** `P(q)` by cosine `u_q · u_v`, descending. Take the top `k` (default `k = 10`) → `N_k(q)`.
3. **Score** `s(q) = mean_{v ∈ N_k(q)} δ(LCA(q, v))`, the depth of the most recent common ancestor.

**Why it is radius-free.** Step 2 uses directions only. Step 1 fixes depth, and in the canonical
arm the planted radius is a function of depth alone. So even a Poincaré-distance ranking inside
`P(q)` would equal the cosine ranking if radii were at target. Nothing in `s(q)` reads a norm.

**Why it measures learned structure.** No same-depth pair is ever a positive training pair.
Training pairs are ancestor→descendant only, and same-depth nodes are exactly what the sampler
draws as *negatives*. A high `s(q)` means that the directions of **cousins** are ordered by how
related they are. That structure arises only through shared ancestors. This is the "lateral"
evidence of spec v3 §1.6, now with a valid null.

### 2.1 Exact null (random directions)

Under i.i.d. uniform directions, independent of the tree (**this is the initialization state**,
`train_hierarchical.py:103-112`), every `k`-subset of `P(q)` is equally likely. So

`μ0(q) = E[δ(LCA(q, v))]` for `v` uniform on `P(q)`.

Let `a_0 = root, …, a_δ = q` be `q`'s ancestor chain, and `c_j = |{v ∈ P(q) : v ∈ subtree(a_j)}|`.
Then `c_0 = |P(q)|`, `c_δ = 0`, and exactly `c_j − c_{j+1}` pool members have LCA depth `j`, so

`μ0(q) = Σ_{j ≥ 1} c_j / c_0`  — closed form, no sampling, no seed.

Each `c_j` is one `searchsorted` over the depth-`δ` pool pre-sorted by `tin`, because a subtree is
a contiguous `tin` block (the same trick as plan v2 Task 6).

### 2.2 Exact ceiling (oracle)

`μ*(q)` = the best mean LCA depth any `k` members of `P(q)` can reach: fill greedily from
`j = δ−1` downward, taking `min(remaining, c_j − c_{j+1})` nodes at depth `j`.

### 2.3 Reported quantity

`S = (mean_q s − mean_q μ0) / (mean_q μ* − mean_q μ0)`, a ratio of means over the query set.
It is more stable than a mean of per-query ratios. `S = 0` is random directions (the untrained
model) and `S = 1` is a perfect cousin ordering. Report `mean s`, `mean μ0`, `mean μ*` beside it.
Also report `S` by depth band: shallow (δ ≤ 10), mid (11-27), deep (≥ 28). These are the spec's own
quartiles Q1 = 11, Q3 = 28.

**Query set.** A fixed-seed uniform sample of `Q = 10,000` nodes. The sample excludes nodes with
`|P(q)| < k`, and nodes with `μ* = μ0` (every pool member equally related, so nothing to rank).
**The same query set is used for every checkpoint and both arms**, so every comparison is paired.

**Uncertainty.** A paired percentile bootstrap over queries (1,000 resamples) for `S` and for
`S_canonical − S_prior`. ⚠ This is **query-sampling** uncertainty for one trained run. Run-to-run
spread comes only from the seeded array (job 5802007). Label the two differently in every table.

## 3. Companion numbers (same query set, same pools)

| key | what | why |
|---|---|---|
| `S_angle` | §2, cosine ranking | **primary** |
| `S_poincare` | §2 with Poincaré distance on the raw embedding instead of cosine | `S_angle − S_poincare` = how much **radial drift** damages same-depth retrieval. ≈ 0 for canonical by construction |
| `depth_norm_r` | Pearson(depth, ‖x‖) | planted axis; quote beside the metazoa init floor (computed, not assumed) |
| `radial_dev` | mean \|‖x‖ − target_radius(δ)\| | direct measure of radial drift |
| trainer fields | `loss`, `knn_purity`, `class_sep_ratio`, `depth_norm_corr` read from the checkpoint | continuity with Fig 4, and the training gate |

## 4. Self-checks that must MOVE A NUMBER (run before any real reading)

1. **Null:** an embedding with planted radius + random directions (a reconstructed epoch 0) must give
   `|S_angle| < 3 × bootstrap SE`. The closed-form `μ0` must agree with a Monte-Carlo estimate on a
   small tree.
2. **Ceiling:** a hand-built tree whose directions encode it exactly must give `S_angle = 1.0`.
3. **Radius blindness:** multiplying every norm by an arbitrary positive per-node factor must leave
   `S_angle` **bit-identical**, and must change `S_poincare`.
4. **Sensitivity:** on the local mollusca checkpoints, where the in-trainer kNN% already separates the
   arms (prior 99.6 % vs canonical 73.4 %), `S_angle` must be clearly above 0 for the arm that
   trained. This is a check that the scorer can *see* structure. It is **not evidence about the
   recipe** (canonical was not converged there).

## 5. Reading (carried over from Task 9 Step 4, unchanged, plus the gate)

**Validity gate first**, per arm. The checkpoint `loss` must fall by more than its own
last-10-epoch noise span, and `S_angle` must trend over the milestones rather than wander. Otherwise
UNINFORMATIVE, in both directions.

Then, at epoch 200:
1. canonical `S_angle` > prior, paired CI excludes 0 → Fig 4's claim **holds on learned structure**;
   re-plot Fig 4 on `S_angle`.
2. indistinguishable on `S_angle`, differing only on depth-norm / `S_poincare` / separation →
   the recipe's benefit is **preservation of the radial prior**; rewrite the caption and the
   "collapse" language.
3. prior > canonical → escalate.

Additionally read the **trajectory**. Fig 4 shows prior's depth-norm peaking at +0.875, then
collapsing to +0.669 at the curriculum transition. Does prior's `S_angle` collapse at the same epoch?
If `S_angle` holds while depth-norm collapses, the collapse is radial only.

For the historical Fig 4 runs (n = 1 per arm, unseeded) this is a **provisional** reading. The
seeded array (3 per arm) confirms or overturns it.

## 6. Inputs and compute

- Fig 4 prior: `artifacts/tags/metazoa_softmax_milestones/…_milestone_epoch{10..200}.pth`
- Fig 4 canonical: `artifacts/tags/metazoa_lower_lr_bigger_batch/…_milestone_epoch{10..200}.pth`
- Closure: `/data/taxonomy_edges_metazoa_33208_clean_transitive.npz` (LRZ); depths via
  `scripts/radial_init_floor.py:node_depths_from_closure`; `max_depth` = closure max descendant
  depth (= `TrainingPairs.max_depth`, which is what the trainer used).
- Cost per checkpoint: load 399 MB; cosine over the same-depth pools, roughly
  `Q · Σ|P_δ|² / N · d` ≈ 3.5e10 flop. Seconds to a minute on CPU. 40 checkpoints ≈ < 1 h on one
  `lrz-cpu` node. **LRZ, not the Mac** (Rule 18). Local runs are limited to unit tests and the 32k-node
  mollusca checkpoints (≈ seconds).

## 7. Out of scope

Rank-labelled purity (phylum/class/order/family) needs NCBI rank names. LCA depth is the rank-free
equivalent, and the spec notes `nodes.dmp` is absent locally. Global (all-depth) retrieval is out
too, because it re-admits radius via the depth mix.

---

## Addendum 2026-09-22 — hostile review, and what changed

An independent review found 3 blockers, 6 should-fix and 4 notes. The **decision rules now live in
`results/recipe_angular_comparison.json` → `preregistration_v2_20260922`**, frozen before any
metazoa checkpoint was scored. §5 above is superseded by it.

Applied in code:
- The bootstrap is over **depth-3 clades**, not queries, because queries in one clade are not
  independent.
- Strata: per depth band **and** per clade. The clade level is fixed from the tree and query set
  before any embedding is read.
- k ∈ {1, 10, 100}.
- Hubness report: distinct neighbours and modal share.
- Poincaré ranking uses the hyperbolic law of cosines on `‖z‖`.
- Single-root guard.
- Each run is scored as the mean over its rolling epoch 196-200 checkpoints, with jitter reported.

Verified rather than taken on trust:
- The band split: on mollusca, canonical wins the shallow band and loses the mid band.
- The seed-vs-clade noise scales: seed SD ≈ 0.0005, while the clade-cluster CI on the arm difference
  spans 0. **The stratum sign rule, not the seed rule, is the binding one.**
- One review item was inaccurate. The design predates the mollusca run. Only the reading rules
  postdate it, and that is declared.

§4.1/§4.2 already have independent brute-force oracles: naive parent-walk LCA on random recursive
trees whose lineages end at different depths. §4.3 guards only against a regression to raw dot
products. It cannot detect radius-direction coupling that arises during training.

**Owed:** per-level AUC, the radius-transplant 2×2 on the trainer's own metrics, dose-response
validation, and the registered secondary metrics. All four are listed in the pre-registration
block.
