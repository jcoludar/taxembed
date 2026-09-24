# FINDING: protocols imported from other benchmarks are vacuous on a taxonomy tree

**Scope:** P2 held-out evaluation design (spec `docs/specs/2026-08-11-taxembed-overfitting-and-plm-showcase-design-v3.md`
§P2.1, §3.1, §3.2, §3.3, §3.4, §3.6). Written as Task 9 of
`docs/plans/2026-09-24-taxembed-p2-heldout-evaluation.md`.

**Evidence, read-only, re-run 2026-09-24 against commit `9316095`:**
- `helpers/p2_check_closure_is_a_tree.py --expand`
- `helpers/p2_eligible_node_definitions.py`
- `helpers/p2_randomdag_changes_the_chance_floor.py`

Every number quoted below was reproduced by one of those three runs on the date at the top of this
file, not restated from memory or from an earlier session's log. Where the same number also appears
in `docs/plans/2026-09-24-taxembed-p2-heldout-evaluation.md` or
`results/p2_heldout_preregistration.json`, that is noted as corroboration, not as the source.

🛑 **The manuscript is not versioned (USER, 2026-09-24).** `manuscript/manuscript.v3_draft.md` states
the final, correct method only — a leaf parent-edge holdout with a chance-normalised cross-tree
metric — never a before/after narrative. **This note is where the reasoning lives.** The paper gets
one sentence of conclusion, citing back here for "why": *we use a leaf parent-edge holdout and a
chance-normalised control because the standard alternatives are vacuous on a taxonomy tree.* A future
editor should paste that sentence, never this note's narrative, into the draft.

Two findings, same shape: a protocol or control imported from another paper's benchmark encodes
assumptions about *that paper's data*. Both survive an invariant check and both fail a second,
unstated invariant that is exactly what made the import meaningful at its source. §"The unifying
lesson" below makes this explicit, because it is the point of the note, not incidental framing.

---

## Finding 1 — the Ganea split is vacuous on a taxonomy tree

### 1.1 The protocol as specified

Spec v3 §3.1: compute the transitive reduction ("basic" edges), **always keep those in training**,
split the remaining non-basic edges into validation 5% / test 5% / train, and vary how much of the
closure is visible (0%, 10%, 25%, 50%). §3.2 makes the Vendrov trivial baseline — a rule that scores
a pair positive iff it is in the closure of whatever is in the training/validation set — mandatory
beside every learned number.

### 1.2 All six clade closures on disk are strict trees

`helpers/p2_check_closure_is_a_tree.py --expand`, all six clades, two independent sources per clade:

| clade | n_nodes | n_basic (dd==1) | n_nodes − 1 | parents-per-node histogram | manifest `edges` |
|---|---:|---:|---:|---|---:|
| echinodermata_7586_clean | 3,965 | 3,964 | 3,964 | `{1: 3964}` | 3,964 |
| mollusca_6447_clean | 32,017 | 32,016 | 32,016 | `{1: 32016}` | 32,016 |
| arthropoda_6656_clean | 324,983 | 324,982 | 324,982 | `{1: 324982}` | 324,982 |
| metazoa_33208_clean | 498,246 | 498,245 | 498,245 | `{1: 498245}` | 498,245 |
| eukaryota_2759_clean | 877,584 | 877,583 | 877,583 | `{1: 877583}` | 877,583 |
| cellular_organisms_131567_clean | 1,102,163 | 1,102,162 | 1,102,162 | `{1: 1102162}` | 1,102,162 |

Every histogram is `{1: N}` — zero nodes with more than one parent among the basic edges — and
`n_basic == n_nodes − 1` exactly in every row. The `manifest edges` column is read straight from each
clade's own `taxonomy_edges_<clade>_manifest.json` (`nodes`/`edges` fields), an artifact written by
the closure builder, independently of the npz's own `depth_diff` array that the helper otherwise
uses. Both sources agree to the integer on all six clades.

### 1.3 The reduction reproduces the stored closure bit-for-bit

`--expand` walks the basic (parent-child) edges to their own transitive closure and diffs the result
against the closure actually stored in each npz:

| clade | derived pairs | stored pairs | in derived not stored | in stored not derived |
|---|---:|---:|---:|---:|
| echinodermata_7586_clean | 34,938 | 34,938 | 0 | 0 |
| mollusca_6447_clean | 278,695 | 278,695 | 0 | 0 |
| metazoa_33208_clean | 11,840,908 | 11,840,908 | 0 | 0 |

(arthropoda, eukaryota, cellular_organisms also ran with `--expand` this session and are identical;
the three above are quoted because they are the set the design gate originally checked and are cheap
enough to re-verify on every re-run of this note.)

Symmetric difference is 0/0 on every clade checked. The single-parent reduction is not merely
*correlated* with the closure — it **determines** it exactly, because a tree has one path from every
node to the root, so every ancestor-descendant fact is a walk up that one path.

### 1.4 Consequence: the mandatory trivial baseline is 100.00%

Once the reduction determines the closure exactly, "always keep the reduction in training" hands the
Vendrov rule perfect information: for any held-out non-basic edge `(a, d)`, walking `d`'s single
parent chain in the (always-present) training reduction recovers `a` if and only if `(a, d)` is a real
ancestor-descendant pair. The helper's own verdict line, printed for every clade in §1.3's table:

```
>>> IDENTICAL. The reduction determines the ENTIRE closure.
>>> Vendrov trivial baseline on ANY held-out non-basic edge = 100.00%
```

No learned number can beat a five-line closure computation. Running P2 under §P2.1 as literally
written would not have produced evidence about the model, in either direction — it produces a
ceiling every arm ties.

### 1.5 Why WordNet differs — a dataset difference, not a misreading of Ganea

Spec v3 §3.1's own verified citation table records Ganea et al. 2018's WordNet counts: 82,114 nodes,
661,127 edges, 578,477 non-basic. So Ganea's basic-edge count is

```
661,127 − 578,477 = 82,650
```

A tree on 82,114 nodes has exactly 82,113 basic edges (`n − 1`). WordNet's reduction carries **537
more** than that. Each of those 537 extra edges is a second immediate parent on some node in the
reduction itself — multiple inheritance, the signature of a DAG, not a tree. (`is-a` relations such
as *"seven overlaps with hexagon and heptagon"* genuinely have more than one immediate hypernym; a
species has exactly one parent taxon. That is the structural difference, not an artifact of either
dataset's construction.)

Vendrov's own trivial-baseline run (§3.2) scored **88.2%**, not 100%, on their WordNet split — a real
prediction task, beatable, because Vendrov's split is a plain random edge holdout that does **not**
enforce "always keep the whole reduction in training." Some of WordNet's own basic edges can end up
held out under Vendrov's design, so a query cannot always be answered by walking a training-time
parent chain that may itself be incomplete.

The trap this finding is named for is specifically the instruction *"always keep the transitive
reduction in training."* On a DAG with genuine multiple inheritance, that instruction is Ganea's
useful contribution — a protocol upgrade over Vendrov's plain random split. On a tree, the very same
instruction is not an upgrade; the reduction already contains everything, so nothing is left to
predict. Same clause, opposite effect, because it interacts with a structural property (tree vs. DAG)
the clause itself never mentions.

### 1.6 The mirror image: Vendrov's rule collapses to 0% under our own split

The degeneracy runs in both directions on our data. Under the leaf parent-edge holdout described in
§1.7, the held-out edge for a test node `v` is the single edge `(parent(v), v)` — and that edge is, by
construction, the one edge removed from every training-visible closure. Vendrov's rule (“positive iff
in the closure of what's visible”) therefore recalls **0%** of held-out parent edges: the one fact it
needs is exactly the one fact withheld. Both directions of the same trivial baseline are uninformative
on this data, for opposite reasons — 100% when the full reduction is always kept, 0% when the
held-out edges are drawn from the reduction itself. Neither number is a result; both are why the
comparison this evaluation actually needs is chance-normalised MRR against `sibling_chance`
(Finding 2), not a closure-membership rule in either direction.

### 1.7 What we do instead: a leaf parent-edge holdout over the interquartile depth band

`helpers/p2_eligible_node_definitions.py`, run against `cellular_organisms_131567_clean`:

- Per-node depth quartiles over the basic edges: Q1 = 11.0, median = 19.0, Q3 = 28.0 — reproducing
  spec §P2.3's stated Q1/Q3 exactly (the same quartiles over *all* closure pairs, not just basic
  edges, are 18/25/31 — a different number computed a different way; per-node is the reading that
  matches the spec).
- `depth ∈ [11, 28]` **inclusive** ⇒ **593,576** nodes — an exact match to spec §P2.3's asserted
  figure, delta **+0** against every one of the **13** rival eligibility rules the helper checks
  (half-open bands, leaf/non-leaf restrictions, sibling-count restrictions, `depth >= 2`; closest
  rival misses by 29,168, furthest by 508,586).
- Restricted further to leaves (`depth in [11, 28] AND leaf`): **501,037** nodes, 84.4% of the
  band-eligible set.
- At metazoa scale (`metazoa_33208_clean`): `n_eligible_leaves` = **284,181**; at a 10% test fraction,
  `n_test` = **28,418**.

The precedent for a **leaf** holdout specifically, not an arbitrary-node holdout, is spec §3.4:
**TaxoExpan** (WWW 2020) holds out 20% of leaf concepts; **Arborist** (WWW 2020) holds out 15% of leaf
nodes over a production taxonomy; **Octet** (KDD 2020) holds out 64/16/20 over leaf nodes on three of
its four taxonomies. Every cited precedent that removes taxonomy nodes for evaluation removes leaves.

### 1.8 Why it has to be leaves: the internal-node leak

Holding out the parent edge `(p, v)` of an **internal** node `v` does not hide `p`. `v`'s own
descendants `w` keep their `(p, w)` ancestor edges in the training-visible closure (only `v`'s own
`dd == 1` edge is removed, not `v`'s subtree's higher-`dd` edges to `p`). So `p` stays recoverable: it
is the unique child of `v`'s grandparent that is also an ancestor of `w`, for any surviving descendant
`w` of `v`. The held-out fact leaks back in through the very subtree that was supposed to make it
non-trivial. Restricting the holdout to **leaves** closes this — a leaf has no descendants to carry
the leaked edges — which is the same reason every §3.4 precedent above holds out leaves rather than
arbitrary internal nodes.

---

## Finding 2 — a structure-preserving randomisation is not a difficulty-preserving control

### 2.1 What RandomDAG preserves, exactly

`src/taxembed/eval/randomdag.py::randomize_parents` rewires every non-root node to a uniformly random
parent one level up, keeping the node's own depth unchanged. The module's own docstring and a unit
test both state and check the consequence: "because each node's ancestor count then equals its depth,
the closure's pair count is `Sum(depth)` either way — preserved EXACTLY"
(`tests/eval/test_randomdag.py::test_randomization_preserves_the_closure_pair_count_exactly`). This
session's re-run confirms it is not merely asserted:

```
closure pair count, both trees = sum(depth) = 278,695  (preserved by construction)
depths identical after rewiring: True  (parents all drawn from depth-1: True)
```

on `mollusca_6447_clean`, seed 0. Total pair count and per-node depth are real, proved, tested
invariants of the rewiring.

### 2.2 What it does not preserve: fan-out

`helpers/p2_randomdag_changes_the_chance_floor.py`, `mollusca_6447_clean`, seed 0:

| quantity | real tree | randomised |
|---|---:|---:|
| mean fan-out (nodes with ≥1 child) | 4.60927 | 1.95900 |
| max fan-out | 187 | 14 |
| internal nodes | 6,946 | 16,343 |
| leaves | 25,071 | 15,674 |
| band-eligible leaves (depth ∈ [11,28]) | 7,405 | 6,070 |
| mean `sibling_chance` (1 / grandparent fan-out) | 0.10330 | 0.55816 |
| median `sibling_chance` | 0.0417 | 0.5000 |

Mean chance-floor ratio randomised/real = **5.403**. The helper's own verdict:

```
🛑 THE CONTROL IS A DIFFERENT TASK: raw MRR is NOT comparable across arms.
   Compare a chance-ADJUSTED quantity (MRR - sibling_chance, or normalized_rank), never raw MRR.
```

A real taxonomy's fan-out is heavy-tailed (a handful of genera with dozens to hundreds of children,
most nodes with one or two) precisely because that is how biological classification works. Drawing
each node's new parent *uniformly* from the level above throws that shape away: the randomised tree
has 2.35x as many internal nodes, less than half the real mean fan-out, and a candidate pool
(`sibling_chance`'s denominator, the true parent's grandchild-of-grandparent fan-out) that is on
average 5.4x smaller — meaning easier to guess right by chance.

### 2.3 Consequence: the control is a ~5.4x easier task at chance, not a same-difficulty scramble

P2's candidate pool for a held-out node is its grandparent's children, so the chance rate of guessing
the true parent is `1 / fanout(grandparent)` — exactly `sibling_chance`. If the RandomDAG arm's own
`sibling_chance` floor is 5.4x higher on average than the real arm's, comparing raw MRR between the
two arms would let the control win almost regardless of what either model learned, and a real arm
that generalised well could still register a **false "the model memorises tree shape" verdict** —
the wrong answer, produced specifically by the control that exists to catch memorisation. This is the
same failure mode a memorisation control is meant to prevent, arrived at through the control itself.

### 2.4 The fix: chance-normalised cross-tree comparison

Within-tree comparisons (an arm vs. its own `sibling_chance_mean`; vis00 vs. vis50, both real trees,
same pool-size distribution) keep raw MRR — it is comparable there. Cross-tree comparisons (an arm vs.
RandomDAG) use

```
normalized_rank = (rank − 1) / (pool − 1)
```

whose expectation under a uniformly random rank is exactly **0.5 for every pool size**: `E[rank] =
(pool+1)/2` under uniform ranking, so `E[normalized_rank] = ((pool−1)/2)/(pool−1) = 0.5`, independent
of fan-out. This is pool-size (and so fan-out) invariant in exactly the way raw MRR is not.

Recorded as `p2_amendment_1_20260924` in `results/p2_heldout_preregistration.json`
(`src/taxembed/eval/preregistration.py::p2_verdict(..., amendment_1=True)`), which amends only the
cross-tree `GENERALISES`/`MEMORISES`/`MIXED` reading; each arm's own within-tree validity gates
(learning gate, floor gate, completion check) are untouched, and the default call
(`amendment_1=False`) still reproduces the frozen 2026-09-24 reading on the frozen test fixtures, so
neither reading was deleted in favour of the other.

---

## The unifying lesson

Both findings are the same shape, and it is the point of this note, not incidental framing: **a
protocol or control imported from another paper encodes assumptions about that paper's data.**

- Ganea's split assumes the transitive reduction does not, by itself, determine the whole closure —
  true on WordNet's DAG (537 nodes with a second immediate parent), false on a taxonomy tree (every
  node has exactly one).
- RandomDAG assumes that preserving structure (depth, and so total pair count) preserves task
  difficulty — true only if fan-out is close to uniform across the structure being preserved, false
  for a heavy-tailed real taxonomy where a handful of genera hold most of the fan-out.

In both cases the imported object was **checked against a real invariant, and it held** — the
reduction really does encode the closure (mathematically, by definition of transitive reduction); the
rewiring really does preserve depth and pair count (proved, unit-tested). That correctness is exactly
what made the *unchecked* invariant easy to miss: a component that visibly keeps one promise reads as
trustworthy, and nobody asks what else changed. **Ask what a transformation changes, not only what it
preserves.** A protocol audit that stops at "does it preserve the thing its docstring claims" will
pass both of these and still ship a vacuous evaluation.

This is the mirror case of the closure-leakage literature surveyed in spec §3.3 (Mashkova,
Zhapa-Camacho & Hoehndorf 2024, arXiv:2405.04868, is the one paper there with genuine support for a
leakage thesis). That literature argues benchmarks leak *too much* structure into the evaluation set.
These two findings are the opposite failure: the standard remedies — "always keep the reduction",
"scramble the structure for a control" — can make the held-out task leak *nothing left to predict*
(Finding 1) or compare two tasks of different difficulty under one metric (Finding 2). Same family of
bug, opposite sign.

---

## Verification record

Both required helpers were re-run in full this session, against commit `9316095`, and every number in
this note was read from that output, not from an earlier log:

| helper | invocation | numbers matched |
|---|---|---|
| `helpers/p2_check_closure_is_a_tree.py` | `--expand` (all six clades) | all six `{1: N}` histograms, all six `n_basic == n_nodes−1`, all three cited 0/0 symmetric differences, all six manifest `edges == nodes−1` |
| `helpers/p2_eligible_node_definitions.py` | (default, `cellular_organisms_131567_clean`) | 593,576 exact match / +0 vs. 13 rival rules; 501,037 leaf-restricted; Q1/median/Q3 = 11.0/19.0/28.0 over basic edges |
| `helpers/p2_randomdag_changes_the_chance_floor.py` | (default, `mollusca_6447_clean`, seed 0) | mean/max fan-out, internal/leaf counts, band-eligible leaves, mean/median `sibling_chance`, ratio 5.403 — all to the printed digit |

No number in this note disagreed with a helper's own output. `284,181` / `28,418` (metazoa-scale
eligible leaves / 10% test count) are read from
`docs/plans/2026-09-24-taxembed-p2-heldout-evaluation.md`, not re-derived here, since they were
already the plan's own measured figures at the time this note cites them.

## Pointers

- Spec: `docs/specs/2026-08-11-taxembed-overfitting-and-plm-showcase-design-v3.md` §3.1 (Ganea),
  §3.2 (Vendrov), §3.3 (leakage literature), §3.4 (leaf-holdout precedents), §3.6 (RandomDAG/GRAM),
  §P2.1–§P2.4.
- Plan: `docs/plans/2026-09-24-taxembed-p2-heldout-evaluation.md` ("Why this plan departs from the
  spec — read this before Task 1").
- Pre-registration: `results/p2_heldout_preregistration.json` (`p2_amendment_1_20260924`).
- Split builder: `src/taxembed/eval/p2_split.py` (`eligible_nodes`, `leaf_mask`, `select_holdout`).
- RandomDAG: `src/taxembed/eval/randomdag.py` (`randomize_parents`),
  `tests/eval/test_randomdag.py`.
- Manuscript target (not versioned): `manuscript/manuscript.v3_draft.md`, via
  `docs/MANUSCRIPT_CORRECTIONS_PENDING.md`.
