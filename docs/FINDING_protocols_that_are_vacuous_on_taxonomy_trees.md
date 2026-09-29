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

**Evidence for Finding 3, read-only, run 2026-09-29** against the banked P2 array
(`results/p2_verdict_20260929.json`, `artifacts/p2_scoring/*.json`, submodule `e04c478`):
- `helpers/_p2_cosine_vs_poincare.py`
- `helpers/_p2_curriculum_vs_collapse.py`
- `helpers/_p2_peak_real_vs_control.py`

Four findings, same shape: a protocol or control imported from another paper's benchmark encodes
assumptions about *that paper's data*. All four survive an invariant check and all four fail a
second, unstated invariant that is exactly what made the import meaningful at its source. §"The
unifying lesson" below makes this explicit, because it is the point of the note, not incidental
framing.

⚠ **Finding 3 is about the replacement adopted in §1.7 of this note.** Findings 1 and 2 rejected two
imported designs and §1.7 adopted a third; Finding 3 reports that the third is vacuous too, for a
reason neither earlier finding covers. **The one-sentence manuscript conclusion prescribed above
therefore no longer holds as written** — see §3.6.

⚠⚠ **Finding 4 is about the replacement for THAT replacement, and the recursion is the point.**
§3.7 concluded that a valid test needs information from outside the tree, and the P3 placement arm
was built the same day to supply it. Finding 4 reports that P3 is vacuous too — it satisfies §3.7's
stated requirement and still measures nothing but candidate **subtree size**, a static property of
the training tree. Its `ANTICIPATES` verdict is **withdrawn** (§4.5); §3.7's requirement is
**necessary but not sufficient** and is amended, not retracted, in §4.7. **Three protocols adopted as
remedies, three vacuous** — which is itself the strongest form of the unifying lesson.

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

## Finding 3 — the leaf parent-edge holdout is vacuous for a FEATURE-FREE TRANSDUCTIVE embedding

§1.7 adopted the leaf parent-edge holdout because Findings 1 and 2 disqualified the alternatives.
It closes the internal-node leak (§1.8) and it is what every §3.4 precedent does. It is still
vacuous here — not because of what it leaks, but because of what it leaves the model to work with.

### 3.1 The protocol as adopted, and the model it is applied to

For a held-out leaf `v` with true parent `p`, the parent edge `(p, v)` is removed from training and
the model must rank `p` among `candidate_pool(strategy="grandparent_children")`
(`src/taxembed/eval/linkpred.py`) — the children of `v`'s grandparent `G`, i.e. `p` **and `p`'s
siblings**.

TaxEmbed is **transductive and feature-free**: a node has no attributes, no text, no sequence. Its
coordinate is a free parameter learned *only* from the closure rows it appears in. The plan records
the transductive property at `docs/plans/2026-09-24-taxembed-p2-heldout-evaluation.md:59` — "removed
taxa have no coordinates and placement would need the pLM" — and the design's response was to keep
the node and remove only its parent edge, so that it still *has* a coordinate. What was never
checked is whether that surviving coordinate can carry the answer.

### 3.2 What a held-out leaf actually retains — the information argument

`scripts/build_p2_split.py:103` exempts held-out rows from visibility thinning:

```python
hide_deep = deep & ~is_heldout_row & (rng.random(len(pairs)) >= visibility)
```

So a held-out leaf keeps its **full** `depth_diff >= 2` ancestry in every arm, and loses exactly one
row: its `depth_diff == 1` parent edge. Confirmed rather than assumed —
`helpers/_p2_curriculum_vs_collapse.py` reports **28,418 / 28,418 (100.00 %) of held-out nodes
present in the training closure as descendants, in BOTH vis00 and vis50**.

⚠ `vis00`/`vis50` do **not** describe the held-out node. Visibility thins the deep rows of every
*other* node; the held-out node's ancestry is exempt either way. `n pairs at dd == 1` is identical
(**469,827**) in both arms for exactly this reason.

The retained rows for `v` are therefore precisely `(G, v), (GG, v), (GGG, v), …` — its ancestor
chain **above** `p`. And every node in that chain is an ancestor of **all** the candidates equally,
because the candidates are by definition the children of `G`. Therefore:

> **No retained closure row distinguishes the true parent from its siblings.** The discriminative
> information is exactly, and only, the edge that was withheld.

This is §1.6's sentence — *"the one fact it needs is exactly the one fact withheld"* — which was
written about Vendrov's trivial baseline. It applies verbatim to the learned model, and that was not
noticed, because §1.6 read it as a statement about a five-line rule rather than about the
information content of the split.

### 3.3 The falsifiable prediction, and its confirmation

If §3.2 is right, whatever an arm scores must come from `v`-independent structure (geometry,
fan-out), never from taxonomy — so **a real tree and a degree-matched shuffle must be
indistinguishable at every epoch**, not merely at the pre-registered read point.

`helpers/_p2_peak_real_vs_control.py`, normalized_rank (lower better, 0.5 = chance for any pool
size), each arm read at its **own best** milestone over all 20 checkpoints:

| pair | real best | @ep | ctrl best | @ep | real − ctrl |
|---|---:|---:|---:|---:|---:|
| vis50_s0 vs degmatch_vis50_s0 | 0.3951 | 80 | 0.3937 | 60 | **+0.0015** |
| vis50_s1 vs degmatch_vis50_s1 | 0.3917 | 90 | 0.3933 | 70 | −0.0015 |
| vis50_s2 vs degmatch_vis50_s2 | 0.3928 | 90 | 0.3950 | 60 | −0.0023 |
| vis00_s0 vs degmatch_vis00_s0 | 0.4997 | 10 | 0.5049 | 30 | −0.0053 |
| vis00_s1 vs degmatch_vis00_s1 | 0.5004 | 20 | 0.5040 | 20 | −0.0036 |
| vis00_s2 vs degmatch_vis00_s2 | 0.5016 | 10 | 0.5021 | 30 | −0.0005 |

Mean vis50 delta = **−0.0008**, against pooled within-arm seed SD of **0.00626** (vis50) and
**0.01056** (degmatch_vis50) from `results/p2_verdict_20260929.json` — an order of magnitude larger
than the effect, and the effect **changes sign across seeds**. The vis00 rows never leave chance in
either arm (best ≈ 0.50, and at epochs 10–30, i.e. before the curriculum admits any row a held-out
node appears in).

**Reading these at their best epoch is a post-hoc rescue and is NOT a result.** It is admissible
here only in the negative direction: it establishes that nothing is being masked by the
epoch-200 reading, so no amendment or re-run can recover a verdict from this design.

### 3.4 Why the leaf-holdout precedents (spec §3.4) do not transfer

TaxoExpan (WWW 2020), Arborist (WWW 2020) and Octet (KDD 2020) all hold out leaves, which is why
§1.7 cited them. But every one of them attaches **features to the held-out node** — surface names,
definitions, corpus embeddings — and predicts its position *from those features*. The held-out
concept still has an input after its edge is removed.

A TaxEmbed leaf has no input but its edges, and §3.2 shows the only informative one is gone. The
unstated invariant behind "hold out leaves" is therefore **"held-out nodes retain a
node-level signal"** — true by construction in all three precedents, false by construction here.
Same clause, opposite effect, third time.

### 3.5 What the array measured instead

Two artifacts of the run, recorded so they are not mistaken for results about the model:

1. **A curriculum confound.** `train_small.py::auto_curriculum_phases(max_depth=37, n_epochs=200)`
   gives phases at epochs **1 / 40 / 80 / 120** with caps `dd<=1`, `dd<=9`, `dd<=18`, all. P2 scores
   the `dd == 1` relation at epoch 200 (`P2_FINAL_EPOCH`), after 80 epochs optimising the global
   structure that competes with it. Every inflection in the measured curve lands on a phase
   boundary: flat at chance through epoch 40 (a held-out node's only `dd == 1` row is its withheld
   parent edge, so it receives **no gradient at all in phase 1**), learning from 40, peak at 80,
   monotone decay to below chance by 200. All 12 runs end worse than their own random
   initialization.
2. **A dominant `v`-independent prior.** The non-embedding `degree_prior` baseline scores
   normalized_rank **0.154** (MRR 0.560, hits@1 0.404) through the same pool and harness. The task
   is thus easily winnable by a prior over *which parent is likely*, while carrying no signal about
   *which parent this leaf has*. That combination — strong marginal prior, zero discriminative
   signal — is the precise sense in which it is vacuous as an overfitting test.

Eliminated by measurement, so they are not available as alternative explanations: the metric
(`helpers/_p2_cosine_vs_poincare.py` — cosine and Poincaré agree to ~0.001 at every checkpoint of
every arm); radius blow-up (`radius_overflow_risk` False throughout, `max_radius` stable at
3.66–3.84); the candidate pool and eval loop (the degree prior's 0.154 through the same harness, and
normalized_rank sitting on exactly 0.500 at initialization); ranking direction
(`rank_of_true_parent` sorts ascending, nearest-first).

⚠ **Separate defect, logged here because it was found on the way:** `chance_mrr_mean` (0.2711)
disagrees with the empirical untrained MRR (0.2457) by ~10 %, while `normalized_rank` sits on its
0.5 floor exactly. Do not quote `chance_mrr_mean` as a floor until it is re-derived. Nothing in this
note depends on it — every cross-tree reading here uses `normalized_rank` (amendment_1).

### 3.6 Consequence: the manuscript sentence changes

The conclusion prescribed at the top of this note — *"we use a leaf parent-edge holdout and a
chance-normalised control because the standard alternatives are vacuous on a taxonomy tree"* —
asserts a working evaluation. It cannot be used as written.

What the paper can say, one sentence, citing back here for "why": **held-out link prediction cannot
test generalisation for a transductive, feature-free taxonomy embedding, because removing a leaf's
parent edge removes the only training row that distinguishes its parent from that parent's
siblings.** The `UNINFORMATIVE` verdict of `results/p2_verdict_20260929.json` stands and should not
be amended or re-run; §3.3 shows no epoch, and therefore no schedule fix, recovers it.

🛑 **This leaves Burkhard's "TaxEmbed: overfitting?" unanswered by P2**, which was P2's whole
purpose. The question splits, and only one half is answerable with what is on disk:
- *Is the headline planted at initialization?* **Yes, and it is already documented** — depth-norm
  init floor 0.957151 vs trained 0.957244. The remedy is to report S_angle (null = 0 by
  construction for random directions; canonical 0.9726 vs init null −0.0015) and to print the floor
  beside any depth-norm number. Local, no GPU. Already `MANUSCRIPT_CORRECTIONS_PENDING.md` #3 plus
  the Task 9 Figure 4 re-plot.
- *Does it generalise or memorise?* **Not answerable by any node-holdout on this model class.** It
  needs an inductive, feature-bearing probe — spec §P4, the pLM showcase, which is Burkhard's own
  suggestion 1. Until that runs, this is a stated limitation, not a result.

### 3.7 The general statement: on a strict tree, EVERY single-edge holdout is trivial or vacuous

Findings 1 and 3 are two halves of one exhaustive argument. Take any single closure row removed from
training, for a transductive feature-free embedding of a strict tree:

- **The row is not in the reduction (`dd >= 2`).** Finding 1: the reduction determines the closure
  exactly (0/0 symmetric difference on every clade checked), so walking the retained parent chain
  recovers it. **Trivially recoverable** — Vendrov's rule scores 100.00 % (§1.4).
- **The row is in the reduction (`dd == 1`, a parent edge) and its child is INTERNAL.** §1.8: the
  child's own descendants keep their `(p, w)` rows, so `p` is the unique child of the grandparent
  ancestral to any surviving `w`. **Trivially recoverable.**
- **The row is in the reduction and its child is a LEAF.** §3.2: the retained rows are the ancestor
  chain above `p`, every member of which is an ancestor of all candidates equally. **No
  discriminative signal.**

Root edges do not exist, so the three cases are exhaustive over node types. **Either the held-out
fact is recoverable from what remains by a rule no model is needed for, or nothing that remains
distinguishes the answer from its alternatives.** The only way out is to remove enough of a node's
rows that it has no coordinate at all — the transductive limit, which the USER ruled out of scope
(`docs/plans/2026-09-24-taxembed-p2-heldout-evaluation.md:59`, `:452`).

⚠ **Stated as an argument plus a measurement, not as a theorem.** "No discriminative row" does not
by itself forbid the global geometry from correlating with the answer; §3.3 is what closes that gap
empirically, and it closes it at every epoch, on three seeds, against a matched control.

**This is why the imports failed and why no fourth protocol will fix it.** Ganea's split works on
WordNet because WordNet is a DAG (537 basic edges beyond a tree's count — §1.5). TaxoExpan, Arborist
and Octet work because their held-out concepts keep text features (§3.4). Both escape routes are
properties the importing paper had and this data does not. **A held-out test of a taxonomy embedding
needs information from outside the tree — later NCBI releases (spec §P3), protein sequence
(spec §P4), or divergence times (spec §4.5). Inside the tree there is nothing left to hold out.**

🛑 **AMENDED BY FINDING 4 (2026-09-29, same day) — this condition is NECESSARY BUT NOT SUFFICIENT,
and the first of the three routes named above has already failed it.** The P3 placement arm drew its
labels from an NCBI release three months after training, satisfying the requirement exactly, and
still measured nothing but candidate **subtree size** — a static property of the training tree
(§4.4–§4.5, decomposition 94–113 %). The sentence above is kept as written because its
inside-the-tree exhaustion stands; **read §4.7 with it.** Outside information must enter the
**answer**, not merely the **question**: a label can be unseen and still be predictable from a cheap
statistic of the training data.

---

## Finding 4 — outside-the-tree information is NECESSARY BUT NOT SUFFICIENT: the placement test measured subtree size

### 4.1 The protocol, and what it was built to escape

§3.7 closes by naming the only way out of the three-case exhaustion: *"a held-out test of a taxonomy
embedding needs information from outside the tree — later NCBI releases (spec §P3), protein sequence
(spec §P4), or divergence times (spec §4.5)."* The **P3 placement arm** was built the same day to be
exactly that. It takes its labels from the 2026-09-01 NCBI release, three months after the
2026-06-09 training snapshot, and asks:

> For a taxon `v` that NCBI later **moved** from parent `p_old` to parent `p_new`, does the shipped
> embedding already sit nearer `p_new` than the equally-local alternatives NCBI could have chosen?

Primary: normalized_rank of `p_new` in a locality-matched pool — nodes at `depth(p_new)` inside the
subtree of `L = LCA(p_old, p_new)`, after amendment 2 excluding `p_old`'s entire child branch of `L`.
Chance 0.5, lower better. It returned **0.4400, CI95 [0.4214, 0.4588], n = 1,241**, with four
pre-registered gates passing, and was read as `ANTICIPATES`
(`results/p3_placement_result_20260929.json`).

**It satisfies §3.7's stated requirement and it is still vacuous.** That is Finding 4, and it is not
covered by Findings 1–3.

Evidence line, all run 2026-09-29 against the shipped artifact, read-only, no GPU:
- `helpers/_p3_confound_diagnostics.py` → `results/p3_confound_diagnostics_20260929.json`
- `helpers/_p3_combinatorial_baselines.py` → `results/p3_combinatorial_baselines_20260929.json`
- `helpers/_p3_size_residual_control_v3.py` → `results/p3_size_residual_control_v3_20260929.json`
- superseding verdict `results/p3_placement_result_v2_20260929.json`; the full record is
  `p3_amendment_3_20260929` inside `results/p3_placement_preregistration.json`.

### 4.2 The prescribed control could not have failed

The result closed with one uncontrolled residual and a prescribed fix: *reclassifications are local,
the pool excludes `p_old`'s branch but not branches adjacent to it, so **match each pseudo-parent on
tree distance from `p_old`.*** 

**Falsifiable prediction, recorded before the measurement** (`_p3_confound_diagnostics.py` docstring,
P1): the matching cannot move a candidate, because the pool is *depth-homogeneous* — every `q` sits
at `depth(p_new)` — and, after amendment 2, *branch-homogeneous*: no `q` lies in `p_old`'s child
branch of `L`, so `LCA(p_old, q) == L` for every `q`. Hence

```
path(p_old, q) = (depth(p_old) − depth(L)) + (depth(p_new) − depth(L))
```

which contains no term varying with `q`.

**Confirmed, 1,241 / 1,241 pools (100.00 %)**, and the constant equals `path(p_old, p_new)` in
1,241 / 1,241. Implementing the prescription as written would have produced a clean-looking control
**incapable of failing** ([[feedback_a_check_that_could_not_have_failed_is_not_evidence]]). The
residual it pointed at is real; the instrument was not.

### 4.3 The moved taxon is inert — P3 was never a held-out-taxon test

Path length is not what drives the statistic; hyperbolic position is. Sibling branches of `L` are
equidistant in path length but not in the embedding. The decisive instrument is therefore a
**substitution**, not a matching: rank the *identical* pool from `p_old` instead of from `v`.

| query point | mean normalized_rank | CI95 |
|---|---:|---|
| `v`, the moved taxon (the PRIMARY) | 0.4400 | [0.4214, 0.4588] |
| `p_old`, its old parent, substituted | 0.4444 | [0.4257, 0.4628] |
| **paired delta `v − p_old`** | **−0.0044** | **[−0.0145, +0.0062]** |

The paired CI contains zero. The whole effect is `0.5 − 0.4400 = 0.0600`; **`p_old` alone delivers
0.0556 of it — 92.7 %** — and the upper bound caps any true contribution from `v` at ~24 %.

⇒ **The moved taxon's own coordinate contributes nothing measurable.** P3 is not a test of a held-out
taxon at all; it is a statement about the pair `(p_old, p_new)`. Every sentence attributing the
signal to the reclassified taxon — including the verdict sentence as originally drafted — is false.

### 4.4 A one-line tree statistic beats the embedding outright

If the discrete old tree cannot separate candidates by depth (matched) or tree distance (constant,
§4.2), can it separate them by any *other* cheap property? This is the question P2's `degree_prior`
baseline answered for that array, and it is asked here for the same reason.

On the identical pools, n = 1,241:

| score | mean | CI95 |
|---|---:|---|
| EMBEDDING from `v` (the primary) | 0.4400 | [0.4214, 0.4588] |
| EMBEDDING from `p_old` | 0.4444 | [0.4257, 0.4628] |
| **B0 random** (harness check) | **0.4937** | [0.4761, 0.5110] |
| **B1 subtree size, larger first** | **0.3146** | [0.2982, 0.3313] |
| B2 `\|taxid − taxid(p_old)\|`, nearer first | 0.5571 | [0.5392, 0.5747] |
| B3 `n_children`, more first | 0.3188 | [0.3020, 0.3359] |

**Paired, `emb_pold − B1` = +0.1298, CI95 [+0.1107, +0.1488].** Positive, and more than twice the
embedding's entire effect over chance. **"Pick the largest candidate branch" predicts NCBI's choice
far better than the geometry does.** The mechanism is plain: NCBI moves taxa into big, actively
curated groups — the target's mean within-pool size rank is **0.3218** against a background of
**0.5013** (0 = largest) — and `p_new` is a direct sibling of `p_old` in **80.2 %** of moves.

B2 deserves a line because its prediction was refuted in the *reassuring* direction: I predicted ~0.5
on the argument that sibling-branch ordering around `L` is set by random initialization rather than
accession order, and got 0.5571 — `p_new` is *further* in taxid than a pool draw. **There is no
accession-order layout artifact.** Its mirror (0.4429) is nonetheless level with the embedding: a
taxid subtraction is as predictive here as the trained geometry.

### 4.5 Conditioned on subtree size, nothing remains

Two size-**matching** designs were built and both failed their own pre-declared gates, for one shared
reason recorded here because it generalises: **subtree size is heavy-tailed and `p_new` lives in the
upper tail**, so no pool can be constructed in which size does not matter.

- Fixed log2 bands (`_p3_size_matched_control.py`, DEPRECATED): size still predicted **inside** the
  band at every width — 0.4746 [0.455, 0.495] even at ±0.5 log2 (~1.4×).
- k-nearest-in-size pools (`_p3_size_matched_control_v2.py`, DEPRECATED): **worse**, 0.2445 / 0.2575 /
  0.2907 at k = 4/8/16. The k nodes nearest in log-size to an upper-tail target are drawn
  asymmetrically *from below*, so the target stays among the largest in its own "matched" pool. That
  also invalidates that helper's matched-procedure null, whose pseudo-target is drawn uniformly and
  is therefore typically small; its `BEYOND SIZE` readings (−0.0792, −0.0356) are **withdrawn**.

v3 therefore **conditions** on size instead of matching on it. For every candidate compute two
within-pool normalized ranks — `r_s` by subtree size (larger first) and `r_e` by Poincaré distance
from `p_old` — estimate the calibration curve `E[r_e | r_s]` by binned mean over the **166,852
non-target** candidates (weighted `1/n_q`, so the curve is fit under the same weighting the
query-averaged statistic uses), and test each target's residual `r_e(p_new) − curve(r_s(p_new))`.

| bins | G1 null residual (must contain 0) | G2 curve spread (>0.05) | **G3 target residual, from `p_old`** | from `v` |
|---:|---|---|---|---|
| 10 | +0.0010 [−0.0026, +0.0045] **PASS** | 0.3067 **PASS** | **−0.0031 [−0.0203, +0.0137]** | −0.0075 |
| 20 | +0.0011 [−0.0025, +0.0044] **PASS** | 0.3626 **PASS** | **+0.0019 [−0.0149, +0.0185]** | −0.0025 |
| 40 | +0.0009 [−0.0026, +0.0042] **PASS** | 0.4068 **PASS** | **+0.0076 [−0.0091, +0.0241]** | +0.0032 |

Both harness gates pass at all three binnings; **G3 contains zero at all three and its sign changes
across them.** The `v` column tracks the `p_old` column to ~0.004, independently re-confirming §4.3.

**The decomposition is the finding.** Of the 0.0556 raw effect over chance, candidate subtree size
alone accounts for **0.0525 / 0.0575 / 0.0631** at 10/20/40 bins — **94 % / 103 % / 113 %.** Nothing
is left over.

> **The P3 placement signal is a lossy encoding of subtree size.** Conditional on how large a
> candidate subtree is, the embedding carries no information about where NCBI will move a taxon.

`ANTICIPATES` is withdrawn; the superseding verdict is **UNINFORMATIVE**
(`results/p3_placement_result_v2_20260929.json`). The original verdict file is kept unedited as the
record of what amendment 2 produced (Rule 5: verdicts are append-only at the file level).

### 4.6 The gate that reversed the verdict — a method lesson worth more than the result

The **first** v3 run returned G3 = **−0.0325 / −0.0340 / −0.0269 with CIs excluding zero** — i.e.
`BEYOND SIZE`, a positive, reportable result. Its null gate G1 passed: +0.0069, CI95
[−0.0101, +0.0240].

It passed because it used **one** pseudo-target per query, giving it a CI half-width of **0.017
against an effect of 0.034**. Raising the replicate count to 20 — **for power, with nothing looking
wrong** — made G1 **FAIL** at +0.0122, CI95 [+0.0081, +0.0163], exposing a **weighting mismatch in
the estimator**: the calibration curve was fit candidate-weighted while the statistic is
query-weighted, so large pools dominated the curve. Fitting with `1/n_q` weights fixed it, G1 now
passes at +0.0010 — **and the verdict flipped from positive to null.**

🧨 **A gate whose CI half-width is comparable to the effect it certifies is not certifying it.** This
is distinct from §4.2's failure mode: that control *could not* fail; this one *could* and merely
lacked the resolution to. **Report a gate's resolution beside its verdict** — `PASS` alone is not a
result, `PASS, resolution ±0.004, effect 0.034` is — and keep resolution to ≲⅓ of the effect. Banked
as `feedback_a_gate_must_resolve_finer_than_the_effect_it_certifies`.

🎯 The correction moved the headline the **unwelcome** way: a publishable positive became a null.
That is the evidence it was made for the right reason, and it is the contrast the P2 diagnosis drew —
P2 ran twelve arms for 200 epochs below chance with no control able to reveal why; here three
successive gates caught a vacuous instrument, two mis-specified controls and a bug in the estimator,
all before a manuscript sentence was written.

### 4.7 The general statement: §3.7 is necessary but not sufficient

§3.7 concluded that a held-out test of a taxonomy embedding **needs information from outside the
tree**. P3 had it — labels drawn from a release three months after training — and still measured
nothing but a static property of the training tree. So the requirement must be strengthened:

> **Outside-the-tree information must enter the ANSWER, not merely the QUESTION.** A protocol may
> source its labels from outside and still be vacuous, if the discriminative content of those labels
> is predictable from a cheap statistic of the training data alone. The label being unseen is not the
> same as the label being unpredictable.

⚠ This does **not** retract §3.7 — its three-case exhaustion over *inside*-the-tree holdouts stands,
and outside information remains necessary. It adds the second condition §3.7 did not state.

**The operational test, applicable before a protocol is run rather than after:** name the cheapest
statistic of the training data that could predict the outside-sourced label, compute it on the same
pools, and report it beside the embedding. If it wins, the protocol is measuring that statistic. For
P3 that statistic was subtree size, and it won by 0.1298.

Applied to the two remaining escape routes, the news is better than it looks:
- **§4.5 TimeTree already has the right shape.** It specifies *"report Spearman(embedded distance,
  divergence time) **beside** Spearman(NCBI path length, divergence time)"* — the cheap tree
  statistic is named, computed and reported by construction. It is, further, the only remaining
  instrument that can touch steelman (ii), *embedded distance tracks NCBI convention rather than
  relatedness*, which §4.5 already notes no NCBI-internal test can refute. Finding 4 makes that
  sharper, not weaker: P3 was the last NCBI-internal candidate.
- **§P4 already has its baseline, and is currently losing to it.** C5 records the bridge at 0.6029
  against a majority-class baseline of 0.6448. Finding 4 says that comparison is not a formality to
  be explained away; it is the same test P3 failed.

---

## The unifying lesson

All four findings are the same shape, and it is the point of this note, not incidental framing: **a
protocol or control imported from another paper encodes assumptions about that paper's data.**

- Ganea's split assumes the transitive reduction does not, by itself, determine the whole closure —
  true on WordNet's DAG (537 nodes with a second immediate parent), false on a taxonomy tree (every
  node has exactly one).
- RandomDAG assumes that preserving structure (depth, and so total pair count) preserves task
  difficulty — true only if fan-out is close to uniform across the structure being preserved, false
  for a heavy-tailed real taxonomy where a handful of genera hold most of the fan-out.
- The leaf holdout assumes a held-out node still carries a node-level signal after its edge is
  removed — true for TaxoExpan/Arborist/Octet, whose held-out concepts keep names, definitions and
  corpus embeddings, false for a feature-free transductive embedding whose node is nothing but the
  rows it appears in.
- The placement test assumes that a label sourced from outside the training data is therefore not
  predictable from inside it — true wherever the outside label is independent of the training
  structure, false for NCBI reclassification, whose destination is largely determined by candidate
  **subtree size**, a statistic computable from the training tree alone and one that beats the
  embedding outright (0.3146 vs 0.4444, paired +0.1298).

In all three cases the imported object was **checked against a real invariant, and it held** — the
reduction really does encode the closure (mathematically, by definition of transitive reduction); the
rewiring really does preserve depth and pair count (proved, unit-tested); the leaf restriction really
does close the internal-node leak (§1.8, and every cited precedent does the same). That correctness is
exactly what made the *unchecked* invariant easy to miss: a component that visibly keeps one promise
reads as trustworthy, and nobody asks what else changed. **Ask what a transformation changes, not only
what it preserves.** A protocol audit that stops at "does it preserve the thing its docstring claims"
will pass all three of these and still ship a vacuous evaluation.

🧨🧨 **Finding 4 closes the recursion, and it is the hardest version of the lesson in this note.**
§1.7 replaced the rejected Ganea split; Finding 3 killed it. §3.7 prescribed what the *next*
replacement must have — outside-the-tree information — and P3 was built to that prescription the same
day; Finding 4 killed that too, for a condition §3.7 did not state. **Three remedies, three
vacuities**, each one adopted *because* the previous failure had been analysed carefully. The
analysis was not the problem; the analyses were right as far as they went. What recurs is that a
diagnosis names the condition that broke *last time* and is then read as a specification of
sufficiency ([[feedback_a_guard_carried_across_a_boundary_is_a_new_guard]]). **A remedy derived from
a failure inherits exactly the coverage of that failure, and no more** — so the question to ask of
any replacement protocol is not "does it fix what broke?" but "what is the cheapest thing that could
pass it?". For P3 that question had a one-line answer, `subtree size`, available before a single GPU
hour or even a single distance was spent.

🧨 **Finding 3 sharpens this into the harder lesson: a fix for a vacuity is itself an import, and
inherits the same duty.** Findings 1 and 2 were diagnosed carefully and the replacement adopted in
§1.7 was justified by four cited precedents — and it still shipped vacuous, because the precedents
were checked for *what they hold out* (leaves, closing §1.8's leak) and never for *what their
held-out nodes keep* (features). **The audit a rejected design receives must also be given to the
design that replaces it**; being the considered response to a known failure is not evidence, and a
protocol that arrives as a remedy is the one least likely to be re-audited.

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

Finding 3's helpers were run 2026-09-29 against the banked array (submodule `e04c478`), read-only:

| helper | invocation | numbers matched |
|---|---|---|
| `helpers/_p2_cosine_vs_poincare.py` | (default, all 9 `artifacts/p2_scoring/*.json`) | cosine vs Poincaré agree to ~0.001 at all 20 milestone + 5 roll checkpoints of all 12 runs; `radius_overflow_risk` False and `max_radius` 3.66–3.84 throughout; the epoch-10/40/80/120 inflections |
| `helpers/_p2_curriculum_vs_collapse.py` | (default, `vis00`/`vis50` seed 0 splits) | phases 1/40/80/120 at caps 1/9/18/None, **asserted equal to the production `train_small.py::auto_curriculum_phases`**; `n pairs at dd==1` = 469,827 identical in both arms; 28,418/28,418 held-out nodes present as descendants in both arms; 1,071,269 vs 6,442,714 pairs |
| `helpers/_p2_peak_real_vs_control.py` | (default, milestone series) | the six best-epoch rows of §3.3 and their epochs |

Finding 4's helpers were run 2026-09-29 against the shipped release artifact
(`release/taxembed-cellular-v1/`) and the 2026-07-01 / 2026-09-01 taxdump snapshots, read-only, CPU
only:

| helper | invocation | numbers matched |
|---|---|---|
| `helpers/_p3_confound_diagnostics.py` | (default) | P1 constancy 1,241/1,241 and target-equality 1,241/1,241; rank from `v` 0.4400 and from `p_old` 0.4444; paired −0.0044 [−0.0145, +0.0062]. **Imports its pool construction from `_p3_placement_score.py` rather than re-implementing it, and aborts unless the primary reproduces 0.4400 ±0.005** — so the pool measured is the pool that was scored |
| `helpers/_p3_combinatorial_baselines.py` | (default) | B0 0.4937 / B1 0.3146 / B2 0.5571 / B3 0.3188 and their CIs; paired `emb_pold − B1` +0.1298 [+0.1107, +0.1488]; direct-sibling fraction 80.2 % |
| `helpers/_p3_size_residual_control_v3.py` | (default, bins 10/20/40, `R_NULL=20`) | G1 +0.0010/+0.0011/+0.0009, G2 0.3067/0.3626/0.4068, G3 −0.0031/+0.0019/+0.0076 and their CIs; target mean size rank 0.3218 vs background 0.5013; 166,852 background candidates; the 94/103/113 % decomposition |
| `helpers/_p3_size_matched_control.py` · `_v2.py` | (default) | §4.5's withdrawn figures only — both **DEPRECATED in their own docstrings**, kept because the reason each failed is itself the finding |
| `helpers/_p3_record_amendment_3.py` | (default) | asserts all 13 pre-existing pre-registration keys survive byte-identically and that the superseded verdict file still reads `ANTICIPATES` before writing anything |

⚠ **§4.6's reversal is reproducible from the helper as it stands, via two INDEPENDENT one-line
knobs** — kept separable deliberately, so the claim can be checked rather than believed:

| curve weighting | `R_NULL` | G1 | G3 | what it was |
|---|---:|---|---|---|
| unweighted bin mean (drop `weights=bg_w`) | 1 | +0.0069 [−0.0101, +0.0240] **PASS** | **−0.0325** | run 1 — the withdrawn positive |
| unweighted bin mean | 20 | +0.0122 [+0.0081, +0.0163] **FAIL** | −0.0325 | run 2 — the gate, once it could resolve |
| `weights=bg_w` (`1/n_q`) | 20 | +0.0010 [−0.0026, +0.0045] **PASS** | **−0.0031** | run 3 — as shipped, §4.5 |

The curve weighting alone moves **G3** (the verdict); `R_NULL` alone moves **G1's resolution** and
therefore whether the bias is visible at all. Run 1 is positive only because both knobs were in the
wrong position simultaneously, and only the second one was load-bearing for the answer.

⚠ `rank_by`'s unbiasedness was verified synthetically before any band result was read — 20,000
replicates at each of 15 pool sizes from 2 to 120, all in 0.4941–0.5036 — which is what established
that v1's two failing random controls were noise rather than a rank bug
([[feedback_a_verifier_you_wrote_shares_your_blind_spot]]).

⚠ `_p2_curriculum_vs_collapse.py` pins its local copy of `auto_curriculum_phases` against the
imported production function over five `max_depth` values and fails if they diverge — the phase
boundaries in §3.5 are therefore the trainer's own, not a re-implementation
([[feedback_a_verifier_you_wrote_shares_your_blind_spot]]).

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
- Split builder: `src/taxembed/eval/p2_split.py` (`eligible_nodes`, `leaf_mask`, `select_holdout`);
  visibility thinning and the held-out-row exemption at `scripts/build_p2_split.py:103`.

**Finding 4:**
- Pre-registration: `results/p3_placement_preregistration.json`. 🛑 **Read
  `p3_amendment_3_20260929` first** — it is self-contained, carries all four sub-findings with their
  measurements, both superseded matching attempts with their withdrawn numbers, and the integrity
  note on the gate that reversed the verdict. Amendments 1 and 2 precede it.
- Verdict: `results/p3_placement_result_v2_20260929.json` (**UNINFORMATIVE**). ⛔ **Do not quote
  `results/p3_placement_result_20260929.json`** — it is the superseded `ANTICIPATES` record, kept
  unedited on purpose (Rule 5).
- Spec: §P3 (placement arm), §4.5 (TimeTree — the remaining instrument, and the one that already
  carries its own cheap-statistic baseline by construction), §P4/C5 (the pLM showcase, currently
  0.6029 against a 0.6448 majority-class baseline — the same test P3 failed).
- Scorer and pool construction: `helpers/_p3_placement_score.py` (amendment 2's branch exclusion at
  `:196-212`); machinery tests `tests/eval/test_p3_placement_machinery.py` (13, 0.56 s).
- Session log: `SpeciesEmbedding/docs/sessions/2026-09-29-taxembed-p3-tree-distance-control.md`.
- Method lesson banked as
  `feedback_a_gate_must_resolve_finer_than_the_effect_it_certifies` (indexed in `MEMORY.md`).
- Finding 3's evidence: `results/p2_verdict_20260929.json` (verdict `UNINFORMATIVE`),
  `artifacts/p2_scoring/*.json`, candidate pool at `src/taxembed/eval/linkpred.py::candidate_pool`
  (`grandparent_children`), curriculum at `train_small.py::auto_curriculum_phases`, training recipe
  at `scripts/p2_lrz_train.sh`. Session log:
  `SpeciesEmbedding/docs/sessions/2026-09-29-taxembed-p2-below-chance-diagnosis.md`.
- RandomDAG: `src/taxembed/eval/randomdag.py` (`randomize_parents`),
  `tests/eval/test_randomdag.py`.
- Manuscript target (not versioned): `manuscript/manuscript.v3_draft.md`, via
  `docs/MANUSCRIPT_CORRECTIONS_PENDING.md`.
