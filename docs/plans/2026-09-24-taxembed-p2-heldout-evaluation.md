# TaxEmbed P2 — Held-Out Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a held-out link-prediction evaluation that answers Burkhard's "TaxEmbed: overfitting?" challenge with a number measured on relations the model never saw during training.

**Architecture:** The split is a **data artifact, not a trainer feature**. A builder masks a `TrainingPairs` closure and writes a new `.npz` in the identical schema; the trainer then runs completely unmodified via `taxembed.cli.main train --file <split.npz>`. This keeps 1,308 lines of load-bearing training code untouched, and — critically — guarantees the negative sampler's ancestry index is built from the *visible* edges only, so held-out relations cannot leak into training as ancestry knowledge. Evaluation is three new pure-function modules under `src/taxembed/eval/` plus one scoring driver, mirroring the existing `score_recipe_checkpoints.py` shape.

**Tech Stack:** Python 3.13, numpy, PyTorch (trainer only), pytest. No new dependencies.

**Spec:** `docs/specs/2026-08-11-taxembed-overfitting-and-plm-showcase-design-v3.md` §P2, §3.1, §3.2, §3.4, §3.5, §3.6.
**Session that produced the redesign:** `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/docs/sessions/2026-09-24-taxembed-p2-heldout-build.md`

---

## Why this plan departs from the spec — read this before Task 1

🛑 **§P2.1's inherited Ganea split is degenerate on our data. Do not implement it.**

Measured 2026-09-24 by `helpers/p2_check_closure_is_a_tree.py`, before any code was written:

- All six NCBI clade closures are strict **trees** — parents-per-node histogram is `{1: N}` in every
  one, and `n_basic == n_nodes − 1` exactly in every one. Independently confirmed from the six
  `*_manifest.json` files, where `edges == nodes − 1`.
- Expanding the parent-child edges **reproduces the stored closure bit-for-bit** — 0 pairs in either
  direction of the symmetric difference, on echinodermata (34,938), mollusca (278,695) and metazoa
  (11,840,908).

⇒ §P2.1 says *keep the transitive reduction always in training*. On a tree the reduction determines
every ancestor-descendant pair, so §3.2's **mandatory Vendrov trivial baseline scores 100.00 %** on
any held-out non-basic edge. No learned number can beat a five-line closure computation. The
experiment could not have produced evidence for the model.

**This is a dataset difference, not a misreading of Ganea, and the spec's own numbers prove it.**
§3.1 records Ganea's WordNet as 82,114 nodes / 661,127 edges / 578,477 non-basic ⇒ basic = **82,650**
against a tree's 82,113. Those **537 extra basic edges are multiple inheritance**: WordNet is a DAG.
That is why Vendrov measured 88.2 % there. The trap is specifically *"always keep the reduction"* —
Vendrov did not keep it, which is why their baseline was beatable at all.

✅ **§P2.3 is the sound instrument, and both of its prose figures reproduce exactly.** Measured by
`helpers/p2_eligible_node_definitions.py` against 14 candidate rules:

- per-node depth quartiles over basic edges = **Q1 11.0 / median 19.0 / Q3 28.0** — the spec's
  numbers. (Over *all* closure pairs it is 18/25/31. The two differ; we use per-node.)
- `depth ∈ [11, 28] inclusive` ⇒ **593,576 nodes, delta +0, exact.** The 13 other rules missed by
  29,168 to 508,586. "Eligible" means the **interquartile depth band**, which is what §P2.3's
  contrast with *"not HiG2Vec's 6-11"* was signalling.
- 10 % of 593,576 = 59,357 ⇒ **one test link per node** = a **parent-edge** holdout.

⚠ **And it must be restricted to leaves.** Removing only `(p, v)` for an *internal* `v` does not hide
`p`: `v`'s descendants `w` keep their `(p, w)` edges, so `p` is recoverable as the unique child of
`v`'s grandparent that is an ancestor of `w`. That is why every §3.4 precedent holds out leaves —
TaxoExpan 20 % of leaf concepts, Arborist 15 % of leaf nodes, Octet 64/16/20 over leaf nodes.

Of the 593,576 band-eligible nodes, **501,037 are leaves** (84.4 %) ⇒ **50,104 test links at 10 %**.

**USER decisions already taken (2026-09-24), do not re-litigate:**
- Held-out **edges only**. The strong form (retrain without a clade) is **not** in scope — the
  embedding is transductive, so removed taxa have no coordinates and placement would need the pLM
  bridge, which currently sits below the majority-class baseline (0.6029 vs 0.6448).
- Closure-visibility sweep is **two endpoints, 0 % and 50 %**, not 0/10/25/50 %.

## Global Constraints

- **Rule 16 — never submit untested code to LRZ.** Every `sbatch` is preceded by `bash -n` on the job
  script, `python -m py_compile` on every module it imports, a **flag-reality check** against
  `taxembed.cli.main train --help`, and a passing smoke on the cheap partition. Copy the mechanism
  verbatim from `scripts/task8_lrz_fixed_sampler_smoke.sh:37-49`.
- **Rule 18 — no training-class compute on the Mac.** Split building, unit tests and scoring of a
  single small checkpoint are local; anything with an epoch loop goes to LRZ.
- **Pre-registration is written and committed BEFORE any array is submitted**, following
  `results/objective_integrity_delta_preregistration.json`. Amend only by adding a dated block;
  never edit a frozen one.
- **Gates must be able to fail, and must fail for the right reason.** Task 9's gate (b) could not
  tell a *converged* arm from a *never-trained* one. Every gate in Task 7 carries a test that it
  fires on the lesion **and** a test that it stays silent on the healthy-but-unusual case.
- **Fixtures must jitter.** Pinning a dead arm's statistic to exactly `0.0` gives it zero variance,
  which collapses any `k × jitter` threshold to zero and makes the gate unfailable.
- **Shell hygiene:** no compound Bash commands — no pipes, `&&`, `;`, heredocs, or
  `echo`/`printf` redirection into files. Commit messages go via
  `Write` → `/tmp/<topic>_commit_msg.txt` → `git commit -F`.
- **Scale:** production is `metazoa_33208_clean` (498,246 nodes / 11,840,908 pairs). Smoke is
  `mollusca_6447_clean` (32,017 nodes / 278,695 pairs). Both are already on disk under `data/taxopy/`.
- **Venv python (absolute):**
  `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python`

## File Structure

| path | responsibility |
|---|---|
| `src/taxembed/eval/p2_split.py` | **new** — pure functions: leaf detection, band eligibility, holdout selection, leak assertion |
| `src/taxembed/eval/baselines.py` | **new** — Vendrov trivial rule, sibling chance, majority-parent. §P4.1 also wants this file; build the link-prediction half now |
| `src/taxembed/eval/linkpred.py` | **new** — candidate generation, filtered ranking, MR / MRR / Hits@k, stratification |
| `src/taxembed/eval/randomdag.py` | **new** — depth-preserving closure randomisation (§P2.4) |
| `scripts/build_p2_split.py` | **new** — CLI writing train/val/test `.npz` + manifest JSON with md5s |
| `scripts/score_p2_linkpred.py` | **new** — scoring driver over checkpoint globs, mirrors `score_recipe_checkpoints.py:121-131` |
| `results/p2_heldout_preregistration.json` | **new** — frozen before any run |
| `scripts/p2_lrz_train.sh`, `scripts/p2_lrz_train_smoke.sh`, `scripts/p2_lrz_score.sh` | **new** — LRZ jobs, Task 8/9 pattern |
| `docs/FINDING_ganea_split_degenerate_on_trees.md` | **new** — the methodological write-up |
| `tests/eval/test_p2_split.py`, `test_baselines.py`, `test_linkpred.py`, `test_randomdag.py` | **new** — unit tests |

Nothing under `train_small.py`, `train_hierarchical.py` or `cli/main.py` is modified. That is a
deliberate property of the design, not an oversight.

---

### Task 1: Split primitives — eligibility and leak-freedom

**Files:**
- Create: `src/taxembed/eval/p2_split.py`
- Test: `tests/eval/test_p2_split.py`

**Interfaces:**
- Consumes: `taxembed.eval.subtree.parent_from_closure(ancestor_idx, descendant_idx, depth_diff, n_nodes) -> np.ndarray`, `taxembed.eval.subtree.euler_intervals(parent) -> (tin, tout)`, `taxembed.utils.training_pairs.TrainingPairs`
- Produces:
  - `leaf_mask(parent: np.ndarray) -> np.ndarray` (bool, length n)
  - `depth_from_closure(descendant_idx, descendant_depth, ancestor_idx, ancestor_depth, n_nodes) -> np.ndarray` (int64, length n)
  - `eligible_nodes(parent, depth, band=(11, 28), leaves_only=True) -> np.ndarray` (int64 node indices, sorted)
  - `select_holdout(eligible, frac_test=0.10, frac_val=0.0, seed=0) -> dict` with keys `test`, `val`, `train` (int64 arrays, disjoint, union == eligible)
  - `parent_edge_mask(pairs: TrainingPairs, held_out: np.ndarray) -> np.ndarray` (bool, True for rows to REMOVE)

- [ ] **Step 1: Write the failing tests**

```python
# tests/eval/test_p2_split.py
import numpy as np
import pytest

from taxembed.eval.p2_split import (
    depth_from_closure,
    eligible_nodes,
    leaf_mask,
    parent_edge_mask,
    select_holdout,
)
from taxembed.utils.training_pairs import TrainingPairs


def tiny_tree():
    """
    0 (root, depth 0)
    |- 1 (depth 1) -- internal
    |  |- 3 (depth 2) -- leaf
    |  |- 4 (depth 2) -- leaf
    |- 2 (depth 1) -- leaf
    """
    parent = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    depth = np.array([0, 1, 1, 2, 2], dtype=np.int64)
    return parent, depth


def tiny_pairs():
    """Full closure of tiny_tree, as TrainingPairs."""
    anc = np.array([0, 0, 0, 0, 1, 1], dtype=np.int32)
    dsc = np.array([1, 2, 3, 4, 3, 4], dtype=np.int32)
    ad = np.array([0, 0, 0, 0, 1, 1], dtype=np.int16)
    dd_ = np.array([1, 1, 2, 2, 2, 2], dtype=np.int16)
    ddiff = (dd_ - ad).astype(np.int16)
    return TrainingPairs(
        ancestor_idx=anc, descendant_idx=dsc, depth_diff=ddiff,
        ancestor_depth=ad, descendant_depth=dd_,
        ancestor_taxid=anc.copy(), descendant_taxid=dsc.copy(),
    )


def test_leaf_mask_marks_exactly_the_childless_nodes():
    parent, _ = tiny_tree()
    assert leaf_mask(parent).tolist() == [False, False, True, True, True]


def test_root_is_not_a_leaf_even_though_it_self_parents():
    parent, _ = tiny_tree()
    assert not leaf_mask(parent)[0]


def test_depth_from_closure_recovers_every_node_depth():
    pairs = tiny_pairs()
    depth = depth_from_closure(
        pairs.descendant_idx, pairs.descendant_depth,
        pairs.ancestor_idx, pairs.ancestor_depth, n_nodes=5,
    )
    assert depth.tolist() == [0, 1, 1, 2, 2]


def test_eligible_nodes_applies_band_and_leaf_restriction():
    parent, depth = tiny_tree()
    # band [2, 2] keeps only depth-2 nodes; both are leaves
    assert eligible_nodes(parent, depth, band=(2, 2)).tolist() == [3, 4]
    # band [1, 2] with leaves_only drops internal node 1, keeps leaf 2
    assert eligible_nodes(parent, depth, band=(1, 2)).tolist() == [2, 3, 4]
    # without the leaf restriction node 1 returns
    assert eligible_nodes(parent, depth, band=(1, 2), leaves_only=False).tolist() == [1, 2, 3, 4]


def test_eligible_nodes_never_includes_the_root():
    parent, depth = tiny_tree()
    assert 0 not in eligible_nodes(parent, depth, band=(0, 99), leaves_only=False).tolist()


def test_select_holdout_partitions_eligible_exactly_and_is_seed_stable():
    eligible = np.arange(100, dtype=np.int64)
    a = select_holdout(eligible, frac_test=0.10, frac_val=0.10, seed=0)
    b = select_holdout(eligible, frac_test=0.10, frac_val=0.10, seed=0)
    c = select_holdout(eligible, frac_test=0.10, frac_val=0.10, seed=1)

    assert len(a["test"]) == 10 and len(a["val"]) == 10 and len(a["train"]) == 80
    union = np.concatenate([a["test"], a["val"], a["train"]])
    assert np.array_equal(np.sort(union), eligible)      # partition, nothing lost
    assert len(np.unique(union)) == 100                  # and nothing duplicated
    assert np.array_equal(a["test"], b["test"])          # same seed, same split
    assert not np.array_equal(a["test"], c["test"])      # different seed, different split


def test_parent_edge_mask_removes_only_the_dd1_row_of_held_out_nodes():
    pairs = tiny_pairs()
    mask = parent_edge_mask(pairs, held_out=np.array([3], dtype=np.int64))
    # only the (1 -> 3) dd==1 row is marked
    removed = [(int(pairs.ancestor_idx[i]), int(pairs.descendant_idx[i]))
               for i in np.flatnonzero(mask)]
    assert removed == [(1, 3)]


def test_parent_edge_mask_leaves_the_grandparent_edge_intact():
    """The held-out node must keep a coordinate: its dd>=2 ancestry stays in training."""
    pairs = tiny_pairs()
    mask = parent_edge_mask(pairs, held_out=np.array([3], dtype=np.int64))
    kept = pairs[~mask]
    assert (0, 3) in list(zip(kept.ancestor_idx.tolist(), kept.descendant_idx.tolist()))


def test_a_held_out_leaf_parent_is_NOT_recoverable_from_the_retained_closure():
    """The load-bearing property. Node 3 is a leaf; removing (1,3) must hide parent 1."""
    pairs = tiny_pairs()
    mask = parent_edge_mask(pairs, held_out=np.array([3], dtype=np.int64))
    kept = pairs[~mask]
    reachable_to_3 = {int(a) for a, d in zip(kept.ancestor_idx, kept.descendant_idx) if d == 3}
    assert 1 not in reachable_to_3


def test_the_SAME_check_FAILS_for_an_internal_node_the_negative_control():
    """
    A check that could not have failed is not evidence. Holding out internal node 1's
    parent edge does NOT hide parent 0, because 1's descendants keep their (0, 3) and
    (0, 4) edges. This is exactly why the holdout is restricted to leaves.
    """
    pairs = tiny_pairs()
    mask = parent_edge_mask(pairs, held_out=np.array([1], dtype=np.int64))
    kept = pairs[~mask]
    # 0 is no longer a direct ancestor of 1 ...
    reachable_to_1 = {int(a) for a, d in zip(kept.ancestor_idx, kept.descendant_idx) if d == 1}
    assert 0 not in reachable_to_1
    # ... but 0 is still the unique child-side ancestor of 1's whole subtree, so it leaks.
    subtree_of_1 = {3, 4}
    ancestors_of_subtree = {
        int(a) for a, d in zip(kept.ancestor_idx, kept.descendant_idx) if int(d) in subtree_of_1
    }
    assert 0 in ancestors_of_subtree


def test_eligible_nodes_rejects_a_band_whose_bounds_are_inverted():
    parent, depth = tiny_tree()
    with pytest.raises(ValueError):
        eligible_nodes(parent, depth, band=(28, 11))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest tests/eval/test_p2_split.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'taxembed.eval.p2_split'`

- [ ] **Step 3: Write the implementation**

```python
# src/taxembed/eval/p2_split.py
"""Held-out split construction for P2 (spec v3 §P2.3, as corrected 2026-09-24).

WHY A PARENT-EDGE HOLDOUT AND NOT GANEA'S NON-BASIC-EDGE HOLDOUT.

Every NCBI closure in this repo is a strict TREE (one parent per node; measured across all six
clades). On a tree the transitive reduction determines the entire closure, so Ganea's protocol --
"keep the reduction always in training" -- hands Vendrov's trivial baseline 100% of any held-out
non-basic edge. See docs/FINDING_ganea_split_degenerate_on_trees.md.

What IS predictive on a tree is the parent edge itself, withheld for LEAF nodes only. For an
internal node v the parent p stays recoverable, because v's descendants keep their (p, w) edges;
restricting to leaves closes that path. This matches TaxoExpan / Arborist / Octet (spec v3 §3.4),
which all hold out leaves.
"""

from __future__ import annotations

import numpy as np

from taxembed.utils.training_pairs import TrainingPairs

DEFAULT_BAND = (11, 28)  # per-node depth interquartile range, Q1=11 Q3=28 (spec v3 §P2.3)


def leaf_mask(parent: np.ndarray) -> np.ndarray:
    """True for nodes that are nobody's parent.

    `parent_from_closure` makes the root point to itself, so a self-edge must not count as
    parenthood -- otherwise the root would be marked internal for the wrong reason and, in a
    single-node tree, never marked at all.
    """
    parent = np.asarray(parent, dtype=np.int64)
    n = len(parent)
    has_real_parent = np.arange(n, dtype=np.int64) != parent
    is_parent = np.zeros(n, dtype=bool)
    is_parent[parent[has_real_parent]] = True
    return ~is_parent


def depth_from_closure(descendant_idx, descendant_depth, ancestor_idx, ancestor_depth,
                       n_nodes: int) -> np.ndarray:
    """Per-node depth, read off whichever closure column mentions the node."""
    depth = np.full(n_nodes, -1, dtype=np.int64)
    depth[np.asarray(ancestor_idx, dtype=np.int64)] = np.asarray(ancestor_depth, dtype=np.int64)
    depth[np.asarray(descendant_idx, dtype=np.int64)] = np.asarray(descendant_depth, dtype=np.int64)
    return depth


def eligible_nodes(parent: np.ndarray, depth: np.ndarray, band=DEFAULT_BAND,
                   leaves_only: bool = True) -> np.ndarray:
    """Node indices eligible for parent-edge holdout: depth inside `band`, non-root, and leaf.

    `band` is INCLUSIVE at both ends -- that is the reading that reproduces spec §P2.3's
    593,576 eligible nodes exactly (delta +0 against 13 rival rules).
    """
    lo, hi = band
    if lo > hi:
        raise ValueError(f"band lower bound {lo} exceeds upper bound {hi}")
    parent = np.asarray(parent, dtype=np.int64)
    depth = np.asarray(depth, dtype=np.int64)
    n = len(parent)
    mask = (np.arange(n, dtype=np.int64) != parent)      # exclude the root
    mask &= (depth >= lo) & (depth <= hi)
    if leaves_only:
        mask &= leaf_mask(parent)
    return np.flatnonzero(mask).astype(np.int64)


def select_holdout(eligible: np.ndarray, frac_test: float = 0.10, frac_val: float = 0.0,
                   seed: int = 0) -> dict:
    """Partition `eligible` into disjoint test / val / train node sets.

    Uses a dedicated Generator, never the legacy global numpy RNG -- §1.2 records that the
    unseeded global RNG is exactly how this project lost reproducibility before.
    """
    if frac_test < 0 or frac_val < 0 or frac_test + frac_val > 1:
        raise ValueError(f"invalid fractions: test={frac_test} val={frac_val}")
    eligible = np.asarray(eligible, dtype=np.int64)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(eligible))
    n_test = int(len(eligible) * frac_test)
    n_val = int(len(eligible) * frac_val)
    test = np.sort(eligible[perm[:n_test]])
    val = np.sort(eligible[perm[n_test:n_test + n_val]])
    train = np.sort(eligible[perm[n_test + n_val:]])
    return {"test": test, "val": val, "train": train}


def parent_edge_mask(pairs: TrainingPairs, held_out: np.ndarray) -> np.ndarray:
    """True for closure rows to REMOVE: the depth_diff==1 row of each held-out node."""
    held = np.zeros(pairs.n_nodes, dtype=bool)
    held[np.asarray(held_out, dtype=np.int64)] = True
    return (pairs.depth_diff == 1) & held[pairs.descendant_idx]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest tests/eval/test_p2_split.py -v`
Expected: PASS, 10 passed

- [ ] **Step 5: Add the real-data regression test**

This pins the two spec figures to code so a later refactor cannot silently move them. It is skipped
when the closure is absent, so the suite stays runnable on a machine without `data/`.

```python
# append to tests/eval/test_p2_split.py
from pathlib import Path

from taxembed.eval.subtree import parent_from_closure

CELLULAR = Path(__file__).resolve().parents[2] / (
    "data/taxopy/cellular_organisms_131567_clean/"
    "taxonomy_edges_cellular_organisms_131567_clean_transitive.npz"
)


@pytest.mark.skipif(not CELLULAR.exists(), reason="cellular closure not on this machine")
def test_band_eligibility_reproduces_the_spec_figures_on_the_real_closure():
    pairs = TrainingPairs.load(CELLULAR)
    n = pairs.n_nodes
    parent = parent_from_closure(
        pairs.ancestor_idx, pairs.descendant_idx, pairs.depth_diff, n
    )
    depth = depth_from_closure(
        pairs.descendant_idx, pairs.descendant_depth,
        pairs.ancestor_idx, pairs.ancestor_depth, n,
    )
    # spec v3 §P2.3: "593,576 eligible nodes" == depth in [11, 28] inclusive, all nodes
    assert len(eligible_nodes(parent, depth, leaves_only=False)) == 593_576
    # and the leaf-restricted set this plan actually holds out
    assert len(eligible_nodes(parent, depth, leaves_only=True)) == 501_037
```

- [ ] **Step 6: Run it**

Run: `.../.venv/bin/python -m pytest tests/eval/test_p2_split.py -v`
Expected: PASS, 11 passed (0 skipped on the dev Mac, where `data/` is present)

- [ ] **Step 7: Commit**

Write the message to a file first — no heredocs (Rule 15).

```bash
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings add src/taxembed/eval/p2_split.py tests/eval/test_p2_split.py
```
```bash
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings commit -F /tmp/p2_task1_commit_msg.txt
```

---

### Task 2: Split builder CLI

**Files:**
- Create: `scripts/build_p2_split.py`
- Test: `tests/eval/test_build_p2_split.py`

**Interfaces:**
- Consumes: everything Task 1 produces; `TrainingPairs.load/save/__getitem__`
- Produces: on disk, `<outdir>/p2_<clade>_vis<NN>_seed<S>_train.npz` (the trainer's `--file`),
  `<outdir>/p2_<clade>_seed<S>_heldout.npz` (node arrays `test`, `val`), and
  `<outdir>/p2_<clade>_vis<NN>_seed<S>_manifest.json`
- Manifest keys (consumed by Tasks 6 and 7):
  `clade, source_npz, source_md5, visibility, seed, band, n_nodes, n_pairs_source, n_eligible_all, n_eligible_leaves, n_test, n_val, n_train_nodes, n_pairs_train, n_pairs_removed_parent_edges, n_pairs_removed_visibility, n_pairs_heldout_ancestry_kept, train_md5, heldout_md5, built_at`

🛑 **CORRECTED 2026-09-24 after Task 2's implementer found the original wording to be false.**

The **visibility** knob applies to `depth_diff >= 2` rows **whose descendant is NOT held out**.
Held-out nodes keep their full `dd >= 2` ancestry at every visibility level.

The original text claimed "parent edges of retained nodes are always kept — that is what gives
held-out nodes their `dd >= 2` coordinates". **That is wrong.** A held-out node loses its `dd == 1`
row to the holdout and, without the exemption, its `dd >= 2` rows to the thinning. Held-out nodes are
leaves, so they are never an ancestor either. Measured consequence: at `visibility 0.0` **all 28,418**
held-out metazoa nodes appeared in **zero** training rows, and at `0.5` it still hit 2 of 28,418 by
chance. Those nodes would carry an untrained, random embedding, so the arm would have measured
initialization — and it silently becomes the transductive strong form the USER ruled out of scope.

⭐ The exemption is not just a patch, it is more correct: each held-out node then retains exactly
`depth − 1` ancestry rows at every visibility, so (i) the 0 % and 50 % arms stay **comparable** —
otherwise they would differ in the *test nodes' own input* as well as in the training graph — and
(ii) "retained-ancestor count" stops being a coin flip, removing a nuisance covariate from the
stratification §P2 asks for.

⚠ **Declared asymmetry for Task 7's `declared_confounds`:** held-out nodes therefore carry more of
their own ancestry than a random retained node does at visibility 0.5.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_build_p2_split.py
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from taxembed.utils.training_pairs import TrainingPairs

REPO = Path(__file__).resolve().parents[2]
MOLLUSCA = REPO / "data/taxopy/mollusca_6447_clean/taxonomy_edges_mollusca_6447_clean_transitive.npz"


@pytest.mark.skipif(not MOLLUSCA.exists(), reason="mollusca closure not on this machine")
def test_builder_writes_a_loadable_split_and_a_manifest_that_adds_up(tmp_path):
    rc = subprocess.run(
        [sys.executable, str(REPO / "scripts/build_p2_split.py"),
         "--npz", str(MOLLUSCA), "--outdir", str(tmp_path),
         "--visibility", "0.5", "--seed", "0", "--frac-test", "0.10", "--frac-val", "0.05"],
        capture_output=True, text=True,
    )
    assert rc.returncode == 0, rc.stderr

    manifests = list(tmp_path.glob("*_manifest.json"))
    assert len(manifests) == 1
    m = json.loads(manifests[0].read_text())

    train = TrainingPairs.load(tmp_path / m["train_npz"])
    held = np.load(tmp_path / m["heldout_npz"])

    # the manifest's arithmetic must close
    assert len(train) == m["n_pairs_train"]
    assert m["n_pairs_source"] == (
        m["n_pairs_train"] + m["n_pairs_removed_parent_edges"] + m["n_pairs_removed_visibility"]
    )
    # test and val are disjoint
    assert len(np.intersect1d(held["test"], held["val"])) == 0
    # NO held-out node has a depth_diff==1 row left anywhere in the training file
    held_all = np.concatenate([held["test"], held["val"]])
    dd1_children = set(train.descendant_idx[train.depth_diff == 1].tolist())
    assert dd1_children.isdisjoint(set(held_all.tolist()))
    # every held-out node still appears (it needs a coordinate)
    appearing = set(train.descendant_idx.tolist()) | set(train.ancestor_idx.tolist())
    assert set(held_all.tolist()).issubset(appearing)


@pytest.mark.skipif(not MOLLUSCA.exists(), reason="mollusca closure not on this machine")
def test_visibility_zero_keeps_only_parent_edges(tmp_path):
    rc = subprocess.run(
        [sys.executable, str(REPO / "scripts/build_p2_split.py"),
         "--npz", str(MOLLUSCA), "--outdir", str(tmp_path),
         "--visibility", "0.0", "--seed", "0", "--frac-test", "0.10", "--frac-val", "0.05"],
        capture_output=True, text=True,
    )
    assert rc.returncode == 0, rc.stderr
    m = json.loads(next(tmp_path.glob("*_manifest.json")).read_text())
    train = TrainingPairs.load(tmp_path / m["train_npz"])
    assert (train.depth_diff == 1).all()


@pytest.mark.skipif(not MOLLUSCA.exists(), reason="mollusca closure not on this machine")
def test_same_seed_reproduces_the_same_split_md5(tmp_path):
    outs = []
    for sub in ("a", "b"):
        d = tmp_path / sub
        d.mkdir()
        subprocess.run(
            [sys.executable, str(REPO / "scripts/build_p2_split.py"),
             "--npz", str(MOLLUSCA), "--outdir", str(d),
             "--visibility", "0.5", "--seed", "7", "--frac-test", "0.10", "--frac-val", "0.05"],
            capture_output=True, text=True, check=True,
        )
        outs.append(json.loads(next(d.glob("*_manifest.json")).read_text())["heldout_md5"])
    assert outs[0] == outs[1]
```

- [ ] **Step 2: Run to verify failure**

Run: `.../.venv/bin/python -m pytest tests/eval/test_build_p2_split.py -v`
Expected: FAIL — `scripts/build_p2_split.py` does not exist, non-zero returncode

- [ ] **Step 3: Implement the builder**

```python
#!/usr/bin/env python
"""Build a P2 held-out split: withhold the parent edge of band-eligible LEAF nodes.

The output is a closure .npz in the IDENTICAL TrainingPairs schema, so the trainer consumes it
unmodified:  taxembed train --file <train.npz> --mapping <clade>.mapping.tsv ...

That is the point of the design. The negative sampler builds its ancestry index from whatever
closure it is handed (train_hierarchical.py:263-294), so feeding it the split file means held-out
relations are invisible to training -- as ancestry knowledge as well as as positive pairs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from taxembed.eval.p2_split import (
    DEFAULT_BAND,
    depth_from_closure,
    eligible_nodes,
    parent_edge_mask,
    select_holdout,
)
from taxembed.eval.subtree import parent_from_closure
from taxembed.utils.training_pairs import TrainingPairs


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", required=True, type=Path, help="source transitive closure .npz")
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--visibility", type=float, required=True,
                    help="fraction of depth_diff>=2 closure rows visible in training (0.0 or 0.5)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frac-test", type=float, default=0.10)
    ap.add_argument("--frac-val", type=float, default=0.05)
    ap.add_argument("--band", type=int, nargs=2, default=list(DEFAULT_BAND),
                    metavar=("LO", "HI"), help="inclusive per-node depth band")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    pairs = TrainingPairs.load(args.npz)
    n = pairs.n_nodes

    parent = parent_from_closure(pairs.ancestor_idx, pairs.descendant_idx, pairs.depth_diff, n)
    depth = depth_from_closure(pairs.descendant_idx, pairs.descendant_depth,
                               pairs.ancestor_idx, pairs.ancestor_depth, n)
    band = (args.band[0], args.band[1])

    elig_all = eligible_nodes(parent, depth, band=band, leaves_only=False)
    elig_leaf = eligible_nodes(parent, depth, band=band, leaves_only=True)
    split = select_holdout(elig_leaf, args.frac_test, args.frac_val, args.seed)
    held_all = np.concatenate([split["test"], split["val"]])

    drop_parent = parent_edge_mask(pairs, held_all)

    rng = np.random.default_rng(args.seed + 10_000)  # independent stream from the node split
    deep = pairs.depth_diff >= 2
    # Held-out nodes are EXEMPT from thinning: they have already lost their dd==1 row, they are
    # leaves so they are never an ancestor, and without this they would appear in no training row
    # at all -- an untrained embedding at evaluation time. Draw over the full length so the
    # thinning of non-held-out rows stays bit-identical for a given seed.
    held_mask = np.zeros(n, dtype=bool)
    held_mask[held_all] = True
    is_heldout_row = held_mask[pairs.descendant_idx]
    hide_deep = deep & ~is_heldout_row & (rng.random(len(pairs)) >= args.visibility)

    keep = ~(drop_parent | hide_deep)
    train = pairs[keep]

    clade = args.npz.parent.name
    vis_tag = f"vis{int(round(args.visibility * 100)):02d}"
    train_name = f"p2_{clade}_{vis_tag}_seed{args.seed}_train.npz"
    held_name = f"p2_{clade}_seed{args.seed}_heldout.npz"
    train.save(args.outdir / train_name)
    np.savez_compressed(args.outdir / held_name, test=split["test"], val=split["val"])

    manifest = {
        "clade": clade,
        "source_npz": str(args.npz),
        "source_md5": md5(args.npz),
        "visibility": args.visibility,
        "seed": args.seed,
        "band": list(band),
        "n_nodes": int(n),
        "n_pairs_source": int(len(pairs)),
        "n_eligible_all": int(len(elig_all)),
        "n_eligible_leaves": int(len(elig_leaf)),
        "n_test": int(len(split["test"])),
        "n_val": int(len(split["val"])),
        "n_train_nodes": int(len(split["train"])),
        "n_pairs_train": int(len(train)),
        "n_pairs_removed_parent_edges": int(drop_parent.sum()),
        "n_pairs_removed_visibility": int(hide_deep.sum()),
        "train_npz": train_name,
        "heldout_npz": held_name,
        "train_md5": md5(args.outdir / train_name),
        "heldout_md5": md5(args.outdir / held_name),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    out = args.outdir / f"p2_{clade}_{vis_tag}_seed{args.seed}_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

⚠ Note the two RNG streams. The node split uses `seed`; the visibility sample uses `seed + 10_000`.
If both used `seed`, the visibility mask would be correlated with the holdout permutation and the
"same seed, different visibility" arms would not be comparable.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.../.venv/bin/python -m pytest tests/eval/test_build_p2_split.py -v`
Expected: PASS, 3 passed

- [ ] **Step 5: Build the four production splits and eyeball the manifests**

Run each as its own Bash call (no `&&` chaining):

```bash
/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/scripts/build_p2_split.py --npz /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/data/taxopy/metazoa_33208_clean/taxonomy_edges_metazoa_33208_clean_transitive.npz --outdir /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/data/p2_splits --visibility 0.0 --seed 0
```

Repeat with `--visibility 0.5`, and both again for the mollusca smoke closure.
**Check before continuing:** the metazoa manifest's `n_eligible_all` and `n_eligible_leaves` are
recorded; `n_pairs_source` equals `n_pairs_train + n_pairs_removed_parent_edges +
n_pairs_removed_visibility`; and at `visibility 0.0`,
`n_pairs_train == (n_nodes - 1 - n_test - n_val) + n_pairs_heldout_ancestry_kept`.

🛑 The second identity was originally written without the `n_pairs_heldout_ancestry_kept` term. That
form is only correct if held-out nodes are thinned to nothing — i.e. it encoded the defect. Measured
metazoa values for reference: `n_eligible_leaves` **284,181**, `n_test` **28,418**.

- [ ] **Step 6: Commit**

```bash
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings add scripts/build_p2_split.py tests/eval/test_build_p2_split.py
```
```bash
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings commit -F /tmp/p2_task2_commit_msg.txt
```

---

### Task 3: Baselines — the numbers every learned score is reported beside

**Files:**
- Create: `src/taxembed/eval/baselines.py`
- Test: `tests/eval/test_baselines.py`

**Interfaces:**
- Produces:
  - `vendrov_closure_rule(visible_ancestor, visible_descendant, queries, candidates) -> np.ndarray` (bool, `queries × candidates`) — positive iff the pair is in the transitive closure of the visible edges
  - `sibling_chance(parent, held_out) -> np.ndarray` (float, per query = `1 / n_children(grandparent)`)
  - `majority_parent_rate(parent, held_out) -> float`

§3.2 makes Vendrov's rule mandatory beside every learned number. On this split it is expected to
score **0 % recall on the positives** — the held-out parent is by construction unreachable in the
visible graph. That is the *opposite* degeneracy from the one that killed §P2.1, and reporting it is
the honest thing: it shows the task is not solvable by closure computation in either direction.
`sibling_chance` is therefore the load-bearing chance floor, and it is what the learned MRR must beat.

- [ ] **Step 1: Write the failing tests**

```python
# tests/eval/test_baselines.py
import numpy as np

from taxembed.eval.baselines import majority_parent_rate, sibling_chance, vendrov_closure_rule


def test_vendrov_rule_is_true_exactly_on_visible_closure_pairs():
    anc = np.array([0, 0, 1])
    dsc = np.array([1, 3, 3])
    got = vendrov_closure_rule(anc, dsc, queries=np.array([3, 3]), candidates=np.array([0, 2]))
    assert got.tolist() == [[True, False], [True, False]]


def test_vendrov_rule_returns_all_false_when_the_parent_edge_was_withheld():
    """The held-out parent is unreachable by construction: the rule cannot recover it."""
    anc = np.array([0])          # only (0 -> 1) visible; (1 -> 3) was withheld
    dsc = np.array([1])
    got = vendrov_closure_rule(anc, dsc, queries=np.array([3]), candidates=np.array([1]))
    assert got.tolist() == [[False]]


def test_sibling_chance_is_one_over_the_grandparents_fanout():
    # 0 -> 1, 0 -> 2 ; 1 -> 3, 1 -> 4. Node 3's parent is 1, whose parent 0 has 2 children.
    parent = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    assert sibling_chance(parent, np.array([3])).tolist() == [0.5]


def test_sibling_chance_is_one_when_the_true_parent_is_an_only_child():
    parent = np.array([0, 0, 1], dtype=np.int64)   # 0 -> 1 -> 2; 0 has one child
    assert sibling_chance(parent, np.array([2])).tolist() == [1.0]


def test_majority_parent_rate_is_the_share_of_the_commonest_true_parent():
    parent = np.array([0, 0, 0, 1, 1, 2], dtype=np.int64)
    # held out 3, 4, 5 -> true parents 1, 1, 2 -> majority share 2/3
    assert majority_parent_rate(parent, np.array([3, 4, 5])) == 2 / 3
```

- [ ] **Step 2: Run to verify failure**

Run: `.../.venv/bin/python -m pytest tests/eval/test_baselines.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'taxembed.eval.baselines'`

- [ ] **Step 3: Implement**

```python
# src/taxembed/eval/baselines.py
"""No-learning baselines reported beside every P2 link-prediction number (spec v3 §3.2).

Vendrov et al. ICLR 2016 §3.4 classify a pair positive iff it lies in the transitive closure of
the training+validation edges, scoring 88.2% on WordNet. Reporting it is mandatory because a
learned number is only interesting relative to what no learning achieves. On our tree-shaped split
it collapses to 0% recall -- see the module test -- which is itself a reportable property of the
task, not a bug in the baseline.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from taxembed.eval.subtree import euler_intervals, is_descendant


def vendrov_closure_rule(visible_ancestor, visible_descendant, queries, candidates) -> np.ndarray:
    """(queries x candidates) bool: is candidate an ancestor of query in the VISIBLE graph?"""
    visible = set(zip(np.asarray(visible_ancestor).tolist(),
                      np.asarray(visible_descendant).tolist()))
    q = np.asarray(queries).tolist()
    c = np.asarray(candidates).tolist()
    return np.array([[(cand, node) in visible for cand in c] for node in q], dtype=bool)


def sibling_chance(parent: np.ndarray, held_out: np.ndarray) -> np.ndarray:
    """Per held-out node, 1 / (number of children its GRANDPARENT has).

    This is the chance rate of guessing the true parent uniformly among the candidates that the
    retained closure still admits -- the grandparent's children. It is the floor a learned score
    must clear to mean anything.
    """
    parent = np.asarray(parent, dtype=np.int64)
    held_out = np.asarray(held_out, dtype=np.int64)
    fanout = np.bincount(parent[np.arange(len(parent)) != parent], minlength=len(parent))
    grandparent = parent[parent[held_out]]
    n_candidates = np.maximum(fanout[grandparent], 1)
    return 1.0 / n_candidates


def majority_parent_rate(parent: np.ndarray, held_out: np.ndarray) -> float:
    """Accuracy of always answering the commonest true parent among the held-out nodes."""
    parent = np.asarray(parent, dtype=np.int64)
    held_out = np.asarray(held_out, dtype=np.int64)
    if len(held_out) == 0:
        return 0.0
    counts = Counter(parent[held_out].tolist())
    return counts.most_common(1)[0][1] / len(held_out)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.../.venv/bin/python -m pytest tests/eval/test_baselines.py -v`
Expected: PASS, 5 passed

- [ ] **Step 5: Commit**

```bash
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings add src/taxembed/eval/baselines.py tests/eval/test_baselines.py
```
```bash
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings commit -F /tmp/p2_task3_commit_msg.txt
```

---

### Task 4: Link-prediction metrics

**Files:**
- Create: `src/taxembed/eval/linkpred.py`
- Test: `tests/eval/test_linkpred.py`

**Interfaces:**
- Produces:
  - `candidate_pool(parent, depth, node, strategy="grandparent_children") -> np.ndarray`
  - `rank_of_true_parent(emb, node, true_parent, candidates, metric="poincare") -> int` (1-based)
  - `linkpred_metrics(ranks: np.ndarray, n_candidates: np.ndarray) -> dict` with keys
    `mean_rank`, `mrr`, `hits_at_1`, `hits_at_10`, `normalized_rank`, `n`
  - `stratify(ranks, key, bins) -> dict`

🧨 **With exactly one positive per query, MAP is algebraically identical to MRR.** §P2 asks for
"MR/MAP"; reporting both as if they were two pieces of evidence would be double counting. Report
**MR, MRR and Hits@k**, and state the identity in the Methods. This is the honest reading of §3.5.

- [ ] **Step 1: Write the failing tests**

```python
# tests/eval/test_linkpred.py
import numpy as np
import pytest

from taxembed.eval.linkpred import (
    candidate_pool,
    linkpred_metrics,
    rank_of_true_parent,
    stratify,
)


def test_candidate_pool_is_the_grandparents_children():
    # 0 -> {1, 2}; 1 -> {3, 4}; node 3's grandparent is 0, whose children are 1 and 2
    parent = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    depth = np.array([0, 1, 1, 2, 2], dtype=np.int64)
    assert sorted(candidate_pool(parent, depth, node=3).tolist()) == [1, 2]


def test_candidate_pool_never_contains_the_query_itself():
    parent = np.array([0, 0, 1, 1], dtype=np.int64)
    depth = np.array([0, 1, 2, 2], dtype=np.int64)
    assert 2 not in candidate_pool(parent, depth, node=2).tolist()


def test_rank_is_one_when_the_true_parent_is_nearest():
    emb = np.array([[0.0, 0.0], [0.1, 0.0], [0.9, 0.0], [0.11, 0.0]])
    r = rank_of_true_parent(emb, node=3, true_parent=1, candidates=np.array([1, 2]))
    assert r == 1


def test_rank_is_last_when_the_true_parent_is_furthest():
    emb = np.array([[0.0, 0.0], [0.9, 0.0], [0.1, 0.0], [0.11, 0.0]])
    r = rank_of_true_parent(emb, node=3, true_parent=1, candidates=np.array([1, 2]))
    assert r == 2


def test_metrics_on_a_hand_computed_case():
    ranks = np.array([1, 2, 4])
    n_cand = np.array([10, 10, 10])
    m = linkpred_metrics(ranks, n_cand)
    assert m["n"] == 3
    assert m["mean_rank"] == pytest.approx(7 / 3)
    assert m["mrr"] == pytest.approx((1 + 0.5 + 0.25) / 3)
    assert m["hits_at_1"] == pytest.approx(1 / 3)
    assert m["hits_at_10"] == pytest.approx(1.0)


def test_normalized_rank_accounts_for_uneven_pool_sizes():
    """A rank of 2 out of 2 is chance; a rank of 2 out of 1000 is near-perfect.
    Un-normalized mean rank would call them similar."""
    m_small = linkpred_metrics(np.array([2]), np.array([2]))
    m_large = linkpred_metrics(np.array([2]), np.array([1000]))
    assert m_small["normalized_rank"] > m_large["normalized_rank"]
    assert m_small["mean_rank"] == m_large["mean_rank"]     # the reason normalization is needed


def test_metrics_on_an_empty_input_do_not_divide_by_zero():
    m = linkpred_metrics(np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    assert m["n"] == 0
    assert np.isnan(m["mrr"])


def test_stratify_groups_ranks_by_key_into_the_named_bins():
    ranks = np.array([1, 5, 1, 9])
    key = np.array([2, 2, 50, 50])
    out = stratify(ranks, key, bins=[(1, 10), (11, 1000)])
    assert out["1-10"]["n"] == 2
    assert out["11-1000"]["n"] == 2
    assert out["1-10"]["mrr"] == pytest.approx((1 + 0.2) / 2)
```

- [ ] **Step 2: Run to verify failure**

Run: `.../.venv/bin/python -m pytest tests/eval/test_linkpred.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'taxembed.eval.linkpred'`

- [ ] **Step 3: Implement**

```python
# src/taxembed/eval/linkpred.py
"""Held-out parent prediction: candidate pools, filtered ranking, MR / MRR / Hits@k.

PROTOCOL. For each held-out LEAF node v whose parent edge was withheld, rank the candidate
parents by embedded distance to v and record the 1-based rank of the true parent p. Candidates
are the children of v's grandparent -- the set the retained closure still admits -- which makes
the task sibling disambiguation and gives a well-defined chance rate of 1/|candidates|
(taxembed.eval.baselines.sibling_chance).

WHY NOT "MAP". With exactly one positive per query, mean average precision is algebraically
identical to mean reciprocal rank. Spec §P2 asks for "MR/MAP"; reporting both would present one
number twice. We report MR, MRR, Hits@k and a pool-size-normalized rank, and say so in Methods.
"""

from __future__ import annotations

import numpy as np


def _poincare_distance(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Poincare ball distance from one point u to each row of v."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    sq_u = np.sum(u * u)
    sq_v = np.sum(v * v, axis=-1)
    sq_d = np.sum((v - u) ** 2, axis=-1)
    denom = np.clip((1.0 - sq_u) * (1.0 - sq_v), 1e-12, None)
    return np.arccosh(1.0 + 2.0 * sq_d / denom)


def _cosine_distance(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v, axis=-1)
    return 1.0 - (v @ u) / np.clip(nu * nv, 1e-12, None)


def candidate_pool(parent: np.ndarray, depth: np.ndarray, node: int,
                   strategy: str = "grandparent_children") -> np.ndarray:
    """Admissible parents for `node` given that its own parent edge was withheld."""
    parent = np.asarray(parent, dtype=np.int64)
    depth = np.asarray(depth, dtype=np.int64)
    if strategy == "grandparent_children":
        grandparent = int(parent[int(parent[node])])
        has_parent = np.arange(len(parent), dtype=np.int64) != parent
        pool = np.flatnonzero(has_parent & (parent == grandparent))
    elif strategy == "same_depth":
        pool = np.flatnonzero(depth == depth[node] - 1)
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    return pool[pool != node].astype(np.int64)


def rank_of_true_parent(emb: np.ndarray, node: int, true_parent: int,
                        candidates: np.ndarray, metric: str = "poincare",
                        tie_seed: int = 0) -> int:
    """1-based rank of `true_parent` among `candidates`, nearest first.

    Ties are broken by seeded jitter rather than by array order: index order correlates with
    taxonomic order in this data, so argsort ties would systematically favour low indices.
    """
    candidates = np.asarray(candidates, dtype=np.int64)
    if true_parent not in set(candidates.tolist()):
        raise ValueError(f"true parent {true_parent} absent from candidate pool")
    dist_fn = _poincare_distance if metric == "poincare" else _cosine_distance
    d = dist_fn(emb[node], emb[candidates])
    rng = np.random.default_rng(tie_seed + int(node))
    d = d + rng.random(len(d)) * 1e-12
    order = np.argsort(d, kind="stable")
    position = int(np.flatnonzero(candidates[order] == true_parent)[0])
    return position + 1


def linkpred_metrics(ranks: np.ndarray, n_candidates: np.ndarray) -> dict:
    """MR, MRR, Hits@1, Hits@10 and a pool-size-normalized rank."""
    ranks = np.asarray(ranks, dtype=np.float64)
    n_candidates = np.asarray(n_candidates, dtype=np.float64)
    if len(ranks) == 0:
        return {"n": 0, "mean_rank": float("nan"), "mrr": float("nan"),
                "hits_at_1": float("nan"), "hits_at_10": float("nan"),
                "normalized_rank": float("nan")}
    # (rank - 1) / (pool - 1) is 0 for a perfect call and 1 for the worst possible one.
    denom = np.clip(n_candidates - 1.0, 1.0, None)
    return {
        "n": int(len(ranks)),
        "mean_rank": float(ranks.mean()),
        "mrr": float((1.0 / ranks).mean()),
        "hits_at_1": float((ranks <= 1).mean()),
        "hits_at_10": float((ranks <= 10).mean()),
        "normalized_rank": float(((ranks - 1.0) / denom).mean()),
    }


def stratify(ranks: np.ndarray, key: np.ndarray, bins: list[tuple[int, int]]) -> dict:
    """Metrics per inclusive [lo, hi] bin of `key` (e.g. candidate-pool size, node depth)."""
    ranks = np.asarray(ranks)
    key = np.asarray(key)
    out = {}
    for lo, hi in bins:
        sel = (key >= lo) & (key <= hi)
        out[f"{lo}-{hi}"] = linkpred_metrics(ranks[sel], np.full(int(sel.sum()), hi))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.../.venv/bin/python -m pytest tests/eval/test_linkpred.py -v`
Expected: PASS, 8 passed

- [ ] **Step 5: Commit**

```bash
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings add src/taxembed/eval/linkpred.py tests/eval/test_linkpred.py
```
```bash
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings commit -F /tmp/p2_task4_commit_msg.txt
```

---

### Task 5: RandomDAG control (§P2.4)

**Files:**
- Create: `src/taxembed/eval/randomdag.py`
- Test: `tests/eval/test_randomdag.py`

**Interfaces:**
- Produces: `randomize_parents(parent, depth, seed) -> np.ndarray`, and
  `closure_from_parent(parent, depth) -> TrainingPairs`

🎯 **The pair count is preserved exactly, and that is provable rather than approximate.** If every
node keeps its depth and is rewired to a uniformly random parent at `depth − 1`, then each node's
ancestor count equals its depth, so total pairs = `Σ depth(v)` — unchanged by construction. §P2.4
asks for "randomize the closure, holding pair count fixed"; this satisfies it exactly. Assert it.

- [ ] **Step 1: Write the failing tests**

```python
# tests/eval/test_randomdag.py
import numpy as np

from taxembed.eval.randomdag import closure_from_parent, randomize_parents


def chain_and_bush():
    # 0 -> {1,2,3}; 1 -> {4,5}; 2 -> {6}
    parent = np.array([0, 0, 0, 0, 1, 1, 2], dtype=np.int64)
    depth = np.array([0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
    return parent, depth


def test_randomization_preserves_every_node_depth():
    parent, depth = chain_and_bush()
    rp = randomize_parents(parent, depth, seed=0)
    assert (depth[rp[depth > 0]] == depth[depth > 0] - 1).all()


def test_randomization_preserves_the_closure_pair_count_exactly():
    parent, depth = chain_and_bush()
    before = len(closure_from_parent(parent, depth))
    after = len(closure_from_parent(randomize_parents(parent, depth, seed=0), depth))
    assert before == after == int(depth.sum())


def test_randomization_actually_moves_something():
    """A control that returns the input is not a control.

    Needs a tree wide enough that a random rewire is unlikely to reproduce the original:
    10 depth-1 nodes and 190 depth-2 nodes, so each depth-2 node picks among 10 parents.
    """
    n = 201
    parent = np.zeros(n, dtype=np.int64)
    depth = np.zeros(n, dtype=np.int64)
    depth[1:11] = 1                      # nodes 1..10 hang off the root
    depth[11:] = 2                       # nodes 11..200 hang off nodes 1..10
    parent[11:] = 1 + (np.arange(n - 11) % 10)
    rp = randomize_parents(parent, depth, seed=0)
    assert not np.array_equal(rp, parent)


def test_the_root_stays_the_root():
    parent, depth = chain_and_bush()
    assert randomize_parents(parent, depth, seed=3)[0] == 0


def test_randomization_is_seed_stable():
    parent, depth = chain_and_bush()
    assert np.array_equal(
        randomize_parents(parent, depth, seed=5), randomize_parents(parent, depth, seed=5)
    )
```

- [ ] **Step 2: Run to verify failure**

Run: `.../.venv/bin/python -m pytest tests/eval/test_randomdag.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'taxembed.eval.randomdag'`

- [ ] **Step 3: Implement**

```python
# src/taxembed/eval/randomdag.py
"""RandomDAG memorization control (spec v3 §P2.4, after GRAM, Choi et al. KDD'17).

Rewire every non-root node to a uniformly random parent at depth-1, keeping its own depth. Because
each node's ancestor count then equals its depth, the closure's pair count is Sum(depth) either
way -- preserved EXACTLY, which is what §P2.4 requires, not approximately.

The control answers: does the recipe's held-out performance depend on the taxonomy's real
structure, or would any tree of the same shape do as well? If a model trained on the randomized
closure scores as well on ITS OWN held-out parents, the geometry is memorizing tree shape rather
than learning taxonomy.
"""

from __future__ import annotations

import numpy as np

from taxembed.utils.training_pairs import TrainingPairs


def randomize_parents(parent: np.ndarray, depth: np.ndarray, seed: int = 0) -> np.ndarray:
    """Uniformly resample each non-root node's parent from the nodes one level above it."""
    parent = np.asarray(parent, dtype=np.int64)
    depth = np.asarray(depth, dtype=np.int64)
    rng = np.random.default_rng(seed)
    out = parent.copy()
    by_depth = {d: np.flatnonzero(depth == d) for d in np.unique(depth)}
    for d in sorted(by_depth):
        if d == 0:
            continue
        above = by_depth.get(d - 1)
        if above is None or len(above) == 0:
            continue                       # no level above: leave these nodes alone
        nodes = by_depth[d]
        out[nodes] = above[rng.integers(0, len(above), size=len(nodes))]
    return out


def closure_from_parent(parent: np.ndarray, depth: np.ndarray) -> TrainingPairs:
    """Expand a parent array into the full ancestor-descendant closure as TrainingPairs."""
    parent = np.asarray(parent, dtype=np.int64)
    depth = np.asarray(depth, dtype=np.int64)
    anc, dsc, ddiff, adep, ddep = [], [], [], [], []
    for node in range(len(parent)):
        if depth[node] == 0:
            continue                       # the root has no ancestors
        cur, step = int(parent[node]), 1
        while True:
            anc.append(cur); dsc.append(node)
            ddiff.append(step); adep.append(int(depth[cur])); ddep.append(int(depth[node]))
            if depth[cur] == 0:            # reached the root; stop before self-looping
                break
            cur, step = int(parent[cur]), step + 1
    if not anc:
        raise ValueError("empty closure")
    to32 = lambda x: np.asarray(x, dtype=np.int32)   # noqa: E731
    to16 = lambda x: np.asarray(x, dtype=np.int16)   # noqa: E731
    return TrainingPairs(
        ancestor_idx=to32(anc), descendant_idx=to32(dsc), depth_diff=to16(ddiff),
        ancestor_depth=to16(adep), descendant_depth=to16(ddep),
        ancestor_taxid=to32(anc), descendant_taxid=to32(dsc),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.../.venv/bin/python -m pytest tests/eval/test_randomdag.py -v`
Expected: PASS, 5 passed

- [ ] **Step 5: Commit**

```bash
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings add src/taxembed/eval/randomdag.py tests/eval/test_randomdag.py
```
```bash
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings commit -F /tmp/p2_task5_commit_msg.txt
```

---

### Task 6: Scoring driver

**Files:**
- Create: `scripts/score_p2_linkpred.py`
- Test: `tests/eval/test_score_p2_cli.py`

**Interfaces:**
- Consumes: Tasks 1, 3, 4 modules; the manifest and `*_heldout.npz` from Task 2; checkpoint globs
- Produces: one JSON per invocation with, per arm and per checkpoint,
  `{mean_rank, mrr, hits_at_1, hits_at_10, normalized_rank, n}` plus `baselines`
  (`sibling_chance_mean`, `vendrov_recall`, `majority_parent_rate`) and stratifications
  `by_pool_size` and `by_depth`
- CLI mirrors `scripts/score_recipe_checkpoints.py:121-131`:
  `--manifest --heldout --checkpoints ARM=GLOB (append, required) --out --metric {poincare,cosine} --max-checkpoints --seed`

- [ ] **Step 1: Write the failing CLI test**

```python
# tests/eval/test_score_p2_cli.py
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]


def test_scorer_runs_end_to_end_on_a_synthetic_tree(tmp_path):
    """A 3-level tree, an embedding that PERFECTLY encodes it, and one that is noise.
    The perfect arm must score MRR 1.0; the noise arm must land near sibling chance."""
    # 0 -> {1,2}; 1 -> {3,4}; 2 -> {5,6}   (3..6 are leaves at depth 2)
    parent = np.array([0, 0, 0, 1, 1, 2, 2], dtype=np.int64)
    depth = np.array([0, 1, 1, 2, 2, 2, 2], dtype=np.int64)

    held = tmp_path / "heldout.npz"
    np.savez(held, test=np.array([3, 5], dtype=np.int64), val=np.array([], dtype=np.int64))

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "clade": "synthetic", "band": [0, 99], "n_nodes": 7,
        "parent": parent.tolist(), "depth": depth.tolist(),
    }))

    # perfect: each leaf sits on top of its true parent
    perfect = np.array([[0.0, 0.0], [0.5, 0.0], [-0.5, 0.0],
                        [0.5, 0.01], [0.5, -0.01], [-0.5, 0.01], [-0.5, -0.01]])
    ck = tmp_path / "perfect_epoch1.pth"
    torch.save({"embeddings": torch.tensor(perfect)}, ck)

    out = tmp_path / "scores.json"
    rc = subprocess.run(
        [sys.executable, str(REPO / "scripts/score_p2_linkpred.py"),
         "--manifest", str(manifest), "--heldout", str(held),
         "--checkpoints", f"perfect={ck}", "--out", str(out)],
        capture_output=True, text=True,
    )
    assert rc.returncode == 0, rc.stderr

    res = json.loads(out.read_text())
    assert res["arms"]["perfect"]["checkpoints"][0]["metrics"]["mrr"] == 1.0
    assert res["arms"]["perfect"]["checkpoints"][0]["metrics"]["hits_at_1"] == 1.0
    # the mandatory baselines are present and the Vendrov rule finds nothing
    assert res["baselines"]["vendrov_recall"] == 0.0
    assert res["baselines"]["sibling_chance_mean"] == 0.5
```

- [ ] **Step 2: Run to verify failure**

Run: `.../.venv/bin/python -m pytest tests/eval/test_score_p2_cli.py -v`
Expected: FAIL — script does not exist

- [ ] **Step 3: Implement the driver**

Read `scripts/score_recipe_checkpoints.py` first and match its structure (argparse block at
`:121-131`, per-checkpoint loop, md5 of outputs printed at the end, final line
`scoring complete:`). The new script must:

1. Load `parent`/`depth` from the manifest when present, else rebuild them from the manifest's
   `source_npz` via `parent_from_closure` + `depth_from_closure`.
2. Load held-out node arrays from `--heldout`.
3. For each arm glob, for each checkpoint (sorted by epoch, capped by `--max-checkpoints`):
   load `embeddings`, and for each held-out node build `candidate_pool`, call
   `rank_of_true_parent`, then `linkpred_metrics`.
4. Compute `by_pool_size` via `stratify(ranks, n_candidates, bins=[(2,2),(3,5),(6,20),(21,10**6)])`
   and `by_depth` via `stratify(ranks, depth[held], bins=[(11,15),(16,21),(22,28)])`.
5. Compute the three baselines once (they do not depend on the checkpoint).
6. Write the JSON, print each output's md5, and end with `scoring complete:`.

The checkpoint key is `embeddings`; confirm against an existing checkpoint before relying on it:

```bash
/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -c "import torch,sys; print(list(torch.load(sys.argv[1], map_location='cpu').keys()))" /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/results/task9_runs/<a-real-checkpoint>.pth
```

⚠ If the key differs, fix the scorer AND the test fixture — do not special-case one of them.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.../.venv/bin/python -m pytest tests/eval/test_score_p2_cli.py -v`
Expected: PASS, 1 passed

- [ ] **Step 5: Run the whole suite — nothing may regress**

Run: `.../.venv/bin/python -m pytest tests/ -q`
Expected: 167 prior tests still green, plus the ~32 added here

- [ ] **Step 6: Commit**

```bash
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings add scripts/score_p2_linkpred.py tests/eval/test_score_p2_cli.py
```
```bash
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings commit -F /tmp/p2_task6_commit_msg.txt
```

---

### Task 7: Pre-registration — frozen before any array is submitted

**Files:**
- Create: `results/p2_heldout_preregistration.json`
- Modify: `src/taxembed/eval/preregistration.py` — add `p2_verdict(result, seeds=(0,1,2)) -> dict`
- Test: `tests/eval/test_preregistration.py` (extend)

**Interfaces:**
- Consumes: the Task 6 JSON
- Produces: `p2_verdict(result, seeds=(0, 1, 2)) -> dict` with `verdict` ∈
  `{GENERALISES, MEMORISES, UNINFORMATIVE, MIXED}`

Model the JSON on `results/objective_integrity_delta_preregistration.json`: `status` (frozen-at
commit-time sentence), `task`, `why`, `arms`, `declared_confounds`, `metrics`, `validity_gate`,
`readings`, `manuscript_hooks`.

**Arms:** `vis00_s{0,1,2}` and `vis50_s{0,1,2}` (metazoa, the Task 2 splits), and — if the USER
approves the extra queue — `randomdag_s{0,1,2}`.

**Primary metric:** MRR on the held-out test nodes at the final checkpoint, per run.

**Validity gates — each one must be able to fail, and must be tested both ways:**
- (a) **Learning gate.** `max_t MRR(t) − MRR(init) > 10 × within-run jitter SD`. Unchanged in spirit
  from the frozen v2 gate (a), which passed by 32× on the weakest arm.
- (b) **Convergence gate, amendment-2 form from the outset.** The final-phase (epoch ≥ 130) loss
  must **not RISE** by more than its within-run jitter. 🛑 Do **not** write the original "must
  fall" form: Task 9 proved it cannot distinguish a converged arm from a never-trained one, and
  `prior_s2` failed it on the sign of a 4th-decimal fluctuation.
- (c) **Floor gate, new.** An arm whose MRR does not exceed `sibling_chance_mean` is
  `UNINFORMATIVE`, not a result. A model at chance has told us nothing about generalisation.

**Readings, declared now:**
- `GENERALISES` — all 3 seeds' MRR above `sibling_chance_mean` by ≥ 2× pooled within-arm seed SD,
  **and** above the RandomDAG arm if it was run, **and** the sign holds in every depth stratum with
  n ≥ 500. ⇒ the model predicts relations it never saw. This is the answer to Burkhard.
- `MEMORISES` — MRR at or below `sibling_chance_mean`, or indistinguishable from RandomDAG.
  ⇒ Figure 4's structure is in-sample only; the manuscript must say so.
- `MIXED` — sign flips across strata. Report per stratum, no aggregate claim.
- `UNINFORMATIVE` — any validity gate fails.

🛑 **Declare in `declared_confounds` before the run:** (i) visibility 0 % trains on ~23× fewer pairs
than 50 % at metazoa, so the two arms take very different numbers of optimizer steps per epoch —
they are **not** a controlled contrast on compute, and the comparison is descriptive; (ii) the
candidate pool is the grandparent's children, so pool size varies from 2 to tens of thousands and
`mean_rank` is not comparable across strata — `normalized_rank` is; (iii) held-out leaves are by
construction the *shallowest* leaves in the band, since deep leaves fall outside [11, 28].

- [ ] **Step 1: Write the failing gate tests**

Write, at minimum, one test per gate that it **fires** on the lesion, and one that it **stays
silent** on the healthy-but-unusual case — the pair Task 9's suite was missing:

```python
def test_gate_b_fires_when_the_final_phase_loss_RISES():
    ...          # must be UNINFORMATIVE

def test_gate_b_stays_silent_on_a_CONVERGED_arm():
    """The case that broke Task 9: flat loss is convergence, not failure to train."""
    ...          # must NOT be UNINFORMATIVE

def test_gate_c_fires_when_mrr_sits_at_sibling_chance():
    ...          # must be UNINFORMATIVE

def test_gate_a_is_not_trivially_passed_by_a_dead_arm_with_realistic_jitter():
    """Fixtures must jitter. A dead arm pinned to exactly 0.0 has zero SD, which collapses
    the 10 x jitter threshold to zero and lets any rise through."""
    ...          # dead arm jittering at its own scale must still FAIL gate (a)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.../.venv/bin/python -m pytest tests/eval/test_preregistration.py -v -k p2`
Expected: FAIL — `p2_verdict` not defined

- [ ] **Step 3: Write `p2_verdict` and the JSON**

- [ ] **Step 4: Run tests to verify they pass**

Run: `.../.venv/bin/python -m pytest tests/eval/test_preregistration.py -v`
Expected: PASS — all prior preregistration tests plus the new ones

- [ ] **Step 5: Commit — this commit is the freeze**

The commit timestamp is what makes "pre-registered" checkable. Do not submit any array before it.

```bash
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings add results/p2_heldout_preregistration.json src/taxembed/eval/preregistration.py tests/eval/test_preregistration.py
```
```bash
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings commit -F /tmp/p2_task7_commit_msg.txt
```

---

### Task 8: LRZ jobs and the smoke that gates them

**Files:**
- Create: `scripts/p2_lrz_train_smoke.sh`, `scripts/p2_lrz_train.sh`, `scripts/p2_lrz_score.sh`

**Interfaces:**
- Consumes: the Task 2 split `.npz` files (must be uploaded to `/data` on LRZ first)
- Produces: tags `p2_vis00_s{0,1,2}`, `p2_vis50_s{0,1,2}` under `artifacts/tags/`

Copy the structure from `scripts/task9_lrz_recipe_contrast.sh` (GPU array) and
`scripts/task8_lrz_fixed_sampler_smoke.sh` (smoke). Preserve every habit: `set -euo pipefail`,
`export MKL_THREADING_LAYER=GNU`, the input pre-flight loop, the trailing `md5sum`, and the header
block naming the pre-registration file and the reading rule.

Array index → arm mapping, following `task9_lrz_recipe_contrast.sh:63-71`:

```bash
IDX="${SLURM_ARRAY_TASK_ID}"
if [ "${IDX}" -lt 3 ]; then
  VIS="00"; SEED="${IDX}"
else
  VIS="50"; SEED="$((IDX-3))"
fi
TAG="p2_vis${VIS}_s${SEED}"
```

🛑 **The smoke must include the flag-reality check** (`task8_lrz_fixed_sampler_smoke.sh:40-49`):
dump `taxembed.cli.main train --help` and grep for every flag the full job passes, failing loudly
with `SMOKE FAILED: ${flag} is not a real option on the CLI`. The ESMFold incident (job 5710076,
dead 37 s after a scarce H100) is what this is for.

- [ ] **Step 1: Write the three job scripts**

- [ ] **Step 2: Static-check all three (Rule 16, step 1)**

```bash
bash -n /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/scripts/p2_lrz_train.sh
```
Repeat for `p2_lrz_train_smoke.sh` and `p2_lrz_score.sh`. Expected: no output, exit 0.

- [ ] **Step 3: Byte-compile every module the jobs import**

```bash
/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m py_compile /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/scripts/build_p2_split.py /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/scripts/score_p2_linkpred.py
```

- [ ] **Step 4: Upload the splits and verify md5 on both sides**

Copy the four `.npz` files to LRZ `/data`, then compare `md5sum` output against the Task 2
manifests' `train_md5`. **Do not proceed on a mismatch.**

- [ ] **Step 5: Submit the smoke to `lrz-cpu` and READ ITS LOG**

🛑 Exit 0 is not "passed". The log must end with the expected completion string and the flag check
must have run. Task 9's session confirmed this by reading the log before submitting.

- [ ] **Step 6: Only after the smoke passes, submit the full array**

Report the job ID and expected wall-clock to the USER before submitting. 6 runs at metazoa scale;
Task 9's canonical arm took ~11 h and its prior arm ~18.6 h, so budget ~11-19 h per run.

- [ ] **Step 7: Commit the job scripts**

```bash
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings add scripts/p2_lrz_train.sh scripts/p2_lrz_train_smoke.sh scripts/p2_lrz_score.sh
```
```bash
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings commit -F /tmp/p2_task8_commit_msg.txt
```

---

### Task 9: The degeneracy write-up

**Files:**
- Create: `docs/FINDING_ganea_split_degenerate_on_trees.md`
- Modify: `docs/MANUSCRIPT_CORRECTIONS_PENDING.md` — add an entry pointing at it

This costs zero GPU, is already fully evidenced, and pre-empts a referee proposing the very protocol
we rejected. It is also a genuine contribution: the leakage literature (§3.3) argues benchmarks leak
*too much* structure; this is the mirror case, where the standard fix makes the task vacuous.

**Must contain, with the numbers already measured:**
1. All six clades are trees — the parents-per-node histogram and `edges == nodes − 1` from the
   manifests, two independent sources.
2. Expansion is bit-identical on three clades (0 / 0 symmetric difference).
3. The consequence: Vendrov's trivial baseline = 100.00 % under "reduction always in training".
4. Why WordNet differs — 661,127 − 578,477 = 82,650 basic vs a tree's 82,113, i.e. **537 extra
   edges of multiple inheritance** — and that Vendrov did *not* retain the reduction, which is why
   their baseline was beatable at 88.2 %.
5. What we do instead, and the §3.4 precedent for holding out leaves.
6. The internal-node leak that forces the leaf restriction.

🛑 **The manuscript is not versioned** (USER, 2026-09-24): it states the final correct thing. This
finding note is where the reasoning lives; the paper gets the *conclusion* (we use a leaf parent-edge
holdout, here is why the obvious alternative is vacuous on a taxonomy), not a before/after narrative.

- [ ] **Step 1: Write the finding note**

- [ ] **Step 2: Verify every number in it against the two helpers**

Run both and diff the numbers by eye against the note. A write-up that restates a figure from memory
is how a wrong number ships.

```bash
/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/helpers/p2_check_closure_is_a_tree.py
```
```bash
/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/helpers/p2_eligible_node_definitions.py
```

- [ ] **Step 3: Add the `MANUSCRIPT_CORRECTIONS_PENDING.md` entry**

- [ ] **Step 4: Commit**

```bash
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings add docs/FINDING_ganea_split_degenerate_on_trees.md docs/MANUSCRIPT_CORRECTIONS_PENDING.md helpers/p2_check_closure_is_a_tree.py helpers/p2_eligible_node_definitions.py
```
```bash
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings commit -F /tmp/p2_task9_commit_msg.txt
```

---

## Open questions for the USER — do not decide these unilaterally

1. **RandomDAG arm costs 3 more runs.** The approved budget was 6 (2 visibility × 3 seeds). A
   RandomDAG control needs its own training runs (~11 h each) and is the single strongest piece of
   evidence against memorisation. Ask before adding it; do not silently expand the queue.
2. **Whether to hold out val at all.** `--frac-val 0.05` is in the builder, but nothing in this plan
   *uses* val — there is no model selection to do, since the pre-registration fixes the reading.
   Holding it out costs test power for nothing. Recommend `--frac-val 0.0` unless a use appears.
3. **Task 8 (LRZ 5807547) is still unharvested** and is item 1 of the previous session's handoff.
   It is independent of this plan and should be picked up when the job lands.

## Self-review

**Spec coverage.** §P2.1 → Task 2's visibility knob, with the Ganea split deliberately *not*
implemented and the reason documented in Task 9. §P2.2 Vendrov → Task 3. §P2.3 level-stratified
holdout → Tasks 1, 2, and the `by_depth` stratification in Task 6. §P2.4 RandomDAG → Task 5 (code)
and open question 1 (runs). §3.5 metrics → Task 4, with the MAP≡MRR identity stated. "Stratify by
retained-ancestor count and true-parent sibling count" → Task 6's `by_pool_size` (pool size **is**
the true parent's sibling count) and `by_depth`; ⚠ retained-ancestor count is constant at
`depth − 1` under a leaf holdout at visibility 100 %, and varies only through the visibility knob —
recorded here rather than silently dropped.

**Placeholders.** None: every code step carries runnable code. Task 6 step 3 and Tasks 7-9 specify
structure plus the exact properties to satisfy rather than full listings, because each is either a
close copy of a named existing file (`score_recipe_checkpoints.py`,
`task9_lrz_recipe_contrast.sh`, `objective_integrity_delta_preregistration.json`) or prose.

**Type consistency.** `parent`/`depth` are `int64` arrays of length `n_nodes` throughout; node index
arrays are `int64`; `TrainingPairs` fields keep their declared `int32`/`int16` dtypes on save.
`eligible_nodes` → `select_holdout` → `parent_edge_mask` → `TrainingPairs.__getitem__(bool mask)`
chains without conversion. `linkpred_metrics` returns the same six keys everywhere, including the
empty case.
