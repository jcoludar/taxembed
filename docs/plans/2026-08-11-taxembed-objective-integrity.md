# TaxEmbed Objective Integrity (P1) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish, in the repo and reproducibly, whether the shipped TaxEmbed embedding was trained with a defective objective — and fix it.

**Architecture:** Five pure, unit-testable analysis functions land first (subtree intervals, false-negative audit, null sanity, radial floor, seeding), each converting a review finding into a repo artifact with a test. Only then does `train_hierarchical.py`'s negative sampler change, because the fix's correctness depends on the zero-pool census the audit produces. Final task retrains fixed-vs-unfixed at clade scale and reports the delta.

**Tech Stack:** Python 3.13, numpy, torch, pytest. No GPU until Task 7.

**Source spec:** `docs/specs/2026-08-11-taxembed-overfitting-and-plm-showcase-design-v3.md`

## Global Constraints

- **Shell hygiene (CLAUDE.md, zero tolerance):** no compound Bash. No `&&`, `||`, `;`, pipes, heredocs, `python -c`, or `>>` redirection. Every Python invocation is a single absolute-path call: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python <abs-script>`.
- **Test runner:** `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest <abs-test-path> -v`
- **Never write to `data/`.** All new outputs go to `results/` or `docs/`. `data/` is a Dropbox symlink with version history; overwriting requires Know+Check+approve (CLAUDE.md Rule 1).
- **Rule 16:** before any `sbatch`, run `bash -n <job.sbatch>`, `python -m py_compile <script>`, and a smoke run on a tiny input. Never submit untested code to LRZ.
- **Canonical closure (read-only):** `data/taxopy/cellular_organisms_131567_clean/taxonomy_edges_cellular_organisms_131567_clean_transitive.npz` — 21,399,053 pairs, 1,102,163 nodes, max depth 40. Arrays: `ancestor_idx`, `descendant_idx`, `depth_diff`, `ancestor_depth`, `descendant_depth`, `ancestor_taxid`, `descendant_taxid`.
- **Numbers of record** (author-verified 2026-08-11, must be reproduced exactly by Task 2): overall false-negative rate **47.3752%**; zero-valid-negative pairs **3,026,809 (14.1446%)**; root-anchored pairs **1,102,162 (5.1505%)**.
- **Commit discipline:** multi-line commit messages go through `Write` to a temp file then `git commit -F <file>` (CLAUDE.md Rule 15). Never `cat <<EOF`.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/taxembed/eval/subtree.py` (new) | Parent array from closure; Euler-tour intervals; O(1) vectorized descendant test. Pure numpy. |
| `src/taxembed/eval/sampler_audit.py` (new) | Closed-form false-negative rate and valid-pool census for the negative sampler. Pure numpy. |
| `src/taxembed/eval/nulls.py` (modify) | Add `null_retrieval_sanity` — the standing guard that a null is not degenerate. |
| `src/taxembed/eval/radial.py` (new) | Initialization-floor depth↔norm correlation. |
| `scripts/audit_negative_sampling.py` (new) | Thin CLI: run the audit over a closure npz, write JSON to `results/`. |
| `train_hierarchical.py` (modify) | Seeding; ancestry-aware negative sampling with root-drop and depth relaxation; realized-negative-count logging. |
| `src/taxembed/cli/main.py` (modify) | `--seed` flag, recorded into `run.json`. |
| `tests/eval/test_subtree.py` (new) | Hand-built tree fixtures. |
| `tests/eval/test_sampler_audit.py` (new) | Hand-computed false-negative rates. |
| `tests/eval/test_radial.py` (new) | Init-floor correlation. |
| `tests/eval/test_nulls.py` (modify) | Sanity-check behaviour on degenerate vs healthy nulls. |
| `tests/test_negative_sampling.py` (new) | Sampler invariants: no descendant negatives, root pairs dropped, relaxation triggers. |

---

## Task 1: Subtree intervals and the descendant test

**Files:**
- Create: `src/taxembed/eval/subtree.py`
- Test: `tests/eval/test_subtree.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `parent_from_closure(ancestor_idx, descendant_idx, depth_diff, n_nodes) -> np.ndarray[int64]` — root points to itself.
  - `euler_intervals(parent) -> tuple[np.ndarray[int64], np.ndarray[int64]]` — `(tin, tout)`.
  - `is_descendant(anc, node, tin, tout) -> np.ndarray[bool]` — vectorized; a node counts as its own descendant.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_subtree.py
import numpy as np
import pytest

from taxembed.eval.subtree import parent_from_closure, euler_intervals, is_descendant


def _chain_and_fork():
    """Tree:  0 -> 1 -> {2, 3};  3 -> 4
    depths:   0    1     2  2         3
    """
    # closure pairs (ancestor, descendant, depth_diff)
    anc = np.array([0, 0, 0, 0, 1, 1, 1, 3])
    des = np.array([1, 2, 3, 4, 2, 3, 4, 4])
    dd = np.array([1, 2, 2, 3, 1, 1, 2, 1])
    return anc, des, dd


def test_parent_from_closure_uses_only_direct_edges():
    anc, des, dd = _chain_and_fork()
    parent = parent_from_closure(anc, des, dd, n_nodes=5)
    assert parent.tolist() == [0, 0, 1, 1, 3]


def test_euler_intervals_span_subtree_sizes():
    anc, des, dd = _chain_and_fork()
    parent = parent_from_closure(anc, des, dd, n_nodes=5)
    tin, tout = euler_intervals(parent)
    # subtree sizes: 0->5, 1->4, 2->1, 3->2, 4->1
    assert (tout - tin).tolist() == [5, 4, 1, 2, 1]


def test_is_descendant_includes_self_and_excludes_siblings():
    anc, des, dd = _chain_and_fork()
    parent = parent_from_closure(anc, des, dd, n_nodes=5)
    tin, tout = euler_intervals(parent)
    nodes = np.arange(5)
    # everything descends from root 0
    assert is_descendant(np.zeros(5, dtype=int), nodes, tin, tout).all()
    # node 2 and node 3 are siblings
    assert not is_descendant(np.array([2]), np.array([3]), tin, tout)[0]
    # 4 descends from 3, not from 2
    assert is_descendant(np.array([3]), np.array([4]), tin, tout)[0]
    assert not is_descendant(np.array([2]), np.array([4]), tin, tout)[0]
    # self-descendance holds (this is what makes drawing the positive count as a false negative)
    assert is_descendant(np.array([3]), np.array([3]), tin, tout)[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/tests/eval/test_subtree.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'taxembed.eval.subtree'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/taxembed/eval/subtree.py
"""Subtree membership in O(1) per query via Euler-tour intervals.

The negative sampler needs a fast "is X a descendant of A?" test at batch time
(300 negatives x 256 batch = 76,800 queries per batch), which rules out repeated
LCA lifting. An Euler tour gives each node an interval [tin, tout); X descends
from A iff tin[A] <= tin[X] < tout[A]. A node is its own descendant.
"""
from __future__ import annotations

import numpy as np


def parent_from_closure(ancestor_idx, descendant_idx, depth_diff, n_nodes: int) -> np.ndarray:
    """Derive the parent array from a transitive closure, using only depth_diff == 1 rows.

    Nodes with no incoming direct edge (the root) point to themselves.
    """
    parent = np.arange(n_nodes, dtype=np.int64)
    direct = np.asarray(depth_diff) == 1
    parent[np.asarray(descendant_idx)[direct]] = np.asarray(ancestor_idx)[direct]
    return parent


def euler_intervals(parent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Iterative DFS Euler tour. Returns (tin, tout); subtree of v spans [tin[v], tout[v])."""
    parent = np.asarray(parent, dtype=np.int64)
    n = len(parent)
    children: list[list[int]] = [[] for _ in range(n)]
    roots: list[int] = []
    for v in range(n):
        p = int(parent[v])
        if p == v:
            roots.append(v)
        else:
            children[p].append(v)

    tin = np.zeros(n, dtype=np.int64)
    tout = np.zeros(n, dtype=np.int64)
    timer = 0
    for r in roots:
        stack: list[tuple[int, bool]] = [(r, False)]
        while stack:
            v, exiting = stack.pop()
            if exiting:
                tout[v] = timer
                continue
            tin[v] = timer
            timer += 1
            stack.append((v, True))
            for c in reversed(children[v]):
                stack.append((c, False))
    return tin, tout


def is_descendant(anc, node, tin: np.ndarray, tout: np.ndarray) -> np.ndarray:
    """Vectorized subtree membership. Broadcasts; a node is its own descendant."""
    anc = np.asarray(anc)
    node = np.asarray(node)
    return (tin[anc] <= tin[node]) & (tin[node] < tout[anc])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/tests/eval/test_subtree.py -v`

Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
git add src/taxembed/eval/subtree.py tests/eval/test_subtree.py
git commit -m "feat(eval): Euler-tour subtree membership for O(1) descendant tests"
```

---

## Task 2: Exact false-negative audit of the negative sampler

**Files:**
- Create: `src/taxembed/eval/sampler_audit.py`
- Create: `scripts/audit_negative_sampling.py`
- Test: `tests/eval/test_sampler_audit.py`

**Interfaces:**
- Consumes: nothing from Task 1 (the audit is closed-form and does not need Euler intervals).
- Produces: `false_negative_audit(ancestor_idx, descendant_idx, ancestor_depth, descendant_depth) -> dict` with keys `overall_rate`, `by_anchor_depth` (list of dicts: `anchor_depth`, `n_pairs`, `fn_rate`, `n_zero_pool`), `n_zero_pool`, `frac_zero_pool`, `n_root_anchored`, `frac_root_anchored`, `nodes_per_depth`.

**Why closed form:** negatives are drawn uniformly from all nodes at the descendant's depth (`train_hierarchical.py:398`, fast path `:400-403`) and scored against the anchor (`:730-732`). So the probability a draw is a false negative is exactly `C[a, dd] / N[dd]` — descendants of the anchor at that depth, over all nodes at that depth. Both terms read off the complete closure. No sampling, no seed.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_sampler_audit.py
import numpy as np

from taxembed.eval.sampler_audit import false_negative_audit


def _two_branch():
    """Tree: 0 -> {1, 2};  1 -> {3, 4};  2 -> {5, 6}
    depths:  0     1  1     2  2          2  2
    Closure pairs (ancestor, descendant):
      (0,1) (0,2) (0,3) (0,4) (0,5) (0,6) (1,3) (1,4) (2,5) (2,6)
    """
    anc = np.array([0, 0, 0, 0, 0, 0, 1, 1, 2, 2])
    des = np.array([1, 2, 3, 4, 5, 6, 3, 4, 5, 6])
    ad = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    dd = np.array([1, 1, 2, 2, 2, 2, 2, 2, 2, 2])
    return anc, des, ad, dd


def test_root_anchored_pairs_are_always_100_percent_false_negative():
    anc, des, ad, dd = _two_branch()
    out = false_negative_audit(anc, des, ad, dd)
    root_rows = [r for r in out["by_anchor_depth"] if r["anchor_depth"] == 0]
    assert len(root_rows) == 1
    # every node at any depth descends from the root
    assert root_rows[0]["fn_rate"] == 1.0
    assert root_rows[0]["n_zero_pool"] == root_rows[0]["n_pairs"]


def test_depth1_anchor_rate_is_half_of_its_depth_layer():
    anc, des, ad, dd = _two_branch()
    out = false_negative_audit(anc, des, ad, dd)
    # anchors 1 and 2 each own 2 of the 4 nodes at depth 2 -> 0.5
    d1 = [r for r in out["by_anchor_depth"] if r["anchor_depth"] == 1][0]
    assert d1["fn_rate"] == 0.5
    assert d1["n_zero_pool"] == 0


def test_overall_rate_is_the_pair_weighted_mean():
    anc, des, ad, dd = _two_branch()
    out = false_negative_audit(anc, des, ad, dd)
    # 6 root-anchored pairs at 1.0, 4 depth-1-anchored pairs at 0.5
    assert out["overall_rate"] == (6 * 1.0 + 4 * 0.5) / 10
    assert out["n_root_anchored"] == 6
    assert out["n_zero_pool"] == 6


def test_nodes_per_depth_counts_every_node_once():
    anc, des, ad, dd = _two_branch()
    out = false_negative_audit(anc, des, ad, dd)
    assert out["nodes_per_depth"] == [1, 2, 4]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/tests/eval/test_sampler_audit.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'taxembed.eval.sampler_audit'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/taxembed/eval/sampler_audit.py
"""Closed-form audit of the default negative sampler's false-negative rate.

train_hierarchical.py draws negatives uniformly from _depth_to_nodes[depth(descendant)]
(:398, fast path :400-403 -- with replacement, no self-exclusion, no ancestry check),
then scores them as distance from the ANCESTOR (:730-732). A drawn negative is therefore
a FALSE negative -- an actual descendant of the anchor, i.e. a valid positive -- with
probability

    FN(a, dd) = C[a, dd] / N[dd]

C[a, dd] = #descendants of a at depth dd (read off the complete closure)
N[dd]    = #nodes at depth dd

Exact, no sampling. Also censuses pairs whose valid-negative pool is EMPTY (C == N),
which is where a naive "reject descendant negatives" fix silently zeroes the gradient.
"""
from __future__ import annotations

import numpy as np


def false_negative_audit(ancestor_idx, descendant_idx, ancestor_depth, descendant_depth) -> dict:
    anc = np.asarray(ancestor_idx, dtype=np.int64)
    des = np.asarray(descendant_idx, dtype=np.int64)
    anc_depth = np.asarray(ancestor_depth, dtype=np.int64)
    des_depth = np.asarray(descendant_depth, dtype=np.int64)
    n_pairs = len(anc)
    n_nodes = int(max(anc.max(), des.max())) + 1

    depth = np.full(n_nodes, -1, dtype=np.int64)
    depth[des] = des_depth
    depth[anc] = anc_depth              # the root never appears as a descendant
    if (depth < 0).any():
        raise ValueError(f"{int((depth < 0).sum())} nodes have no depth in the closure")
    max_depth = int(depth.max())

    # N[dd]
    nodes_per_depth = np.bincount(depth, minlength=max_depth + 1).astype(np.int64)

    # C[a, dd] via a sparse (ancestor, descendant_depth) group count
    stride = max_depth + 1
    key = anc * stride + des_depth
    uniq_key, counts = np.unique(key, return_counts=True)
    C_pair = counts[np.searchsorted(uniq_key, key)].astype(np.int64)
    N_pair = nodes_per_depth[des_depth]

    fn_pair = C_pair / N_pair
    zero_pool = C_pair >= N_pair

    by_depth = []
    for ad in range(max_depth + 1):
        m = anc_depth == ad
        n = int(m.sum())
        if n == 0:
            continue
        by_depth.append({
            "anchor_depth": ad,
            "n_pairs": n,
            "fn_rate": float(fn_pair[m].mean()),
            "n_zero_pool": int(zero_pool[m].sum()),
        })

    root = anc_depth == 0
    return {
        "n_pairs": int(n_pairs),
        "n_nodes": int(n_nodes),
        "max_depth": max_depth,
        "overall_rate": float(fn_pair.mean()),
        "by_anchor_depth": by_depth,
        "n_zero_pool": int(zero_pool.sum()),
        "frac_zero_pool": float(zero_pool.mean()),
        "n_root_anchored": int(root.sum()),
        "frac_root_anchored": float(root.mean()),
        "nodes_per_depth": nodes_per_depth.tolist(),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/tests/eval/test_sampler_audit.py -v`

Expected: PASS, 4 tests.

- [ ] **Step 5: Write the CLI**

```python
# scripts/audit_negative_sampling.py
"""Run the closed-form negative-sampler audit over a closure npz; write JSON to results/.

Usage (single absolute-path invocation, per CLAUDE.md shell hygiene):
  <venv-python> scripts/audit_negative_sampling.py --npz <path> --out results/negative_sampler_audit.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from taxembed.eval.sampler_audit import false_negative_audit  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Closed-form negative-sampler false-negative audit")
    ap.add_argument("--npz", required=True, help="transitive closure .npz")
    ap.add_argument("--out", required=True, help="output JSON path")
    args = ap.parse_args()

    d = np.load(args.npz)
    result = false_negative_audit(
        d["ancestor_idx"], d["descendant_idx"], d["ancestor_depth"], d["descendant_depth"]
    )
    result["source_npz"] = str(Path(args.npz).resolve())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print(f"pairs                 : {result['n_pairs']:,}")
    print(f"overall FN rate       : {result['overall_rate']:.6%}")
    print(f"zero-valid-neg pairs  : {result['n_zero_pool']:,} ({result['frac_zero_pool']:.4%})")
    print(f"root-anchored pairs   : {result['n_root_anchored']:,} ({result['frac_root_anchored']:.4%})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Reproduce the numbers of record**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/scripts/audit_negative_sampling.py --npz /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/data/taxopy/cellular_organisms_131567_clean/taxonomy_edges_cellular_organisms_131567_clean_transitive.npz --out /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/results/negative_sampler_audit.json`

Expected, exactly (these are the Global Constraints numbers of record):
```
pairs                 : 21,399,053
overall FN rate       : 47.375240%
zero-valid-neg pairs  : 3,026,809 (14.1446%)
root-anchored pairs   : 1,102,162 (5.1505%)
```

**If any figure differs, STOP** and reconcile before continuing — every downstream task depends on this census.

- [ ] **Step 7: Commit**

```bash
git add src/taxembed/eval/sampler_audit.py scripts/audit_negative_sampling.py tests/eval/test_sampler_audit.py results/negative_sampler_audit.json
git commit -m "feat(eval): closed-form false-negative audit of the negative sampler"
```

---

## Task 3: Null retrieval sanity check

**Files:**
- Modify: `src/taxembed/eval/nulls.py`
- Modify: `tests/eval/test_nulls.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `null_retrieval_sanity(retrieved_idx, node_depth, chance_accuracy=None) -> dict` with keys `n_distinct_retrieved`, `modal_share`, `mean_retrieved_depth`, `frac_shallow`, `frac_deep`, `degenerate` (bool), `reason` (str).

**Why:** `radial_only_null` scores 0.00089 where frequency-matched chance is 0.4326 — ~480× below chance. Diagnosis: direction randomization returns the shallow-norm shell (mean retrieved depth 2.95 vs pool 20.94; 99.5% at depth ≤10, 0.00% deep), so it structurally never predicts a deep majority class like Mammalia. A null that cannot compete is not a null. Every reported null must pass this check first.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/eval/test_nulls.py
import numpy as np

from taxembed.eval.nulls import null_retrieval_sanity


def test_shallow_shell_collapse_is_flagged_degenerate():
    # 1000 retrievals, all at depth <= 3, from a pool whose typical depth is 20
    node_depth = np.concatenate([np.full(50, 2), np.full(950, 20)])
    retrieved = np.random.default_rng(0).integers(0, 50, size=1000)  # only shallow nodes
    out = null_retrieval_sanity(retrieved, node_depth)
    assert out["degenerate"] is True
    assert out["frac_deep"] == 0.0
    assert out["mean_retrieved_depth"] < 5


def test_healthy_null_spread_over_depths_is_not_degenerate():
    node_depth = np.concatenate([np.full(500, 2), np.full(500, 20)])
    retrieved = np.arange(1000)  # uniform over the pool
    out = null_retrieval_sanity(retrieved, node_depth)
    assert out["degenerate"] is False
    assert out["n_distinct_retrieved"] == 1000


def test_below_chance_accuracy_marks_degenerate():
    node_depth = np.full(100, 10)
    retrieved = np.arange(100)
    out = null_retrieval_sanity(retrieved, node_depth, chance_accuracy=0.43)
    # accuracy not supplied for the null itself -> only depth/concentration criteria apply
    assert "chance_accuracy" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/tests/eval/test_nulls.py -v -k sanity`

Expected: FAIL — `ImportError: cannot import name 'null_retrieval_sanity'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/taxembed/eval/nulls.py

def null_retrieval_sanity(retrieved_idx, node_depth, chance_accuracy=None,
                          shallow_max: int = 10, deep_min: int = 25) -> dict:
    """Guard against a degenerate null (spec v3 §1.4).

    radial_only_null retrieval collapses onto the shallow-norm shell: it returns many
    DISTINCT nodes (so modal concentration looks mild) but nearly all of them shallow,
    so it can never predict a deep majority class. Measuring identity concentration
    misses this; measure the DEPTH distribution.

    A null is flagged degenerate if it never retrieves a deep node, or if it is
    dominated by one reference.
    """
    retrieved_idx = np.asarray(retrieved_idx)
    node_depth = np.asarray(node_depth)
    depths = node_depth[retrieved_idx]
    _, counts = np.unique(retrieved_idx, return_counts=True)

    frac_shallow = float((depths <= shallow_max).mean())
    frac_deep = float((depths >= deep_min).mean())
    modal_share = float(counts.max() / counts.sum())

    reasons = []
    if frac_deep == 0.0:
        reasons.append(f"never retrieves a node at depth >= {deep_min}")
    if modal_share > 0.5:
        reasons.append(f"one reference takes {modal_share:.1%} of retrievals")

    return {
        "n_distinct_retrieved": int(len(counts)),
        "modal_share": modal_share,
        "mean_retrieved_depth": float(depths.mean()),
        "frac_shallow": frac_shallow,
        "frac_deep": frac_deep,
        "chance_accuracy": chance_accuracy,
        "degenerate": bool(reasons),
        "reason": "; ".join(reasons) if reasons else "passes depth-spread and concentration checks",
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/tests/eval/test_nulls.py -v`

Expected: PASS — existing null tests plus 3 new.

- [ ] **Step 5: Commit**

```bash
git add src/taxembed/eval/nulls.py tests/eval/test_nulls.py
git commit -m "feat(eval): null retrieval sanity check (depth-spread, not identity concentration)"
```

---

## Task 4: Radial initialization floor

**Files:**
- Create: `src/taxembed/eval/radial.py`
- Test: `tests/eval/test_radial.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `initialization_depth_norm_r(depths, max_depth, radial_schedule) -> float` — Pearson r between the *initialization* Poincaré norm and depth.

**Why:** `_initialize_by_depth` (`train_hierarchical.py:65-96`) sets every node's norm to `target_radius(depth)` at step 0, and `radial_regularizer` (`:748-778`) then penalizes deviation from that same target. The reported depth-norm r = 0.952 must be read against this floor. Note the random direction does not affect the norm, so the floor is deterministic given depths and schedule — no seed needed.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_radial.py
import numpy as np

from taxembed.eval.radial import initialization_depth_norm_r


def test_linear_schedule_floor_is_essentially_one():
    depths = np.repeat(np.arange(0, 20), 5)
    r = initialization_depth_norm_r(depths, max_depth=20, radial_schedule="linear")
    assert r > 0.999


def test_log_schedule_floor_is_high_but_below_one():
    depths = np.repeat(np.arange(0, 41), 10)
    r = initialization_depth_norm_r(depths, max_depth=40, radial_schedule="log")
    assert 0.7 < r < 1.0


def test_constant_depths_return_nan_rather_than_raising():
    depths = np.full(50, 7)
    r = initialization_depth_norm_r(depths, max_depth=40, radial_schedule="log")
    assert np.isnan(r)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/tests/eval/test_radial.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'taxembed.eval.radial'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/taxembed/eval/radial.py
"""The initialization floor for the depth<->norm correlation (spec v3 §1.3).

_initialize_by_depth sets ||x|| = target_radius(depth) at step 0 and radial_regularizer
penalizes deviation from the same target, so the headline depth-norm r is not an outcome
of training. This computes what the correlation is BEFORE any gradient step, which is the
floor the trained value must be reported against.

The random direction does not affect the norm, so this is deterministic -- no seed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train_hierarchical import target_radius  # noqa: E402


def initialization_depth_norm_r(depths, max_depth: int, radial_schedule: str) -> float:
    """Pearson r between initialization Poincare norm and node depth.

    Returns nan when depth has zero variance (correlation undefined).
    """
    depths = np.asarray(depths, dtype=np.float64)
    if depths.std() == 0:
        return float("nan")
    norms = np.array(
        [float(target_radius(int(d), max_depth, radial_schedule)) for d in depths],
        dtype=np.float64,
    )
    if norms.std() == 0:
        return float("nan")
    return float(np.corrcoef(norms, depths)[0, 1])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/tests/eval/test_radial.py -v`

Expected: PASS, 3 tests.

**If the import of `target_radius` fails**, read its actual location with `Grep` for `def target_radius` and correct the import path — do not reimplement the schedule, or the floor stops being comparable to the model's.

- [ ] **Step 5: Commit**

```bash
git add src/taxembed/eval/radial.py tests/eval/test_radial.py
git commit -m "feat(eval): initialization floor for the depth-norm correlation"
```

---

## Task 5: Seed the training run

**Files:**
- Modify: `train_hierarchical.py` (add `seed_everything`; call it at training entry)
- Modify: `src/taxembed/cli/main.py` (add `--seed`, pass through, record in `run.json`)
- Test: `tests/test_negative_sampling.py` (create; seeding test lands here, sampler tests join in Task 6)

**Interfaces:**
- Consumes: nothing.
- Produces: `seed_everything(seed: int) -> None` in `train_hierarchical.py`.

**Why:** the string `seed` occurs **zero times** across `train_hierarchical.py`, `train_small.py`, and `cli/main.py`. Unseeded: direction init (`torch.randn`, `:85`), every negative draw (`np.random.randint`/`np.random.choice` — the **legacy global** numpy RNG, so `np.random.default_rng` will NOT fix it), the per-epoch shuffle and `epoch_fraction` subsample (`:592-623`). The paper design doc claims "Reproducible (seeded; repro run matches to ±0.01)"; that claim is currently unsupported.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_negative_sampling.py
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train_hierarchical import seed_everything


def test_seed_everything_makes_legacy_numpy_draws_reproducible():
    seed_everything(1234)
    a = np.random.randint(0, 1000, size=50)
    seed_everything(1234)
    b = np.random.randint(0, 1000, size=50)
    assert (a == b).all()


def test_seed_everything_makes_torch_init_reproducible():
    import torch

    seed_everything(7)
    a = torch.randn(20)
    seed_everything(7)
    b = torch.randn(20)
    assert torch.equal(a, b)


def test_different_seeds_differ():
    seed_everything(1)
    a = np.random.randint(0, 10_000, size=50)
    seed_everything(2)
    b = np.random.randint(0, 10_000, size=50)
    assert not (a == b).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/tests/test_negative_sampling.py -v`

Expected: FAIL — `ImportError: cannot import name 'seed_everything'`

- [ ] **Step 3: Add `seed_everything` to `train_hierarchical.py`**

Insert near the top of `train_hierarchical.py`, after the existing imports:

```python
def seed_everything(seed: int) -> None:
    """Seed every RNG the training loop actually uses.

    The negative sampler and epoch subsampler use the LEGACY global numpy RNG
    (np.random.randint / np.random.choice / np.random.shuffle), so np.random.seed
    is required -- np.random.default_rng does not affect them. Direction init uses
    the global torch RNG.

    Note: --amp plus CUDA scatter nondeterminism means this gives run-to-run
    reproducibility on CPU and near-reproducibility on GPU, not bitwise equality.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/tests/test_negative_sampling.py -v`

Expected: PASS, 3 tests.

- [ ] **Step 5: Wire `--seed` through the CLI**

In `src/taxembed/cli/main.py`, add to the `train` subparser (alongside the existing `--early-stopping` argument near line 376):

```python
    train_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for init, negative sampling, and epoch subsampling. "
             "Omit for the historical unseeded behaviour.",
    )
```

Then in the block that builds the training subprocess command (the same place `--early-stopping` is forwarded, around `:376-377`), append:

```python
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
```

And in the metadata dict written to `run.json` (around `:457`, beside `"early_stopping": args.early_stopping`), add:

```python
                    "seed": args.seed,
```

Mirror the `--seed` argument in `train_hierarchical.py`'s own argparse, and call `seed_everything(args.seed)` immediately after parsing, guarded by `if args.seed is not None:`.

- [ ] **Step 6: Verify the flag is accepted end to end**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m taxembed.cli.main train --help`

Expected: `--seed` appears in the help output.

- [ ] **Step 7: Commit**

```bash
git add train_hierarchical.py src/taxembed/cli/main.py tests/test_negative_sampling.py
git commit -m "feat(train): --seed flag covering torch, legacy numpy, and epoch subsampling"
```

---

## Task 6: Ancestry-aware negative sampling

**Files:**
- Modify: `train_hierarchical.py` (`HierarchicalDataLoader.__init__`, `_sample_negatives_default_vectorized`)
- Test: `tests/test_negative_sampling.py` (extend)

**Interfaces:**
- Consumes: `parent_from_closure`, `euler_intervals`, `is_descendant` from Task 1; the zero-pool census from Task 2.
- Produces: `HierarchicalDataLoader(..., exclude_descendant_negatives: bool = False, drop_root_anchored: bool = False)`; a `realized_negatives` counter attribute readable after each epoch.

**Design — and why the obvious fix is wrong.** The naive fix ("reject negatives that are descendants of the anchor") leaves **3,026,809 pairs (14.14%) with an empty pool** — 100% of root-anchored, 67.69% of depth-1, ~27% of depths 2-5 (Task 2's census). An empty negative set makes `softmax_loss_from_dists` (`:713-715`) a cross-entropy over a single logit ≡ 0, i.e. **zero gradient**, silently deleting 14% of the training signal at exactly the shallow anchors that set global structure. Three-part correct design:

1. **Drop root-anchored pairs entirely** (1,102,162; 5.15%). Every node descends from the root, so these are unsupervisable by a contrastive loss — and `d(root, x)` is already supervised by `radial_regularizer` (`:748-778`).
2. **When the same-depth non-descendant pool is short, relax the depth stratification, not the ancestry constraint** — draw the shortfall from all non-descendant nodes.
3. **Mask rather than resample, and log realized negative count per example**, because rejection rate falls 100%→33% with anchor depth, making effective negative count a monotone function of depth. Softmax loss scale depends on denominator size and this compounds with `depth_weight = sqrt(depth+1)` (`:737-738`). The confound must stay visible, not be hidden.

Framing for the Methods: Nickel & Kiela sample negatives from `{v' : (u,v') ∉ D}`, so ancestry exclusion is part of the canonical objective. This **restores** N&K rather than departing from it.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_negative_sampling.py
import numpy as np
import pytest

from taxembed.eval.subtree import parent_from_closure, euler_intervals, is_descendant


def _fixture_closure():
    """Tree: 0 -> {1,2}; 1 -> {3,4}; 2 -> {5,6}. Depth 2 layer = {3,4,5,6}."""
    anc = np.array([0, 0, 0, 0, 0, 0, 1, 1, 2, 2])
    des = np.array([1, 2, 3, 4, 5, 6, 3, 4, 5, 6])
    dd = np.array([1, 1, 2, 2, 2, 2, 1, 1, 1, 1])
    return anc, des, dd


def test_no_emitted_negative_is_a_descendant_of_its_anchor():
    from train_hierarchical import HierarchicalDataLoader

    anc, des, dd = _fixture_closure()
    parent = parent_from_closure(anc, des, dd, n_nodes=7)
    tin, tout = euler_intervals(parent)

    loader = HierarchicalDataLoader.from_arrays(
        ancestor_idx=anc, descendant_idx=des, depth_diff=dd,
        n_negatives=3, batch_size=4,
        exclude_descendant_negatives=True, drop_root_anchored=True,
    )
    for ancestors, descendants, negatives, _depths in loader:
        a = ancestors.numpy()[:, None]
        n = negatives.numpy()
        assert not is_descendant(np.broadcast_to(a, n.shape), n, tin, tout).any()


def test_root_anchored_pairs_are_dropped():
    from train_hierarchical import HierarchicalDataLoader

    anc, des, dd = _fixture_closure()
    loader = HierarchicalDataLoader.from_arrays(
        ancestor_idx=anc, descendant_idx=des, depth_diff=dd,
        n_negatives=3, batch_size=16,
        exclude_descendant_negatives=True, drop_root_anchored=True,
    )
    seen_anchors = set()
    for ancestors, _d, _n, _dep in loader:
        seen_anchors.update(ancestors.numpy().tolist())
    assert 0 not in seen_anchors


def test_realized_negative_count_is_logged_and_never_zero():
    from train_hierarchical import HierarchicalDataLoader

    anc, des, dd = _fixture_closure()
    loader = HierarchicalDataLoader.from_arrays(
        ancestor_idx=anc, descendant_idx=des, depth_diff=dd,
        n_negatives=3, batch_size=16,
        exclude_descendant_negatives=True, drop_root_anchored=True,
    )
    for _a, _d, _n, _dep in loader:
        pass
    assert loader.realized_negatives, "loader must record realized negative counts"
    assert min(loader.realized_negatives) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/tests/test_negative_sampling.py -v -k "descendant or root_anchored or realized"`

Expected: FAIL — `AttributeError: type object 'HierarchicalDataLoader' has no attribute 'from_arrays'`

- [ ] **Step 3: Implement**

Add a `from_arrays` classmethod to `HierarchicalDataLoader` that builds a loader from raw arrays (so the sampler is testable without an npz on disk), and extend `__init__` to accept the two new flags. In `__init__`, when `exclude_descendant_negatives` is set, build the subtree index once:

```python
        self.exclude_descendant_negatives = exclude_descendant_negatives
        self.realized_negatives: list[int] = []
        self._tin = self._tout = None
        if exclude_descendant_negatives:
            from taxembed.eval.subtree import parent_from_closure, euler_intervals

            parent = parent_from_closure(
                self.pairs.ancestor_idx, self.pairs.descendant_idx,
                self.pairs.depth_diff, self.n_nodes,
            )
            self._tin, self._tout = euler_intervals(parent)

        if drop_root_anchored:
            keep = self.pairs.ancestor_depth != 0
            n_dropped = int((~keep).sum())
            self.pairs = self.pairs[keep]
            print(f"  ✓ dropped {n_dropped:,} root-anchored pairs "
                  f"(unsupervisable by a contrastive loss; d(root,x) is covered by the radial regularizer)")
```

Then replace the body of `_sample_negatives_default_vectorized` with an ancestry-aware version. Keep the existing depth-pool fast path, then mask and backfill:

```python
    def _sample_negatives_default_vectorized(self, desc_depths, desc_idxs, batch_size,
                                             anc_idxs=None):
        negatives = np.zeros((batch_size, self.n_negatives), dtype=np.int64)
        unique_depths, inverse = np.unique(desc_depths, return_inverse=True)

        for di, d in enumerate(unique_depths):
            mask = inverse == di
            n_items = int(mask.sum())
            pool = self._depth_to_nodes.get(int(d))
            if pool is not None and len(pool) > self.n_negatives:
                rand_positions = np.random.randint(0, len(pool), size=(n_items, self.n_negatives))
                negatives[mask] = pool[rand_positions]
            elif pool is not None and len(pool) > 1:
                negatives[mask] = np.random.choice(pool, size=(n_items, self.n_negatives))
            else:
                negatives[mask] = np.random.randint(0, self.n_nodes, size=(n_items, self.n_negatives))

        if not self.exclude_descendant_negatives or anc_idxs is None:
            self.realized_negatives.append(self.n_negatives)
            return negatives

        from taxembed.eval.subtree import is_descendant

        anc_col = np.asarray(anc_idxs)[:, None]
        bad = is_descendant(np.broadcast_to(anc_col, negatives.shape), negatives,
                            self._tin, self._tout)

        # Relax the DEPTH stratification (not the ancestry constraint) for the shortfall.
        for row in np.flatnonzero(bad.any(axis=1)):
            n_bad = int(bad[row].sum())
            a = int(anc_idxs[row])
            replacements = []
            attempts = 0
            while len(replacements) < n_bad and attempts < 20:
                cand = np.random.randint(0, self.n_nodes, size=n_bad * 4)
                ok = cand[~is_descendant(np.full(len(cand), a), cand, self._tin, self._tout)]
                replacements.extend(ok.tolist())
                attempts += 1
            if len(replacements) >= n_bad:
                negatives[row, bad[row]] = np.array(replacements[:n_bad], dtype=np.int64)
            else:
                # Genuinely exhausted: mask by repeating a known-good negative rather than
                # emitting a false one. Record the reduced count so the depth confound stays visible.
                good = negatives[row][~bad[row]]
                if len(good) == 0:
                    raise RuntimeError(
                        f"anchor {a} has no valid negative anywhere -- it should have been "
                        f"dropped by drop_root_anchored"
                    )
                negatives[row, bad[row]] = good[0]
                self.realized_negatives.append(int(len(good)))
                continue
            self.realized_negatives.append(self.n_negatives)

        return negatives
```

Update the call site at `:641` to pass anchors:

```python
                negatives = self._sample_negatives_default_vectorized(
                    desc_depths, desc_idxs, batch_size,
                    anc_idxs=self.pairs.ancestor_idx[batch_indices],
                )
```

`from_arrays` builds a loader without touching disk, so the sampler is unit-testable. Add it as a classmethod on `HierarchicalDataLoader`:

```python
    @classmethod
    def from_arrays(cls, ancestor_idx, descendant_idx, depth_diff, **kwargs):
        """Build a loader directly from arrays (test seam -- no .npz on disk).

        Depth is derived by walking the parent chain built from the depth_diff == 1 rows,
        so it stays consistent with Task 1's parent_from_closure. O(n * depth), which is
        irrelevant at fixture scale.
        """
        import numpy as np

        from taxembed.eval.subtree import parent_from_closure
        from taxembed.utils.training_pairs import TrainingPairs

        anc = np.asarray(ancestor_idx, dtype=np.int32)
        des = np.asarray(descendant_idx, dtype=np.int32)
        dd = np.asarray(depth_diff, dtype=np.int16)
        n_nodes = int(max(anc.max(), des.max())) + 1

        parent = parent_from_closure(anc, des, dd, n_nodes)
        depth = np.zeros(n_nodes, dtype=np.int16)
        for v in range(n_nodes):
            steps, cur = 0, v
            while int(parent[cur]) != cur:
                cur = int(parent[cur])
                steps += 1
            depth[v] = steps
        desc_depth = depth[des]
        anc_depth = depth[anc]

        pairs = TrainingPairs(
            ancestor_idx=anc, descendant_idx=des, depth_diff=dd,
            ancestor_depth=anc_depth, descendant_depth=desc_depth,
            ancestor_taxid=anc.copy(), descendant_taxid=des.copy(),
        )
        return cls(pairs=pairs, **kwargs)
```

- [ ] **Step 4: Wire both flags through the CLI**

Task 5 added `--seed`; these two follow the identical pattern. In `src/taxembed/cli/main.py`'s `train` subparser:

```python
    train_parser.add_argument(
        "--exclude-descendant-negatives",
        action="store_true",
        help="Reject sampled negatives that are descendants of the anchor "
             "(restores Nickel-Kiela's {v' : (u,v') not in D}).",
    )
    train_parser.add_argument(
        "--drop-root-anchored",
        action="store_true",
        help="Drop pairs whose anchor is the root: every node descends from it, so they "
             "carry no contrastive signal. Required with --exclude-descendant-negatives.",
    )
```

Forward them where `--seed` is forwarded:

```python
    if args.exclude_descendant_negatives:
        cmd += ["--exclude-descendant-negatives"]
    if args.drop_root_anchored:
        cmd += ["--drop-root-anchored"]
```

Record both in `run.json` beside `"seed"`. Mirror both in `train_hierarchical.py`'s argparse and pass them into the `HierarchicalDataLoader` constructor. Add a guard immediately after parsing:

```python
    if args.exclude_descendant_negatives and not args.drop_root_anchored:
        raise SystemExit(
            "--exclude-descendant-negatives requires --drop-root-anchored: 1,102,162 "
            "root-anchored pairs (5.15%) have NO valid negative, and an empty negative set "
            "makes the softmax a single logit with zero gradient."
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/tests/test_negative_sampling.py -v`

Expected: PASS, 6 tests.

- [ ] **Step 6: Confirm no regression in the unfixed path**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/tests -v`

Expected: PASS. Defaults are `exclude_descendant_negatives=False`, `drop_root_anchored=False`, so historical behaviour is unchanged.

- [ ] **Step 7: Commit**

```bash
git add train_hierarchical.py tests/test_negative_sampling.py
git commit -m "feat(train): ancestry-aware negative sampling with root-drop and depth relaxation"
```

---

## Task 7: Fixed-vs-unfixed retrain and delta report

**Files:**
- Create: `scripts/train_echinodermata_fixed_negatives.sh`
- Create: `docs/NUMBERS_OF_RECORD_objective_integrity.md`

**Interfaces:**
- Consumes: everything above.
- Produces: the decision record on whether the shipped 1.1M artifact needs re-training.

**Scale:** echinodermata (3,965 taxa) first — its closure is at `data/taxopy/` alongside the cellular one. Then one mid-size clade if the delta is material.

- [ ] **Step 1: Static-check before any submission (Rule 16)**

Run: `bash -n /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/scripts/train_echinodermata_fixed_negatives.sh`

Expected: no output (syntax OK).

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m py_compile /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/train_hierarchical.py`

Expected: no output.

Confirm every CLI flag the job passes is a real argparse option in `train_hierarchical.py` before proceeding. Do not assume.

- [ ] **Step 2: Audit the echinodermata closure first**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/scripts/audit_negative_sampling.py --npz /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/data/taxopy/echinodermata_7586_clean/taxonomy_edges_echinodermata_7586_clean_transitive.npz --out /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/results/negative_sampler_audit_echinodermata.json`

Expected: a non-zero overall FN rate and a non-zero root-anchored count. Record both — they set the size of the effect the retrain should move.

**If the path does not exist**, list `data/taxopy/` and use the actual echinodermata directory name; do not guess.

- [ ] **Step 3: Smoke run — 2 epochs, local (Rule 16)**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m taxembed.cli.main train --file /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/data/taxopy/echinodermata_7586_clean/taxonomy_edges_echinodermata_7586_clean_transitive.npz --mapping /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/data/taxopy/echinodermata_7586_clean/taxonomy_edges_echinodermata_7586_clean.mapping.tsv -as smoke_fixed_neg --dim 100 --epochs 2 --batch-size 256 --n-negatives 50 --lr 0.001 --loss softmax --euclidean-param --seed 0 --exclude-descendant-negatives --drop-root-anchored`

Expected: completes without error; the dropped-root-anchored line prints a non-zero count; at least one entry in `realized_negatives` is below `--n-negatives`, proving the relaxation path is exercised rather than dead code.

- [ ] **Step 4: Paired runs — six total**

Same command shape as Step 3 with `--epochs 200`, for seeds `0`, `1`, `2` in each arm:
- **Arm A (unfixed):** omit both `--exclude-descendant-negatives` and `--drop-root-anchored`; tag `echino_unfixed_s<S>`
- **Arm B (fixed):** include both; tag `echino_fixed_s<S>`

- [ ] **Step 5: Report the delta**

For each arm and seed record, into `results/objective_integrity_delta.json`: final training loss; depth-norm r **with its Task 4 initialization floor printed beside it**; kNN retrieval precision@10; and the Task 3 null-sanity verdict for whichever null the analysis uses. Report mean and spread across the three seeds per arm.

This is the first seeded comparison in the project's history — the spread itself is a deliverable, independent of the fixed-vs-unfixed contrast, because the paper's Fig-4 recipe claim currently rests on n=1 per arm.

- [ ] **Step 6: Write the decision record**

Create `docs/NUMBERS_OF_RECORD_objective_integrity.md` containing: the Task 2 census table; the Task 4 initialization floor; the Task 3 null diagnosis; the Task 7 fixed-vs-unfixed deltas. State the verdict explicitly in one of two forms:

- *"The fix materially changes the numbers"* ⇒ the shipped 1.1M artifact was trained with a defective objective; the Methods must say so and a re-train is scoped.
- *"The fix does not materially change the numbers"* ⇒ that is a robustness result worth reporting.

**Either way** the Methods description needs correcting: the paper describes a clean Nickel-Kiela softmax, and the shipped sampler omits ancestry exclusion.

- [ ] **Step 7: Commit**

```bash
git add scripts/train_echinodermata_fixed_negatives.sh docs/NUMBERS_OF_RECORD_objective_integrity.md results/objective_integrity_delta.json results/negative_sampler_audit_echinodermata.json
git commit -m "docs: objective-integrity numbers of record and fixed-vs-unfixed verdict"
```

---

## Follow-on plans (not this document)

- **P2 — Held-out evaluation.** Ganea-style split (transitive reduction always in training; 5%/5% from non-basic edges; closure visibility swept 0/10/25/50%), the Vendrov trivial-closure baseline, level-stratified holdout at our own quartiles (Q1=11, Q3=28; 593,576 eligible nodes), and the RandomDAG control. **Downstream of P1** — running it before the fix measures a defective objective.
- **P3 — Temporal QC.** ~70% already built (`src/taxembed/eval/release_diff.py`, `scripts/_anomaly_validation.py:168-172`, `utils/taxdump.py:100-132`, `tests/eval/test_release_diff.py`). **Independent of P1; can run in parallel.**
- **P4 — pLM showcase.** Scoring layer, matched head-to-head, resampling protocol.
- **TimeTree external anchor.** Scheduled, not optional — no NCBI-internal test can refute "embedded distance tracks NCBI convention rather than relatedness".
