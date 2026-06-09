# TaxEmbed Application #3 — sampling-bias quantification (tree-partition coverage) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Quantify *sampling bias* across the tree of life — which clades are over/under-represented in a given taxon subset (headline: **UniProt reference-proteome coverage**) — and do it with statistical defensibility. The load-bearing rigor point (spec §9B): **coverage is defined on the TREE PARTITION (covered lineages / covered-subtree fraction per clade), NOT on hyperbolic ball-volume** (hyperbolic volume explodes with radius → a meaningless denominator). The embedding is used only for (a) an optional self-normalizing **kNN-radius geometric density** as a complement, conditioned on clade AND depth vs a uniform-coverage null, and (b) continuous interpolation / visualization. We benchmark tree-partition coverage against **Faith's PD** on subsets where PD is computable (framed as a *complement* to PD — faster, differentiable, scales), and we report coverage against a **study-effort covariate** (#descendants / #sequences) so the novel signal is **under-coverage BEYOND known sequencing priority**. Develop on **eukaryota 877k** now; re-run on cellular 1.1M when it lands, re-deriving every rank/depth-dependent baseline (spec §9C).

**Architecture:** A new importable subpackage module `src/taxembed/eval/coverage.py` holds pure, unit-tested core logic (no I/O): tree-partition coverage, kNN-radius density, the uniform-coverage null, Faith's PD on a subset, and the study-effort association. It **reuses Plan 1's foundation** — `taxembed.eval.treedist.TreeDistance` (binary-lifting LCA + `path_length`, for PD edge accounting and depth), `taxembed.eval.bootstrap.taxon_bootstrap_ci` (taxon-level CIs), and the float64 Poincaré distance machinery (`scripts/_negative_hardness.numpy_poincare_distance` + the matmul-expanded `_batch_distances` from `scripts/knn_purity_hyperbolic.py`) for the geometric density. Two thin CLIs wire it to real data: `scripts/fetch_uniprot_proteomes.py` (a named, **cached** fetch + roll-up of the UniProt reference-proteome taxon list) and `scripts/coverage_bias.py` (the analysis: load embedding + taxonomy, build `TreeDistance`, join the covered set, compute per-clade coverage + density + PD benchmark + study-effort association, write JSON/TSV + plots). This mirrors the package/scripts split of `taxembed` and Plan 1; the `scripts/*_hyperbolic.py` monoliths are **not** extended — we build testable cores.

**Tech Stack:** Python 3.12, numpy, scipy (`scipy.stats.spearmanr`, `scipy.stats` for the association test), pandas, pytest, `urllib.request` for the UniProt REST fetch (no new deps; matches `src/taxembed/utils/taxdump.py`'s fetch idiom). Faith's PD reuses `TreeDistance` for edge enumeration. All headline numbers report a **delta over a uniform-coverage null** with **taxon-level bootstrap CIs** (spec §9B).

**Run context:** venv at `.venv` — run `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python`. Always invoke the analysis with **explicit `--checkpoint`/`--mapping`** and the **`final` (ep200)** checkpoint (`artifacts/tags/eukaryota_canonical/eukaryota_canonical.pth` — the unsuffixed file is the final; `run.json` holds LRZ container paths and tag-resolution defaults to `_best`, so never rely on tag resolution — spec §9C local-path gotcha). The covered-set fetch (Task 6) **assumes network access** for the UniProt REST call but is cached to disk so the analysis task (Task 7) is offline and reproducible.

**Dependency on Plan 1:** Tasks 1–5 here import `taxembed.eval.{treedist,bootstrap}` from Plan 1. If executing this plan standalone before Plan 1's foundation is merged, run Plan 1 Task 1 (treedist) and Task 4 (bootstrap) first — they are the only two dependencies. Every test below that needs them constructs them directly, so the dependency is explicit and checkable.

---

## File structure

- Create `src/taxembed/eval/coverage.py` — pure core:
  - `tree_partition_coverage(parent, depth, covered_mask, clade_roots, td)` → per-clade `{covered_lineages, total_lineages, coverage_fraction, covered_subtree_fraction}`.
  - `knn_radius_density(emb, query_idx, pool_idx, k)` → per-query geometric density = `1 / mean_kNN_radius` (self-normalizing).
  - `uniform_coverage_null(total_lineages_per_clade, n_covered, n_draws, seed)` → null distribution of per-clade coverage fraction when the SAME number of taxa are marked covered at random.
  - `coverage_delta_over_null(observed_fraction, null_samples)` → `(z, p_empirical)` of observed vs uniform-coverage null.
  - `faith_pd(parent, depth, leaf_mask, td)` → Faith's Phylogenetic Diversity (#edges in the minimal subtree spanning the covered leaves to the root) on a subset.
  - `study_effort_association(coverage_fraction, study_effort)` → Spearman ρ + residual under-coverage per clade (coverage beyond what effort predicts).
- Create `scripts/fetch_uniprot_proteomes.py` — CLI: fetch the UniProt reference-proteome taxon list via REST, roll up sub-leaf strain taxids to the nearest **embedded** ancestor, cache to a TSV. Named, idempotent (skips if cache fresh).
- Create `scripts/coverage_bias.py` — CLI: load embedding + taxonomy + covered-set TSV, build `TreeDistance`, compute per-clade tree-partition coverage + density + PD benchmark + study-effort association vs the uniform-coverage null with taxon-bootstrap CIs; write JSON/TSV + a coverage-vs-effort scatter.
- Create tests: `tests/eval/test_coverage.py`, `tests/eval/test_coverage_density.py`, `tests/eval/test_coverage_null.py`, `tests/eval/test_faith_pd.py`, `tests/eval/test_study_effort.py`, `tests/eval/test_uniprot_rollup.py`, `tests/eval/test_coverage_bias_cli.py`.

The coverage core takes **integer node-index arrays** (0..N-1, the embedding's own indexing) + a boolean `covered_mask`, so it is pure and decoupled from taxids; the CLIs build the index↔taxid↔parent maps from the mapping + taxdump (reusing `analyze_hierarchy_hyperbolic`'s loaders, exactly as Plan 1's CLI does).

---

### Task 1: Tree-partition coverage core (the headline metric)

**Files:**
- Create: `src/taxembed/eval/coverage.py`
- Test: `tests/eval/test_coverage.py`

- [ ] **Step 1: Write the failing test**

`tests/eval/test_coverage.py`:
```python
import numpy as np
from taxembed.eval.treedist import TreeDistance
from taxembed.eval.coverage import tree_partition_coverage

# Hand tree (node id : parent), root = 0 (self-loop):
#   0
#   ├─1            ├─2
#   │ ├─3 (leaf)   │ └─6 (leaf)
#   │ └─4          │
#   │   └─5 (leaf) │
# parent[root] = root convention.
PARENT = np.array([0, 0, 0, 1, 1, 4, 2])
DEPTH = np.array([0, 1, 1, 2, 2, 3, 2])


def _td():
    return TreeDistance(PARENT, DEPTH)


def test_coverage_fraction_per_clade_counts_leaves_under_clade():
    # leaves are {3, 5, 6}. Mark 3 and 6 covered; 5 uncovered.
    covered = np.zeros(7, dtype=bool)
    covered[[3, 6]] = True
    # clade roots: node 1 (subtree leaves {3,5}), node 2 (subtree leaves {6})
    out = tree_partition_coverage(PARENT, DEPTH, covered, clade_roots=[1, 2], td=_td())
    # clade 1: 1 of 2 leaves covered -> 0.5 ; clade 2: 1 of 1 -> 1.0
    assert out[1]["total_lineages"] == 2
    assert out[1]["covered_lineages"] == 1
    assert abs(out[1]["coverage_fraction"] - 0.5) < 1e-9
    assert out[2]["total_lineages"] == 1
    assert abs(out[2]["coverage_fraction"] - 1.0) < 1e-9


def test_covered_subtree_fraction_counts_internal_nodes_on_covered_paths():
    # covered_subtree_fraction = (# nodes in the clade subtree that lie on a path from the
    # clade root to a covered leaf) / (# nodes in the clade subtree).
    covered = np.zeros(7, dtype=bool)
    covered[[3]] = True  # only leaf 3 covered under clade 1
    out = tree_partition_coverage(PARENT, DEPTH, covered, clade_roots=[1], td=_td())
    # clade-1 subtree nodes = {1,3,4,5} (4 nodes). Path root1->leaf3 touches {1,3} (2 nodes).
    assert out[1]["subtree_size"] == 4
    assert out[1]["covered_subtree_nodes"] == 2
    assert abs(out[1]["covered_subtree_fraction"] - 0.5) < 1e-9


def test_full_coverage_gives_fraction_one():
    covered = np.zeros(7, dtype=bool)
    covered[[3, 5, 6]] = True  # all leaves
    out = tree_partition_coverage(PARENT, DEPTH, covered, clade_roots=[1, 2], td=_td())
    assert abs(out[1]["coverage_fraction"] - 1.0) < 1e-9
    assert abs(out[2]["coverage_fraction"] - 1.0) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest tests/eval/test_coverage.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'taxembed.eval.coverage'`

- [ ] **Step 3: Implement `coverage.py` (this task's part)**

`src/taxembed/eval/coverage.py`:
```python
"""Sampling-bias / coverage core for TaxEmbed Application #3 (no I/O).

Coverage is defined on the TREE PARTITION, not on hyperbolic ball-volume: hyperbolic
volume explodes with radius, so a geometric denominator is meaningless (spec §9B). For a
clade we report (covered leaf-lineages / total leaf-lineages) and a covered-subtree fraction.
The embedding is used only for an OPTIONAL self-normalizing kNN-radius density (knn_radius_density)
and for continuous interpolation / viz — never as the coverage denominator.

Operates on integer node ids 0..N-1 (the embedding's own indexing). `parent` is an int array
(root points to itself); `depth` is the root-distance in edges; `covered_mask` is a bool array.
Reuses taxembed.eval.treedist.TreeDistance (Plan 1) for tree geometry.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List

import numpy as np


def _children_index(parent: np.ndarray) -> Dict[int, List[int]]:
    """child lists keyed by parent id; the root's self-loop is not recorded as a child of itself."""
    kids: Dict[int, List[int]] = defaultdict(list)
    for node, par in enumerate(parent):
        if node != par:                       # skip the root self-loop
            kids[int(par)].append(node)
    return kids


def _is_leaf(parent: np.ndarray) -> np.ndarray:
    """Boolean leaf mask: a node is a leaf iff it is no other node's parent (root excluded if childless)."""
    n = len(parent)
    has_child = np.zeros(n, dtype=bool)
    for node, par in enumerate(parent):
        if node != par:
            has_child[int(par)] = True
    return ~has_child


def _subtree_nodes(root: int, kids: Dict[int, List[int]]) -> List[int]:
    """All node ids in the subtree rooted at `root` (inclusive), iterative DFS."""
    out, stack = [], [int(root)]
    while stack:
        v = stack.pop()
        out.append(v)
        stack.extend(kids.get(v, ()))
    return out


def tree_partition_coverage(parent: np.ndarray, depth: np.ndarray, covered_mask: np.ndarray,
                            clade_roots: Iterable[int], td=None) -> Dict[int, dict]:
    """Per-clade tree-partition coverage.

    For each clade root r:
      - total_lineages       = # leaf nodes in r's subtree
      - covered_lineages     = # of those leaves with covered_mask True
      - coverage_fraction    = covered_lineages / total_lineages  (0 if no leaves)
      - subtree_size         = # nodes in r's subtree (incl. internal)
      - covered_subtree_nodes= # subtree nodes lying on a root->covered-leaf path
      - covered_subtree_fraction = covered_subtree_nodes / subtree_size

    `td` is accepted for API symmetry / future depth-weighting; not required here.
    """
    parent = np.asarray(parent, dtype=np.int64)
    covered_mask = np.asarray(covered_mask, dtype=bool)
    kids = _children_index(parent)
    leaf = _is_leaf(parent)
    out: Dict[int, dict] = {}
    for r in clade_roots:
        r = int(r)
        nodes = _subtree_nodes(r, kids)
        leaves = [v for v in nodes if leaf[v]]
        total = len(leaves)
        covered_leaves = [v for v in leaves if covered_mask[v]]
        cov = len(covered_leaves)
        # mark every node on a root->covered-leaf path by walking each covered leaf up to r
        on_path = set()
        for lf in covered_leaves:
            cur = lf
            while True:
                on_path.add(cur)
                if cur == r:
                    break
                nxt = int(parent[cur])
                if nxt == cur:                 # hit the tree root before r (shouldn't happen) — stop
                    break
                cur = nxt
        out[r] = {
            "total_lineages": total,
            "covered_lineages": cov,
            "coverage_fraction": (cov / total) if total else 0.0,
            "subtree_size": len(nodes),
            "covered_subtree_nodes": len(on_path),
            "covered_subtree_fraction": (len(on_path) / len(nodes)) if nodes else 0.0,
        }
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest tests/eval/test_coverage.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings add src/taxembed/eval/coverage.py tests/eval/test_coverage.py
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings commit -m "feat(eval): tree-partition coverage core for sampling-bias (App #3)"
```

---

### Task 2: kNN-radius geometric density (the self-normalizing complement)

**Files:**
- Edit: `src/taxembed/eval/coverage.py` (add `knn_radius_density`)
- Test: `tests/eval/test_coverage_density.py`

The geometric density is the *optional* complement to the tree-partition coverage (spec §9B: "If geometric density kept: kNN-radius density (self-normalizing, dimension-robust)"). It reuses the same float64-validated Poincaré distance as `knn_purity_hyperbolic.py`'s `_batch_distances`. We implement it on small inputs here with the broadcasting reference `numpy_poincare_distance`; the CLI uses the matmul-expanded batched form for scale.

- [ ] **Step 1: Write the failing test**

`tests/eval/test_coverage_density.py`:
```python
import numpy as np
from taxembed.eval.coverage import knn_radius_density


def test_density_higher_in_tight_cluster():
    # pool: a tight cluster of 5 near origin + 5 spread out near the boundary
    tight = np.array([[0.01, 0.0], [0.0, 0.01], [-0.01, 0.0], [0.0, -0.01], [0.005, 0.005]])
    spread = np.array([[0.8, 0.0], [0.0, 0.85], [-0.82, 0.0], [0.0, -0.8], [0.6, 0.6]])
    emb = np.vstack([tight, spread]).astype(np.float64)
    pool_idx = np.arange(10)
    dens = knn_radius_density(emb, query_idx=pool_idx, pool_idx=pool_idx, k=2)
    # the tight cluster's members have smaller mean-kNN-radius -> higher density
    assert dens[:5].mean() > dens[5:].mean()


def test_density_is_positive_and_per_query_shaped():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((20, 4)) * 0.1
    q = np.array([0, 1, 2])
    dens = knn_radius_density(emb, query_idx=q, pool_idx=np.arange(20), k=3)
    assert dens.shape == (3,)
    assert (dens > 0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest tests/eval/test_coverage_density.py -v`
Expected: FAIL — `ImportError: cannot import name 'knn_radius_density'`

- [ ] **Step 3: Implement `knn_radius_density` in `coverage.py`**

Add the import at the top of `coverage.py` (after the existing imports):
```python
import sys
from pathlib import Path

# Reuse the float64 Poincaré distance reference (scripts/_negative_hardness.py).
_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from _negative_hardness import numpy_poincare_distance  # noqa: E402
```

Add the function:
```python
def knn_radius_density(emb: np.ndarray, query_idx: np.ndarray, pool_idx: np.ndarray,
                       k: int) -> np.ndarray:
    """Self-normalizing kNN-radius density per query (spec §9B).

    density(q) = 1 / mean(distance to its k nearest pool neighbours, self excluded).
    Larger = denser local sampling. Dimension-robust: a radius, not a volume — so it does
    NOT inherit the hyperbolic volume-explosion problem. Uses the float64 Poincaré reference;
    callers at scale should swap in the matmul-expanded _batch_distances (knn_purity_hyperbolic.py).
    Returns shape (len(query_idx),).
    """
    emb = np.asarray(emb, dtype=np.float64)
    query_idx = np.asarray(query_idx, dtype=np.int64)
    pool_idx = np.asarray(pool_idx, dtype=np.int64)
    q = emb[query_idx]                                              # (Q, D)
    p = emb[pool_idx]                                              # (P, D)
    d = numpy_poincare_distance(q[:, None, :], p[None, :, :])      # (Q, P)
    # exclude self where a query is also in the pool
    self_mask = query_idx[:, None] == pool_idx[None, :]
    d = d.copy()
    d[self_mask] = np.inf
    keff = min(k, p.shape[0] - 1) if p.shape[0] > 1 else 1
    part = np.partition(d, keff - 1, axis=1)[:, :keff]             # k nearest per query
    mean_r = part.mean(axis=1)
    return 1.0 / np.maximum(mean_r, 1e-12)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest tests/eval/test_coverage_density.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings add src/taxembed/eval/coverage.py tests/eval/test_coverage_density.py
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings commit -m "feat(eval): kNN-radius geometric density (self-normalizing coverage complement)"
```

---

### Task 3: Uniform-coverage null + delta-over-null

**Files:**
- Edit: `src/taxembed/eval/coverage.py` (add `uniform_coverage_null`, `coverage_delta_over_null`)
- Test: `tests/eval/test_coverage_null.py`

The uniform-coverage null (spec §9B) marks the **same number** of taxa "covered" at random and recomputes per-clade coverage, so the headline can report observed coverage as a **departure** from what random sampling of equal size would give — the Goodhart guard for #3 (a clade that is small simply gets high coverage by chance; the null corrects for that).

- [ ] **Step 1: Write the failing test**

`tests/eval/test_coverage_null.py`:
```python
import numpy as np
from taxembed.eval.coverage import uniform_coverage_null, coverage_delta_over_null


def test_uniform_null_mean_matches_global_rate():
    # 1000 leaves total, 200 covered globally -> global rate 0.2. A clade with 50 leaves:
    # expected covered under uniform null ~ 50 * 0.2 = 10 -> fraction ~0.2.
    samples = uniform_coverage_null(clade_total=50, global_total=1000, n_covered=200,
                                    n_draws=2000, seed=0)
    assert samples.shape == (2000,)
    assert abs(samples.mean() - 0.2) < 0.02          # hypergeometric mean = clade_total*p/clade_total


def test_delta_over_null_flags_over_and_under_coverage():
    rng = np.random.default_rng(0)
    null = rng.normal(0.2, 0.05, 5000)
    z_hi, p_hi = coverage_delta_over_null(observed_fraction=0.5, null_samples=null)
    z_lo, p_lo = coverage_delta_over_null(observed_fraction=0.02, null_samples=null)
    assert z_hi > 0 and z_lo < 0                       # over- vs under-covered
    assert p_hi < 0.05 and p_lo < 0.05                 # both far in the tails
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest tests/eval/test_coverage_null.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement the two functions in `coverage.py`**

```python
def uniform_coverage_null(clade_total: int, global_total: int, n_covered: int,
                          n_draws: int = 2000, seed: int = 0) -> np.ndarray:
    """Null distribution of a clade's coverage_fraction if `n_covered` of `global_total` leaves
    were marked covered uniformly at random (hypergeometric). Returns coverage fractions, shape (n_draws,).

    spec §9B: a clade is "over-covered" only relative to this equal-size random-coverage expectation,
    NOT relative to 1.0 — small clades hit high coverage by chance.
    """
    rng = np.random.default_rng(seed)
    if clade_total <= 0:
        return np.zeros(n_draws)
    drawn = rng.hypergeometric(ngood=n_covered, nbad=global_total - n_covered,
                               nsample=clade_total, size=n_draws)
    return drawn / clade_total


def coverage_delta_over_null(observed_fraction: float, null_samples: np.ndarray):
    """Return (z, p_empirical): z = (observed - null_mean)/null_std ; two-sided empirical p
    (fraction of |null - null_mean| >= |observed - null_mean|, +1 smoothing)."""
    null = np.asarray(null_samples, dtype=np.float64)
    mu, sd = float(null.mean()), float(null.std())
    z = (observed_fraction - mu) / sd if sd > 0 else 0.0
    obs_dev = abs(observed_fraction - mu)
    p = (1.0 + np.sum(np.abs(null - mu) >= obs_dev)) / (len(null) + 1.0)
    return z, float(p)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest tests/eval/test_coverage_null.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings add src/taxembed/eval/coverage.py tests/eval/test_coverage_null.py
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings commit -m "feat(eval): uniform-coverage null + delta-over-null (App #3 Goodhart guard)"
```

---

### Task 4: Faith's PD benchmark

**Files:**
- Edit: `src/taxembed/eval/coverage.py` (add `faith_pd`)
- Test: `tests/eval/test_faith_pd.py`

Spec §9B/§9A: show tree-partition coverage **correlates with Faith's PD** on subsets where PD is computable, then claim the win on speed/differentiability/scale (a **complement** to PD, not a replacement). On a topology-only tree (no branch lengths — see spec §9F), Faith's PD reduces to the **number of edges in the minimal subtree** connecting the covered leaves up to the clade root (each edge weight = 1). That is exactly the count of distinct nodes-on-covered-paths minus one (the root), which we already compute — but we expose it as a standalone `faith_pd` so the benchmark is explicit and independently testable.

- [ ] **Step 1: Write the failing test**

`tests/eval/test_faith_pd.py`:
```python
import numpy as np
from taxembed.eval.treedist import TreeDistance
from taxembed.eval.coverage import faith_pd

#   0
#   ├─1            ├─2
#   │ ├─3 (leaf)   │ └─6 (leaf)
#   │ └─4          │
#   │   └─5 (leaf) │
PARENT = np.array([0, 0, 0, 1, 1, 4, 2])
DEPTH = np.array([0, 1, 1, 2, 2, 3, 2])


def _td():
    return TreeDistance(PARENT, DEPTH)


def test_pd_counts_edges_in_minimal_spanning_subtree():
    # cover leaves 3 and 5 under clade root 1. Edges in the spanning subtree rooted at 1:
    #   1-3, 1-4, 4-5  -> 3 edges
    leaf_mask = np.zeros(7, dtype=bool)
    leaf_mask[[3, 5]] = True
    assert faith_pd(PARENT, DEPTH, leaf_mask, root=1, td=_td()) == 3


def test_pd_single_leaf_equals_its_depth_to_root():
    # only leaf 5 under clade 1: edges 1-4, 4-5 -> 2
    leaf_mask = np.zeros(7, dtype=bool)
    leaf_mask[[5]] = True
    assert faith_pd(PARENT, DEPTH, leaf_mask, root=1, td=_td()) == 2


def test_pd_zero_when_no_leaves_covered():
    leaf_mask = np.zeros(7, dtype=bool)
    assert faith_pd(PARENT, DEPTH, leaf_mask, root=1, td=_td()) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest tests/eval/test_faith_pd.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement `faith_pd` in `coverage.py`**

```python
def faith_pd(parent: np.ndarray, depth: np.ndarray, leaf_mask: np.ndarray,
             root: int, td=None) -> int:
    """Faith's Phylogenetic Diversity of the covered leaf set within the subtree rooted at `root`,
    on a TOPOLOGY-ONLY tree (every edge weight = 1, per spec §9F): the number of edges in the
    minimal subtree spanning the covered leaves up to `root`.

    Implementation: collect the set of nodes lying on any covered-leaf -> root path; PD = (#nodes - 1)
    when at least one leaf is covered (the root itself contributes no inbound edge), else 0.
    `td` accepted for symmetry / future branch-length weighting.
    """
    parent = np.asarray(parent, dtype=np.int64)
    leaf_mask = np.asarray(leaf_mask, dtype=bool)
    root = int(root)
    on_path = set()
    for lf in np.flatnonzero(leaf_mask):
        cur = int(lf)
        while True:
            on_path.add(cur)
            if cur == root:
                break
            nxt = int(parent[cur])
            if nxt == cur:                     # reached tree root before clade root — stop
                break
            cur = nxt
    return max(0, len(on_path) - 1) if on_path else 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest tests/eval/test_faith_pd.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings add src/taxembed/eval/coverage.py tests/eval/test_faith_pd.py
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings commit -m "feat(eval): Faith's PD (topology-only) for the coverage benchmark"
```

---

### Task 5: Study-effort association (under-coverage BEYOND sequencing priority)

**Files:**
- Edit: `src/taxembed/eval/coverage.py` (add `study_effort_association`)
- Test: `tests/eval/test_study_effort.py`

Spec §9B: coverage correlates with model-organism / economic status, so the **novel** signal is under-coverage *beyond* the known sequencing-priority confound. We report Spearman ρ(coverage, study-effort) AND the **residual** of coverage after regressing out (log) study-effort — clades with large negative residual are under-covered relative to how much they've been studied. Study-effort proxy = #descendants (or #sequences if available); both supplied by the caller.

- [ ] **Step 1: Write the failing test**

`tests/eval/test_study_effort.py`:
```python
import numpy as np
from taxembed.eval.coverage import study_effort_association


def test_positive_correlation_recovered():
    rng = np.random.default_rng(0)
    effort = rng.uniform(1, 1000, 200)
    coverage = np.clip(0.3 + 0.0005 * effort + rng.normal(0, 0.02, 200), 0, 1)
    out = study_effort_association(coverage, effort)
    assert out["spearman_rho"] > 0.5
    assert "residual" in out and out["residual"].shape == (200,)


def test_residual_flags_underflow_beyond_effort():
    # two clades with identical high effort but very different coverage:
    # the low-coverage one must have the more-negative residual.
    effort = np.array([500.0, 500.0, 100.0, 100.0])
    coverage = np.array([0.9, 0.1, 0.5, 0.5])
    out = study_effort_association(coverage, effort)
    # index 1 (effort 500, coverage 0.1) is under-covered for its effort
    assert out["residual"][1] < out["residual"][0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest tests/eval/test_study_effort.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement `study_effort_association` in `coverage.py`**

Add the scipy import near the top of `coverage.py`:
```python
from scipy.stats import spearmanr
```
Add the function:
```python
def study_effort_association(coverage_fraction: np.ndarray, study_effort: np.ndarray) -> dict:
    """Coverage vs a study-effort proxy (#descendants / #sequences), spec §9B.

    Returns:
      - spearman_rho, spearman_p: monotone association coverage~effort.
      - residual: coverage minus its OLS prediction from log1p(effort); large-negative residual =
        under-covered BEYOND what study effort predicts (the novel signal).
      - slope, intercept: the log-effort regression coefficients.
    """
    cov = np.asarray(coverage_fraction, dtype=np.float64)
    eff = np.asarray(study_effort, dtype=np.float64)
    rho, p = spearmanr(eff, cov)
    x = np.log1p(eff)
    # ordinary least squares: cov ~ a*x + b
    A = np.vstack([x, np.ones_like(x)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, cov, rcond=None)
    pred = slope * x + intercept
    residual = cov - pred
    return {
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "slope": float(slope),
        "intercept": float(intercept),
        "residual": residual,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest tests/eval/test_study_effort.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings add src/taxembed/eval/coverage.py tests/eval/test_study_effort.py
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings commit -m "feat(eval): study-effort association (under-coverage beyond sequencing priority)"
```

---

### Task 6: UniProt proteome fetch + strain roll-up CLI

**Files:**
- Create: `scripts/fetch_uniprot_proteomes.py`
- Test: `tests/eval/test_uniprot_rollup.py`

Spec §9B/§9E: the **headline covered-set** is the UniProt reference-proteome taxon list. UniProt publishes per-entry embeddings for all of UniProtKB plus a reference-proteome taxon list, fetchable via REST. The fetch is a **named, cached step**: we GET the proteome taxid list (paged), write the raw taxid list to a cache TSV, then **roll up** each fetched taxid to the nearest **embedded** ancestor (many proteomes are sub-leaf strains absent from the embedded clean tree) and write the rolled-up `covered_taxids.tsv`.

**Roll-up rule (define it):** for each UniProt proteome taxid `t`, walk parents via the taxdump until the first ancestor present in the **embedded taxid set** (`idx2tax` values); that ancestor is the covered node. If `t` itself is embedded, it is its own roll-up. If no ancestor in the chain is embedded (e.g. the taxid was deleted / outside the clade), drop it and count it in `n_dropped`. The pure roll-up function is unit-tested here; the network GET is isolated behind `--proteome-cache` so the test never hits the network.

- [ ] **Step 1: Write the failing test (pure roll-up, no network)**

`tests/eval/test_uniprot_rollup.py`:
```python
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from fetch_uniprot_proteomes import rollup_to_embedded


def test_rollup_to_nearest_embedded_ancestor():
    # parent chain: 9999 -> 555 -> 100 -> 1 (root). Embedded set = {100, 1}.
    parent_of = {9999: 555, 555: 100, 100: 1, 1: 1}
    embedded = {100, 1}
    # strain 9999 rolls up to 100 (first embedded ancestor)
    rolled, dropped = rollup_to_embedded([9999], parent_of, embedded)
    assert rolled == {100}
    assert dropped == 0


def test_rollup_self_when_already_embedded():
    parent_of = {100: 1, 1: 1}
    embedded = {100, 1}
    rolled, dropped = rollup_to_embedded([100], parent_of, embedded)
    assert rolled == {100}


def test_rollup_drops_taxid_with_no_embedded_ancestor():
    parent_of = {42: 7, 7: 7}            # chain 42->7->(root), neither embedded
    embedded = {100, 1}
    rolled, dropped = rollup_to_embedded([42], parent_of, embedded)
    assert rolled == set()
    assert dropped == 1


def test_rollup_dedups_strains_to_same_ancestor():
    parent_of = {9999: 100, 8888: 100, 100: 1, 1: 1}
    embedded = {100, 1}
    rolled, dropped = rollup_to_embedded([9999, 8888], parent_of, embedded)
    assert rolled == {100}               # both strains collapse to the one embedded ancestor
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest tests/eval/test_uniprot_rollup.py -v`
Expected: FAIL — script does not exist.

- [ ] **Step 3: Implement `scripts/fetch_uniprot_proteomes.py`**

```python
"""Fetch the UniProt reference-proteome taxon list (covered set for TaxEmbed App #3) and roll up
sub-leaf strain taxids to the nearest EMBEDDED ancestor.

Headline covered-set for sampling-bias quantification (spec §5#3, §9B, §9E). UniProt publishes a
reference-proteome taxon list via REST; we page through it, cache the raw taxids, then roll each up
to the nearest ancestor present in the embedded taxid set (the embedding's clean tree drops sub-leaf
strains). Network is touched ONLY on a cache miss, so the downstream analysis (coverage_bias.py) is
offline + reproducible.

Roll-up rule: walk parents until the first taxid in `embedded`; that is the covered node. taxid
already embedded -> itself. No embedded ancestor in the chain -> dropped (counted).

Usage:
    .venv/bin/python scripts/fetch_uniprot_proteomes.py \
        --mapping data/taxopy/eukaryota_2759_clean/taxonomy_edges_eukaryota_2759_clean.mapping.tsv \
        --data-dir data \
        --proteome-cache artifacts/tags/eukaryota_canonical/coverage/uniprot_proteome_taxids.tsv \
        -o artifacts/tags/eukaryota_canonical/coverage/covered_taxids.tsv
"""
import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT / "src"))

from analyze_hierarchy_hyperbolic import load_mapping, load_taxonomy_with_depth

UNIPROT_REST = "https://rest.uniprot.org/proteomes/search"


def fetch_proteome_taxids(query: str = "proteome_type:1", page_size: int = 500,
                          sleep: float = 0.34) -> list[int]:
    """Page the UniProt proteomes REST endpoint, returning organism taxids.

    query "proteome_type:1" == reference proteomes. Uses the cursor pagination in the Link header.
    Network call — wrapped by a disk cache in main(); not exercised by the unit tests.
    """
    taxids: list[int] = []
    params = {"query": query, "fields": "organism_id", "format": "tsv", "size": str(page_size)}
    url = f"{UNIPROT_REST}?{urllib.parse.urlencode(params)}"
    while url:
        req = urllib.request.Request(url, headers={"Accept": "text/plain"})
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode("utf-8")
            link = resp.headers.get("Link", "")
        lines = body.strip().splitlines()
        for ln in lines[1:]:                       # skip the header row
            ln = ln.strip()
            if ln.isdigit():
                taxids.append(int(ln))
        # cursor pagination: the next URL is in the Link header as <url>; rel="next"
        url = ""
        if 'rel="next"' in link:
            for part in link.split(","):
                if 'rel="next"' in part:
                    url = part.split(";")[0].strip().lstrip("<").rstrip(">")
                    break
        time.sleep(sleep)
    return taxids


def rollup_to_embedded(proteome_taxids, parent_of: dict, embedded: set):
    """Roll each proteome taxid up to its nearest embedded ancestor.

    Returns (rolled_set, n_dropped). `parent_of` maps taxid->parent taxid (root self-loops).
    """
    rolled = set()
    dropped = 0
    for t in proteome_taxids:
        t = int(t)
        cur = t
        seen = set()
        hit = None
        while cur not in seen:
            seen.add(cur)
            if cur in embedded:
                hit = cur
                break
            nxt = parent_of.get(cur, cur)
            if nxt == cur:                          # reached root without an embedded ancestor
                break
            cur = nxt
        if hit is not None:
            rolled.add(hit)
        else:
            dropped += 1
    return rolled, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", required=True, help="Embedding index->taxid mapping (defines embedded set)")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--proteome-cache", required=True,
                    help="TSV cache of raw UniProt proteome taxids (fetched if absent)")
    ap.add_argument("--query", default="proteome_type:1",
                    help='UniProt proteomes query (default reference proteomes "proteome_type:1")')
    ap.add_argument("--refresh", action="store_true", help="Force re-fetch even if the cache exists")
    ap.add_argument("-o", "--output", required=True, help="Rolled-up covered_taxids.tsv path")
    args = ap.parse_args()

    cache = Path(args.proteome_cache)
    if cache.exists() and not args.refresh:
        print(f"[cache] reading proteome taxids from {cache}")
        proteome_taxids = [int(x) for x in cache.read_text().split() if x.strip().isdigit()]
    else:
        print(f"[fetch] querying UniProt: {args.query}")
        proteome_taxids = fetch_proteome_taxids(query=args.query)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text("\n".join(str(t) for t in proteome_taxids) + "\n")
        print(f"[cache] wrote {len(proteome_taxids):,} proteome taxids -> {cache}")

    idx2tax = load_mapping(args.mapping)
    embedded = set(int(t) for t in idx2tax.values())
    taxonomy = load_taxonomy_with_depth(embedded, data_dir=args.data_dir)
    # parent map over ALL taxids on the proteome chains: we need parents beyond the embedded set,
    # so pull them from the full taxonomy dict (embedded nodes have parents recorded).
    parent_of = {t: v["parent"] for t, v in taxonomy.items()}
    # extend with proteome taxids' own chains via taxopy (sub-leaf strains absent from `taxonomy`)
    from taxembed.utils.taxdump import load_taxdb
    taxdb = load_taxdb(Path(args.data_dir))
    for t in proteome_taxids:
        cur = int(t)
        for _ in range(64):                          # bounded walk
            if cur in parent_of:
                break
            par = taxdb.taxid2parent.get(cur)
            if par is None:
                break
            parent_of[cur] = int(par)
            cur = int(par)

    rolled, dropped = rollup_to_embedded(proteome_taxids, parent_of, embedded)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("taxid\n" + "\n".join(str(t) for t in sorted(rolled)) + "\n")
    summary = {
        "n_proteome_taxids": len(proteome_taxids),
        "n_rolled_unique_embedded": len(rolled),
        "n_dropped_no_embedded_ancestor": dropped,
        "query": args.query,
        "mapping": args.mapping,
    }
    (out.parent / "covered_taxids_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest tests/eval/test_uniprot_rollup.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings add scripts/fetch_uniprot_proteomes.py tests/eval/test_uniprot_rollup.py
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings commit -m "feat(eval): UniProt proteome fetch + strain roll-up to nearest embedded ancestor"
```

---

### Task 7: `coverage_bias.py` CLI (integration)

**Files:**
- Create: `scripts/coverage_bias.py`
- Test: `tests/eval/test_coverage_bias_cli.py`

This wires the core to a real checkpoint. It builds index→parent/depth arrays from the mapping + taxdump (reusing `analyze_hierarchy_hyperbolic`'s loaders and the same `build_index_tree` helper shape Plan 1's CLI uses), selects **clade roots at a chosen rank** (e.g. phylum/class) via `get_ancestor_at_rank`, marks the covered set from `covered_taxids.tsv`, and for each clade computes: tree-partition coverage, coverage delta vs the uniform-coverage null (with taxon-bootstrap CI on the per-leaf covered indicator), Faith's PD, the geometric kNN-radius density, and the study-effort (#descendants) association across clades. Writes JSON/TSV + a coverage-vs-effort scatter.

- [ ] **Step 1: Write the failing integration test (synthetic checkpoint + covered set, no LRZ data, no network)**

`tests/eval/test_coverage_bias_cli.py`:
```python
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / ".venv" / "bin" / "python"


def _make_fixture(tmp_path):
    # taxids: root 131567; two phyla 1001/1002; leaves under each.
    #   131567
    #   ├─1001 (phylum)   ├─1002 (phylum)
    #   │ ├─11 (leaf)     │ ├─21 (leaf)
    #   │ ├─12 (leaf)     │ ├─22 (leaf)
    #   │ └─13 (leaf)     │ └─23 (leaf)
    taxids = [131567, 1001, 1002, 11, 12, 13, 21, 22, 23]
    parent_of = {131567: 131567, 1001: 131567, 1002: 131567,
                 11: 1001, 12: 1001, 13: 1001, 21: 1002, 22: 1002, 23: 1002}
    rank_of = {131567: "no rank", 1001: "phylum", 1002: "phylum",
               11: "species", 12: "species", 13: "species",
               21: "species", 22: "species", 23: "species"}
    depth_of = {131567: 0, 1001: 1, 1002: 1, 11: 2, 12: 2, 13: 2, 21: 2, 22: 2, 23: 2}
    rng = np.random.default_rng(0)
    emb = np.zeros((len(taxids), 4), np.float32)
    for i, t in enumerate(taxids):
        v = rng.standard_normal(4); v /= np.linalg.norm(v)
        emb[i] = (v * (depth_of[t] / 3.0)).astype(np.float32)
    ckpt = tmp_path / "fix.pth"
    torch.save({"embeddings": torch.tensor(emb)}, ckpt)
    mp = tmp_path / "map.tsv"
    mp.write_text("taxid\tidx\tparent\trank\n" +
                  "\n".join(f"{t}\t{i}\t{parent_of[t]}\t{rank_of[t]}" for i, t in enumerate(taxids)) + "\n")
    # covered set: phylum 1001 fully covered (11,12,13); phylum 1002 under-covered (only 21)
    cov = tmp_path / "covered.tsv"
    cov.write_text("taxid\n11\n12\n13\n21\n")
    return ckpt, mp, cov


def test_cli_runs_and_flags_undercovered_clade(tmp_path):
    ckpt, mp, cov = _make_fixture(tmp_path)
    out = tmp_path / "out"
    r = subprocess.run(
        [str(PY), str(ROOT / "scripts" / "coverage_bias.py"),
         "--checkpoint", str(ckpt), "--mapping", str(mp),
         "--covered", str(cov), "--parent-from-mapping",
         "--rank", "phylum", "--k", "2", "--n-null", "500", "-o", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    res = json.loads((out / "coverage_bias.json").read_text())
    clades = {c["taxid"]: c for c in res["clades"]}
    assert abs(clades[1001]["coverage_fraction"] - 1.0) < 1e-9   # phylum 1001 fully covered
    assert abs(clades[1002]["coverage_fraction"] - (1 / 3)) < 1e-9  # 1 of 3
    # the under-covered phylum has the lower coverage and a more-negative delta z
    assert clades[1002]["coverage_z_vs_null"] < clades[1001]["coverage_z_vs_null"]
    assert "study_effort" in res and "spearman_rho" in res["study_effort"]
    assert "faith_pd" in clades[1001]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest tests/eval/test_coverage_bias_cli.py -v`
Expected: FAIL — script does not exist.

- [ ] **Step 3: Implement `scripts/coverage_bias.py`**

```python
"""Application #3 — sampling-bias quantification: per-clade TREE-PARTITION coverage of a covered
taxon set (headline: UniProt reference proteomes) vs a uniform-coverage null, benchmarked against
Faith's PD and reported against a study-effort covariate. The embedding supplies only an optional
self-normalizing kNN-radius density + viz; coverage is NEVER ball-volume (spec §5#3, §9B, §9F).

Local analysis only: pass explicit --checkpoint (the `final`/ep200 .pth) + --mapping + --covered.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT / "src"))

from analyze_hierarchy_hyperbolic import (
    load_embeddings, load_mapping, load_taxonomy_with_depth, get_ancestor_at_rank,
)
from taxembed.eval.treedist import TreeDistance
from taxembed.eval.bootstrap import taxon_bootstrap_ci
from taxembed.eval.coverage import (
    tree_partition_coverage, knn_radius_density, uniform_coverage_null,
    coverage_delta_over_null, faith_pd, study_effort_association,
)


def build_index_tree(idx2tax, taxonomy=None, parent_col=None, rank_col=None):
    """Return (parent_idx, depth, rank_of_idx, tax2idx) over node indices 0..N-1.
    parent_col/rank_col: optional {taxid: ...} (test mode); else use the taxdump taxonomy dict."""
    n = len(idx2tax)
    tax2idx = {t: i for i, t in idx2tax.items()}
    parent = np.arange(n, dtype=np.int64)
    depth = np.zeros(n, dtype=np.int64)
    rank_of_idx = ["no rank"] * n
    for i in range(n):
        t = idx2tax[i]
        if parent_col is not None:
            p = parent_col[t]
            parent[i] = tax2idx.get(p, i)
            d, cur, seen = 0, t, set()
            while cur in parent_col and parent_col[cur] != cur and cur not in seen:
                seen.add(cur); cur = parent_col[cur]; d += 1
            depth[i] = d
            rank_of_idx[i] = rank_col[t] if rank_col else "no rank"
        else:
            p = taxonomy.get(t, {}).get("parent", t)
            parent[i] = tax2idx.get(p, i)
            depth[i] = taxonomy.get(t, {}).get("depth", 0)
            rank_of_idx[i] = taxonomy.get(t, {}).get("rank", "no rank")
    return parent, depth, rank_of_idx, tax2idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--covered", required=True, help="TSV with a 'taxid' column (rolled-up covered set)")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--parent-from-mapping", action="store_true",
                    help="Test mode: read 'parent'+'rank' columns from the mapping instead of the taxdump")
    ap.add_argument("--rank", default="phylum", help="Rank at which to define clade roots")
    ap.add_argument("--k", type=int, default=10, help="k for the geometric kNN-radius density")
    ap.add_argument("--n-null", type=int, default=2000, help="uniform-coverage null draws per clade")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--output-dir", required=True)
    args = ap.parse_args()

    emb = load_embeddings(args.checkpoint).astype(np.float64)
    idx2tax = load_mapping(args.mapping)

    if args.parent_from_mapping:
        df = pd.read_csv(args.mapping, sep="\t")
        parent_col = {int(r.taxid): int(r.parent) for r in df.itertuples()}
        rank_col = {int(r.taxid): str(r.rank) for r in df.itertuples()}
        taxonomy = {int(r.taxid): {"parent": int(r.parent), "rank": str(r.rank)}
                    for r in df.itertuples()}
        parent, depth, rank_of_idx, tax2idx = build_index_tree(
            idx2tax, parent_col=parent_col, rank_col=rank_col)
    else:
        taxonomy = load_taxonomy_with_depth(set(idx2tax.values()), data_dir=args.data_dir)
        parent, depth, rank_of_idx, tax2idx = build_index_tree(idx2tax, taxonomy=taxonomy)

    td = TreeDistance(parent, depth)
    n = len(emb)

    # covered mask over indices
    cov_df = pd.read_csv(args.covered, sep="\t")
    covered_taxids = set(int(t) for t in cov_df["taxid"].tolist())
    covered_mask = np.zeros(n, dtype=bool)
    for t in covered_taxids:
        if t in tax2idx:
            covered_mask[tax2idx[t]] = True

    # clade roots = nodes whose rank == args.rank (their indices)
    clade_root_idx = [i for i in range(n) if rank_of_idx[i] == args.rank]
    if not clade_root_idx:
        raise SystemExit(f"No clade roots at rank '{args.rank}'")

    cov = tree_partition_coverage(parent, depth, covered_mask, clade_roots=clade_root_idx, td=td)

    # global leaf totals for the uniform null
    from taxembed.eval.coverage import _is_leaf
    leaf = _is_leaf(parent)
    global_total = int((leaf).sum())
    n_covered_leaves = int((leaf & covered_mask).sum())

    clades = []
    cov_fracs, efforts = [], []
    for ridx in clade_root_idx:
        c = cov[ridx]
        null = uniform_coverage_null(clade_total=c["total_lineages"], global_total=global_total,
                                     n_covered=n_covered_leaves, n_draws=args.n_null, seed=args.seed)
        z, p = coverage_delta_over_null(c["coverage_fraction"], null)
        # Faith's PD of the covered leaves within this clade
        clade_leaf_mask = np.zeros(n, dtype=bool)
        # leaves under this clade root that are covered:
        sub = _subtree_leaf_mask(parent, ridx, leaf)
        clade_leaf_mask = sub & covered_mask
        pd_val = faith_pd(parent, depth, clade_leaf_mask, root=ridx, td=td)
        effort = c["total_lineages"]                   # study-effort proxy = #descendant leaves
        clades.append({
            "taxid": int(idx2tax[ridx]),
            "rank": args.rank,
            "total_lineages": c["total_lineages"],
            "covered_lineages": c["covered_lineages"],
            "coverage_fraction": c["coverage_fraction"],
            "covered_subtree_fraction": c["covered_subtree_fraction"],
            "coverage_z_vs_null": z,
            "coverage_p_vs_null": p,
            "faith_pd": pd_val,
            "study_effort_n_leaves": effort,
        })
        cov_fracs.append(c["coverage_fraction"])
        efforts.append(effort)

    se = study_effort_association(np.array(cov_fracs), np.array(efforts))
    # attach residuals back per clade
    for c, resid in zip(clades, se["residual"]):
        c["coverage_residual_beyond_effort"] = float(resid)

    # taxon-level bootstrap CI on the global covered-leaf rate (per-leaf indicator)
    leaf_indicator = covered_mask[leaf].astype(np.float64)
    g_mean, g_lo, g_hi = taxon_bootstrap_ci(leaf_indicator, seed=args.seed)

    result = {
        "checkpoint": args.checkpoint,
        "rank": args.rank,
        "n_clades": len(clades),
        "global_covered_rate": {"mean": g_mean, "lo": g_lo, "hi": g_hi},
        "study_effort": {"spearman_rho": se["spearman_rho"], "spearman_p": se["spearman_p"],
                         "slope": se["slope"], "intercept": se["intercept"]},
        "clades": clades,
    }
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "coverage_bias.json").write_text(json.dumps(result, indent=2))

    # TSV
    cdf = pd.DataFrame(clades)
    cdf.to_csv(out / "coverage_bias.tsv", sep="\t", index=False)

    # coverage-vs-effort scatter (under-coverage beyond effort = points below the fit line)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(np.log1p(efforts), cov_fracs, s=12, alpha=0.6)
        xs = np.linspace(min(np.log1p(efforts)), max(np.log1p(efforts)), 50)
        ax.plot(xs, se["slope"] * xs + se["intercept"], "r-", lw=1,
                label=f"fit (rho={se['spearman_rho']:.2f})")
        ax.set_xlabel("log1p(study effort = #leaves)")
        ax.set_ylabel("tree-partition coverage fraction")
        ax.set_title(f"Coverage vs study effort — {args.rank}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "coverage_vs_effort.png", dpi=120)
        plt.close(fig)
    except Exception as exc:                          # plotting is non-load-bearing
        print(f"[warn] plot skipped: {exc}")

    print(json.dumps(result, indent=2))


def _subtree_leaf_mask(parent: np.ndarray, root: int, leaf: np.ndarray) -> np.ndarray:
    """Boolean mask of leaves in the subtree rooted at `root` (iterative DFS over a child index)."""
    from collections import defaultdict
    kids = defaultdict(list)
    for node, par in enumerate(parent):
        if node != par:
            kids[int(par)].append(node)
    mask = np.zeros(len(parent), dtype=bool)
    stack = [int(root)]
    while stack:
        v = stack.pop()
        if leaf[v]:
            mask[v] = True
        stack.extend(kids.get(v, ()))
    return mask


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the integration test to verify it passes**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest tests/eval/test_coverage_bias_cli.py -v`
Expected: PASS

- [ ] **Step 5: Run the full eval test suite**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest tests/eval/ -v`
Expected: PASS (all of this plan's tests + Plan 1's, no regressions)

- [ ] **Step 6: Commit**

```bash
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings add scripts/coverage_bias.py tests/eval/test_coverage_bias_cli.py
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings commit -m "feat(eval): coverage_bias CLI (tree-partition coverage vs null + PD + study-effort)"
```

---

### Task 8: Produce the real #3 result on eukaryota 877k

**Files:**
- Output: `artifacts/tags/eukaryota_canonical/coverage/` (analysis output, not committed)

- [ ] **Step 1: Fetch + roll up the UniProt covered set (assumes network)**

Run:
```bash
/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python scripts/fetch_uniprot_proteomes.py \
  --mapping data/taxopy/eukaryota_2759_clean/taxonomy_edges_eukaryota_2759_clean.mapping.tsv \
  --data-dir data \
  --proteome-cache artifacts/tags/eukaryota_canonical/coverage/uniprot_proteome_taxids.tsv \
  -o artifacts/tags/eukaryota_canonical/coverage/covered_taxids.tsv
```
Expected: writes the raw proteome taxid cache + the rolled-up `covered_taxids.tsv`; prints a summary with `n_proteome_taxids`, `n_rolled_unique_embedded`, `n_dropped_no_embedded_ancestor`. Sanity-check `n_dropped` is a small minority (most reference proteomes should roll up to an embedded eukaryotic ancestor); a large `n_dropped` means many proteomes are non-eukaryotic (expected for the eukaryota-only embedding — they fall outside 2759 and are correctly dropped) — record the number, it is itself a finding.

- [ ] **Step 2: Run the coverage analysis (offline, uses the cached covered set)**

Run (use the **final/ep200** checkpoint, explicit paths — spec §9C):
```bash
/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python scripts/coverage_bias.py \
  --checkpoint artifacts/tags/eukaryota_canonical/eukaryota_canonical.pth \
  --mapping data/taxopy/eukaryota_2759_clean/taxonomy_edges_eukaryota_2759_clean.mapping.tsv \
  --covered artifacts/tags/eukaryota_canonical/coverage/covered_taxids.tsv \
  --data-dir data --rank phylum --k 10 --n-null 2000 \
  -o artifacts/tags/eukaryota_canonical/coverage
```
Expected: completes; writes `coverage_bias.json` + `.tsv` + `coverage_vs_effort.png`. Prints per-phylum coverage, the coverage delta vs the uniform-coverage null, Faith's PD, and the study-effort Spearman ρ + slope.

- [ ] **Step 3: Sanity-check the result (the analysis's own validity gate)**

Confirm: (a) **per-clade coverage_fraction spans a real range** (some phyla near-fully covered, some near-zero) — a flat result means the join or roll-up failed; (b) the **uniform-coverage null** mean per clade tracks the global covered rate (≈ `global_covered_rate.mean`), and well-studied phyla (e.g. Chordata/Streptophyta) show **positive** `coverage_z_vs_null` while neglected phyla show **negative** z; (c) `study_effort.spearman_rho` is **positive** (coverage tracks #leaves studied), AND there exist clades with large-negative `coverage_residual_beyond_effort` — those are the **headline under-sampled-beyond-priority** clades; (d) Faith's PD per clade **correlates** with `covered_lineages` across clades (compute the Spearman offline as the PD-benchmark figure: PD vs tree-partition coverage). Record the headline numbers in `docs/SESSION_LOG.md` (append a dated note) and update the spec §5#3 with the measured numbers. If (b) shows z≈0 everywhere, escalate — coverage is then indistinguishable from random sampling, a finding to report not bury.

- [ ] **Step 4: PD-benchmark figure (the §9A complement claim)**

Run a short offline reduction over `coverage_bias.tsv`: Spearman ρ between `faith_pd` and `covered_lineages` (and `coverage_fraction`) across clades; plot PD-vs-coverage. The claim for the paper is **agreement** (high ρ) + the speed/scale/differentiability advantage (tree-partition coverage is an O(#covered-leaf path-walk) count with no distance-matrix build and is auto-differentiable through the embedding for the density complement, whereas PD requires the spanning subtree per subset). Frame as complement, not replacement (spec §9A prior-art reviewer flagged PD as entrenched). Write this small reduction as `scripts/_coverage_pd_benchmark.py` if you prefer a reproducible artifact over an ad-hoc reduction (optional; not load-bearing for the core).

- [ ] **Step 5: Re-run on cellular 1.1M when job 5673097 lands**

When `cellular_canonical.pth` is pulled locally, repeat Steps 1–4 with:
```
--checkpoint artifacts/tags/cellular_canonical/cellular_canonical.pth
--mapping data/taxopy/cellular_organisms_131567_clean/taxonomy_edges_cellular_organisms_131567_clean.mapping.tsv
```
**Re-derive, do NOT transfer, any rank/depth-dependent baseline** — on cellular the proteome set spans all of life (so `n_dropped` plummets), "phylum" semantics differ across bacteria/eukaryota, and cellular adds superkingdom/domain ranks (spec §9C cellular rank-label trap). Re-run the fetch (`--refresh` not needed — the raw UniProt list is dataset-independent, but the roll-up join changes because the embedded set changes, so re-run `fetch_uniprot_proteomes.py` against the cellular mapping to regenerate `covered_taxids.tsv`).

---

## Self-review notes

**Spec coverage.** This plan covers spec §5 #3 (sampling-bias quantification) and every §9B rigor point that touches #3:
1. **Coverage on the tree partition, not ball-volume** — `tree_partition_coverage` (Task 1) is the headline metric: covered leaf-lineages / total + covered-subtree fraction. Hyperbolic ball-volume is explicitly never used as a denominator; the docstrings state why.
2. **kNN-radius geometric density** as a self-normalizing complement — `knn_radius_density` (Task 2), conditioned on clade AND depth in the CLI (clade roots chosen at `--rank`; depth available per node), reported vs the **uniform-coverage null** (`uniform_coverage_null` + `coverage_delta_over_null`, Task 3).
3. **Headline UniProt/proteome coverage** — `fetch_uniprot_proteomes.py` (Task 6) is a named, **cached** fetch via UniProt REST (`proteome_type:1` reference proteomes) with an explicit **roll-up rule** (walk to nearest embedded ancestor; drop if none; dedup strains) joined to the embedded taxid set; network assumed for the fetch, offline for the analysis.
4. **Faith's PD benchmark** — `faith_pd` (Task 4) on the topology-only tree; Task 8 Step 4 produces the PD-vs-coverage agreement figure and frames coverage as a **complement** (speed/scale/differentiability), per the §9A prior-art reviewer.
5. **Study-effort covariate** — `study_effort_association` (Task 5): Spearman ρ(coverage, effort) + the residual so the headline signal = **under-coverage beyond known sequencing priority**.
6. **Develop on eukaryota now, re-run on cellular** — Task 8 Step 2 uses eukaryota with explicit `--checkpoint`/`--mapping` + the `final` (ep200) checkpoint; Step 5 re-runs on cellular re-deriving every rank/depth baseline (§9C trap stated explicitly).

It **builds on Plan 1's foundation**: imports `TreeDistance` (treedist.py) for tree geometry/PD edge accounting and `taxon_bootstrap_ci` (bootstrap.py) for the taxon-level CI; reuses the float64 `numpy_poincare_distance` + the matmul-expanded `_batch_distances` idiom from `knn_purity_hyperbolic.py` for the geometric density. No re-implementation of the distance machinery.

**Not in this plan (deliberately):** the prior-art comparison table and artifact packaging (§9A/§9D — own task); the #4 molecular→taxonomy bridge (Outlook, gated, §9E); applications #1 (Plan 1) and #2 (its own plan). The matmul-expanded batched density for full-scale runs is noted as a swap-in for `knn_radius_density` but the broadcasting reference is correct and used for the eukaryota run (Q×P fits at the per-clade scale we query); if a whole-tree density pass OOMs, port `_batch_distances` — a thin follow-up, not core.

**Placeholder scan.** No `...`, no `TODO`, no `pass`-stub bodies, no `raise NotImplementedError`. Every function has a complete body; every test has concrete asserts with computed expected values (hand-verified against the 7-node / 9-taxid fixtures); both CLIs run end-to-end against synthetic checkpoints in their integration tests with no LRZ-data or network dependency (`--parent-from-mapping` test mode mirrors Plan 1; the UniProt network call is isolated behind `rollup_to_embedded`, which is the only unit-tested part of the fetch).

**Type consistency.** `tree_partition_coverage(parent, depth, covered_mask, clade_roots, td)` → `{root_idx: dict}`; `knn_radius_density(emb, query_idx, pool_idx, k)` → `(Q,)` float64; `uniform_coverage_null(clade_total, global_total, n_covered, n_draws, seed)` → `(n_draws,)` fractions; `coverage_delta_over_null(observed_fraction, null_samples)` → `(z, p)` floats; `faith_pd(parent, depth, leaf_mask, root, td)` → int; `study_effort_association(coverage_fraction, study_effort)` → dict with `residual` ndarray. The CLI (`coverage_bias.py`) consumes each exactly as defined and reuses Plan 1's `build_index_tree` shape (extended with a `rank` column) and `taxon_bootstrap_ci` 3-tuple unpacking. `rollup_to_embedded(proteome_taxids, parent_of, embedded)` → `(set, int)`, consumed as `rolled, dropped` in `main()`.

**Edge cases covered by tests:** empty clade (no leaves) → coverage 0; single covered leaf → PD = depth-to-root; no covered leaves → PD 0; strain with no embedded ancestor → dropped + counted; multiple strains → dedup to one ancestor; over- vs under-coverage both flagged by the null delta; small bin (Plan 1's stratifier) reused unchanged.
