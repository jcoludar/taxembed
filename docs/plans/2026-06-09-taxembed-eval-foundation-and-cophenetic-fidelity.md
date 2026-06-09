# TaxEmbed eval foundation + Application #1 (cophenetic fidelity) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the reusable analysis foundation (tree-distance/LCA, null-model embeddings, taxon-level bootstrap, stratified pair sampling, fidelity metrics) and use it to produce Application #1 — *cophenetic fidelity*: how faithfully embedded Poincaré distance recovers true tree distance, measured by local rank fidelity vs a radial-only null, on the eukaryota 877k embedding we already have locally.

**Architecture:** A new importable subpackage `src/taxembed/eval/` holds pure, unit-tested core logic (no I/O); a thin CLI `scripts/cophenetic_fidelity.py` wires it to a checkpoint + taxdump and writes results. This mirrors the package/scripts split already used by `taxembed` and makes the math testable (the existing `scripts/*_hyperbolic.py` analyzers are untested monoliths — we do NOT extend them; we build testable cores). All numbers report a **delta over the radial-only null** with **taxon-level block-bootstrap CIs** (per the spec §9B rigor fold).

**Tech Stack:** Python 3.12, numpy, scipy (`scipy.stats.spearmanr`, already a transitive dep via sklearn), pytest. Poincaré distance reuses `scripts/_negative_hardness.numpy_poincare_distance` (float64). Embedding/mapping/taxonomy loaders reuse `scripts/analyze_hierarchy_hyperbolic.py`.

**Run context:** venv at `.venv` (run `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python`). Always invoke analysis with explicit `--checkpoint`/`--mapping` and the `final` (ep200) checkpoint — `run.json` holds LRZ container paths and tag-resolution defaults to `_best` (spec §9C).

---

## File structure

- Create `src/taxembed/eval/__init__.py` — subpackage marker.
- Create `src/taxembed/eval/treedist.py` — `TreeDistance` (binary-lifting LCA + path-length + lca-depth distances). **Heaviest new code.**
- Create `src/taxembed/eval/nulls.py` — `radial_only_null`, `shuffled_label_null`, `random_ball_null`.
- Create `src/taxembed/eval/pairs.py` — `sample_pairs_stratified` (by true-distance bin).
- Create `src/taxembed/eval/bootstrap.py` — `taxon_bootstrap_ci` (resamples per-taxon values).
- Create `src/taxembed/eval/fidelity.py` — `multiplicative_distortion`, `knn_retrieval_precision`, `within_clade_rank_corr`.
- Create `scripts/cophenetic_fidelity.py` — CLI: load embedding + taxonomy, build `TreeDistance`, sample pairs, compute fidelity for model vs radial-only null, bootstrap CIs, write JSON/TSV + a hexbin plot.
- Create tests: `tests/eval/__init__.py`, `tests/eval/test_treedist.py`, `tests/eval/test_nulls.py`, `tests/eval/test_pairs.py`, `tests/eval/test_bootstrap.py`, `tests/eval/test_fidelity.py`.

The eval core takes **integer node-index arrays** (0..N-1, the embedding's own indexing) so it is pure and decoupled from taxids; the CLI builds the index↔taxid↔parent maps from the mapping + taxdump.

---

### Task 1: Tree-distance core (binary-lifting LCA)

**Files:**
- Create: `src/taxembed/eval/__init__.py`
- Create: `src/taxembed/eval/treedist.py`
- Create: `tests/eval/__init__.py`
- Test: `tests/eval/test_treedist.py`

- [ ] **Step 1: Create empty package markers**

```bash
mkdir -p /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/src/taxembed/eval /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/tests/eval
```
Write `src/taxembed/eval/__init__.py` with a one-line docstring:
```python
"""Reusable, unit-tested evaluation core for TaxEmbed paper analyses (no I/O)."""
```
Write `tests/eval/__init__.py` as an empty file.

- [ ] **Step 2: Write the failing test**

`tests/eval/test_treedist.py`:
```python
import numpy as np
from taxembed.eval.treedist import TreeDistance

# Hand tree (node id : parent):  0=root
#   0
#   ├─1        ├─2
#   │ ├─3      │
#   │ │ └─5    │
#   │ └─4      │
# parent[root] = root (self-loop convention)
PARENT = np.array([0, 0, 0, 1, 1, 3])
DEPTH  = np.array([0, 1, 1, 2, 2, 3])


def _td():
    return TreeDistance(PARENT, DEPTH)


def test_lca_basic():
    td = _td()
    a = np.array([3, 5, 5, 1, 4])
    b = np.array([4, 4, 2, 1, 5])
    assert list(td.lca(a, b)) == [1, 1, 0, 1, 1]


def test_path_length():
    td = _td()
    a = np.array([3, 5, 5, 1])
    b = np.array([4, 4, 2, 1])
    # d(3,4)=2; d(5,4)=3; d(5,2)=4; d(1,1)=0
    assert list(td.path_length(a, b)) == [2, 3, 4, 0]


def test_lca_depth_distance():
    td = _td()
    a = np.array([5, 5])
    b = np.array([4, 2])
    # relatedness distance = (depth[a]-depth[lca]) ... we test the shallow-split form:
    # n_levels_to_split = depth[a] + depth[b] - 2*depth[lca] is path_length; lca_depth returns depth[lca]
    assert list(td.lca_depth(a, b)) == [1, 0]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest tests/eval/test_treedist.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'taxembed.eval.treedist'`

- [ ] **Step 4: Implement `treedist.py`**

```python
"""Exact tree distances over the NCBI taxonomy via binary-lifting LCA.

Operates on integer node ids 0..N-1 (the embedding's own indexing). `parent` is an
int array (root points to itself); `depth` is the root-distance in edges. Vectorized:
all queries take numpy arrays of node ids and return numpy arrays.
"""
from __future__ import annotations

import numpy as np


class TreeDistance:
    def __init__(self, parent: np.ndarray, depth: np.ndarray):
        self.parent = np.asarray(parent, dtype=np.int64)
        self.depth = np.asarray(depth, dtype=np.int64)
        n = len(self.parent)
        max_depth = int(self.depth.max()) if n else 0
        self.maxlog = max(1, int(np.ceil(np.log2(max(2, max_depth + 1)))) + 1)
        # up[k, v] = the (2^k)-th ancestor of v
        self.up = np.zeros((self.maxlog, n), dtype=np.int64)
        self.up[0] = self.parent
        for k in range(1, self.maxlog):
            self.up[k] = self.up[k - 1][self.up[k - 1]]

    def lca(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.int64)
        b = np.asarray(b, dtype=np.int64)
        # ensure a is the deeper-or-equal node
        swap = self.depth[a] < self.depth[b]
        a, b = np.where(swap, b, a), np.where(swap, a, b)
        # lift a up to b's depth
        diff = self.depth[a] - self.depth[b]
        for k in range(self.maxlog):
            move = ((diff >> k) & 1).astype(bool)
            a = np.where(move, self.up[k][a], a)
        # lift both until their ancestors meet
        for k in range(self.maxlog - 1, -1, -1):
            au, bu = self.up[k][a], self.up[k][b]
            move = au != bu
            a = np.where(move, au, a)
            b = np.where(move, bu, b)
        return np.where(a == b, a, self.parent[a])

    def lca_depth(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self.depth[self.lca(a, b)]

    def path_length(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Cophenetic distance = #edges on the a→b path through the LCA (the PRIMARY tree distance)."""
        l = self.lca(a, b)
        return self.depth[np.asarray(a)] + self.depth[np.asarray(b)] - 2 * self.depth[l]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest tests/eval/test_treedist.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings add src/taxembed/eval/__init__.py src/taxembed/eval/treedist.py tests/eval/__init__.py tests/eval/test_treedist.py
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings commit -m "feat(eval): binary-lifting LCA + tree-distance core"
```

---

### Task 2: Null-model embeddings

**Files:**
- Create: `src/taxembed/eval/nulls.py`
- Test: `tests/eval/test_nulls.py`

- [ ] **Step 1: Write the failing test**

`tests/eval/test_nulls.py`:
```python
import numpy as np
from taxembed.eval.nulls import radial_only_null, shuffled_label_null, random_ball_null


def _emb():
    rng = np.random.default_rng(1)
    e = rng.standard_normal((50, 8)) * 0.1
    return e


def test_radial_only_preserves_norms_changes_direction():
    e = _emb()
    null = radial_only_null(e, seed=0)
    assert np.allclose(np.linalg.norm(e, axis=1), np.linalg.norm(null, axis=1))
    assert not np.allclose(e, null)               # directions scrambled
    assert null.shape == e.shape


def test_shuffled_label_is_a_permutation():
    e = _emb()
    null = shuffled_label_null(e, seed=0)
    # every row of null is some row of e (a permutation of the SAME vectors)
    assert sorted(null.sum(axis=1).round(6)) == sorted(e.sum(axis=1).round(6))


def test_random_ball_inside_ball():
    null = random_ball_null(50, 8, seed=0)
    assert null.shape == (50, 8)
    assert (np.linalg.norm(null, axis=1) < 1.0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/eval/test_nulls.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `nulls.py`**

```python
"""Null-model embeddings for baseline deltas (spec §9B: the radial-only null is the key Goodhart guard).

- radial_only_null: keep each node's norm (=depth structure), randomize its direction → fidelity
  attributable to RADIUS alone. The model must beat this.
- shuffled_label_null: permute which embedding vector belongs to which node → destroys all
  label-structure while preserving the marginal point cloud.
- random_ball_null: uniform-ish random points strictly inside the Poincaré ball.
"""
from __future__ import annotations

import numpy as np


def radial_only_null(emb: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    dirs = rng.standard_normal(emb.shape)
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
    return dirs * norms


def shuffled_label_null(emb: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(emb))
    return emb[perm]


def random_ball_null(n: int, dim: int, seed: int = 0, max_norm: float = 0.95) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dirs = rng.standard_normal((n, dim))
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
    r = max_norm * rng.random((n, 1)) ** (1.0 / dim)
    return dirs * r
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/eval/test_nulls.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings add src/taxembed/eval/nulls.py tests/eval/test_nulls.py
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings commit -m "feat(eval): null-model embeddings (radial-only/shuffled/random)"
```

---

### Task 3: Stratified pair sampler

**Files:**
- Create: `src/taxembed/eval/pairs.py`
- Test: `tests/eval/test_pairs.py`

- [ ] **Step 1: Write the failing test**

`tests/eval/test_pairs.py`:
```python
import numpy as np
from taxembed.eval.pairs import sample_pairs_stratified


def test_stratified_balances_bins_and_counts():
    # true distances 1..6; bin edges [0,2,4,7) -> 3 bins. Ask for 90 pairs.
    rng = np.random.default_rng(0)
    n = 2000
    a = rng.integers(0, 500, n)
    b = rng.integers(0, 500, n)
    dist = rng.integers(1, 7, n)            # pretend tree distances
    idx = sample_pairs_stratified(dist, bin_edges=[0, 2, 4, 7], per_bin=30, seed=0)
    assert len(idx) == 90
    binned = np.digitize(dist[idx], [2, 4])  # 0,1,2
    counts = np.bincount(binned, minlength=3)
    assert (counts == 30).all()             # exactly balanced


def test_handles_small_bin_without_replacement_error():
    dist = np.array([1, 1, 5])              # bin0 has 2, bin2 has 1, bin1 empty
    idx = sample_pairs_stratified(dist, bin_edges=[0, 2, 4, 7], per_bin=30, seed=0)
    # takes all available in undersized bins, never errors
    assert set(idx).issubset({0, 1, 2})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/eval/test_pairs.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `pairs.py`**

```python
"""Stratified pair sampling by true tree-distance bin (spec §9B: uniform sampling is swamped by
trivially-far cross-kingdom pairs, inflating global correlation). Returns indices into the input
distance array; the caller holds the parallel (a, b) endpoint arrays.
"""
from __future__ import annotations

import numpy as np


def sample_pairs_stratified(tree_dist: np.ndarray, bin_edges, per_bin: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tree_dist = np.asarray(tree_dist)
    inner = list(bin_edges)[1:-1]               # digitize uses interior edges
    which = np.digitize(tree_dist, inner)        # bin id per pair
    n_bins = len(bin_edges) - 1
    picks = []
    for b in range(n_bins):
        members = np.flatnonzero(which == b)
        if members.size == 0:
            continue
        take = min(per_bin, members.size)
        picks.append(rng.choice(members, size=take, replace=False))
    return np.concatenate(picks) if picks else np.array([], dtype=np.int64)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/eval/test_pairs.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings add src/taxembed/eval/pairs.py tests/eval/test_pairs.py
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings commit -m "feat(eval): stratified-by-distance pair sampler"
```

---

### Task 4: Taxon-level bootstrap CI

**Files:**
- Create: `src/taxembed/eval/bootstrap.py`
- Test: `tests/eval/test_bootstrap.py`

- [ ] **Step 1: Write the failing test**

`tests/eval/test_bootstrap.py`:
```python
import numpy as np
from taxembed.eval.bootstrap import taxon_bootstrap_ci


def test_ci_brackets_mean_and_is_ordered():
    rng = np.random.default_rng(0)
    vals = rng.normal(0.9, 0.02, 500)       # per-taxon metric values
    mean, lo, hi = taxon_bootstrap_ci(vals, n_boot=1000, seed=0)
    assert lo < mean < hi
    assert abs(mean - vals.mean()) < 1e-9    # point estimate is the plain mean
    assert (hi - lo) < 0.02                   # tight CI for n=500, low variance


def test_ci_widens_with_fewer_units():
    rng = np.random.default_rng(0)
    big = taxon_bootstrap_ci(rng.normal(0.9, 0.05, 1000), n_boot=500, seed=0)
    small = taxon_bootstrap_ci(rng.normal(0.9, 0.05, 30), n_boot=500, seed=0)
    assert (small[2] - small[1]) > (big[2] - big[1])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/eval/test_bootstrap.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `bootstrap.py`**

```python
"""Taxon-level bootstrap CIs (spec §9B: the unit of analysis is the TAXON, not the pair — pairs are
non-independent, so effective N ≈ #taxa). Caller reduces its analysis to one value per taxon (e.g.
per-taxon kNN-retrieval precision); this resamples taxa with replacement.
"""
from __future__ import annotations

import numpy as np


def taxon_bootstrap_ci(per_taxon_values: np.ndarray, n_boot: int = 1000, seed: int = 0,
                       alpha: float = 0.05):
    """Return (point_mean, lo, hi) where lo/hi are the (alpha/2, 1-alpha/2) percentile-bootstrap CI."""
    v = np.asarray(per_taxon_values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(v)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        boot[i] = v[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/eval/test_bootstrap.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings add src/taxembed/eval/bootstrap.py tests/eval/test_bootstrap.py
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings commit -m "feat(eval): taxon-level bootstrap CI"
```

---

### Task 5: Fidelity metrics

**Files:**
- Create: `src/taxembed/eval/fidelity.py`
- Test: `tests/eval/test_fidelity.py`

- [ ] **Step 1: Write the failing test**

`tests/eval/test_fidelity.py`:
```python
import numpy as np
from taxembed.eval.fidelity import multiplicative_distortion, knn_retrieval_precision


def test_distortion_perfect_when_proportional():
    d_emb = np.array([1.0, 2.0, 4.0])
    d_tree = np.array([2.0, 4.0, 8.0])      # exactly 0.5x — proportional
    out = multiplicative_distortion(d_emb, d_tree)
    # multiplicative distortion is scale-invariant after global rescale -> ~1.0
    assert abs(out["median"] - 1.0) < 1e-9


def test_distortion_detects_disorder():
    d_emb = np.array([1.0, 5.0, 2.0])
    d_tree = np.array([1.0, 2.0, 5.0])      # ranks scrambled
    out = multiplicative_distortion(d_emb, d_tree)
    assert out["median"] > 1.0


def test_knn_retrieval_precision_perfect_and_chance():
    # 4 query nodes; emb-NN order identical to tree-NN order -> precision 1.0
    # distance matrices (row=query, col=candidate), diagonal = self = inf
    INF = np.inf
    d_tree = np.array([[INF, 1, 2, 3],
                       [1, INF, 2, 3],
                       [2, 1, INF, 3],
                       [3, 2, 1, INF]], float)
    d_emb = d_tree.copy()
    p = knn_retrieval_precision(d_emb, d_tree, k=1)
    assert np.allclose(p, 1.0)
    # if emb distances are reversed, precision at k=1 should drop
    p_bad = knn_retrieval_precision(d_tree[:, ::-1].copy(), d_tree, k=1)
    assert p_bad.mean() < 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/eval/test_fidelity.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `fidelity.py`**

```python
"""Fidelity metrics for embedded-vs-tree distance (spec §9B: local rank fidelity, NOT global Pearson).

- multiplicative_distortion: the standard metric-embedding quality measure. Scale-invariant: we
  divide out the median ratio first, then report max(r, 1/r) summary stats. 1.0 == perfect.
- knn_retrieval_precision: per-query overlap between the k nearest by embedding and the k nearest by
  tree. The headline LOCAL metric; returns one value per query (feed to taxon_bootstrap_ci).
- within_clade_rank_corr: per-query Spearman of d_emb vs d_tree over a local candidate set.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def multiplicative_distortion(d_emb: np.ndarray, d_tree: np.ndarray) -> dict:
    d_emb = np.asarray(d_emb, float)
    d_tree = np.asarray(d_tree, float)
    m = (d_tree > 0) & (d_emb > 0)
    ratio = d_emb[m] / d_tree[m]
    ratio = ratio / np.median(ratio)            # remove the arbitrary global scale
    dist = np.maximum(ratio, 1.0 / ratio)       # >= 1, symmetric
    return {"median": float(np.median(dist)), "mean": float(np.mean(dist)),
            "p95": float(np.percentile(dist, 95)), "max": float(np.max(dist))}


def knn_retrieval_precision(d_emb_mat: np.ndarray, d_tree_mat: np.ndarray, k: int) -> np.ndarray:
    """d_*_mat: (Q, C) query-to-candidate distances (self-distance must be +inf). Returns (Q,) precision@k."""
    emb_nn = np.argsort(d_emb_mat, axis=1)[:, :k]
    tree_nn = np.argsort(d_tree_mat, axis=1)[:, :k]
    out = np.empty(len(d_emb_mat))
    for i in range(len(d_emb_mat)):
        out[i] = len(set(emb_nn[i]).intersection(tree_nn[i])) / k
    return out


def within_clade_rank_corr(d_emb_mat: np.ndarray, d_tree_mat: np.ndarray) -> np.ndarray:
    """Per-query Spearman rho of embedded vs tree distance over the candidate set. Returns (Q,)."""
    out = np.empty(len(d_emb_mat))
    for i in range(len(d_emb_mat)):
        finite = np.isfinite(d_emb_mat[i]) & np.isfinite(d_tree_mat[i])
        out[i] = spearmanr(d_emb_mat[i][finite], d_tree_mat[i][finite]).correlation
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/eval/test_fidelity.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings add src/taxembed/eval/fidelity.py tests/eval/test_fidelity.py
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings commit -m "feat(eval): distortion + kNN-retrieval + within-clade rank fidelity metrics"
```

---

### Task 6: `cophenetic_fidelity.py` CLI (integration)

**Files:**
- Create: `scripts/cophenetic_fidelity.py`
- Test: `tests/eval/test_cophenetic_cli.py`

This wires the core to a real checkpoint. It builds index→parent/depth arrays from the mapping + taxdump (reusing `analyze_hierarchy_hyperbolic`'s loaders), constructs `TreeDistance`, samples stratified pairs for the global-distortion view, and for a seeded set of **query taxa** computes per-query kNN-retrieval precision@k (embedding-NN vs tree-NN over a candidate sample) for the **model** and the **radial-only null**, then bootstraps CIs over queries.

- [ ] **Step 1: Write the failing integration test (uses a tiny synthetic checkpoint, no LRZ data)**

`tests/eval/test_cophenetic_cli.py`:
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
    # 6-node tree from Task 1; embed each node at radius=depth/4 along a random-but-clade-coherent dir
    parent = {1: 131567, 2: 131567, 3: 1, 4: 1, 5: 3, 6: 3}  # taxids; 131567 root
    # mapping idx->taxid
    taxids = [131567, 1, 2, 3, 4, 5]
    emb = np.zeros((6, 4), np.float32)
    rng = np.random.default_rng(0)
    for i, t in enumerate(taxids):
        d = {131567: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3}[t]
        v = rng.standard_normal(4); v /= np.linalg.norm(v)
        emb[i] = (v * (d / 5.0)).astype(np.float32)
    ckpt = tmp_path / "fix.pth"
    torch.save({"embeddings": torch.tensor(emb)}, ckpt)
    mp = tmp_path / "map.tsv"
    mp.write_text("taxid\tidx\n" + "\n".join(f"{t}\t{i}" for i, t in enumerate(taxids)) + "\n")
    return ckpt, mp


def test_cli_runs_and_emits_model_vs_null(tmp_path):
    ckpt, mp = _make_fixture(tmp_path)
    out = tmp_path / "out"
    r = subprocess.run(
        [str(PY), str(ROOT / "scripts" / "cophenetic_fidelity.py"),
         "--checkpoint", str(ckpt), "--mapping", str(mp),
         "--parent-from-mapping",          # test mode: read parent from a sidecar instead of taxdump
         "--k", "1", "--n-queries", "6", "--n-pairs", "10", "-o", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    res = json.loads((out / "cophenetic_fidelity.json").read_text())
    assert "model" in res and "radial_only_null" in res
    assert "knn_precision" in res["model"]
    # model should be at least as faithful as the radial-only null on this clade-coherent fixture
    assert res["model"]["knn_precision"]["mean"] >= res["radial_only_null"]["knn_precision"]["mean"]
```
Note: the `--parent-from-mapping` flag keeps the test self-contained (no taxdump). In that mode the script reads a `parent` column if present; for the fixture, extend the mapping writer to include it:
```python
    # in _make_fixture, replace the mapping line with parent column:
    parent_of = {131567: 131567, 1: 131567, 2: 131567, 3: 1, 4: 1, 5: 3}
    mp.write_text("taxid\tidx\tparent\n" +
                  "\n".join(f"{t}\t{i}\t{parent_of[t]}" for i, t in enumerate(taxids)) + "\n")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/eval/test_cophenetic_cli.py -v`
Expected: FAIL — script does not exist.

- [ ] **Step 3: Implement `scripts/cophenetic_fidelity.py`**

```python
"""Application #1 — cophenetic fidelity: how faithfully embedded Poincaré distance recovers true tree
distance. Reports LOCAL rank fidelity (kNN-retrieval precision@k, per-query) + global multiplicative
distortion, for the MODEL vs a RADIAL-ONLY null, with taxon-level bootstrap CIs. See
docs/specs/2026-06-09-taxembed-paper-design.md §5#1 + §9B.

Local analysis only: pass explicit --checkpoint (the `final`/ep200 .pth) + --mapping.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))            # for _negative_hardness, analyze_hierarchy_hyperbolic
sys.path.insert(0, str(ROOT / "src"))

from _negative_hardness import numpy_poincare_distance
from analyze_hierarchy_hyperbolic import load_embeddings, load_mapping, load_taxonomy_with_depth
from taxembed.eval.treedist import TreeDistance
from taxembed.eval.nulls import radial_only_null
from taxembed.eval.pairs import sample_pairs_stratified
from taxembed.eval.bootstrap import taxon_bootstrap_ci
from taxembed.eval.fidelity import multiplicative_distortion, knn_retrieval_precision


def build_index_tree(idx2tax, taxonomy=None, parent_col=None):
    """Return (parent_idx, depth) int arrays over node indices 0..N-1.
    parent_col: optional {taxid: parent_taxid} (test mode); else use taxonomy dict from taxdump."""
    n = len(idx2tax)
    tax2idx = {t: i for i, t in idx2tax.items()}
    parent = np.arange(n, dtype=np.int64)
    depth = np.zeros(n, dtype=np.int64)
    for i in range(n):
        t = idx2tax[i]
        p = (parent_col[t] if parent_col is not None else taxonomy.get(t, {}).get("parent", t))
        parent[i] = tax2idx.get(p, i)
        if parent_col is not None:
            # depth via walk (small test trees only)
            d, cur, seen = 0, t, set()
            while cur in parent_col and parent_col[cur] != cur and cur not in seen:
                seen.add(cur); cur = parent_col[cur]; d += 1
            depth[i] = d
        else:
            depth[i] = taxonomy.get(t, {}).get("depth", 0)
    return parent, depth


def _fidelity_block(emb, td, query_idx, cand_idx, k):
    """Per-query kNN-retrieval precision@k over a fixed candidate pool."""
    q = emb[query_idx]                                       # (Q, D)
    c = emb[cand_idx]                                        # (C, D)
    d_emb = numpy_poincare_distance(q[:, None, :], c[None, :, :])   # (Q, C)
    # tree distances query x candidate
    Q, C = len(query_idx), len(cand_idx)
    aa = np.repeat(query_idx, C); bb = np.tile(cand_idx, Q)
    d_tree = td.path_length(aa, bb).reshape(Q, C).astype(float)
    # mask self
    self_mask = query_idx[:, None] == cand_idx[None, :]
    d_emb = d_emb.copy(); d_tree = d_tree.copy()
    d_emb[self_mask] = np.inf; d_tree[self_mask] = np.inf
    return knn_retrieval_precision(d_emb, d_tree, k=k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--parent-from-mapping", action="store_true",
                    help="Test mode: read a 'parent' column from the mapping instead of the taxdump")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-queries", type=int, default=3000)
    ap.add_argument("--n-cands", type=int, default=2000)
    ap.add_argument("--n-pairs", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--output-dir", required=True)
    args = ap.parse_args()

    emb = load_embeddings(args.checkpoint)
    idx2tax = load_mapping(args.mapping)
    if args.parent_from_mapping:
        df = pd.read_csv(args.mapping, sep="\t")
        parent_col = {int(r.taxid): int(r.parent) for r in df.itertuples()}
        parent, depth = build_index_tree(idx2tax, parent_col=parent_col)
    else:
        taxonomy = load_taxonomy_with_depth(set(idx2tax.values()), args.data_dir)
        parent, depth = build_index_tree(idx2tax, taxonomy=taxonomy)

    td = TreeDistance(parent, depth)
    rng = np.random.default_rng(args.seed)
    n = len(emb)
    query_idx = rng.choice(n, size=min(args.n_queries, n), replace=False)
    cand_idx = rng.choice(n, size=min(args.n_cands, n), replace=False)

    null = radial_only_null(emb, seed=args.seed)

    result = {}
    for name, E in [("model", emb), ("radial_only_null", null)]:
        per_q = _fidelity_block(E, td, query_idx, cand_idx, args.k)
        mean, lo, hi = taxon_bootstrap_ci(per_q, seed=args.seed)
        # global distortion on a stratified pair sample
        a = rng.integers(0, n, args.n_pairs); b = rng.integers(0, n, args.n_pairs)
        dt = td.path_length(a, b)
        keep = sample_pairs_stratified(dt, bin_edges=[0, 2, 4, 6, 8, 100], per_bin=args.n_pairs // 10, seed=args.seed)
        de = numpy_poincare_distance(E[a[keep]], E[b[keep]])
        dist = multiplicative_distortion(de, dt[keep].astype(float))
        result[name] = {"knn_precision": {"mean": mean, "lo": lo, "hi": hi, "k": args.k},
                        "distortion": dist}

    result["delta_knn_precision"] = result["model"]["knn_precision"]["mean"] - \
        result["radial_only_null"]["knn_precision"]["mean"]

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "cophenetic_fidelity.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the integration test to verify it passes**

Run: `.venv/bin/python -m pytest tests/eval/test_cophenetic_cli.py -v`
Expected: PASS

- [ ] **Step 5: Run the full eval test suite**

Run: `.venv/bin/python -m pytest tests/eval/ -v`
Expected: PASS (all tasks' tests)

- [ ] **Step 6: Commit**

```bash
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings add scripts/cophenetic_fidelity.py tests/eval/test_cophenetic_cli.py
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings commit -m "feat(eval): cophenetic_fidelity CLI (model vs radial-only null, bootstrap CIs)"
```

---

### Task 7: Produce the real #1 result on eukaryota 877k

**Files:**
- Output: `artifacts/tags/eukaryota_canonical/cophenetic_fidelity/` (analysis output, not committed)

- [ ] **Step 1: Run on the eukaryota final checkpoint**

Run:
```bash
/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python scripts/cophenetic_fidelity.py \
  --checkpoint artifacts/tags/eukaryota_canonical/eukaryota_canonical.pth \
  --mapping data/taxopy/eukaryota_2759_clean/taxonomy_edges_eukaryota_2759_clean.mapping.tsv \
  --k 10 --n-queries 3000 --n-cands 2000 --n-pairs 2000000 \
  -o artifacts/tags/eukaryota_canonical/cophenetic_fidelity
```
Expected: completes; prints JSON with `model.knn_precision.mean`, `radial_only_null.knn_precision.mean`, and `delta_knn_precision`.

- [ ] **Step 2: Sanity-check the result (the analysis's own validity gate)**

Confirm: (a) `model.knn_precision.mean` is high (expect ≳0.7 at k=10 given family kNN-purity was 0.91) with a tight bootstrap CI; (b) **`delta_knn_precision` is clearly positive** — the model beats the radial-only null (if it's ≈0, the fidelity is a radial artifact and that is a finding to escalate, not bury); (c) `model.distortion.median` is low (close to 1). Record the three numbers in `docs/SESSION_LOG.md` (append a dated note) and update the spec's §5#1 with the measured delta.

- [ ] **Step 3: Re-run on cellular when job 5673097 lands**

When `cellular_canonical.pth` is pulled locally, repeat Step 1 with `--checkpoint artifacts/tags/cellular_canonical/cellular_canonical.pth --mapping data/taxopy/cellular_organisms_131567_clean/taxonomy_edges_cellular_organisms_131567_clean.mapping.tsv`. Re-derive, do NOT transfer, any rank-dependent baseline (spec §9C cellular rank-label trap).

---

## Self-review notes

- **Spec coverage:** This plan covers spec §5 #1 (cophenetic fidelity) + the §9B rigor requirements that touch #1 (local rank fidelity over global Pearson; radial-only null delta; taxon bootstrap CI; stratified sampling; fair distortion). It builds the shared foundation (treedist/nulls/pairs/bootstrap/fidelity) that the **#2 (taxonomy QC)** and **#3 (sampling-bias)** plans will reuse — those are separate plans, to be written next.
- **NOT in this plan (deliberately):** the speed benchmark with the binary-lifting baseline (small follow-up task; the `TreeDistance` here already IS that baseline — benchmark is a timing harness over it), the prior-art comparison table, the artifact packaging, and #2/#3. Each is its own plan/task.
- **Effort honesty:** Task 1 (LCA) is the heaviest; everything else is thin. The integration test (Task 6) runs with a synthetic checkpoint so the suite has no LRZ-data dependency.
- **Type consistency:** `TreeDistance.path_length` / `.lca` / `.lca_depth` take and return int arrays; `knn_retrieval_precision(d_emb_mat, d_tree_mat, k)` and `multiplicative_distortion(d_emb, d_tree)` signatures are used consistently in `cophenetic_fidelity.py`; `taxon_bootstrap_ci` returns `(mean, lo, hi)` and is unpacked as such.
