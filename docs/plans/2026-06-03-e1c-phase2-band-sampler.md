# E1c Phase 2 — Clade-Band Negative Sampler + Gate Rigor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tunable, clade-relative ("band") negative sampler that restores within-family negative gradient at metazoa scale (Step-0 confirmed the default sampler captures only ~2.4% of achievable within-family negatives at 498k vs 27% at echino), plus the measurement rigor (seeded training, seeded+repeated analyzer, kNN-purity) needed to gate it honestly — and validate it echino-first.

**Architecture:** A new dataloader sampler `_sample_negatives_band_vectorized` draws, per training pair, negatives from the **descendant's clade at a tunable level-band** — nodes sharing the descendant's ancestor `w_far` levels up but NOT sharing its ancestor `w_near` levels up (a per-batch mixture over `w_near`), built from a compact ancestor-lineage index over the transitive closure (no runtime LCA). Guardrails (sentinel-safe, self/ancestor-excluded, band-fill-rate tracked) prevent silent reversion to the default sampler. Selected by `--neg-sampling band`. Gates use a seeded analyzer with `--repeats` (mean±std) and a new kNN-purity column (anti-Goodhart vs radial inflation).

**Tech Stack:** Python 3.13, numpy, torch; the repo's `train_hierarchical.HierarchicalDataLoader`/`TrainingPairs`, `train_small.py`, `src/taxembed/cli/main.py`, `scripts/analyze_hierarchy_hyperbolic.py`. pytest (config in `pyproject.toml`).

**WHERE THIS RUNS:** all files + commits land inside the poincaré submodule `TaxPointCare/poincare-embeddings/` on branch `feat/lrz-readiness-prep` (NOT the superproject). Use `git -C TaxPointCare/poincare-embeddings ...`. The superproject's hooks are not installed here — be deliberate with selective `git add`.

**GATING / what's IN vs OUT:**
- IN (this plan, all CPU/local, RED-LINE-safe): the sampler + index, the seed/analyzer rigor, CLI wiring, tests, and the **echino-first G0 grid** + the **multi-phylum (arthropoda) middle gate G0.5**.
- OUT (separate, user-involved): any **metazoa LRZ run** (G1/G2). The RED-LINE forbids sbatch without a local end-to-end first; this plan produces that local end-to-end. The metazoa run is a follow-up the master launches *after* G0/G0.5 pass and after folding the Experiment-1 (5664609) ep80 readout.

**Step-0 evidence this builds on** (PROJECT_STATE Roadmap E1c, 2026-06-03): within-grandparent (≈family) default negative fraction = 0.129 (echino) → 0.011 (metazoa); tiered/achievable flat ~0.46; uniform/chance ~0.0005. Confirms starvation + ~42× headroom at metazoa, and explains the echino A/B (echino not starved → hard negs over-repel; metazoa starved → they fill the gap).

**OPEN DESIGN QUESTION (resolved empirically, not assumed):** which `w_near` helps metazoa without the over-repel pathology — and note **echino alone cannot answer this** (echino is not starved, so hard negs are expected to *hurt* it). The G0 echino grid measures the per-rank trade-off surface and the NON-regression envelope; the **arthropoda middle gate (G0.5)** — a large single clade where the default sampler is already starved (Step-0: arthropoda default within-gp = 0.015) — is the smallest scale that can show a *lift*. Treat G0 as "did we break the working case" and G0.5 as "does it lift a starved case."

---

## File Structure

- **Modify** `train_hierarchical.py`:
  - `_build_depth_index` (~:221-335): add a compact ancestor-lineage index built from the closure.
  - new method `_sample_negatives_band_vectorized(...)`.
  - `HierarchicalDataLoader.__init__` (:184): replace the `tiered_negatives: bool` arg path with a `neg_sampling: str = "default"` mode (+ `band_w_near`, `band_w_far`); keep `tiered_negatives` as a back-compat alias so existing tests/callers don't break.
  - `__iter__` dispatch (:638-641): 2-way `if self.tiered_negatives` → 3-way on `self.neg_sampling`.
- **Modify** `train_small.py`: `--seed` (determinism); `--neg-sampling`/`--neg-band-near`/`--neg-band-far` argparse; pass to the loader construction (:1199-1208).
- **Modify** `src/taxembed/cli/main.py`: forward + persist the new flags (mirror `--tiered-negatives`/`--loss`).
- **Modify** `scripts/analyze_hierarchy_hyperbolic.py`: `--seed`, `--repeats` (mean±std separation), and a kNN-purity-per-rank column.
- **Create** `tests/test_band_sampler.py`: synthetic-tree fixture; sampler correctness (band membership, sibling/self/ancestor exclusion, `-1` sentinel, band-fill).
- **Create** `tests/test_analyzer_rigor.py`: seeding reproducibility + kNN-purity on a synthetic embedding.
- **Reuse (no change):** `scripts/diagnose_negative_hardness.py` (Phase-1) to spot-check the band sampler's within-gp fraction matches expectations.

---

## Task 1: Deterministic training — `--seed`

**Files:** Modify `train_small.py` (argparse near :1043; seeding at the top of `main()` / `train_with_visualization` setup). Test: `tests/test_analyzer_rigor.py` (seed reproducibility is covered jointly with Task 2; this task adds the flag + global seeding).

- [ ] **Step 1: Add the flag** — in the argparse block (near the other flags at `train_small.py:1043-1052`), add:
```python
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (numpy + torch). Default: None (nondeterministic).')
```
- [ ] **Step 2: Seed early** — at the start of `main()` (before model init / dataloader construction), add:
```python
    if args.seed is not None:
        import random as _random
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        _random.seed(args.seed)
```
- [ ] **Step 3: Persist** — ensure `--seed` is forwarded by the CLI (handled in Task 6) and written to `run.json`.
- [ ] **Step 4: Verify** — run two short identical echino runs with `--seed 0 --epochs 3` and confirm identical final loss:
```
/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/taxembed train --file <echino_npz> --mapping <echino_map> -as _seedtest_a --dim 50 --epochs 3 --loss softmax --euclidean-param --seed 0 --gpu 0
```
Run twice (tags `_seedtest_a`, `_seedtest_b`); the printed "Best loss" must match to ≥5 decimals. (MPS has minor nondeterminism in some ops; if the match is only ~3 decimals, note it — seeding still makes runs *comparable*, which is what the gates need. Delete the two `_seedtest_*` tags after.)
- [ ] **Step 5: Commit** — `git -C TaxPointCare/poincare-embeddings add train_small.py` then `git -C ... commit -m "feat(e1c): --seed for reproducible training"`

---

## Task 2: Analyzer rigor — `--seed` + `--repeats` (mean±std separation)

**Files:** Modify `scripts/analyze_hierarchy_hyperbolic.py` (`main()` argparse :413-478; `analyze_hierarchical_clustering_hyperbolic` :254). Test: `tests/test_analyzer_rigor.py`.

The separation estimator subsamples groups/pairs with `np.random.choice` and is unseeded (BLOCKER from review). Make it seeded + repeatable.

- [ ] **Step 1: Write the failing test** — `tests/test_analyzer_rigor.py`:
```python
"""Analyzer rigor: seeding reproducibility + kNN purity."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import numpy as np
from analyze_hierarchy_hyperbolic import separation_repeats  # added in Task 2

def test_separation_repeats_seeded_reproducible():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((200, 8)) * 0.1     # in-ball points
    labels = np.array([i % 4 for i in range(200)])  # 4 groups
    m1, s1, _ = separation_repeats(emb, labels, repeats=5, seed=0)
    m2, s2, _ = separation_repeats(emb, labels, repeats=5, seed=0)
    assert m1 == m2 and s1 == s2          # same seed -> identical
    assert s1 >= 0.0                       # std defined
    m3, _, _ = separation_repeats(emb, labels, repeats=5, seed=1)
    # different seed -> (almost surely) different mean, but same ballpark
    assert abs(m3 - m1) < 0.5 * max(m1, 1e-6) + 0.5
```
- [ ] **Step 2: Run, confirm fail** (`ImportError: separation_repeats`):
`/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest tests/test_analyzer_rigor.py::test_separation_repeats_seeded_reproducible -v -p no:cacheprovider`
- [ ] **Step 3: Implement** — refactor the separation computation into a seedable, repeatable helper. Add to `analyze_hierarchy_hyperbolic.py`:
```python
def _separation_once(emb, labels, rng):
    """One separation-ratio estimate = mean(inter)/mean(intra), groups sampled with rng."""
    groups = {}
    for idx, lab in enumerate(labels):
        groups.setdefault(int(lab), []).append(idx)
    large = {g: v for g, v in groups.items() if len(v) >= 2}
    if len(large) < 2:
        return float("nan")
    sampled = {g: (list(rng.choice(v, 100, replace=False)) if len(v) > 100 else v)
               for g, v in large.items()}
    intra, inter = [], []
    gl = list(sampled.items())
    for i, (g1, i1) in enumerate(gl):
        if len(i1) >= 2:
            for a in range(len(i1)):
                for b in range(a + 1, min(a + 20, len(i1))):
                    intra.append(poincare_distance(emb[i1[a]], emb[i1[b]]))
        for j in range(i + 1, min(i + 10, len(gl))):
            g2, i2 = gl[j]
            for _ in range(min(100, len(i1) * len(i2))):
                inter.append(poincare_distance(emb[int(rng.choice(i1))], emb[int(rng.choice(i2))]))
    if not intra or not inter:
        return float("nan")
    return float(np.mean(inter) / np.mean(intra))

def separation_repeats(emb, labels, repeats=5, seed=0):
    """Returns (mean, std, list) of the separation ratio over `repeats` seeded estimates."""
    vals = [_separation_once(emb, labels, np.random.default_rng(seed + r)) for r in range(repeats)]
    vals = [v for v in vals if not np.isnan(v)]
    if not vals:
        return float("nan"), float("nan"), []
    return float(np.mean(vals)), float(np.std(vals)), vals
```
Then refactor `analyze_hierarchical_clustering_hyperbolic` to build `labels` for the rank and call `separation_repeats(emb, labels, repeats=args.repeats, seed=args.seed)`, printing `mean ± std`. Add argparse: `--seed` (default 0) and `--repeats` (type=int, default 5).
- [ ] **Step 4: Run test, confirm pass.**
- [ ] **Step 5: Commit** — `git -C ... add scripts/analyze_hierarchy_hyperbolic.py tests/test_analyzer_rigor.py` then commit `-m "feat(e1c): seeded + repeated separation (mean±std)"`

---

## Task 3: Analyzer kNN-purity per rank (anti-Goodhart)

**Files:** Modify `scripts/analyze_hierarchy_hyperbolic.py`. Test: extend `tests/test_analyzer_rigor.py`.

Separation ratio can rise via radial inflation without angular improvement (review finding); add the angular signal (kNN purity, ported from `train_small.py:279`'s `compute_class_separation`) so gates can require purity to move with separation.

- [ ] **Step 1: Write the failing test** — append to `tests/test_analyzer_rigor.py`:
```python
from analyze_hierarchy_hyperbolic import knn_purity  # added in Task 3

def test_knn_purity_separates_clusters():
    # Two tight, well-separated clusters -> purity ~1.0
    a = np.full((50, 4), 0.0); a[:, 0] = 0.6
    b = np.full((50, 4), 0.0); b[:, 0] = -0.6
    a += np.random.default_rng(0).standard_normal((50, 4)) * 0.01
    b += np.random.default_rng(1).standard_normal((50, 4)) * 0.01
    emb = np.vstack([a, b]); labels = np.array([0] * 50 + [1] * 50)
    assert knn_purity(emb, labels, k=10, seed=0) > 0.95
    # Shuffled labels -> purity ~chance (0.5)
    shuf = np.random.default_rng(0).permutation(labels)
    assert knn_purity(emb, shuf, k=10, seed=0) < 0.7
```
- [ ] **Step 2: Run, confirm fail.**
- [ ] **Step 3: Implement** — add to `analyze_hierarchy_hyperbolic.py` (numpy port of the train_small purity; uses module `poincare_distance`):
```python
def knn_purity(emb, labels, k=10, n_sample=500, seed=0):
    """Fraction of each node's k nearest (Poincaré) neighbors sharing its label.
    Sampled within a subset of n_sample labeled nodes (O(n_sample^2))."""
    rng = np.random.default_rng(seed)
    idx = np.where(labels >= 0)[0] if (labels < 0).any() else np.arange(len(labels))
    if len(idx) < 4:
        return float("nan")
    sub = rng.choice(idx, min(n_sample, len(idx)), replace=False)
    e = emb[sub]; lab = labels[sub]
    n = len(sub)
    # pairwise Poincaré distance matrix
    D = np.zeros((n, n))
    for i in range(n):
        D[i] = poincare_distance(e[i][None, :], e)   # broadcast (n,) ; poincare_distance handles it
    np.fill_diagonal(D, np.inf)
    kk = min(k, n - 1)
    nn = np.argsort(D, axis=1)[:, :kk]
    same = (lab[nn] == lab[:, None]).mean()
    return float(same)
```
Call it per rank in `main()` and print `kNN-purity: X.XX` next to the separation line. (Confirm `poincare_distance` broadcasts `(1,dim)` vs `(n,dim)` → `(n,)`; it uses `axis=-1` so it does.)
- [ ] **Step 4: Run test, confirm pass.**
- [ ] **Step 5: Commit** — `-m "feat(e1c): kNN-purity per rank in analyzer (anti-Goodhart gate signal)"`

---

## Task 4: Ancestor-lineage closure index

**Files:** Modify `train_hierarchical.py` `_build_depth_index` (add at the end, ~:330). Test: `tests/test_band_sampler.py`.

Build, from the transitive closure (`self.pairs`), two compact structures the band sampler needs (no runtime LCA, no per-(d,p) dict):
- `_node_anc_by_depth: dict[int, dict[int,int]]` — for each descendant node, `{ancestor_depth: ancestor_idx}` (its lineage). Total entries = closure size; memory ~ closure (acceptable; loader already holds the pairs).
- `_anc_depth_to_nodes: dict[tuple[int,int], np.ndarray]` — `(ancestor_idx, descendant_depth) → int64 array of descendant idxs` (the sampling pools). This GENERALIZES the existing `_gp_depth_to_nodes`.

- [ ] **Step 1: Write the failing test** — start `tests/test_band_sampler.py` with a synthetic 3-level tree fixture + an ancestor-index test:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from train_hierarchical import TrainingPairs, HierarchicalDataLoader

def _tree_pairs():
    # root 0; classes 1,2 (depth1); genera 3,4 under 1 and 5,6 under 2 (depth2);
    # species 7,8 under 3; 9,10 under 4; 11,12 under 5; 13,14 under 6 (depth3)
    parent = {1:0,2:0, 3:1,4:1,5:2,6:2, 7:3,8:3,9:4,10:4,11:5,12:5,13:6,14:6}
    depth  = {0:0, 1:1,2:1, 3:2,4:2,5:2,6:2, 7:3,8:3,9:3,10:3,11:3,12:3,13:3,14:3}
    pairs = []
    for node in range(1, 15):
        cur = node
        while cur in parent:                      # walk to root: emit (ancestor, node) for every ancestor
            anc = parent[cur]
            pairs.append({"ancestor_idx": anc, "descendant_idx": node,
                          "depth_diff": depth[node]-depth[anc],
                          "ancestor_depth": depth[anc], "descendant_depth": depth[node],
                          "ancestor_taxid": 1000+anc, "descendant_taxid": 2000+node})
            cur = anc
    return TrainingPairs.from_list(pairs)

def _loader(neg_sampling="band", w_near=2, w_far=3, n_neg=4):
    tp = _tree_pairs()
    return HierarchicalDataLoader(tp, n_nodes=15, batch_size=8, n_negatives=n_neg,
                                  neg_sampling=neg_sampling, band_w_near=w_near, band_w_far=w_far)

def test_ancestor_index():
    ld = _loader()
    # species 7 (depth3): ancestor at depth2 = genus 3; at depth1 = class 1; at depth0 = root 0
    assert ld._node_anc_by_depth[7] == {0:0, 1:1, 2:3}
    # pool (ancestor=1 i.e. class, descendant_depth=3) = all species under class 1 = {7,8,9,10}
    assert set(ld._anc_depth_to_nodes[(1, 3)].tolist()) == {7,8,9,10}
    # pool (ancestor=3 i.e. genus, descendant_depth=3) = species under genus 3 = {7,8}
    assert set(ld._anc_depth_to_nodes[(3, 3)].tolist()) == {7,8}
```
- [ ] **Step 2: Run, confirm fail** (`__init__` doesn't accept `neg_sampling`/`band_*` yet AND the index attrs don't exist — this also drives Task 5/6 wiring; for THIS task, temporarily construct with the existing signature if needed, or implement the `__init__` kwargs in Step 3 below).
- [ ] **Step 3: Implement the index** — at the end of `_build_depth_index` (after the `_node_class_arr` block ~:329), add:
```python
        # Ancestor-lineage index for band sampling (built from the transitive closure).
        from collections import defaultdict
        self._node_anc_by_depth = defaultdict(dict)
        anc_depth_lists = defaultdict(list)
        a_idx = self.pairs.ancestor_idx; d_idx = self.pairs.descendant_idx
        a_dep = self.pairs.ancestor_depth; d_dep = self.pairs.descendant_depth
        for i in range(len(a_idx)):
            di = int(d_idx[i])
            self._node_anc_by_depth[di][int(a_dep[i])] = int(a_idx[i])
            anc_depth_lists[(int(a_idx[i]), int(d_dep[i]))].append(di)
        self._anc_depth_to_nodes = {k: np.array(sorted(set(v)), dtype=np.int64)
                                    for k, v in anc_depth_lists.items()}
```
(Also implement the `__init__` kwargs `neg_sampling="default"`, `band_w_near=2`, `band_w_far=3` + back-compat `tiered_negatives` → `neg_sampling="tiered"`; see Task 6 for the exact `__init__`/dispatch edit — do that edit here so the test can construct the loader, and Task 6 only adds the CLI surface.)
- [ ] **Step 4: Run test, confirm pass.**
- [ ] **Step 5: Commit** — `-m "feat(e1c): ancestor-lineage closure index for band sampling"`

---

## Task 5: The band sampler

**Files:** Modify `train_hierarchical.py` (new method + `__iter__` dispatch). Test: extend `tests/test_band_sampler.py`.

Per pair `(ancestor, descendant d at depth dd)`: pick `w_near` from the mixture `{w_near, w_near+1, ... up to w_far-1}` (default mixture spans the configured near→far), look up the descendant's ancestor at depth `dd - w` (the EXCLUDE boundary) and at depth `dd - w_far` (the INCLUDE boundary); band pool = nodes in the far-ancestor's depth-`dd` subtree MINUS the near-ancestor's depth-`dd` subtree; sample WITH replacement; exclude self + the pair's ancestor; top up from same-depth default pool if the band is empty; record band-fill.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_band_sampler.py`:
```python
def test_band_excludes_immediate_subclade_and_self():
    # w_near=1, w_far=3 for a depth-3 species anchored at the root (dd large).
    # For species 7 (genus3, class1): far ancestor at depth (3-3)=0 = root -> subtree@depth3 = all species 7..14.
    # near ancestor at depth (3-1)=2 = genus3 -> subtree@depth3 = {7,8}. Band = {9,10,11,12,13,14}.
    tp = _tree_pairs()
    ld = HierarchicalDataLoader(tp, n_nodes=15, batch_size=8, n_negatives=4,
                                neg_sampling="band", band_w_near=1, band_w_far=3)
    negs = ld._sample_negatives_band_vectorized(
        desc_depths=np.array([3]), desc_idxs=np.array([7]),
        anc_idxs=np.array([0]), batch_size=1)
    s = set(negs[0].tolist())
    assert 7 not in s and 8 not in s         # excludes self + immediate sub-clade (genus 3)
    assert s.issubset({9,10,11,12,13,14})    # within the band

def test_band_w_near_2_keeps_cousins_excludes_genus():
    # w_near=2 excludes the class subtree (depth (3-2)=1 = class1 -> {7,8,9,10}); far=root -> all;
    # band = species in OTHER classes = {11,12,13,14}.
    tp = _tree_pairs()
    ld = HierarchicalDataLoader(tp, n_nodes=15, batch_size=8, n_negatives=4,
                                neg_sampling="band", band_w_near=2, band_w_far=3)
    negs = ld._sample_negatives_band_vectorized(
        desc_depths=np.array([3]), desc_idxs=np.array([7]),
        anc_idxs=np.array([0]), batch_size=1)
    assert set(negs[0].tolist()).issubset({11,12,13,14})

def test_band_returns_correct_shape_and_dtype():
    ld = HierarchicalDataLoader(_tree_pairs(), n_nodes=15, batch_size=8, n_negatives=4,
                                neg_sampling="band", band_w_near=1, band_w_far=3)
    negs = ld._sample_negatives_band_vectorized(
        desc_depths=np.array([3,3]), desc_idxs=np.array([7,11]),
        anc_idxs=np.array([0,0]), batch_size=2)
    assert negs.shape == (2, 4) and negs.dtype == np.int64
```
- [ ] **Step 2: Run, confirm fail** (`_sample_negatives_band_vectorized` missing).
- [ ] **Step 3: Implement** — add the method to `HierarchicalDataLoader`:
```python
    def _sample_negatives_band_vectorized(self, desc_depths, desc_idxs, anc_idxs, batch_size):
        """Clade-band negatives: within the descendant's clade at a tunable level-band,
        EXCLUDING the immediate sub-clade. Per-item w_near mixture over [w_near, w_far-1].
        Falls back to same-depth default negatives when a band pool is empty (tracked)."""
        negatives = np.zeros((batch_size, self.n_negatives), dtype=np.int64)
        self._band_fill_hits = getattr(self, "_band_fill_hits", 0)
        self._band_fill_total = getattr(self, "_band_fill_total", 0)
        w_lo, w_far = self.band_w_near, self.band_w_far
        for i in range(batch_size):
            d = int(desc_idxs[i]); dd = int(desc_depths[i]); anc = int(anc_idxs[i])
            lineage = self._node_anc_by_depth.get(d, {})
            # choose w_near from the mixture (levels strictly inside w_far)
            choices = [w for w in range(w_lo, max(w_lo + 1, w_far)) if (dd - w) in lineage]
            band = None
            if choices and (dd - w_far) in lineage:
                w = int(np.random.choice(choices))
                far_anc = lineage[dd - w_far]; near_anc = lineage[dd - w]
                far_pool = self._anc_depth_to_nodes.get((far_anc, dd))
                near_pool = self._anc_depth_to_nodes.get((near_anc, dd))
                if far_pool is not None and len(far_pool):
                    if near_pool is not None and len(near_pool):
                        band = far_pool[~np.isin(far_pool, near_pool)]
                    else:
                        band = far_pool
                    # exclude self + the pair's ancestor
                    band = band[(band != d) & (band != anc)]
            if band is not None and len(band) > 0:
                negatives[i] = np.random.choice(band, self.n_negatives, replace=True)
                self._band_fill_hits += 1
            else:
                # fallback: same-depth default pool (tracked as a band miss)
                pool = self._depth_to_nodes.get(dd)
                if pool is not None and len(pool) > 1:
                    cands = pool[pool != d]
                    negatives[i] = np.random.choice(cands, self.n_negatives, replace=True)
                else:
                    negatives[i] = np.random.randint(0, self.n_nodes, self.n_negatives)
            self._band_fill_total += 1
        return negatives
```
- [ ] **Step 4: Wire the dispatch** — in `__iter__` (`train_hierarchical.py:638-641`), change the 2-way branch to 3-way and pass `anc_idxs`:
```python
            anc_idxs = self.pairs.ancestor_idx[batch_indices]
            if self.neg_sampling == "band":
                negatives = self._sample_negatives_band_vectorized(desc_depths, desc_idxs, anc_idxs, batch_size)
            elif self.neg_sampling == "tiered":
                negatives = self._sample_negatives_tiered_vectorized(desc_depths, desc_idxs, batch_size)
            else:
                negatives = self._sample_negatives_default_vectorized(desc_depths, desc_idxs, batch_size)
```
- [ ] **Step 5: Run tests, confirm pass.** Then spot-check on real echino with the Phase-1 diagnostic (band should raise within-gp fraction toward the band level): run `scripts/diagnose_negative_hardness.py` after Task 6 adds the CLI; for now assert via the unit tests.
- [ ] **Step 6: Commit** — `-m "feat(e1c): clade-band negative sampler (ancestor-anchored, sibling-excluded, w_near mixture)"`

---

## Task 6: CLI wiring + per-epoch band-fill log

**Files:** Modify `train_small.py`, `src/taxembed/cli/main.py`. Test: covered by Tasks 4-5 (loader kwargs) + a smoke run.

Mirror the exact `--tiered-negatives`/`--loss` wiring pattern (grounded line refs below).

- [ ] **Step 1: `train_small.py` argparse** — near :1043 add:
```python
    parser.add_argument('--neg-sampling', choices=['default', 'tiered', 'band'], default='default',
                       help='Negative sampler: default (same-depth), tiered, or band (clade-band, E1c)')
    parser.add_argument('--neg-band-near', type=int, default=2, help='Band: exclude sub-clade within this many levels of the descendant')
    parser.add_argument('--neg-band-far', type=int, default=3, help='Band: include clade up to this many levels above the descendant')
```
- [ ] **Step 2: `train_small.py` loader construction** (:1199-1208) — add kwargs:
```python
        neg_sampling=getattr(args, 'neg_sampling', 'default'),
        band_w_near=getattr(args, 'neg_band_near', 2),
        band_w_far=getattr(args, 'neg_band_far', 3),
```
(Keep the existing `tiered_negatives=getattr(args,'tiered_negatives',False)` line; the `__init__` back-compat maps it. If `--neg-sampling tiered` is given it takes precedence.)
- [ ] **Step 3: Per-epoch band-fill log** — in the epoch loop (after the dataloader pass, near the existing per-epoch print), if `neg_sampling == 'band'`, print the band-fill rate: `hits/total` from `dataloader._band_fill_hits/_band_fill_total`, then reset them to 0 for the next epoch. (Gives the band-fill-rate guard the reviews required.)
- [ ] **Step 4: `main.py` forward + persist** — mirror `--loss` (value-flag) at the three sites: `train_cmd.extend(["--neg-sampling", args.neg_sampling])` etc. near :412; argparse near :746; `run.json` keys near :476. Add `--seed` (Task 1) the same way (forward only if not None). Use the exact patterns from `--loss`/`--tiered-negatives` already in `main.py`.
- [ ] **Step 5: Smoke** — full CLI path end-to-end on echino (this is the RED-LINE "real argv" check):
```
/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/taxembed train --file <echino_npz> --mapping <echino_map> -as _band_smoke --dim 50 --epochs 3 --loss softmax --euclidean-param --neg-sampling band --neg-band-near 2 --neg-band-far 3 --seed 0 --gpu 0
```
Confirm: runs clean; `run.json` shows `"neg_sampling": "band"`, `"neg_band_near": 2`, `"neg_band_far": 3`, `"seed": 0`; band-fill rate printed per epoch. (Delete the `_band_smoke` tag after.)
- [ ] **Step 6: Commit** — `-m "feat(e1c): wire --neg-sampling/--neg-band-* + --seed through CLI; per-epoch band-fill log"`

---

## Task 7: Echino G0 grid + arthropoda middle gate (G0.5)

**Files:** Modify docs (record results). No code. All CPU/local. This is the validation, NOT a metazoa run.

- [ ] **Step 1: Echino paired baseline (already have it)** — the reference is `echino_softmax_abctrl` (order 2.54×, family 1.99×, class 1.46×, depth↔norm +0.956). Re-analyze it with the NEW seeded analyzer (`--repeats 5 --seed 0`) to get mean±std + kNN-purity per rank as the baseline envelope.
- [ ] **Step 2: Echino band grid** — train echino with the winning recipe + `--neg-sampling band` for `(w_near, w_far)` in `{(1,3), (2,3), (2,4), (3,5)}`, each `--seed 0 --early-stopping 0 --epochs 200 --dim 100 --save-every 20`. For each, run the seeded analyzer (`--repeats 5`) → per-rank separation mean±std + kNN-purity, at the peak-separation milestone (not loss-best). Use the exact `taxembed train ... --neg-sampling band` CLI argv (RED-LINE: the same path a future LRZ run uses).
- [ ] **Step 3: G0 verdict (echino)** — record the per-rank trade-off surface. EXPECTATION (per the open-question note): hard bands likely *reduce* echino separation (echino isn't starved). G0 is NOT "band must beat echino" — it's "we understand the regression and at least one band cell is within noise of baseline OR the regression is explained by over-repel at small w_near." Pick the band that's gentlest on echino as a default, but DO NOT down-select on echino alone.
- [ ] **Step 4: Arthropoda middle gate (G0.5) — the lift test** — train arthropoda (324,983 nodes, present locally) with default vs `--neg-sampling band` (the gentlest non-regressing cell + one or two others), `--seed 0`, paired. Arthropoda is a large single clade where Step-0 showed the default is already starved (within-gp 0.015), so it is the smallest scale that can show a LIFT. Analyze with `--repeats 5` + kNN-purity. **G0.5 PASS = band lifts arthropoda per-rank separation above the default-sampler paired control (mean lift > 2σ) with kNN-purity moving up too.** This is the real go-signal for the metazoa run. (Arthropoda 200ep/dim100 on MPS is hours; consider `--epoch-fraction 0.3` to match the metazoa recipe and keep it overnight-scale, or run on the LRZ as the FIRST sbatch only after a local echino end-to-end — master decision.)
- [ ] **Step 5: Record + decide** — write G0/G0.5 results into `docs/PROJECT_STATE.md` (metrics table + Roadmap E1c) and a `docs/SESSION_LOG.md` entry. Decision: G0.5 lift → queue the metazoa run (separate, master-launched, echino-end-to-end-first per RED-LINE, paired control, no `--amp` or AMP-matched, save-every 10, read G1/G2 at ep80–110 window). No lift → diagnose which band/rank, or escalate to E2 (cones) / E3.

---

## Out of plan (master-launched follow-ups, NOT subagent tasks)

- **Metazoa G1/G2 LRZ run** — gated on G0.5 lift + Experiment-1 (5664609) ep80 readout. Requires the RED-LINE local echino end-to-end first (Task 6 smoke + Task 7 echino grid satisfy that), the user's sentinel/LRZ involvement, and a paired same-seed default control. Read against: depth↔norm ≥+0.85 AND sep ≥1.10× AND kNN-purity held, across the ep80–110 window.
- **E2/E3** — separate specs if E1c stalls.

---

## Self-review notes

- **Spec coverage:** implements spec v2's band sampler (ancestor-lineage, sibling-excluded, w_near mixture, precomputed membership — Tasks 4-5), the gate rigor (seeded training T1, seeded+repeated analyzer T2, kNN-purity T3), CLI (T6), and the echino-first + middle gate (T7). The metazoa run + Experiment-1 fold are explicitly out-of-plan (master/LRZ). `--neg-band-near/far` are RE-INCLUDED (spec v2 deferred them) because the G0 grid needs to sweep them — justified by the Step-0 open design question.
- **No placeholders:** complete code for the algorithmic cores (index, sampler, analyzer rigor, purity, seeding) and exact mirror-this-line specs for the mechanical CLI wiring (grounded in the verbatim `--loss`/`--tiered-negatives` pattern).
- **Type/naming consistency:** `neg_sampling`/`band_w_near`/`band_w_far` used identically across `__init__` (T4), sampler (T5), and CLI (T6); `_node_anc_by_depth`/`_anc_depth_to_nodes` defined in T4 and consumed in T5; `separation_repeats`/`knn_purity` defined in T2/T3 and used in T7.
- **Risk:** modifies production training/loss-adjacent code (the sampler + dataloader `__init__`), so the back-compat `tiered_negatives` alias + the existing test suite (`tests/test_training_pairs.py` constructs the loader ~18×) must stay green — run the FULL test suite after T4-T6, not just the new tests. RED-LINE: no metazoa/LRZ in this plan; the only large local run (arthropoda G0.5) is a master decision with an explicit cost note.
- **Honesty:** the open design question (which w_near; echino can't validate a lift) is surfaced, not hidden; G0 is framed as "don't break the working case," G0.5 as "show a lift," and only G0.5 gates the GPU spend.

---

# Plan fan-review — findings ledger + v2 (4 reviewers, 2026-06-03)

**STATUS: BLOCKED — not execution-ready.** Two design BLOCKERs (band anchored to the wrong node; no gradient-mass diagnostic) plus a code BLOCKER (kNN-purity broadcast) mean the core algorithm may measure the wrong thing. The fix is a corrected ancestor-anchored design + a **cheap pre-implementation validation gate (Phase 2.0)** that empirically confirms the mechanism before any training sampler is written. PR1 (algorithm), PR2 (API/perf), PR3 (validity), PR4 (scope/hygiene).

## BLOCKERs
- **[PR1+PR3] Band anchored to the DESCENDANT; the loss measures negatives from the ANCESTOR.** This reverts spec v2's top folded fix. The plan's sampler keys the band off `_node_anc_by_depth[descendant]`, but `softmax_loss`/`ranking_loss` compute `d(anc_emb, neg_emb)` (train_hierarchical.py:730-732/680). Descendant-cousins are mostly far from the ancestor → ~0 softmax gradient → a null/lift result is uninterpretable. PR1 showed the descendant-anchoring is only "hard" when `anc_depth ≤ dd − w_far` (a minority of closure pairs). → **FIX: re-anchor on the ANCESTOR's lineage** (see v2 below).
- **[PR3] No `p_j`-from-ancestor diagnostic ships with the sampler.** Band-fill-rate is structural (did a pool exist), not gradient (did the loss put mass on band negatives). Tiered RAISED hardness and DROPPED separation (E1a) — so hardness without gradient-mass-correlation is not decision-grade. → **FIX: Phase 2.0 validates `p_j`-mass-from-ancestor before implementation.**
- **[PR2] `knn_purity` is broken: `poincare_distance` (analyzer:147) has NO `axis` arg and returns a SCALAR**, not a row vector — the plan's `D[i] = poincare_distance(e[i][None,:], e)` collapses to a scalar → kNN order is noise. → **FIX: add a vectorized `poincare_distance_rowwise(q, M)` (axis=-1, `np.minimum` clip) OR keep a scalar inner `for j` loop (measured 1.4s for 500×500 — acceptable). Same trap exists in the train_small port — check before reusing.**

## MAJORs
- **[PR3] Training-vs-eval geometry mismatch unaddressed.** Sampler shapes ancestor↔negative distances; the gate measures leaf↔leaf separation grouped by rank. The causal chain must be *validated* (correlate the ancestor-anchored within-minibatch proxy with the leaf↔leaf analyzer ratio on echino) — spec v2's "validate the indicator" step the plan dropped. → Phase 2.0 + G0 add this.
- **[PR1] No `w_near < w_far` guard** → equal/inverted silently reverts to the starved default for every pair (the exact silent-revert the guardrails exist to prevent). → validate in `__init__`/CLI.
- **[PR3] The `(w_near=2,w_far=3)` default degenerates to a single-level slice** (`range(2,3)={2}`), not the spec's multi-level mixture, and sits at the family rank where E1a's over-repel lives. → default must be a real multi-level mixture (e.g. `w_far ≥ w_near+2`); add a band-composition-by-rank readout.
- **[PR2] Analyzer refactor is NOT faithful:** `_separation_once` filters groups `>=2` vs the original `>=10` (different numbers) and `separation_repeats` returns a 3-tuple while `main()` unpacks `(sep, qual)` → breaks callers. → preserve `min_size=10` + the `(separation, quality)` contract.
- **[PR3] 2σ over 5 ANALYZER repeats ≠ training-seed variance** (the dominant source; band injects fresh RNG every batch). → gate on the band−default paired delta over **≥3 training seeds**.
- **[PR3] Arthropoda (single phylum) may not transfer to metazoa (38 phyla);** spec v2 specified a *merged multi-phylum* middle gate to exercise inter-phylum crowding. → restore merged gate OR document G0.5 doesn't validate the inter-phylum axis.
- **[PR4] Arthropoda LRZ path floated inside the subagent task** → RED-LINE risk (the S0274 incident shape). → ALL LRZ out-of-plan; hard STOP before any run >~30 min or any sbatch.
- **[PR4] Commit steps are compound bash** (`add … then commit`) → trips the shell-hygiene hook / Rule 15. → `Write` msg to `/tmp` + `git -C … add` (one call) + `git -C … commit -F /tmp/...` (one call).
- **[PR4] No explicit regression checkpoint** that default/tiered negatives are unchanged after the `__init__` refactor (only prose). → add a checkbox: full `pytest tests/` after T4 & T5 + a "default/tiered negatives bit-identical pre/post refactor" test.
- **[PR4] CLI wiring under-specified** ("etc.") → inline the exact 3-site additions for all four flags.

## MINOR/NIT (fold)
- [PR2] **Index memory: plan's "~100MB" is wrong but the truth is FINE** — measured arthropoda ~0.3GB resident / ~0.8GB transient / ~4s; metazoa ~0.5GB / ~1.5GB / ~8s. No OOM, no architectural change; just correct the prose (and a CSR/`lexsort` build is a cheap optional optimization). [PR1] w_near=1 test is a mixture (under-tests) — add a deterministic `(1,2)` equality case + a `w_near==w_far` revert-guard test + a `tiered_negatives=True, neg_sampling="band"` precedence test. [PR2] `--seed` needs an `if args.seed is not None` guard (not the unconditional `--loss` pattern). [PR3] stratify band-fill by dd bucket (confirm active at dd≤18). [PR3] G2 PASS (sep≥1.10×) is below EXCELLENT — add "PASS ≠ project success; E2/E3 still required" to the launch decision. [PR4] replace `<echino_npz>`/`<echino_map>` placeholders with absolute paths; show both `_seedtest_a/_b` commands. [PR3-NIT] optional angular-only purity column.

---

# Spec v2.1 — corrected band design (supersedes Tasks 4-5) + Phase 2.0 gate

## The corrected, ANCESTOR-anchored band
For a training pair `(ancestor a @ depth da, descendant d @ depth dd)`, the hard negative for the
loss `d(a, ·)` is a node **near `a` in the tree but NOT in `a`'s own subtree** (a's subtree = the
positive's clade). So:
- Build `_node_anc_by_depth` keyed as before, but the band uses **`a`'s lineage**, not `d`'s.
- `far_anc = lineage_a[da − w_far]` (go `w_far` levels above the anchor); `excl_anc = lineage_a[da − w_near]` with `0 ≤ w_near < w_far` (`w_near=0` ⇒ exclude `a`'s own subtree; `w_near=1` ⇒ also exclude `a`'s parent's subtree, i.e. drop the closest cousins).
- **band = `_anc_depth_to_nodes[(far_anc, P)]` MINUS `_anc_depth_to_nodes[(excl_anc, P)]`**, where `P` is the negatives' depth (use the descendant's depth `dd`, matching the existing same-depth scheme).
- These are leaves/nodes in **sibling clades of `a`** — pushing them from `a` separates `a`'s clade from its siblings = exactly rank-`da` separation, and they ARE hard w.r.t. `a` by construction. The curriculum (dd≤1→9→18) naturally sweeps `da` from fine to coarse ranks.
- Mixture over `w_far` (or `w_near`) per batch so multiple ranks get cousins. Default e.g. `w_near=0, w_far∈{1,2,3}` mixture.
- Guards unchanged: `w_near < w_far` validated; self/positive-subtree excluded (the `(a, P)` subtract already removes d and its co-clade); with-replacement; band-fill (per dd bucket).

## Phase 2.0 — validate the mechanism BEFORE building the training sampler (cheap, ~0 GPU)
Extend the EXISTING `scripts/diagnose_negative_hardness.py` (Phase-1) with an ancestor-anchored
band mode (it already computes `p_j` from the ANCESTOR — `d_pos=d(anc,desc)`, `d_neg=d(anc,neg)` —
so it directly measures the quantity the BLOCKER is about). On echino + arthropoda, with the
healthy echino checkpoint + arthropoda (train a quick local arthropoda checkpoint OR use embeddings
from a short run), measure for the ancestor-anchored band negatives:
1. **`p_j`-mass-from-ancestor** — must be materially > the default sampler's (proves the band negatives actually receive gradient). If it ISN'T, the anchoring/design is still wrong → stop, redesign.
2. **band composition by rank** (what fraction of band negatives are same-family/order/class of the anchor) — confirms which rank a given `(w_near,w_far)` targets.
3. **correlation** of the within-minibatch ancestor-anchored separation proxy with the leaf↔leaf analyzer ratio (addresses the training-vs-eval geometry MAJOR).
Only if Phase 2.0 confirms (1)+(3) do we implement the training sampler (corrected Tasks 4-6) and
run the G0/G0.5 grid. This mirrors Step-0's "validate before build" discipline and is the gate the
reviews demand.

## Other v2 corrections (apply when Tasks 1-7 are rewritten)
kNN-purity scalar-loop or rowwise fix; analyzer `min_size=10` + `(sep,qual)` contract; `w_near<w_far`
guard; multi-level mixture default; ≥3-training-seed gate; merged multi-phylum middle gate (or
documented limitation); ALL LRZ out-of-plan + hard STOP >30min; commit via `Write`+`git commit -F`
(separate calls); explicit full-suite + bit-identical-default regression checkpoints; inline CLI
wiring for all four flags; absolute paths (no `<placeholders>`); correct the index-memory prose.
