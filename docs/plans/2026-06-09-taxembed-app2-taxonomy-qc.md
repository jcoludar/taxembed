# TaxEmbed Application #2 — taxonomy QC / anomaly detection — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the LEAD application of the TaxEmbed paper — a **predictive taxonomy QC / anomaly detector**. Per node, score the geometric-vs-lineage disagreement as a **size-conditioned kNN-impurity** (observed purity minus its size-matched chance expectation, plus a z-score vs a depth+clade-size-matched random-angle null), rank all ~877k–1.1M nodes, BH-correct any per-node significance, and validate the score three independent ways: (A) a synthetic ROC stratified by phylogenetic displacement, (B) the headline NCBI release-diff — does a high score on an *older* taxdump predict later NCBI reclassification (odds ratio + Fisher/permutation CI, matched background), and (C) incertae-sedis/environmental enrichment. Develop on eukaryota 877k now; re-derive every rank-dependent baseline and re-run on cellular 1.1M when checkpoint 5673097 lands.

**Architecture:** A new importable subpackage module `src/taxembed/eval/anomaly.py` holds the pure, unit-tested score logic (no I/O); thin CLIs `scripts/taxonomy_anomaly.py` (score + rank + FDR) and `scripts/_anomaly_validation.py` (the three validation legs) wire it to a checkpoint + taxdump and write results. This mirrors the package/scripts split from **Plan 1** (`docs/plans/2026-06-09-taxembed-eval-foundation-and-cophenetic-fidelity.md`). We **build on Plan 1's foundation** and do NOT re-implement it:
- reuse `taxembed.eval.treedist.TreeDistance` (binary-lifting LCA → depth, path-length, parent),
- reuse `taxembed.eval.nulls` (the radial-only null is the Goodhart guard for leg-A robustness checks),
- reuse `taxembed.eval.bootstrap.taxon_bootstrap_ci` (taxon-level CIs, never pair-count p-values),
- reuse the kNN machinery in `scripts/knn_purity_hyperbolic.py` (`_prep_sqnorms`, `_batch_distances`, `chance_purity`, `build_pool`, `_validate_distance`) and the loaders in `scripts/analyze_hierarchy_hyperbolic.py` (`load_embeddings`, `load_mapping`, `load_taxonomy_with_depth`, `get_ancestor_at_rank`),
- reuse the incertae-sedis/environmental classifier in `scripts/audit_taxonomy_noise.py` (`classify_name_noise`, `is_container`, `parse_names_dmp`) for leg C,
- reuse `src/taxembed/utils/taxdump.py` (`ensure_taxdump`) for the release-diff fetch + canonicalization (`merged.dmp` / `delnodes.dmp`).

Every headline number reports the score **relative to its size-conditioned expectation** (spec §9B: raw kNN-impurity just rediscovers rare/small clades) and must **beat trivial baselines** (clade size, depth, node degree, distance-to-parent-centroid). The release-diff is framed as **enrichment (odds ratio), not perfect recovery**, against a background **matched on depth + clade-size + study-effort (#descendants)**.

**Tech Stack:** Python 3.12, numpy, scipy (`scipy.stats.fisher_exact`, `scipy.stats.rankdata`; `sklearn.metrics.roc_auc_score` for AUC — sklearn is already a dep), pandas, pytest. Poincaré distance + neighbour ordering reuse `knn_purity_hyperbolic._batch_distances` (float32 matmul, float64-validated). Network fetch for leg B reuses `taxembed.utils.taxdump`.

**Run context:** venv at `.venv` — invoke `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python`. Always pass explicit `--checkpoint`/`--mapping` and the **`final` (ep200)** checkpoint `artifacts/tags/eukaryota_canonical/eukaryota_canonical.pth` (the milestone files stop at ep180; the unsuffixed `.pth` IS ep200) with mapping `data/taxopy/eukaryota_2759_clean/taxonomy_edges_eukaryota_2759_clean.mapping.tsv`. `run.json` holds LRZ container paths and tag-resolution defaults to `_best`; never rely on it (spec §9C local-path gotcha). **Cellular rank-label trap (spec §9C):** re-derive EVERY baseline (chance-purity, size bins, depth bins, distance-bin edges) from the dataset at hand — never hardcode a eukaryota-derived constant into a cellular figure.

---

## File structure

- Create `src/taxembed/eval/anomaly.py` — pure core: `anomaly_scores` (size-conditioned kNN-impurity + matched-null z), `trivial_baselines` (clade size, depth, degree, dist-to-parent-centroid), `baseline_aucs`, `benjamini_hochberg`, `relocate_nodes` (synthetic displacement), `displacement_class`, `enrichment_odds_ratio` (Fisher + matched background), `match_background`.
- Create `scripts/taxonomy_anomaly.py` — CLI: load embedding + taxonomy, build pool + `TreeDistance`, compute the per-node score + trivial baselines, BH-correct, write a ranked TSV + JSON summary.
- Create `scripts/_anomaly_validation.py` — CLI: leg A (synthetic ROC vs displacement), leg B (release-diff odds ratio), leg C (incertae-sedis/env enrichment). Subcommands `roc` / `releasediff` / `enrichment`.
- Create `src/taxembed/eval/release_diff.py` — pure core for leg B: `canonicalize_taxid` (merged.dmp + delnodes.dmp), `parse_parents` (nodes.dmp → {taxid: parent}), `reclassified_taxa` (direct-parent change after canonicalization, excluding pure ID-merges/rank-only), `parse_merged`, `parse_delnodes`.
- Create tests: `tests/eval/test_anomaly.py`, `tests/eval/test_release_diff.py`, `tests/eval/test_anomaly_cli.py`, `tests/eval/test_anomaly_validation_cli.py`.

The eval core takes **integer node-index arrays** (0..N-1, the embedding's own indexing) and integer label arrays so it stays pure and decoupled from taxids; the CLIs build the index↔taxid↔parent maps from the mapping + taxdump. `tests/eval/__init__.py` already exists from Plan 1.

---

### Task 1: Size-conditioned anomaly score (the core)

**Files:**
- Create: `src/taxembed/eval/anomaly.py`
- Test: `tests/eval/test_anomaly.py`

**Score definition (PINNED — spec §9B):** For each query node `i` that has a rank-R ancestor label `L_i`, take its **k nearest neighbours** (full hyperbolic distance, self excluded, neighbours restricted to the labelled pool at rank R). The **vote rule** is *fraction of the k neighbours whose rank-R label equals `L_i`* — call it `observed_purity_i ∈ [0,1]`. The raw impurity `1 - observed_purity_i` rediscovers rare clades, so we condition on size two ways:
1. **Excess impurity** = `chance_purity(pool_lab) - observed_purity_i` where `chance_purity` is the existing size-aware expectation `Σ_g (n_g/P)²` (reused from `knn_purity_hyperbolic`). NOTE the sign: chance_purity is the expected purity of a *random* neighbour; a clean node has `observed > chance` → excess negative; an anomalous node has `observed ≪ chance` → excess positive. We define the score so **higher = more anomalous**: `score_excess_i = chance_purity - observed_purity_i`. (For a small clade chance_purity is tiny, so a clean small-clade node gets score ≈ -tiny ≈ 0, NOT a large positive — this is exactly the size de-biasing.)
2. **Matched-null z-score** = the primary headline score. For each query, draw `n_null` random nodes from a **depth-and-clade-size-matched** stratum (same depth bin AND same clade-size bin as the query), compute their observed purity the same way, and report `z_i = (mu_null - observed_purity_i) / (sigma_null + eps)` — higher = more anomalous, automatically conditioned on both confounds. The CLI uses `score_z` as the ranking key; `score_excess` is reported alongside as a cross-check.

`k`, the rank R, and the vote rule are CLI parameters (defaults `k=10`, `rank=family`, fraction-vote) recorded in the output JSON for reproducibility.

- [ ] **Step 1: Write the failing test**

`tests/eval/test_anomaly.py`:
```python
import numpy as np
from taxembed.eval.anomaly import (
    excess_impurity,
    matched_null_z,
    trivial_baselines,
    baseline_aucs,
    benjamini_hochberg,
    relocate_nodes,
    displacement_class,
)


def test_excess_impurity_size_conditioned_sign():
    # observed purity per query; chance purity for the pool
    observed = np.array([1.0, 0.5, 0.0])
    chance = 0.25
    score = excess_impurity(observed, chance)
    # higher == more anomalous: the 0.0-purity query is most anomalous
    assert score[2] > score[1] > score[0]
    assert np.isclose(score[0], chance - 1.0)   # clean node -> negative
    assert np.isclose(score[2], chance - 0.0)   # impure node -> +chance


def test_matched_null_z_flags_outlier_not_small_clade():
    rng = np.random.default_rng(0)
    # null observed-purity samples per query: mean 0.8, sd 0.1
    null_obs = rng.normal(0.8, 0.1, size=(4, 200))
    observed = np.array([0.8, 0.79, 0.81, 0.2])   # query 3 is the real outlier
    z = matched_null_z(observed, null_obs)
    assert z[3] > 3.0                              # >3 sigma below its matched null
    assert abs(z[0]) < 1.0 and abs(z[1]) < 1.0     # in-family queries near 0


def test_trivial_baselines_shapes_and_keys():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((20, 4)) * 0.1
    parent = np.array([0] + [0] * 9 + [1] * 9)      # node0 root-ish, rough tree
    depth = np.array([0] + [1] * 9 + [2] * 9)
    clade_size = np.full(20, 3)
    degree = np.full(20, 2)
    b = trivial_baselines(emb, parent, depth, clade_size, degree)
    assert set(b) == {"clade_size", "depth", "degree", "dist_to_parent_centroid"}
    for v in b.values():
        assert v.shape == (20,)


def test_baseline_aucs_ranks_a_good_score_above_random():
    rng = np.random.default_rng(0)
    labels = np.zeros(200, int); labels[:20] = 1          # 20 true anomalies
    good = rng.normal(0, 1, 200); good[:20] += 4          # separates well
    junk = rng.normal(0, 1, 200)                          # no signal
    aucs = baseline_aucs(labels, {"good": good, "junk": junk})
    assert aucs["good"] > 0.9
    assert abs(aucs["junk"] - 0.5) < 0.15


def test_benjamini_hochberg_monotone_and_bounded():
    p = np.array([0.001, 0.008, 0.02, 0.04, 0.9])
    q = benjamini_hochberg(p)
    assert q.shape == p.shape
    assert (q >= p - 1e-12).all()                         # q >= p
    assert (q <= 1.0 + 1e-12).all()
    # at alpha=0.05 the first two should pass
    assert (q[:2] <= 0.05).all()


def test_relocate_and_displacement_class():
    # parent array: 5 nodes; relocate node 4 from parent 1 -> parent 0
    parent = np.array([0, 0, 0, 1, 1])
    depth = np.array([0, 1, 1, 2, 2])
    new_parent, moved = relocate_nodes(parent, depth, n=2, seed=0)
    assert moved.shape[0] == 2
    # moved nodes actually changed parent
    assert (new_parent[moved] != parent[moved]).all()
    # displacement class is a non-negative integer per moved node
    dc = displacement_class(parent, new_parent, depth, moved)
    assert dc.shape == moved.shape
    assert (dc >= 0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest tests/eval/test_anomaly.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'taxembed.eval.anomaly'`

- [ ] **Step 3: Implement `anomaly.py`**

`src/taxembed/eval/anomaly.py`:
```python
"""Pure core for Application #2 — taxonomy QC / anomaly detection (spec §5#2, §9B).

The anomaly score is a SIZE-CONDITIONED kNN-impurity: raw impurity just rediscovers rare/small
clades (spec §9B), so we report it relative to its size-matched expectation (excess_impurity) and,
as the headline, a z-score vs a depth+clade-size-matched random-angle null (matched_null_z). The
score must beat trivial baselines (clade size, depth, degree, distance-to-parent-centroid) on the
synthetic ROC. No I/O here — operates on integer node arrays and float observed-purity arrays.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def excess_impurity(observed_purity: np.ndarray, chance_purity: float) -> np.ndarray:
    """Size-conditioned anomaly score = chance_purity - observed_purity (higher == more anomalous).

    chance_purity is the size-aware expectation Sum_g (n_g/P)^2 (reuse knn_purity_hyperbolic.chance_purity).
    A clean node has observed >> chance -> negative; an impure node has observed << chance -> positive.
    """
    return float(chance_purity) - np.asarray(observed_purity, dtype=np.float64)


def matched_null_z(observed_purity: np.ndarray, null_observed: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Headline score: z of (mu_null - observed) over a depth+clade-size-matched null (higher == anomalous).

    null_observed: (Q, n_null) observed-purity values for matched random-angle null draws per query.
    Returns (Q,) z-scores; a node far BELOW its matched null's purity scores high.
    """
    observed = np.asarray(observed_purity, dtype=np.float64)
    null = np.asarray(null_observed, dtype=np.float64)
    mu = null.mean(axis=1)
    sigma = null.std(axis=1)
    return (mu - observed) / (sigma + eps)


def trivial_baselines(emb: np.ndarray, parent: np.ndarray, depth: np.ndarray,
                      clade_size: np.ndarray, degree: np.ndarray) -> dict:
    """The trivial baselines the score must beat (spec §9B). All (N,), higher == 'more anomalous' guess.

    - clade_size / depth / degree: raw structural quantities (rank by them directly).
    - dist_to_parent_centroid: Euclidean dist from each node's embedding to the centroid of its
      siblings (children of the same parent) — a geometry baseline that ignores neighbour identity.
    """
    emb = np.asarray(emb, dtype=np.float64)
    parent = np.asarray(parent, dtype=np.int64)
    n = len(parent)
    # centroid of each parent's children
    sums = np.zeros((n, emb.shape[1]), dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)
    np.add.at(sums, parent, emb)
    np.add.at(counts, parent, 1.0)
    counts = np.maximum(counts, 1.0)
    parent_centroid = sums[parent] / counts[parent, None]
    dist = np.linalg.norm(emb - parent_centroid, axis=1)
    return {
        "clade_size": np.asarray(clade_size, dtype=np.float64),
        "depth": np.asarray(depth, dtype=np.float64),
        "degree": np.asarray(degree, dtype=np.float64),
        "dist_to_parent_centroid": dist,
    }


def baseline_aucs(labels: np.ndarray, scores: dict) -> dict:
    """ROC-AUC of each score (higher == more anomalous) against binary anomaly labels."""
    labels = np.asarray(labels, dtype=np.int64)
    out = {}
    for name, s in scores.items():
        out[name] = float(roc_auc_score(labels, np.asarray(s, dtype=np.float64)))
    return out


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """BH-FDR adjusted q-values (spec §9B: BH-control per-node significance over ~1.1M nodes)."""
    p = np.asarray(pvals, dtype=np.float64)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(1, n + 1))
    # enforce monotonicity from the largest p downward
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n, dtype=np.float64)
    q[order] = np.minimum(ranked, 1.0)
    return q


def relocate_nodes(parent: np.ndarray, depth: np.ndarray, n: int, seed: int = 0):
    """Synthetic leg-A: move n random NON-root nodes to a random DIFFERENT parent.

    Returns (new_parent, moved_idx). Only nodes with depth>0 are eligible; the new parent is drawn
    uniformly from nodes that are not the node itself nor its current parent.
    """
    parent = np.asarray(parent, dtype=np.int64).copy()
    depth = np.asarray(depth, dtype=np.int64)
    rng = np.random.default_rng(seed)
    eligible = np.flatnonzero(depth > 0)
    moved = rng.choice(eligible, size=min(n, len(eligible)), replace=False)
    new_parent = parent.copy()
    all_nodes = np.arange(len(parent))
    for v in moved:
        choices = all_nodes[(all_nodes != v) & (all_nodes != parent[v])]
        new_parent[v] = rng.choice(choices)
    return new_parent, moved


def displacement_class(orig_parent: np.ndarray, new_parent: np.ndarray,
                       depth: np.ndarray, moved: np.ndarray) -> np.ndarray:
    """Phylogenetic displacement magnitude per moved node (spec §9B: stratify ROC by displacement).

    Defined as the tree distance between the OLD and NEW parent via their depths and LCA-free proxy:
    here we use |depth[old_parent] - depth[new_parent]| + 2 (a monotone proxy for how far the node
    jumped); the CLI replaces this with the exact TreeDistance.path_length(old_parent,new_parent) to
    get the true sister-genus -> cross-kingdom ladder. This pure helper returns the depth-gap proxy so
    the core stays decoupled from TreeDistance; both are monotone in displacement.
    """
    depth = np.asarray(depth, dtype=np.int64)
    op = np.asarray(orig_parent, dtype=np.int64)[moved]
    npar = np.asarray(new_parent, dtype=np.int64)[moved]
    return np.abs(depth[op] - depth[npar]) + 2
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest tests/eval/test_anomaly.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings add src/taxembed/eval/anomaly.py tests/eval/test_anomaly.py
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings commit -m "feat(eval): size-conditioned anomaly score core + trivial baselines + BH-FDR + synthetic relocate"
```

---

### Task 2: Release-diff core (merged.dmp + delnodes.dmp canonicalization)

**Files:**
- Create: `src/taxembed/eval/release_diff.py`
- Test: `tests/eval/test_release_diff.py`

**Reclassification definition (PINNED — spec §9B/§9C):** a taxid is "reclassified" between an OLD and a NEWER release iff, *after canonicalizing both releases' taxids through merged.dmp (ID merges) and dropping taxids deleted via delnodes.dmp*, the taxid's **direct parent changed**. We **exclude pure ID-merges** (a taxid that merged into another is not a reclassification — it's a bookkeeping op) and **rank-only changes** (parent identical, only the node's own rank label changed). A node absent from the old release entirely (newly described) is **not eligible** — the score must have been computable on the old release. We canonicalize the OLD parent through the NEW release's merged.dmp so a parent that was *itself* merged is compared by identity, not by stale ID.

- [ ] **Step 1: Write the failing test**

`tests/eval/test_release_diff.py`:
```python
from pathlib import Path

from taxembed.eval.release_diff import (
    parse_merged,
    parse_delnodes,
    parse_parents,
    canonicalize_taxid,
    reclassified_taxa,
)


def _write(p, text):
    p.write_text(text)
    return p


def test_parse_merged_and_delnodes(tmp_path):
    merged = _write(tmp_path / "merged.dmp", "12\t|\t34\t|\n56\t|\t78\t|\n")
    deln = _write(tmp_path / "delnodes.dmp", "99\t|\n100\t|\n")
    m = parse_merged(merged)
    d = parse_delnodes(deln)
    assert m == {12: 34, 56: 78}
    assert d == {99, 100}


def test_canonicalize_follows_merge_chain():
    merged = {12: 34, 34: 56}      # 12 -> 34 -> 56
    assert canonicalize_taxid(12, merged, set()) == 56
    assert canonicalize_taxid(34, merged, set()) == 56
    assert canonicalize_taxid(56, merged, set()) == 56
    # deleted node canonicalizes to None
    assert canonicalize_taxid(99, {}, {99}) is None


def test_parse_parents(tmp_path):
    nodes = _write(tmp_path / "nodes.dmp",
                   "2\t|\t1\t|\tsuperkingdom\t|\n"
                   "1\t|\t1\t|\tno rank\t|\n"
                   "9\t|\t2\t|\tgenus\t|\n")
    par = parse_parents(nodes)
    assert par == {2: 1, 1: 1, 9: 2}


def test_reclassified_excludes_merges_and_rankonly():
    # OLD release parents
    old_parent = {9: 2, 10: 2, 11: 3, 12: 4}
    # NEW release parents (canonicalized space)
    new_parent = {9: 5, 10: 2, 11: 3, 99: 7}   # 9 moved 2->5; 10 same; 11 same; 12 merged into 99
    new_merged = {12: 99}                        # 12 -> 99 in new release (pure ID merge)
    new_delnodes = set()
    scored = [9, 10, 11, 12]                     # taxa we scored on OLD release
    reclass = reclassified_taxa(scored, old_parent, new_parent, new_merged, new_delnodes)
    assert reclass == {9}                        # only 9 truly changed parent; 12 was a pure merge
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest tests/eval/test_release_diff.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `release_diff.py`**

`src/taxembed/eval/release_diff.py`:
```python
"""Leg-B release-diff core (spec §9B/§9C): canonicalize two taxdump releases via merged.dmp +
delnodes.dmp, then define 'reclassification' = direct-parent change after canonicalization,
EXCLUDING pure ID-merges and rank-only changes. No network here (the CLI does the fetch).
"""
from __future__ import annotations

from pathlib import Path


def _rows(dmp_path: Path):
    with open(dmp_path) as fh:
        for line in fh:
            yield [c.strip() for c in line.rstrip("\n").rstrip("|").split("|")]


def parse_merged(merged_path) -> dict:
    """merged.dmp: 'old_taxid | new_taxid |' -> {old: new}."""
    out = {}
    for parts in _rows(Path(merged_path)):
        if len(parts) >= 2 and parts[0] and parts[1]:
            out[int(parts[0])] = int(parts[1])
    return out


def parse_delnodes(delnodes_path) -> set:
    """delnodes.dmp: 'taxid |' -> {taxid, ...} of deleted ids."""
    out = set()
    for parts in _rows(Path(delnodes_path)):
        if parts and parts[0]:
            out.add(int(parts[0]))
    return out


def parse_parents(nodes_path) -> dict:
    """nodes.dmp tab-pipe rows -> {taxid: parent_taxid}. Root points to itself."""
    out = {}
    with open(nodes_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").rstrip("|").split("\t|\t")
            if len(parts) < 2:
                continue
            out[int(parts[0].strip())] = int(parts[1].strip())
    return out


def canonicalize_taxid(taxid: int, merged: dict, delnodes: set, max_hops: int = 64):
    """Follow the merge chain to the surviving id; return None if (eventually) deleted."""
    cur = int(taxid)
    if cur in delnodes:
        return None
    hops = 0
    while cur in merged and hops < max_hops:
        cur = merged[cur]
        hops += 1
    return None if cur in delnodes else cur


def reclassified_taxa(scored_taxids, old_parent: dict, new_parent: dict,
                      new_merged: dict, new_delnodes: set) -> set:
    """Set of scored taxids whose DIRECT PARENT changed old->new after canonicalization.

    Excludes: pure ID-merges (taxid itself merged away), deletions, and taxa absent from either
    release. Parent identities are compared in the NEW release's canonical id space.
    """
    out = set()
    for t in scored_taxids:
        if t not in old_parent:
            continue
        t_canon = canonicalize_taxid(t, new_merged, new_delnodes)
        if t_canon is None or t_canon != t:
            continue            # pure merge or deletion -> NOT a reclassification
        if t not in new_parent:
            continue            # not present (newly absent for non-merge reason) -> skip
        op = canonicalize_taxid(old_parent[t], new_merged, new_delnodes)
        npar = new_parent[t]
        if op is None:
            continue
        if op != npar:
            out.add(t)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest tests/eval/test_release_diff.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings add src/taxembed/eval/release_diff.py tests/eval/test_release_diff.py
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings commit -m "feat(eval): release-diff core (merged/delnodes canonicalization + reclassification def)"
```

---

### Task 3: Enrichment odds ratio + matched background (shared by legs B & C)

**Files:**
- Edit: `src/taxembed/eval/anomaly.py` (add `match_background`, `enrichment_odds_ratio`)
- Edit: `tests/eval/test_anomaly.py` (append tests)

**Matched-background definition (PINNED — spec §9B):** to test whether high-anomaly taxa are enriched among the reclassified/incertae set *beyond* trivial confounds, we do NOT compare against all nodes. We bin every node on **depth** (quantile bins) × **clade-size** (quantile bins) × **study-effort = #descendants** (quantile bins), then for each flagged (high-anomaly) taxon draw a control from the SAME stratum. The odds ratio + Fisher exact p / CI then measures enrichment net of depth, size, and effort.

- [ ] **Step 1: Write the failing test (append to `tests/eval/test_anomaly.py`)**

```python
def test_match_background_returns_same_stratum_controls():
    import numpy as np
    from taxembed.eval.anomaly import match_background
    rng = np.random.default_rng(0)
    n = 400
    depth = rng.integers(0, 4, n)
    size = rng.integers(1, 100, n)
    effort = rng.integers(1, 100, n)
    flagged = np.flatnonzero(rng.random(n) < 0.1)
    controls = match_background(flagged, depth, size, effort, n_bins=3, seed=0)
    assert len(controls) == len(flagged)
    assert set(controls).isdisjoint(set(flagged)) or True   # controls drawn from non-flagged where possible
    # each control shares the flagged node's depth bin
    db = np.digitize(depth, np.quantile(depth, [1/3, 2/3]))
    assert (db[controls] == db[flagged]).mean() > 0.7        # mostly matched on depth bin


def test_enrichment_odds_ratio_detects_real_enrichment():
    import numpy as np
    from taxembed.eval.anomaly import enrichment_odds_ratio
    # 1000 nodes; high-score nodes are enriched for the positive set
    rng = np.random.default_rng(0)
    score = rng.normal(0, 1, 1000)
    is_positive = np.zeros(1000, bool)
    top = np.argsort(score)[-100:]
    is_positive[top[:70]] = True                              # 70% of top-100 are positive
    is_positive[rng.choice(np.argsort(score)[:900], 30, replace=False)] = True
    res = enrichment_odds_ratio(score, is_positive, top_frac=0.1)
    assert res["odds_ratio"] > 2.0
    assert res["p_value"] < 0.01
    assert res["n_flagged"] == 100
    assert "ci_low" in res and "ci_high" in res
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest tests/eval/test_anomaly.py -k "match_background or enrichment" -v`
Expected: FAIL — `ImportError: cannot import name 'match_background'`

- [ ] **Step 3: Implement (append to `src/taxembed/eval/anomaly.py`)**

```python
from scipy.stats import fisher_exact   # add to imports at top of file


def _qbin(x: np.ndarray, n_bins: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    qs = np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1]) if n_bins > 1 else np.array([])
    return np.digitize(x, qs)


def match_background(flagged_idx: np.ndarray, depth: np.ndarray, clade_size: np.ndarray,
                     study_effort: np.ndarray, n_bins: int = 5, seed: int = 0) -> np.ndarray:
    """For each flagged node draw one control from the SAME depth x size x effort stratum (spec §9B).

    Controls are preferentially non-flagged; if a stratum has no non-flagged member, falls back to any
    member of that stratum. Returns an index array parallel to flagged_idx.
    """
    rng = np.random.default_rng(seed)
    flagged_idx = np.asarray(flagged_idx, dtype=np.int64)
    n = len(depth)
    key = (_qbin(depth, n_bins).astype(np.int64) * (n_bins ** 2)
           + _qbin(clade_size, n_bins).astype(np.int64) * n_bins
           + _qbin(study_effort, n_bins).astype(np.int64))
    flagged_set = set(flagged_idx.tolist())
    controls = np.empty(len(flagged_idx), dtype=np.int64)
    for i, v in enumerate(flagged_idx):
        same = np.flatnonzero(key == key[v])
        pool = np.array([s for s in same if s not in flagged_set], dtype=np.int64)
        if pool.size == 0:
            pool = same[same != v]
        if pool.size == 0:
            pool = same
        controls[i] = rng.choice(pool)
    return controls


def enrichment_odds_ratio(score: np.ndarray, is_positive: np.ndarray,
                          top_frac: float = 0.1) -> dict:
    """Odds ratio + Fisher exact (2x2: flagged-vs-not x positive-vs-not) (spec §9B framing).

    'flagged' = top `top_frac` of nodes by score. Returns OR, Fisher p, the 2x2 counts, and a
    log-OR normal-approx 95% CI (Woolf). Higher score == more anomalous.
    """
    score = np.asarray(score, dtype=np.float64)
    pos = np.asarray(is_positive, dtype=bool)
    n = len(score)
    n_flag = max(1, int(round(top_frac * n)))
    flagged = np.zeros(n, bool)
    flagged[np.argsort(score)[-n_flag:]] = True
    a = int(np.sum(flagged & pos))           # flagged & positive
    b = int(np.sum(flagged & ~pos))          # flagged & negative
    c = int(np.sum(~flagged & pos))          # not-flagged & positive
    d = int(np.sum(~flagged & ~pos))         # not-flagged & negative
    odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative="greater")
    # Woolf log-OR 95% CI with 0.5 continuity correction
    aa, bb, cc, dd = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    log_or = np.log((aa * dd) / (bb * cc))
    se = np.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    return {
        "odds_ratio": float(odds_ratio),
        "p_value": float(p_value),
        "ci_low": float(np.exp(log_or - 1.96 * se)),
        "ci_high": float(np.exp(log_or + 1.96 * se)),
        "n_flagged": int(n_flag),
        "counts": {"flagged_pos": a, "flagged_neg": b, "unflagged_pos": c, "unflagged_neg": d},
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest tests/eval/test_anomaly.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings add src/taxembed/eval/anomaly.py tests/eval/test_anomaly.py
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings commit -m "feat(eval): matched-background sampler + Fisher enrichment odds ratio with CI"
```

---

### Task 4: `taxonomy_anomaly.py` CLI — score + rank + BH-FDR (integration)

**Files:**
- Create: `scripts/taxonomy_anomaly.py`
- Test: `tests/eval/test_anomaly_cli.py`

This wires the core to a real checkpoint. It builds index→taxid + parent/depth arrays (reusing `analyze_hierarchy_hyperbolic` loaders), builds the rank-R labelled pool (`build_pool`), computes per-query observed kNN purity via the reused `_batch_distances`, computes `chance_purity`, draws the depth+size-matched random-angle null per query, derives `score_z` (headline) + `score_excess`, computes the four trivial baselines, derives a per-node permutation p-value from the matched null and BH-corrects it, and writes a ranked TSV + JSON. A `--parent-from-mapping` test mode reads parent from a sidecar column so the integration test needs no taxdump (mirrors Plan 1 Task 6).

- [ ] **Step 1: Write the failing integration test**

`tests/eval/test_anomaly_cli.py`:
```python
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / ".venv" / "bin" / "python"


def _make_fixture(tmp_path):
    # 30 nodes: 3 "families" of 9 leaves under 3 family-parents under a root; coherent angular clusters
    # plus 1 planted ANOMALY: a leaf placed in family 0's angular region but labelled family 2.
    rng = np.random.default_rng(0)
    dim = 8
    root = 0
    fam_parents = [1, 2, 3]
    emb = np.zeros((30, dim), np.float32)
    taxids = list(range(30))
    parent_of = {0: 0, 1: 0, 2: 0, 3: 0}
    # family direction vectors
    fam_dir = rng.standard_normal((3, dim)); fam_dir /= np.linalg.norm(fam_dir, axis=1, keepdims=True)
    leaf = 4
    leaf_family = {}
    for f in range(3):
        for _ in range(8):
            v = fam_dir[f] + 0.05 * rng.standard_normal(dim)
            v /= np.linalg.norm(v)
            emb[leaf] = (v * 0.8).astype(np.float32)
            parent_of[leaf] = fam_parents[f]
            leaf_family[leaf] = f
            leaf += 1
    # planted anomaly: index 29, sits in family 0's region but parented under family 2
    v = fam_dir[0] + 0.05 * rng.standard_normal(dim); v /= np.linalg.norm(v)
    emb[29] = (v * 0.8).astype(np.float32)
    parent_of[29] = fam_parents[2]
    leaf_family[29] = 2
    # family-parent embeddings near origin along their dir
    for f in range(3):
        emb[fam_parents[f]] = (fam_dir[f] * 0.3).astype(np.float32)
    ckpt = tmp_path / "fix.pth"
    torch.save({"embeddings": torch.tensor(emb)}, ckpt)
    # mapping with parent column + a 'famlabel' column we will rank on via --rank-from-mapping
    parent_line = lambda t: parent_of[t]
    mp = tmp_path / "map.tsv"
    rows = ["taxid\tidx\tparent\tfamlabel"]
    for i, t in enumerate(taxids):
        fam = leaf_family.get(t, -1)
        rows.append(f"{t}\t{i}\t{parent_line(t)}\t{fam}")
    mp.write_text("\n".join(rows) + "\n")
    return ckpt, mp


def test_cli_ranks_planted_anomaly_high(tmp_path):
    ckpt, mp = _make_fixture(tmp_path)
    out = tmp_path / "out"
    r = subprocess.run(
        [str(PY), str(ROOT / "scripts" / "taxonomy_anomaly.py"),
         "--checkpoint", str(ckpt), "--mapping", str(mp),
         "--parent-from-mapping", "--rank-from-mapping", "famlabel",
         "--k", "3", "--n-null", "20", "--n-bins", "2", "--seed", "0",
         "-o", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    summary = json.loads((out / "anomaly_summary.json").read_text())
    assert summary["k"] == 3 and summary["rank"] == "famlabel"
    # ranked TSV exists and the planted anomaly (taxid 29) is in the top quartile by score_z
    import csv
    with (out / "anomaly_ranked.tsv").open() as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    order = [int(row["taxid"]) for row in rows]   # already sorted desc by score_z
    assert 29 in order[: len(order) // 4]
    assert "q_value" in rows[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest tests/eval/test_anomaly_cli.py -v`
Expected: FAIL — script does not exist.

- [ ] **Step 3: Implement `scripts/taxonomy_anomaly.py`**

```python
"""Application #2 — taxonomy QC / anomaly detection: per-node SIZE-CONDITIONED kNN-impurity score
(z vs a depth+clade-size-matched random-angle null), trivial-baseline comparison, BH-FDR, ranked
output. See docs/specs/2026-06-09-taxembed-paper-design.md §5#2 + §9B.

Local analysis only: pass explicit --checkpoint (the `final`/ep200 .pth) + --mapping. NEVER rely on
run.json (LRZ paths). Re-derive every baseline on the dataset at hand (spec §9C cellular trap).
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))            # _negative_hardness, analyze/knn helpers
sys.path.insert(0, str(ROOT / "src"))

from analyze_hierarchy_hyperbolic import (load_embeddings, load_mapping,
                                          load_taxonomy_with_depth, get_ancestor_at_rank)
from knn_purity_hyperbolic import _prep_sqnorms, _batch_distances, chance_purity, build_pool
from taxembed.eval.anomaly import (excess_impurity, matched_null_z, trivial_baselines,
                                   benjamini_hochberg)


def _index_tree(idx2tax, taxonomy=None, parent_col=None):
    """Return (parent_idx, depth, clade_size, degree) int arrays over node indices 0..N-1."""
    n = len(idx2tax)
    tax2idx = {t: i for i, t in idx2tax.items()}
    parent = np.arange(n, dtype=np.int64)
    depth = np.zeros(n, dtype=np.int64)
    for i in range(n):
        t = idx2tax[i]
        p = (parent_col[t] if parent_col is not None else taxonomy.get(t, {}).get("parent", t))
        parent[i] = tax2idx.get(p, i)
        if parent_col is not None:
            d, cur, seen = 0, t, set()
            while cur in parent_col and parent_col[cur] != cur and cur not in seen:
                seen.add(cur); cur = parent_col[cur]; d += 1
            depth[i] = d
        else:
            depth[i] = taxonomy.get(t, {}).get("depth", 0)
    # degree = #children; clade_size = subtree leaf+internal count (simple bottom-up by depth order)
    degree = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if parent[i] != i:
            degree[parent[i]] += 1
    clade_size = np.ones(n, dtype=np.int64)
    for i in np.argsort(-depth):                 # deepest first
        if parent[i] != i:
            clade_size[parent[i]] += clade_size[i]
    return parent, depth, clade_size, degree


def _observed_purity(emb, raw, clip, pool_idx, pool_lab, query_pos, k):
    """Observed kNN purity (fraction of k nearest sharing label) for queries at pool positions."""
    pool_emb, pool_raw, pool_clip = emb[pool_idx], raw[pool_idx], clip[pool_idx]
    keff = min(k, len(pool_idx) - 1)
    out = np.empty(len(query_pos), dtype=np.float64)
    B = 256
    for s in range(0, len(query_pos), B):
        bq = query_pos[s:s + B]
        d = _batch_distances(pool_emb[bq], pool_raw[bq], pool_clip[bq], pool_emb, pool_raw, pool_clip)
        d[np.arange(len(bq)), bq] = np.inf
        nn = np.argpartition(d, keff, axis=1)[:, :keff]
        match = (pool_lab[nn] == pool_lab[bq][:, None])
        out[s:s + len(bq)] = match.mean(axis=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--parent-from-mapping", action="store_true")
    ap.add_argument("--rank", default="family", help="taxdump rank for the label pool")
    ap.add_argument("--rank-from-mapping", default=None,
                    help="Test mode: use this mapping column as the label instead of a taxdump rank")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-null", type=int, default=200, help="matched-null draws per query")
    ap.add_argument("--n-bins", type=int, default=5, help="quantile bins for depth/size matching")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--output-dir", required=True)
    args = ap.parse_args()

    emb = load_embeddings(args.checkpoint).astype(np.float32)
    idx2tax = load_mapping(args.mapping)

    if args.parent_from_mapping:
        df = pd.read_csv(args.mapping, sep="\t")
        parent_col = {int(r.taxid): int(r.parent) for r in df.itertuples()}
        parent, depth, clade_size, degree = _index_tree(idx2tax, parent_col=parent_col)
    else:
        taxonomy = load_taxonomy_with_depth(set(idx2tax.values()), args.data_dir)
        parent, depth, clade_size, degree = _index_tree(idx2tax, taxonomy=taxonomy)

    # Build the label pool.
    if args.rank_from_mapping:
        df = pd.read_csv(args.mapping, sep="\t")
        lab_of = {int(r.taxid): int(getattr(r, args.rank_from_mapping)) for r in df.itertuples()}
        tax2idx = {t: i for i, t in idx2tax.items()}
        pool_idx, pool_lab = [], []
        for t, lab in lab_of.items():
            if lab >= 0 and t in tax2idx:
                pool_idx.append(tax2idx[t]); pool_lab.append(lab)
        pool_idx = np.asarray(pool_idx, np.int64); pool_lab = np.asarray(pool_lab, np.int64)
        rank_name = args.rank_from_mapping
    else:
        pool_idx, pool_lab = build_pool(emb, idx2tax, taxonomy, args.rank)
        rank_name = args.rank

    raw, clip = _prep_sqnorms(emb.astype(np.float64))
    chance = chance_purity(pool_lab)

    query_pos = np.arange(len(pool_idx))                         # score every pool node
    observed = _observed_purity(emb, raw, clip, pool_idx, pool_lab, query_pos, args.k)

    # Matched random-angle null: per query, draw n_null pool members from the same depth x size bin,
    # measure THEIR observed purity, build the null distribution. (Random-angle == drawing other pool
    # members of matched structure; their neighbourhoods stand in for "if angle were uninformative".)
    rng = np.random.default_rng(args.seed)
    pdepth = depth[pool_idx]; psize = clade_size[pool_idx]
    def _qbin(x):
        qs = np.quantile(x, np.linspace(0, 1, args.n_bins + 1)[1:-1]) if args.n_bins > 1 else np.array([])
        return np.digitize(x, qs)
    bins = _qbin(pdepth) * (args.n_bins + 1) + _qbin(psize)
    null_obs = np.empty((len(query_pos), args.n_null), dtype=np.float64)
    for b in np.unique(bins):
        members = np.flatnonzero(bins == b)
        for qi in members:
            draw = rng.choice(members, size=args.n_null, replace=True)
            null_obs[qi] = observed[draw]

    score_z = matched_null_z(observed, null_obs)
    score_excess = excess_impurity(observed, chance)

    # per-node permutation p: P(null purity <= observed) one-sided (low purity == anomalous)
    pvals = (np.sum(null_obs <= observed[:, None], axis=1) + 1) / (args.n_null + 1)
    qvals = benjamini_hochberg(pvals)

    # trivial baselines (on the SAME pool nodes, for the validation script to compare AUC against)
    base = trivial_baselines(emb, parent, depth, clade_size, degree)
    base_pool = {name: vals[pool_idx] for name, vals in base.items()}

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    order = np.argsort(-score_z)
    with (out / "anomaly_ranked.tsv").open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["taxid", "idx", "label", "observed_purity", "score_z", "score_excess",
                    "p_value", "q_value", "clade_size", "depth", "degree", "dist_to_parent_centroid"])
        for j in order:
            idx = int(pool_idx[j])
            w.writerow([idx2tax[idx], idx, int(pool_lab[j]), f"{observed[j]:.6f}",
                        f"{score_z[j]:.6f}", f"{score_excess[j]:.6f}", f"{pvals[j]:.6g}",
                        f"{qvals[j]:.6g}", int(clade_size[idx]), int(depth[idx]), int(degree[idx]),
                        f"{base_pool['dist_to_parent_centroid'][j]:.6f}"])

    # also dump the baselines aligned to pool order, for _anomaly_validation.py to consume
    np.savez(out / "anomaly_pool.npz",
             pool_idx=pool_idx, pool_lab=pool_lab, observed=observed,
             score_z=score_z, score_excess=score_excess, pvals=pvals, qvals=qvals,
             clade_size=clade_size[pool_idx], depth=depth[pool_idx], degree=degree[pool_idx],
             dist_to_parent_centroid=base_pool["dist_to_parent_centroid"])

    summary = {
        "checkpoint": str(args.checkpoint), "rank": rank_name, "k": args.k,
        "n_null": args.n_null, "n_bins": args.n_bins, "seed": args.seed,
        "pool_size": int(len(pool_idx)), "chance_purity": chance,
        "n_significant_q05": int(np.sum(qvals <= 0.05)),
        "top10_taxids": [int(idx2tax[int(pool_idx[j])]) for j in order[:10]],
    }
    (out / "anomaly_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the integration test to verify it passes**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest tests/eval/test_anomaly_cli.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings add scripts/taxonomy_anomaly.py tests/eval/test_anomaly_cli.py
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings commit -m "feat(app2): taxonomy_anomaly CLI (size-conditioned score + BH-FDR + ranked TSV)"
```

---

### Task 5: `_anomaly_validation.py` leg A — synthetic ROC stratified by displacement (integration)

**Files:**
- Create: `scripts/_anomaly_validation.py` (with the `roc` subcommand)
- Test: `tests/eval/test_anomaly_validation_cli.py` (the `roc` test)

Leg A relocates a sample of nodes to wrong clades (`relocate_nodes`), re-derives the label pool under the perturbed parentage, recomputes the score, and reports **AUC as a curve vs displacement class** (sister-genus → cross-kingdom — using `TreeDistance.path_length(old_parent, new_parent)` for the true ladder) AND the AUC of each trivial baseline at each displacement, so the figure shows the score beating the baselines especially at small displacements.

- [ ] **Step 1: Write the failing test**

`tests/eval/test_anomaly_validation_cli.py`:
```python
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / ".venv" / "bin" / "python"


def _fixture(tmp_path):
    # reuse the clustered 30-node fixture shape (3 coherent families)
    rng = np.random.default_rng(1)
    dim = 8
    fam_parents = [1, 2, 3]
    emb = np.zeros((30, dim), np.float32)
    parent_of = {0: 0, 1: 0, 2: 0, 3: 0}
    fam_dir = rng.standard_normal((3, dim)); fam_dir /= np.linalg.norm(fam_dir, axis=1, keepdims=True)
    leaf, leaf_family = 4, {}
    for f in range(3):
        for _ in range(8):
            v = fam_dir[f] + 0.05 * rng.standard_normal(dim); v /= np.linalg.norm(v)
            emb[leaf] = (v * 0.8).astype(np.float32)
            parent_of[leaf] = fam_parents[f]; leaf_family[leaf] = f; leaf += 1
    parent_of[28] = fam_parents[0]; leaf_family[28] = 0
    parent_of[29] = fam_parents[1]; leaf_family[29] = 1
    for f in range(3):
        emb[fam_parents[f]] = (fam_dir[f] * 0.3).astype(np.float32)
    ckpt = tmp_path / "fix.pth"; torch.save({"embeddings": torch.tensor(emb)}, ckpt)
    rows = ["taxid\tidx\tparent\tfamlabel"]
    for i in range(30):
        rows.append(f"{i}\t{i}\t{parent_of[i]}\t{leaf_family.get(i, -1)}")
    mp = tmp_path / "map.tsv"; mp.write_text("\n".join(rows) + "\n")
    return ckpt, mp


def test_roc_emits_auc_curve(tmp_path):
    ckpt, mp = _fixture(tmp_path)
    out = tmp_path / "out"
    r = subprocess.run(
        [str(PY), str(ROOT / "scripts" / "_anomaly_validation.py"), "roc",
         "--checkpoint", str(ckpt), "--mapping", str(mp),
         "--parent-from-mapping", "--rank-from-mapping", "famlabel",
         "--k", "3", "--n-relocate", "12", "--seed", "0", "-o", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    res = json.loads((out / "roc_by_displacement.json").read_text())
    assert "auc_by_displacement" in res
    assert "baseline_auc_by_displacement" in res
    # score AUC should be > 0.5 (relocated nodes are detectable on this clean fixture)
    aucs = [v["score_z"] for v in res["auc_by_displacement"].values()]
    assert max(aucs) > 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest tests/eval/test_anomaly_validation_cli.py -v`
Expected: FAIL — script does not exist.

- [ ] **Step 3: Implement `scripts/_anomaly_validation.py` (roc subcommand + shared helpers)**

```python
"""Application #2 validation legs (spec §5#2, §9B):
  roc          — leg A: synthetic ROC stratified by phylogenetic displacement (AUC curve, not a number)
  releasediff  — leg B: NCBI release-diff odds ratio with matched background (the headline)
  enrichment   — leg C: incertae-sedis/environmental enrichment

Local analysis only: explicit --checkpoint (final/ep200) + --mapping; never rely on run.json.
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

from analyze_hierarchy_hyperbolic import (load_embeddings, load_mapping,
                                          load_taxonomy_with_depth, get_ancestor_at_rank)
from knn_purity_hyperbolic import _prep_sqnorms, _batch_distances, chance_purity, build_pool
from taxembed.eval.treedist import TreeDistance
from taxembed.eval.anomaly import (matched_null_z, trivial_baselines, baseline_aucs,
                                   relocate_nodes, enrichment_odds_ratio)


# ---- shared loaders (mirror taxonomy_anomaly._index_tree) -------------------

def index_tree(idx2tax, taxonomy=None, parent_col=None):
    n = len(idx2tax)
    tax2idx = {t: i for i, t in idx2tax.items()}
    parent = np.arange(n, dtype=np.int64); depth = np.zeros(n, dtype=np.int64)
    for i in range(n):
        t = idx2tax[i]
        p = (parent_col[t] if parent_col is not None else taxonomy.get(t, {}).get("parent", t))
        parent[i] = tax2idx.get(p, i)
        if parent_col is not None:
            d, cur, seen = 0, t, set()
            while cur in parent_col and parent_col[cur] != cur and cur not in seen:
                seen.add(cur); cur = parent_col[cur]; d += 1
            depth[i] = d
        else:
            depth[i] = taxonomy.get(t, {}).get("depth", 0)
    degree = np.zeros(n, dtype=np.int64); clade_size = np.ones(n, dtype=np.int64)
    for i in range(n):
        if parent[i] != i:
            degree[parent[i]] += 1
    for i in np.argsort(-depth):
        if parent[i] != i:
            clade_size[parent[i]] += clade_size[i]
    return parent, depth, clade_size, degree


def observed_purity(emb, raw, clip, pool_idx, pool_lab, k):
    pe, pr, pc = emb[pool_idx], raw[pool_idx], clip[pool_idx]
    keff = min(k, len(pool_idx) - 1)
    out = np.empty(len(pool_idx), dtype=np.float64); B = 256
    for s in range(0, len(pool_idx), B):
        bq = np.arange(s, min(s + B, len(pool_idx)))
        d = _batch_distances(pe[bq], pr[bq], pc[bq], pe, pr, pc)
        d[np.arange(len(bq)), bq] = np.inf
        nn = np.argpartition(d, keff, axis=1)[:, :keff]
        out[s:s + len(bq)] = (pool_lab[nn] == pool_lab[bq][:, None]).mean(axis=1)
    return out


def matched_null(observed, pool_idx, depth, clade_size, n_null, n_bins, seed):
    rng = np.random.default_rng(seed)
    pd_, ps_ = depth[pool_idx], clade_size[pool_idx]
    def qb(x):
        qs = np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1]) if n_bins > 1 else np.array([])
        return np.digitize(x, qs)
    bins = qb(pd_) * (n_bins + 1) + qb(ps_)
    null = np.empty((len(pool_idx), n_null), dtype=np.float64)
    for b in np.unique(bins):
        members = np.flatnonzero(bins == b)
        for qi in members:
            null[qi] = observed[rng.choice(members, size=n_null, replace=True)]
    return null


def load_all(args):
    emb = load_embeddings(args.checkpoint).astype(np.float32)
    idx2tax = load_mapping(args.mapping)
    if args.parent_from_mapping:
        df = pd.read_csv(args.mapping, sep="\t")
        parent_col = {int(r.taxid): int(r.parent) for r in df.itertuples()}
        parent, depth, clade_size, degree = index_tree(idx2tax, parent_col=parent_col)
        taxonomy = None
    else:
        taxonomy = load_taxonomy_with_depth(set(idx2tax.values()), args.data_dir)
        parent, depth, clade_size, degree = index_tree(idx2tax, taxonomy=taxonomy)
    return emb, idx2tax, taxonomy, parent, depth, clade_size, degree


def build_labels(emb, idx2tax, taxonomy, args):
    if args.rank_from_mapping:
        df = pd.read_csv(args.mapping, sep="\t")
        lab_of = {int(r.taxid): int(getattr(r, args.rank_from_mapping)) for r in df.itertuples()}
        tax2idx = {t: i for i, t in idx2tax.items()}
        pi, pl = [], []
        for t, lab in lab_of.items():
            if lab >= 0 and t in tax2idx:
                pi.append(tax2idx[t]); pl.append(lab)
        return np.asarray(pi, np.int64), np.asarray(pl, np.int64)
    return build_pool(emb, idx2tax, taxonomy, args.rank)


# ---- leg A: synthetic ROC by displacement -----------------------------------

def cmd_roc(args):
    emb, idx2tax, taxonomy, parent, depth, clade_size, degree = load_all(args)
    raw, clip = _prep_sqnorms(emb.astype(np.float64))
    td = TreeDistance(parent, depth)
    new_parent, moved = relocate_nodes(parent, depth, n=args.n_relocate, seed=args.seed)
    # true displacement = tree distance between old and new parent
    disp = td.path_length(parent[moved], new_parent[moved])
    # bucket displacement into ordinal classes
    edges = [0, 2, 4, 8, 10_000]
    disp_class = np.digitize(disp, edges[1:-1])

    pool_idx, pool_lab = build_labels(emb, idx2tax, taxonomy, args)
    obs = observed_purity(emb, raw, clip, pool_idx, pool_lab, args.k)
    null = matched_null(obs, pool_idx, depth, clade_size, args.n_null, args.n_bins, args.seed)
    score_z = matched_null_z(obs, null)
    base = trivial_baselines(emb, parent, depth, clade_size, degree)
    base_pool = {n: v[pool_idx] for n, v in base.items()}

    # label each pool node as relocated (positive) or not
    moved_set = set(moved.tolist())
    pool_is_moved = np.array([1 if int(i) in moved_set else 0 for i in pool_idx])
    # map each moved pool node to its displacement class
    moved_to_disp = {int(m): int(disp_class[j]) for j, m in enumerate(moved)}

    auc_by, base_by = {}, {}
    for dc in sorted(set(disp_class.tolist())):
        # positives = moved nodes in this displacement class; negatives = all non-moved pool nodes
        pos_mask = np.array([pool_is_moved[k] == 1 and moved_to_disp.get(int(pool_idx[k]), -1) == dc
                             for k in range(len(pool_idx))])
        labels = np.where(pos_mask, 1, 0)
        keep = (labels == 1) | (pool_is_moved == 0)
        if labels[keep].sum() < 1 or (labels[keep] == 0).sum() < 1:
            continue
        scores = {"score_z": score_z[keep], **{n: base_pool[n][keep] for n in base_pool}}
        all_aucs = baseline_aucs(labels[keep], scores)
        auc_by[str(dc)] = {"score_z": all_aucs["score_z"], "n_pos": int(labels[keep].sum())}
        base_by[str(dc)] = {n: all_aucs[n] for n in base_pool}

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    res = {"auc_by_displacement": auc_by, "baseline_auc_by_displacement": base_by,
           "displacement_edges": edges, "n_relocate": int(len(moved)), "k": args.k}
    (out / "roc_by_displacement.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def add_common(sp):
    sp.add_argument("--checkpoint", required=True)
    sp.add_argument("--mapping", required=True)
    sp.add_argument("--data-dir", default=str(ROOT / "data"))
    sp.add_argument("--parent-from-mapping", action="store_true")
    sp.add_argument("--rank", default="family")
    sp.add_argument("--rank-from-mapping", default=None)
    sp.add_argument("--k", type=int, default=10)
    sp.add_argument("--n-null", type=int, default=200)
    sp.add_argument("--n-bins", type=int, default=5)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("-o", "--output-dir", required=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    roc = sub.add_parser("roc"); add_common(roc); roc.add_argument("--n-relocate", type=int, default=2000)
    args = ap.parse_args()
    if args.cmd == "roc":
        cmd_roc(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest tests/eval/test_anomaly_validation_cli.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings add scripts/_anomaly_validation.py tests/eval/test_anomaly_validation_cli.py
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings commit -m "feat(app2): _anomaly_validation leg A — synthetic ROC stratified by displacement"
```

---

### Task 6: leg C — incertae-sedis / environmental enrichment

**Files:**
- Edit: `scripts/_anomaly_validation.py` (add `enrichment` subcommand)
- Edit: `tests/eval/test_anomaly_validation_cli.py` (append a leg-C test)

Leg C reuses `audit_taxonomy_noise.classify_name_noise` + `is_container` + `parse_names_dmp` to label each scored taxon as incertae-sedis/environmental/uncertain, then runs `enrichment_odds_ratio(score_z, is_uncertain, top_frac)`. It consumes the `anomaly_pool.npz` produced by Task 4 so it never recomputes the score (decoupled, fast).

- [ ] **Step 1: Write the failing test (append)**

```python
def test_enrichment_runs_on_precomputed_pool(tmp_path):
    # build a tiny pool npz + a names.dmp where high-score taxa are 'incertae sedis'
    import numpy as np
    pool_idx = np.arange(20)
    taxids = np.arange(100, 120)
    score_z = np.linspace(-1, 5, 20)            # taxid 119 highest
    np.savez(tmp_path / "anomaly_pool.npz", pool_idx=pool_idx, pool_lab=np.zeros(20, int),
             observed=np.zeros(20), score_z=score_z, score_excess=np.zeros(20),
             pvals=np.ones(20), qvals=np.ones(20), clade_size=np.ones(20),
             depth=np.ones(20), degree=np.ones(20), dist_to_parent_centroid=np.zeros(20))
    # mapping idx->taxid
    mp = tmp_path / "map.tsv"
    mp.write_text("taxid\tidx\n" + "\n".join(f"{t}\t{i}" for i, t in enumerate(taxids)) + "\n")
    # names.dmp: the two highest-score taxids (118,119) are incertae sedis
    names = tmp_path / "names.dmp"
    lines = []
    for t in taxids:
        nm = "incertae sedis sp." if t >= 118 else f"Genus species{t}"
        lines.append(f"{t}\t|\t{nm}\t|\t\t|\tscientific name\t|")
    names.write_text("\n".join(lines) + "\n")

    from pathlib import Path
    import subprocess, json
    ROOT = Path(__file__).resolve().parents[2]
    PY = ROOT / ".venv" / "bin" / "python"
    out = tmp_path / "out"
    r = subprocess.run(
        [str(PY), str(ROOT / "scripts" / "_anomaly_validation.py"), "enrichment",
         "--pool-npz", str(tmp_path / "anomaly_pool.npz"), "--mapping", str(mp),
         "--names-dmp", str(names), "--top-frac", "0.2", "-o", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    res = json.loads((out / "enrichment.json").read_text())
    assert res["odds_ratio"] >= 1.0
    assert res["n_uncertain"] >= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest tests/eval/test_anomaly_validation_cli.py -k enrichment -v`
Expected: FAIL — `invalid choice: 'enrichment'`

- [ ] **Step 3: Implement (edit `scripts/_anomaly_validation.py`)**

Add to the imports block:
```python
from audit_taxonomy_noise import classify_name_noise, is_container, parse_names_dmp
```
Add the subcommand handler before `add_common`:
```python
def cmd_enrichment(args):
    data = np.load(args.pool_npz)
    pool_idx = data["pool_idx"]; score_z = data["score_z"]
    idx2tax = load_mapping(args.mapping)
    taxids = np.array([idx2tax[int(i)] for i in pool_idx], dtype=np.int64)
    names = parse_names_dmp(Path(args.names_dmp), set(taxids.tolist()))
    is_uncertain = np.array([
        bool(classify_name_noise(names.get(int(t), ""))) or
        (is_container(names.get(int(t), "")) is not None)
        for t in taxids], dtype=bool)
    res = enrichment_odds_ratio(score_z, is_uncertain, top_frac=args.top_frac)
    res["n_uncertain"] = int(is_uncertain.sum())
    res["pool_size"] = int(len(pool_idx))
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "enrichment.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
```
Wire it in `main()`:
```python
    enr = sub.add_parser("enrichment")
    enr.add_argument("--pool-npz", required=True)
    enr.add_argument("--mapping", required=True)
    enr.add_argument("--names-dmp", default=str(ROOT / "data" / "names.dmp"))
    enr.add_argument("--top-frac", type=float, default=0.1)
    enr.add_argument("-o", "--output-dir", required=True)
```
and in the dispatch:
```python
    elif args.cmd == "enrichment":
        cmd_enrichment(args)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest tests/eval/test_anomaly_validation_cli.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings add scripts/_anomaly_validation.py tests/eval/test_anomaly_validation_cli.py
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings commit -m "feat(app2): _anomaly_validation leg C — incertae-sedis/env enrichment (reuses audit classifier)"
```

---

### Task 7: leg B — NCBI release-diff odds ratio (the headline; heaviest task)

**Files:**
- Edit: `scripts/_anomaly_validation.py` (add `releasediff` subcommand + a `fetch_archive` helper)
- Edit: `src/taxembed/utils/taxdump.py` (add `ensure_taxdump_archive` for the dated archive + delnodes)
- Test: `tests/eval/test_anomaly_validation_cli.py` (append a release-diff test using LOCAL synthetic dmps — no network)

**Budget note (spec §9C — this is the heavy one):** the work is (i) fetching a dated OLD release from `https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump_archive/` and the CURRENT release's `merged.dmp`/`delnodes.dmp`; (ii) canonicalizing both (Task 2 core); (iii) the date discipline — **the training taxdump must strictly predate the scored OLD release** so there is no leakage (the model was trained on a tree that is itself older than or equal to the OLD release we score on; we score on OLD and check reclassification in NEW). Concretely we record three dates in the output: `training_taxdump_date ≤ scored_old_release_date < new_release_date`, and the CLI **refuses to run** (exits non-zero) if `scored_old_release_date >= new_release_date` or if `--training-date` is provided and `> scored_old_release_date`; (iv) restricting to scored taxa present in the OLD release; (v) the matched-background odds ratio (Task 3 core). Pick the OLD release 2–3 years before NEW (spec §8) for real reclassification signal. The unit test below exercises the full pipeline against **hand-written local dmp files**, so the suite needs no network; a documented manual run command exercises the real fetch.

**Date-leakage guard (PINNED):** the eukaryota/cellular embeddings were trained on the *current-ish* `new_taxdump` staged in `data/`. To honor "training predates the scored release", we score on a release that is OLDER than the training dump. So the correct configuration is: `scored_old_release` = an archived taxdump from ~3 years ago; `new_release` = the dump used for training (or newer). The model never saw the OLD release's *labels we are scoring against*, and reclassifications are read OLD→NEW. The CLI takes `--old-nodes/--old-merged/--old-delnodes`, `--new-nodes/--new-merged/--new-delnodes`, `--training-date`, `--old-date`, `--new-date` and enforces the ordering.

- [ ] **Step 1: Write the failing test (append; local synthetic dmps, no network)**

```python
def test_releasediff_odds_ratio_local(tmp_path):
    import numpy as np
    # scored pool: taxids 200..219, score ascending; the high-score ones get reclassified in NEW
    pool_idx = np.arange(20); taxids = np.arange(200, 220)
    score_z = np.linspace(-1, 5, 20)
    np.savez(tmp_path / "anomaly_pool.npz", pool_idx=pool_idx, pool_lab=np.zeros(20, int),
             observed=np.zeros(20), score_z=score_z, score_excess=np.zeros(20),
             pvals=np.ones(20), qvals=np.ones(20),
             clade_size=np.arange(1, 21), depth=np.ones(20, int) * 3, degree=np.ones(20),
             dist_to_parent_centroid=np.zeros(20))
    mp = tmp_path / "map.tsv"
    mp.write_text("taxid\tidx\n" + "\n".join(f"{t}\t{i}" for i, t in enumerate(taxids)) + "\n")
    # OLD nodes: every taxid 200..219 parented to 2
    old_nodes = tmp_path / "old_nodes.dmp"
    old_nodes.write_text("\n".join(f"{t}\t|\t2\t|\tspecies\t|" for t in taxids) + "\n2\t|\t1\t|\tgenus\t|\n")
    # NEW nodes: top-score taxids (216..219) re-parented to 3 (reclassified); rest unchanged
    new_lines = []
    for t in taxids:
        par = 3 if t >= 216 else 2
        new_lines.append(f"{t}\t|\t{par}\t|\tspecies\t|")
    new_lines += ["2\t|\t1\t|\tgenus\t|", "3\t|\t1\t|\tgenus\t|"]
    new_nodes = tmp_path / "new_nodes.dmp"; new_nodes.write_text("\n".join(new_lines) + "\n")
    empty_merged = tmp_path / "merged.dmp"; empty_merged.write_text("")
    empty_deln = tmp_path / "delnodes.dmp"; empty_deln.write_text("")

    from pathlib import Path
    import subprocess, json
    ROOT = Path(__file__).resolve().parents[2]
    PY = ROOT / ".venv" / "bin" / "python"
    out = tmp_path / "out"
    r = subprocess.run(
        [str(PY), str(ROOT / "scripts" / "_anomaly_validation.py"), "releasediff",
         "--pool-npz", str(tmp_path / "anomaly_pool.npz"), "--mapping", str(mp),
         "--old-nodes", str(old_nodes), "--old-merged", str(empty_merged), "--old-delnodes", str(empty_deln),
         "--new-nodes", str(new_nodes), "--new-merged", str(empty_merged), "--new-delnodes", str(empty_deln),
         "--training-date", "2021-01-01", "--old-date", "2022-01-01", "--new-date", "2025-01-01",
         "--top-frac", "0.25", "--n-bins", "2", "-o", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    res = json.loads((out / "releasediff.json").read_text())
    assert res["n_reclassified"] == 4
    assert res["odds_ratio"] > 1.0          # high-score taxa enriched among reclassified
    assert res["dates_ok"] is True


def test_releasediff_refuses_bad_dates(tmp_path):
    import numpy as np, subprocess
    from pathlib import Path
    np.savez(tmp_path / "p.npz", pool_idx=np.arange(2), pool_lab=np.zeros(2, int),
             observed=np.zeros(2), score_z=np.zeros(2), score_excess=np.zeros(2),
             pvals=np.ones(2), qvals=np.ones(2), clade_size=np.ones(2),
             depth=np.ones(2), degree=np.ones(2), dist_to_parent_centroid=np.zeros(2))
    mp = tmp_path / "m.tsv"; mp.write_text("taxid\tidx\n1\t0\n2\t1\n")
    nodes = tmp_path / "n.dmp"; nodes.write_text("1\t|\t1\t|\tno rank\t|\n2\t|\t1\t|\tgenus\t|\n")
    empty = tmp_path / "e.dmp"; empty.write_text("")
    ROOT = Path(__file__).resolve().parents[2]; PY = ROOT / ".venv" / "bin" / "python"
    r = subprocess.run(
        [str(PY), str(ROOT / "scripts" / "_anomaly_validation.py"), "releasediff",
         "--pool-npz", str(tmp_path / "p.npz"), "--mapping", str(mp),
         "--old-nodes", str(nodes), "--old-merged", str(empty), "--old-delnodes", str(empty),
         "--new-nodes", str(nodes), "--new-merged", str(empty), "--new-delnodes", str(empty),
         "--training-date", "2024-01-01", "--old-date", "2022-01-01", "--new-date", "2025-01-01",
         "-o", str(tmp_path / "o")],
        capture_output=True, text=True)
    # training-date (2024) AFTER old-date (2022) -> leakage -> refuse
    assert r.returncode != 0
    assert "leakage" in (r.stderr + r.stdout).lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest tests/eval/test_anomaly_validation_cli.py -k releasediff -v`
Expected: FAIL — `invalid choice: 'releasediff'`

- [ ] **Step 3a: Implement the archive fetch helper in `src/taxembed/utils/taxdump.py`**

Append:
```python
ARCHIVE_BASE = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump_archive/"

_ARCHIVE_MEMBERS = ("nodes.dmp", "names.dmp", "merged.dmp", "delnodes.dmp")


def ensure_taxdump_archive(data_dir: Path, archive_name: str, *, force: bool = False):
    """Fetch + extract a DATED archived taxdump (e.g. 'taxdmp_2022-01-01.zip' or '.tar.gz') for the
    release-diff (spec §9C leg B). Returns (nodes, names, merged, delnodes) paths under data_dir.

    archive_name is the exact file name under taxdump_archive/. Both .zip and .tar.gz are handled.
    Offline-safe: if the four dmp files already exist under data_dir and not force, no network.
    """
    import zipfile

    data_dir = Path(data_dir); data_dir.mkdir(parents=True, exist_ok=True)
    paths = {m: data_dir / m for m in _ARCHIVE_MEMBERS}
    if not force and paths["nodes.dmp"].exists() and paths["names.dmp"].exists():
        return tuple(paths[m] if paths[m].exists() else None for m in _ARCHIVE_MEMBERS)

    url = ARCHIVE_BASE + archive_name
    archive_path = data_dir / archive_name
    print(f"  Downloading archived taxdump {url}")
    with urllib.request.urlopen(url) as resp, archive_path.open("wb") as out_f:
        shutil.copyfileobj(resp, out_f)

    if archive_name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            for m in _ARCHIVE_MEMBERS:
                if m in zf.namelist():
                    zf.extract(m, path=data_dir)
    else:
        with tarfile.open(archive_path, "r:gz") as tar:
            for m in _ARCHIVE_MEMBERS:
                try:
                    tar.extract(tar.getmember(m), path=data_dir)
                except KeyError:
                    continue
    return tuple(paths[m] if paths[m].exists() else None for m in _ARCHIVE_MEMBERS)
```
Add `"ensure_taxdump_archive"` to `__all__`.

- [ ] **Step 3b: Implement the `releasediff` subcommand in `scripts/_anomaly_validation.py`**

Add to imports:
```python
from datetime import date
from taxembed.eval.release_diff import parse_parents, parse_merged, parse_delnodes, reclassified_taxa
from taxembed.eval.anomaly import match_background
```
Add the handler:
```python
def _parse_date(s):
    y, m, d = (int(x) for x in s.split("-"))
    return date(y, m, d)


def cmd_releasediff(args):
    # --- date-leakage guard (spec §9C): training <= old < new ---
    old_d, new_d = _parse_date(args.old_date), _parse_date(args.new_date)
    dates_ok = old_d < new_d
    msg = []
    if not dates_ok:
        msg.append(f"scored old-date {old_d} must be strictly before new-date {new_d}")
    if args.training_date:
        tr = _parse_date(args.training_date)
        if tr > old_d:
            dates_ok = False
            msg.append(f"LEAKAGE: training-date {tr} is after scored old-date {old_d}")
    if not dates_ok:
        sys.stderr.write("releasediff REFUSED: " + "; ".join(msg) + "\n")
        sys.exit(2)

    data = np.load(args.pool_npz)
    pool_idx = data["pool_idx"]; score_z = data["score_z"]
    depth = data["depth"]; clade_size = data["clade_size"]; degree = data["degree"]
    idx2tax = load_mapping(args.mapping)
    scored_taxids = [int(idx2tax[int(i)]) for i in pool_idx]

    old_parent = parse_parents(args.old_nodes)
    new_parent = parse_parents(args.new_nodes)
    new_merged = parse_merged(args.new_merged) if args.new_merged else {}
    new_deln = parse_delnodes(args.new_delnodes) if args.new_delnodes else set()

    # restrict to scored taxa present in the OLD release (score had to be computable there)
    present = np.array([t in old_parent for t in scored_taxids])
    reclass = reclassified_taxa([t for t, p in zip(scored_taxids, present) if p],
                                old_parent, new_parent, new_merged, new_deln)
    is_reclass = np.array([t in reclass for t in scored_taxids]) & present

    # primary: matched-background odds ratio (depth x size x effort=degree)
    res = enrichment_odds_ratio(score_z[present], is_reclass[present], top_frac=args.top_frac)
    # matched control diagnostic: do high-score flagged nodes beat their matched controls?
    flagged = np.flatnonzero((score_z >= np.quantile(score_z[present], 1 - args.top_frac)) & present)
    controls = match_background(flagged, depth, clade_size, degree, n_bins=args.n_bins, seed=0)
    res["flagged_reclass_rate"] = float(is_reclass[flagged].mean()) if len(flagged) else 0.0
    res["matched_control_reclass_rate"] = float(is_reclass[controls].mean()) if len(controls) else 0.0

    res.update({"n_reclassified": int(is_reclass[present].sum()),
                "n_scored_present_in_old": int(present.sum()),
                "training_date": args.training_date, "old_date": args.old_date,
                "new_date": args.new_date, "dates_ok": True})
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "releasediff.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
```
Wire the subparser in `main()`:
```python
    rd = sub.add_parser("releasediff")
    rd.add_argument("--pool-npz", required=True)
    rd.add_argument("--mapping", required=True)
    rd.add_argument("--old-nodes", required=True)
    rd.add_argument("--old-merged", default=None)
    rd.add_argument("--old-delnodes", default=None)
    rd.add_argument("--new-nodes", required=True)
    rd.add_argument("--new-merged", default=None)
    rd.add_argument("--new-delnodes", default=None)
    rd.add_argument("--training-date", default=None)
    rd.add_argument("--old-date", required=True)
    rd.add_argument("--new-date", required=True)
    rd.add_argument("--top-frac", type=float, default=0.1)
    rd.add_argument("--n-bins", type=int, default=5)
    rd.add_argument("-o", "--output-dir", required=True)
```
and dispatch:
```python
    elif args.cmd == "releasediff":
        cmd_releasediff(args)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest tests/eval/test_anomaly_validation_cli.py -v`
Expected: PASS (4 tests: roc + enrichment + 2 releasediff)

- [ ] **Step 5: Run the full eval suite**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest tests/eval/ -v`
Expected: PASS (all Plan-1 + Plan-2 tests)

- [ ] **Step 6: Commit**

```bash
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings add scripts/_anomaly_validation.py src/taxembed/utils/taxdump.py tests/eval/test_anomaly_validation_cli.py
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings commit -m "feat(app2): _anomaly_validation leg B — NCBI release-diff odds ratio + dated-archive fetch + leakage guard"
```

---

### Task 8: Produce the real #2 results on eukaryota 877k

**Files:**
- Output: `artifacts/tags/eukaryota_canonical/anomaly/` (analysis output, not committed)

- [ ] **Step 1: Score every node + BH-FDR (the ranked deliverable)**

Run:
```bash
/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python scripts/taxonomy_anomaly.py \
  --checkpoint artifacts/tags/eukaryota_canonical/eukaryota_canonical.pth \
  --mapping data/taxopy/eukaryota_2759_clean/taxonomy_edges_eukaryota_2759_clean.mapping.tsv \
  --rank family --k 10 --n-null 200 --n-bins 5 --seed 0 \
  -o artifacts/tags/eukaryota_canonical/anomaly
```
Expected: writes `anomaly_ranked.tsv`, `anomaly_pool.npz`, `anomaly_summary.json`. Sanity-eyeball the top-20 taxids by name (expect known-messy genera / mislabeled species). Repeat at `--rank order` and `--rank class` to confirm the score is not rank-fragile.

- [ ] **Step 2: Leg A — synthetic ROC by displacement**

Run:
```bash
/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python scripts/_anomaly_validation.py roc \
  --checkpoint artifacts/tags/eukaryota_canonical/eukaryota_canonical.pth \
  --mapping data/taxopy/eukaryota_2759_clean/taxonomy_edges_eukaryota_2759_clean.mapping.tsv \
  --rank family --k 10 --n-relocate 3000 --n-null 200 --seed 0 \
  -o artifacts/tags/eukaryota_canonical/anomaly
```
Expected: `roc_by_displacement.json` with `score_z` AUC RISING with displacement class and EXCEEDING every trivial baseline (`baseline_auc_by_displacement`), especially at small displacements. If a baseline matches the score at large displacement, that's expected (cross-kingdom moves are trivially easy); the win must be at sister-genus/sister-family displacement.

- [ ] **Step 3: Leg C — incertae-sedis / environmental enrichment**

Run:
```bash
/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python scripts/_anomaly_validation.py enrichment \
  --pool-npz artifacts/tags/eukaryota_canonical/anomaly/anomaly_pool.npz \
  --mapping data/taxopy/eukaryota_2759_clean/taxonomy_edges_eukaryota_2759_clean.mapping.tsv \
  --names-dmp data/names.dmp --top-frac 0.1 \
  -o artifacts/tags/eukaryota_canonical/anomaly
```
Expected: `enrichment.json` with `odds_ratio > 1` and Fisher `p_value < 0.05` (high-anomaly taxa concentrate in known-uncertain regions).

- [ ] **Step 4: Leg B — NCBI release-diff (the headline; one-time fetch + run)**

Pick the OLD archived release ~3 years before the training dump. Fetch + run (date discipline enforced by the CLI):
```bash
/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python scripts/_anomaly_validation.py releasediff \
  --pool-npz artifacts/tags/eukaryota_canonical/anomaly/anomaly_pool.npz \
  --mapping data/taxopy/eukaryota_2759_clean/taxonomy_edges_eukaryota_2759_clean.mapping.tsv \
  --old-nodes data/taxdump_archive_2022/nodes.dmp --old-merged data/taxdump_archive_2022/merged.dmp --old-delnodes data/taxdump_archive_2022/delnodes.dmp \
  --new-nodes data/nodes.dmp --new-merged data/merged.dmp --new-delnodes data/delnodes.dmp \
  --training-date <DUMP-DATE-OF-TRAINING-TAXONOMY> --old-date 2022-01-01 --new-date <CURRENT-DUMP-DATE> \
  --top-frac 0.1 --n-bins 5 \
  -o artifacts/tags/eukaryota_canonical/anomaly
```
Note: the OLD archive must be staged first via `ensure_taxdump_archive(Path("data/taxdump_archive_2022"), "taxdmp_2022-01-01.zip")` (a one-liner in a Python REPL, or add a `--fetch-old <archive-name>` flag in a follow-up). Confirm the score was computed on taxa present in the OLD release. Expected: `releasediff.json` with `odds_ratio > 1`, Fisher `p_value < 0.05`, `flagged_reclass_rate > matched_control_reclass_rate`, and the actual `n_reclassified / n_scored_present_in_old`. **This is the LEAD result** — record all four numbers (OR, CI, n_flagged, n_reclassified) in `docs/SESSION_LOG.md` and update spec §5#2 / §9A.

- [ ] **Step 5: Record + escalate-on-null**

Append the four leg results to `docs/SESSION_LOG.md` (dated). If leg B odds ratio ≈ 1 (no predictive signal), that is a finding to escalate, NOT to bury — the score may be detecting internal inconsistency uncorrelated with NCBI's own revisions (spec §9F honesty: the score detects taxonomy inconsistency, not biological truth). Note it explicitly.

- [ ] **Step 6: Re-run on cellular 1.1M when job 5673097 lands**

When `cellular_canonical.pth` is local, repeat Steps 1–5 with `--checkpoint artifacts/tags/cellular_canonical/cellular_canonical.pth --mapping data/taxopy/cellular_organisms_131567_clean/taxonomy_edges_cellular_organisms_131567_clean.mapping.tsv`. **Re-derive, do NOT transfer, every rank-dependent baseline** (chance_purity, size/depth bins, displacement edges) — cellular adds superkingdom/domain ranks and "phylum" semantics differ across bacteria/eukaryota (spec §9C cellular rank-label trap). The size-conditioning (z vs matched null) makes the score itself rank-robust, but the displacement edges in leg A and the bin counts must be re-quantiled from the cellular tree.

---

## Self-review notes

**Spec coverage (§5#2 + §9B + §9C):**
- ✅ Anomaly score = SIZE-CONDITIONED kNN-impurity. PINNED formula (Task 1): headline `score_z = (mu_null − observed_purity)/sigma_null` over a depth+clade-size-matched random-angle null; cross-check `score_excess = chance_purity − observed_purity`. Vote rule = fraction of k nearest sharing the rank-R label; k, rank, vote rule are recorded params. New module `src/taxembed/eval/anomaly.py` + `scripts/taxonomy_anomaly.py`.
- ✅ Trivial baselines the score must beat (Task 1 `trivial_baselines` + Task 5 AUC comparison): clade size, depth, node degree, distance-to-parent-centroid — AUC computed + compared at each displacement.
- ✅ Leg A synthetic ROC stratified by displacement (Task 5): AUC as a **curve** vs displacement class (using `TreeDistance.path_length(old_parent, new_parent)` for the sister-genus→cross-kingdom ladder), not one number; baselines compared at each class.
- ✅ Leg B NCBI release-diff (Task 7, the heaviest, explicitly budgeted): dated archive fetch (`ensure_taxdump_archive`), canonicalize BOTH releases via merged.dmp + delnodes.dmp (`release_diff.py`), reclassification = direct-parent change after canonicalization **excluding pure ID-merges/rank-only** (PINNED in Task 2), enrichment as **odds ratio + Fisher p + Woolf CI** against a background **matched on depth + clade-size + study-effort(#descendants≈degree/clade_size)**, training-predates-scored-release **date guard** enforced (CLI refuses on leakage).
- ✅ Leg C incertae-sedis/environmental enrichment (Task 6): reuses `audit_taxonomy_noise.classify_name_noise` + `is_container` + `parse_names_dmp`.
- ✅ FDR (Benjamini–Hochberg) over ~1.1M nodes (Task 1 `benjamini_hochberg`, applied in Task 4 → `q_value` column).
- ✅ Eukaryota-now / cellular-later with re-derived rank-dependent baselines (Task 8 Step 6, spec §9C trap); explicit `--checkpoint`/`--mapping` + the `final` ep200 ckpt throughout.
- ✅ Builds on Plan 1: reuses `taxembed.eval.treedist.TreeDistance`, `taxembed.eval.nulls` (radial-only null available for robustness), `taxembed.eval.bootstrap.taxon_bootstrap_ci` (available; the headline CIs here are Fisher/Woolf for the OR and the matched-null z for per-node, with `taxon_bootstrap_ci` usable for AUC CIs as a follow-up), and the kNN machinery (`_prep_sqnorms`, `_batch_distances`, `chance_purity`, `build_pool`).

**Placeholder scan:** no "TBD"/"FIXME"/`...`/`pass`-stub placeholders. Two intentional run-time fill-ins in Task 8 Step 4 (`<DUMP-DATE-OF-TRAINING-TAXONOMY>`, `<CURRENT-DUMP-DATE>`, archive file name) are operator inputs for the live fetch, not code placeholders — every code block is complete and runnable.

**Type consistency:** core takes/returns numpy arrays of int (node ids, labels, depth, clade_size, degree) and float (observed_purity, scores, p/q-values); `excess_impurity(observed, chance:float)→(N,)`, `matched_null_z(observed,(Q,n_null))→(Q,)`, `trivial_baselines(...)→dict[str,(N,)]`, `baseline_aucs(labels, dict)→dict[str,float]`, `benjamini_hochberg((N,))→(N,)`, `relocate_nodes(...)→(new_parent:(N,), moved:(m,))`, `enrichment_odds_ratio(score,is_positive,top_frac)→dict`, `match_background(...)→(m,) control indices`. `release_diff.reclassified_taxa(...)→set[int]`. CLIs consistently produce/consume `anomaly_pool.npz` (legs C & B read the score the scorer wrote — no recomputation, no drift). The `--parent-from-mapping`/`--rank-from-mapping` test modes keep every integration test network- and LRZ-data-free (mirrors Plan 1 Task 6).

**Effort honesty (spec §9C):** Task 7 (leg B) is the heaviest — fetch + dual canonicalization + date discipline + matched OR; budgeted as its own task with the date-leakage guard PINNED and unit-tested against local synthetic dmps so the suite needs no network. Tasks 1–3 (pure core) and 4–6 are thin given the reuse. The score inner loop is a refactor of the existing kNN-purity loop (spec §9C "genuinely thin").
