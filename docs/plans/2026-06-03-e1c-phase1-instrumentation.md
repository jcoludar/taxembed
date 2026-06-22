# E1c Phase 1 — Negative-Hardness Diagnostic + Step-0 Premise Validation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only diagnostic that measures, on the *existing default* negative sampler, how much within-clade gradient signal the softmax actually receives — then run it across the dataset-size gradient (echino → mollusca → arthropoda → metazoa) to confirm or falsify the E1c premise (within-clade gradient starves at scale) BEFORE building any new sampler.

**Architecture:** A standalone offline script (`scripts/diagnose_negative_hardness.py`) reuses the production `HierarchicalDataLoader` to draw negatives exactly as training does, loads a checkpoint's Poincaré embeddings via the analyzer's `load_embeddings`, and computes per-anchor (a) the fraction of negatives that are within-clade (within-class / within-grandparent, via the loader's `_node_class_arr` / `_node_gp_arr`) and (b) the softmax probability mass `p_j` those within-clade negatives receive in the NLL loss — stratified by curriculum hop-distance `dd`. Pure, unit-tested helper functions do the labeling and the `p_j` math; a thin driver wires the loader + checkpoint + datasets.

**Tech Stack:** Python 3.13, numpy, torch (CPU/MPS), the repo's `train_hierarchical.HierarchicalDataLoader` + `TrainingPairs`, and `scripts/analyze_hierarchy_hyperbolic.load_embeddings`. pytest (config in `pyproject.toml`).

**Scope:** Phase 1 = instrument + measure + decide. **Phase 2 (the ancestor-anchored LCA-mixture band sampler, its index, CLI wiring, and the seeded/kNN-purity gate harness) is DEFERRED** — it is written only if Step 0 confirms starvation AND the Experiment-1 (5664609) ep80 readout keeps E1c ahead of E2/E3. See `docs/specs/2026-06-03-e1c-hard-negative-sampling-spec.md` §"Spec v2".

**WHERE THIS RUNS (read first):** all files and **all commits land inside the poincaré submodule** `TaxPointCare/poincare-embeddings/` on its own branch `feat/lrz-readiness-prep` (NOT the superproject's `feat/ant-venom-gene-discovery`). Run commands from the submodule dir; make every commit `git -C TaxPointCare/poincare-embeddings ...` (or `cd` into the submodule in your shell). The superproject will then show a dirty submodule pointer — leave that bump for the user unless told otherwise. (Note: the superproject's pre-commit/pre-push hooks are NOT installed in the submodule, so there's no automated guard here — be deliberate.)

**Datasets present locally** (verified 2026-06-03): `data/taxopy/{echinodermata_7586_clean (3,965), mollusca_6447_clean (32,017), arthropoda_6656_clean (324,983), metazoa_33208_clean (498,246)}/`. Checkpoints present for echino (`artifacts/tags/echino_softmax/echino_softmax_best.pth`) and metazoa (`artifacts/tags/metazoa_softmax_milestones/...best.pth`).

---

## File Structure

- **Create** `scripts/_negative_hardness.py` — pure, importable helpers (no I/O): `numpy_poincare_distance`, `softmax_pj`, `label_negatives`. One responsibility: the math + labeling. Importable by both the driver and tests.
- **Create** `scripts/diagnose_negative_hardness.py` — the driver/CLI: loads a dataset + (optional) checkpoint, iterates the loader, aggregates, prints a report. One responsibility: orchestration + I/O.
- **Create** `tests/test_negative_hardness.py` — unit tests for the three pure helpers.
- **Modify** none of the production training/loss code in Phase 1 (read-only diagnostic). `HierarchicalDataLoader` is imported and used as-is.

Rationale: keeping the pure math in `_negative_hardness.py` (leading underscore = helper, per repo convention) makes it unit-testable without loading datasets or checkpoints; the driver stays thin.

---

## Task 1: Pure helpers — Poincaré distance, softmax `p_j`, within-clade labeling

**Files:**
- Create: `scripts/_negative_hardness.py`
- Test: `tests/test_negative_hardness.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_negative_hardness.py`:

```python
"""Unit tests for the negative-hardness diagnostic helpers (Phase 1, E1c)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import numpy as np
import pytest

from _negative_hardness import numpy_poincare_distance, softmax_pj, label_negatives


def test_poincare_distance_origin():
    # Distance from origin to itself is 0; symmetric; grows toward the boundary.
    o = np.zeros(3)
    assert numpy_poincare_distance(o, o) == pytest.approx(0.0, abs=1e-6)
    a = np.array([0.5, 0.0, 0.0])
    b = np.array([-0.5, 0.0, 0.0])
    d_ab = numpy_poincare_distance(a, b)
    assert d_ab > 0
    assert numpy_poincare_distance(b, a) == pytest.approx(d_ab, rel=1e-6)
    # Closer-to-boundary pair is farther than a near-origin pair of equal Euclidean gap.
    near = numpy_poincare_distance(np.array([0.0, 0, 0]), np.array([0.1, 0, 0]))
    far = numpy_poincare_distance(np.array([0.85, 0, 0]), np.array([0.95, 0, 0]))
    assert far > near


def test_softmax_pj_mass():
    # Positive much closer than negatives -> negatives get ~0 p_j mass.
    p_neg = softmax_pj(np.array([0.1]), np.array([[5.0, 5.0, 5.0]]))
    assert p_neg.shape == (1, 3)
    assert p_neg.sum() < 0.05
    # Positive far, negatives near -> negatives carry almost all the mass.
    p_neg2 = softmax_pj(np.array([5.0]), np.array([[0.1, 0.1, 0.1]]))
    assert p_neg2.sum() > 0.9
    # Within a row, the nearer negative gets more mass than the farther one.
    p_neg3 = softmax_pj(np.array([3.0]), np.array([[0.5, 4.0]]))
    assert p_neg3[0, 0] > p_neg3[0, 1]


def test_label_negatives():
    # node_class_arr / node_gp_arr indexed by node id; -1 = "no class/gp".
    node_class = np.array([-1, 10, 10, 20, 20, -1], dtype=np.int64)
    node_gp = np.array([-1, 100, 100, 200, 999, -1], dtype=np.int64)
    descendants = np.array([1, 3], dtype=np.int64)        # batch of 2 anchors
    negatives = np.array([[2, 4, 5],                       # for desc 1 (class 10, gp 100)
                          [4, 1, 0]], dtype=np.int64)      # for desc 3 (class 20, gp 200)
    within_class, within_gp = label_negatives(node_class, node_gp, descendants, negatives)
    # desc 1: neg 2 shares class 10 AND gp 100; neg 4 class 20 (no); neg 5 class -1 (no)
    assert within_class[0].tolist() == [True, False, False]
    assert within_gp[0].tolist() == [True, False, False]
    # desc 3 (class 20, gp 200): neg 4 shares class 20 but gp 999 (class yes, gp no);
    #                            neg 1 class 10 (no); neg 0 class -1 (no)
    assert within_class[1].tolist() == [True, False, False]
    assert within_gp[1].tolist() == [False, False, False]
    # -1 anchors must never count as within-clade (guard against sentinel collisions)
    desc_noclass = np.array([0], dtype=np.int64)
    negs = np.array([[5, 5, 5]], dtype=np.int64)           # also class -1
    wc, wg = label_negatives(node_class, node_gp, desc_noclass, negs)
    assert wc.sum() == 0 and wg.sum() == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_negative_hardness.py -v -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError: No module named '_negative_hardness'`.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/_negative_hardness.py`:

```python
"""Pure helpers for the negative-hardness diagnostic (Phase 1, E1c). No I/O.

Measures how much within-clade gradient the softmax/NLL loss receives from the
negatives a sampler draws. See docs/plans/2026-06-03-e1c-phase1-instrumentation.md.
"""
from __future__ import annotations

import numpy as np


def numpy_poincare_distance(u: np.ndarray, v: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Poincaré-ball geodesic distance, numpy port of model.poincare_distance.

    Supports broadcasting: u,v of shape (..., dim) -> distance of shape (...).
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    sq_u = np.clip(np.sum(u * u, axis=-1), 0.0, 1.0 - eps)
    sq_v = np.clip(np.sum(v * v, axis=-1), 0.0, 1.0 - eps)
    sq_diff = np.sum((u - v) ** 2, axis=-1)
    arg = 1.0 + 2.0 * sq_diff / ((1.0 - sq_u) * (1.0 - sq_v))
    arg = np.maximum(arg, 1.0)  # arccosh domain
    return np.arccosh(arg)


def softmax_pj(d_pos: np.ndarray, d_neg: np.ndarray) -> np.ndarray:
    """Softmax probability mass on each NEGATIVE in the Nickel-Kiela NLL.

    Logits = [-d_pos, -d_neg_1, ..., -d_neg_n]; returns the negative-part
    probabilities, shape (B, n_neg). The loss' gradient on negative j is its p_j,
    so this is the gradient share each negative receives.
    """
    d_pos = np.asarray(d_pos, dtype=np.float64).reshape(-1, 1)   # (B,1)
    d_neg = np.asarray(d_neg, dtype=np.float64)                  # (B,n_neg)
    logits = np.concatenate([-d_pos, -d_neg], axis=1)           # (B,1+n_neg)
    logits -= logits.max(axis=1, keepdims=True)                 # numerical stability
    ex = np.exp(logits)
    p = ex / ex.sum(axis=1, keepdims=True)
    return p[:, 1:]                                             # drop the positive column


def label_negatives(node_class_arr: np.ndarray, node_gp_arr: np.ndarray,
                    descendant_idxs: np.ndarray, negatives: np.ndarray):
    """Boolean masks: is each negative within the anchor's class / grandparent?

    `-1` sentinels (no class / no grandparent) NEVER count as a match — this is the
    guard against the sentinel-collision bug (R2 review finding).
    Returns (within_class, within_gp), each shape (B, n_neg) bool.
    """
    desc_class = node_class_arr[descendant_idxs][:, None]   # (B,1)
    desc_gp = node_gp_arr[descendant_idxs][:, None]         # (B,1)
    neg_class = node_class_arr[negatives]                   # (B,n_neg)
    neg_gp = node_gp_arr[negatives]                         # (B,n_neg)
    within_class = (neg_class == desc_class) & (desc_class != -1) & (neg_class != -1)
    within_gp = (neg_gp == desc_gp) & (desc_gp != -1) & (neg_gp != -1)
    return within_class, within_gp
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_negative_hardness.py -v -p no:cacheprovider`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/_negative_hardness.py tests/test_negative_hardness.py
git commit -m "feat(e1c): pure helpers for negative-hardness diagnostic + tests"
```

---

## Task 2: Driver script — measure within-clade fraction + `p_j` mass on the default sampler

**Files:**
- Create: `scripts/diagnose_negative_hardness.py`
- Uses: `scripts/_negative_hardness.py` (Task 1), `train_hierarchical.HierarchicalDataLoader` + `TrainingPairs`, `scripts/analyze_hierarchy_hyperbolic.load_embeddings`.

- [ ] **Step 1: Write the driver**

Create `scripts/diagnose_negative_hardness.py`:

```python
#!/usr/bin/env python
"""Step-0 diagnostic (E1c): how much within-clade gradient does the DEFAULT sampler give?

Read-only. For one dataset (and optionally a checkpoint), draws negatives via the
production HierarchicalDataLoader exactly as training does, and reports — overall and
stratified by curriculum hop-distance dd:
  - within-class / within-grandparent NEGATIVE FRACTION   (checkpoint-free; sampler-only)
  - softmax p_j MASS on within-clade negatives             (needs --checkpoint)
  - mean negative Poincaré distance                        (needs --checkpoint)

Premise under test: at large N the default same-depth sampler draws few within-clade
negatives, so the softmax (which self-weights to near negatives) has nothing within-clade
to sharpen against -> within-clade p_j mass collapses with scale.

Usage:
  .venv/bin/python scripts/diagnose_negative_hardness.py \
      --file data/taxopy/<ds>/..._transitive.npz \
      --mapping data/taxopy/<ds>/...mapping.tsv \
      [--checkpoint artifacts/tags/<tag>/<tag>_best.pth] \
      --n-negatives 50 --batches 40 --seed 0 --tag <label>
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                 # train_hierarchical
sys.path.insert(0, str(ROOT / "scripts"))     # _negative_hardness, analyze_hierarchy_hyperbolic

from train_hierarchical import TrainingPairs, HierarchicalDataLoader
from _negative_hardness import numpy_poincare_distance, softmax_pj, label_negatives


def _dd_bucket(dd: int) -> str:
    if dd <= 1:
        return "dd<=1"
    if dd <= 9:
        return "dd2-9"
    if dd <= 18:
        return "dd10-18"
    return "dd19+"


def main():
    ap = argparse.ArgumentParser(description="E1c Step-0 negative-hardness diagnostic")
    ap.add_argument("--file", required=True, help="transitive .npz")
    ap.add_argument("--mapping", required=True, help="mapping .tsv (for n_nodes)")
    ap.add_argument("--checkpoint", default=None, help="optional .pth for p_j / distance")
    ap.add_argument("--n-negatives", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--batches", type=int, default=40, help="how many batches to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="label for the report line")
    args = ap.parse_args()

    np.random.seed(args.seed)

    pairs = TrainingPairs.load(Path(args.file))
    # n_nodes from the mapping (one row per node, header 'taxid\tidx')
    n_nodes = sum(1 for _ in open(args.mapping)) - 1
    loader = HierarchicalDataLoader(
        training_data=pairs, n_nodes=n_nodes,
        batch_size=args.batch_size, n_negatives=args.n_negatives,
        depth_stratify=True, tiered_negatives=False,   # DEFAULT sampler = the thing under test
    )

    emb = None
    if args.checkpoint:
        from analyze_hierarchy_hyperbolic import load_embeddings
        emb = np.asarray(load_embeddings(Path(args.checkpoint)), dtype=np.float64)  # (N,dim)

    # accumulators, overall + per dd bucket
    frac_class = defaultdict(list)
    frac_gp = defaultdict(list)
    pj_class = defaultdict(list)
    neg_dist = defaultdict(list)

    seen = 0
    for ancestors, descendants, negatives, depths in loader:
        anc = ancestors.numpy(); desc = descendants.numpy()
        negs = negatives.numpy(); dd = depths.numpy().astype(int)   # depth_diff per pair

        wc, wg = label_negatives(loader._node_class_arr, loader._node_gp_arr, desc, negs)
        # per-anchor fractions
        f_c = wc.mean(axis=1); f_g = wg.mean(axis=1)

        if emb is not None:
            d_pos = numpy_poincare_distance(emb[anc], emb[desc])                 # (B,)
            d_neg = numpy_poincare_distance(emb[anc][:, None, :], emb[negs])     # (B,n_neg)
            p_neg = softmax_pj(d_pos, d_neg)                                     # (B,n_neg)
            pj_in_class = (p_neg * wc).sum(axis=1)                               # mass on within-class negs
            mean_neg_d = d_neg.mean(axis=1)

        for i in range(len(desc)):
            b = _dd_bucket(int(dd[i]))
            frac_class[b].append(f_c[i]); frac_class["ALL"].append(f_c[i])
            frac_gp[b].append(f_g[i]); frac_gp["ALL"].append(f_g[i])
            if emb is not None:
                pj_class[b].append(pj_in_class[i]); pj_class["ALL"].append(pj_in_class[i])
                neg_dist[b].append(mean_neg_d[i]); neg_dist["ALL"].append(mean_neg_d[i])

        seen += 1
        if seen >= args.batches:
            break

    def m(d, k):
        return float(np.mean(d[k])) if d.get(k) else float("nan")

    print(f"\n=== negative-hardness :: {args.tag or args.file} "
          f"(N={n_nodes:,}, n_neg={args.n_negatives}, batches={seen}, ckpt={'yes' if emb is not None else 'no'}) ===")
    print(f"{'bucket':>8} | {'within-class frac':>17} | {'within-gp frac':>14} | "
          f"{'pj-mass within-class':>20} | {'mean neg dist':>13}")
    for b in ["ALL", "dd<=1", "dd2-9", "dd10-18", "dd19+"]:
        if not frac_class.get(b):
            continue
        print(f"{b:>8} | {m(frac_class,b):>17.4f} | {m(frac_gp,b):>14.4f} | "
              f"{m(pj_class,b):>20.4f} | {m(neg_dist,b):>13.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke the driver on echino, checkpoint-free (fast, no model)**

Run:
```bash
.venv/bin/python scripts/diagnose_negative_hardness.py \
  --file data/taxopy/echinodermata_7586_clean/taxonomy_edges_echinodermata_7586_clean_transitive.npz \
  --mapping data/taxopy/echinodermata_7586_clean/taxonomy_edges_echinodermata_7586_clean.mapping.tsv \
  --batches 20 --seed 0 --tag echino_nockpt
```
Expected: a table printing finite `within-class frac` / `within-gp frac` (≈ a few %–tens of %), `pj-mass` = `nan` (no ckpt), one ALL row + dd buckets present in echino (mostly dd<=1, dd2-9). Confirms loader integration + labeling run end-to-end.

- [ ] **Step 3: Smoke with a checkpoint on echino (exercises the p_j path)**

Run:
```bash
.venv/bin/python scripts/diagnose_negative_hardness.py \
  --file data/taxopy/echinodermata_7586_clean/taxonomy_edges_echinodermata_7586_clean_transitive.npz \
  --mapping data/taxopy/echinodermata_7586_clean/taxonomy_edges_echinodermata_7586_clean.mapping.tsv \
  --checkpoint artifacts/tags/echino_softmax/echino_softmax_best.pth \
  --batches 20 --seed 0 --tag echino_softmax
```
Expected: same table but `pj-mass within-class` and `mean neg dist` now finite. On echino (where same-depth ≈ within-clade) within-class `pj-mass` should be NON-trivial (the recipe works here).

- [ ] **Step 4: Commit**

```bash
git add scripts/diagnose_negative_hardness.py
git commit -m "feat(e1c): Step-0 driver — within-clade negative fraction + softmax p_j mass"
```

---

## Task 3: Run Step 0 across the scale gradient and make the go/no-go decision

**Files:**
- Modify: `docs/PROJECT_STATE.md` (metrics/Roadmap — record Step-0 result), `docs/SESSION_LOG.md` (new dated sub-note). Docs are EDIT-in-place only (Rule 7).
- No code.

- [ ] **Step 1: Run the within-clade FRACTION across all four datasets (checkpoint-free)**

The fraction is a pure property of the sampler+tree (no checkpoint needed), so it runs on every dataset. Run each (echino / mollusca / arthropoda / metazoa), e.g. metazoa:
```bash
.venv/bin/python scripts/diagnose_negative_hardness.py \
  --file data/taxopy/metazoa_33208_clean/taxonomy_edges_metazoa_33208_clean_transitive.npz \
  --mapping data/taxopy/metazoa_33208_clean/taxonomy_edges_metazoa_33208_clean.mapping.tsv \
  --batches 40 --seed 0 --tag metazoa_nockpt
```
Repeat for `mollusca_6447_clean` and `arthropoda_6656_clean` (substitute paths). Record the `ALL` and `dd2-9` within-class fractions per dataset.
Expected (premise TRUE): within-class fraction **falls monotonically with N** (echino high → metazoa low). Expected (premise FALSE): fraction roughly flat across scale.

- [ ] **Step 2: Run the `p_j` MASS at the two scales with checkpoints (echino vs metazoa)**

```bash
.venv/bin/python scripts/diagnose_negative_hardness.py \
  --file data/taxopy/metazoa_33208_clean/taxonomy_edges_metazoa_33208_clean_transitive.npz \
  --mapping data/taxopy/metazoa_33208_clean/taxonomy_edges_metazoa_33208_clean.mapping.tsv \
  --checkpoint artifacts/tags/metazoa_softmax_milestones/metazoa_softmax_milestones_best.pth \
  --batches 40 --seed 0 --tag metazoa_softmax
```
(echino p_j was measured in Task 2 Step 3.) If the metazoa `*_best.pth` filename differs, list `artifacts/tags/metazoa_softmax_milestones/` and use the `_best.pth`.
Expected (premise TRUE): within-class `pj-mass` at metazoa ≪ echino (gradient starves at scale). Premise FALSE: comparable mass at both scales.

- [ ] **Step 3: Decide the gate and record it**

Decision rule:
- **Premise CONFIRMED** (fraction falls with N AND metazoa `pj-mass` ≪ echino) → E1c is justified; proceed to write the **Phase 2 plan** (ancestor-anchored LCA-mixture band sampler + seeded/kNN-purity gate harness). Fold in the Experiment-1 ep80 readout to confirm E1c is still ahead of E2/E3.
- **Premise REFUTED** (within-clade `pj-mass` already non-trivial at metazoa) → the sampler is NOT the lever; STOP E1c and pivot the roadmap to E2 (cones) / E3 (optimizer/manifold). This is the ~0-GPU off-ramp the reviews asked for.

Edit `docs/PROJECT_STATE.md`: add a Step-0 result line under the E1c roadmap bullet (the actual fraction/`pj-mass` numbers + the verdict). Edit `docs/SESSION_LOG.md`: append a dated sub-note to the 2026-06-03 entry with the table and the decision.

- [ ] **Step 4: Commit**

```bash
git add docs/PROJECT_STATE.md docs/SESSION_LOG.md
git commit -m "docs(e1c): Step-0 negative-hardness result across scale gradient + go/no-go"
```

---

## Self-review notes

- **Spec coverage:** This plan implements spec v2 §"Step 0" (the ~0-GPU premise validation) + the class/grandparent half of the §3d diagnostic (within-clade fraction, `p_j` mass, mean neg distance, dd-stratified) — exactly the parts that gate whether Phase 2 is built. The order-ancestor metric, the band sampler, seeded training, the analyzer `--seed/--repeats`/kNN-purity, and the experimental gates (G0/G0.5/G1/G2) are spec v2 items deliberately deferred to the Phase 2 plan (they're only needed once Step 0 says "go").
- **No placeholders:** every code step has complete code; every run step has the exact command + expected output.
- **Naming consistency:** `numpy_poincare_distance`, `softmax_pj`, `label_negatives` are defined in Task 1 and used unchanged in Tasks 2–3. `label_negatives` takes the loader's `_node_class_arr`/`_node_gp_arr` (verified to exist, int64, `-1` sentinel) — the driver passes `loader._node_class_arr` / `loader._node_gp_arr`.
- **Risk:** read-only (no training/loss/store writes; only NEW scripts/tests + in-place doc edits). RED-LINE-safe: no LRZ, no GPU training. The metazoa run is a CPU dataloader pass + numpy distances over a sample of batches (minutes).
- **Known soft spot:** `n_nodes` is derived by counting mapping rows; if a mapping has no header the count is off by one — the driver assumes the `taxid\tidx` header that `_build_mapping` writes. Verify the first run's printed `N=` matches the dataset manifest's node count before trusting the numbers. _(Retired in v2 — use `pairs.n_nodes`.)_

---

# Plan fan-review — findings ledger + v2 fold (4 reviewers, 2026-06-03)

P1 (code-correctness), P2 (test-adequacy), P3 (experimental-validity), P4 (scope/hygiene). The reviews **verified the helpers + plumbing run correctly** (broadcasting, dtypes, sentinel guards, imports, no-curriculum iteration, perf — all confirmed), but found that the **checkpoint-`p_j` arm is confounded three ways** and the **fraction signal needs a baseline**. v2 restructures Step 0 so the **decision rests on a confound-free, checkpoint-free signal**, with `p_j` demoted to corroboration.

**BLOCKERs (fixed in v2):**
- **[P1] `load_embeddings` returns raw euclidean-param `z` (norm can be >1), NOT the in-ball point.** Both echino + metazoa were trained `--euclidean-param`; the in-ball point is `tanh(‖z‖/2)·z/‖z‖` (train_hierarchical.py:99-107), which the driver never applied — so `numpy_poincare_distance`'s norm-clip silently corrupts every distance/`p_j`. → **FIX:** map z→ball in the driver before distances (v2 Task 2 patch).
- **[P3] metazoa `p_j` used a COLLAPSED checkpoint.** `metazoa_softmax_milestones_best.pth` is verified +0.784 / 1.05× POOR (mid/post-collapse, loss-selected ≠ peak). Low `p_j` there conflates *bad embeddings* with *bad sampling*. → **FIX:** use the **ep50-70 PEAK milestone** (`..._milestone_epoch60.pth`), health-matched to echino, plus a milestone sweep (ep30/60/90/120). p_j is corroboration-only.
- **[P3] No chance-level baseline → "fraction falls with N" fires on the null.** Uniform sampling's within-class fraction mechanically falls with N regardless of any pathology. → **FIX:** add a **uniform control arm** and a **tiered upper-bound arm**; the decision metric becomes the scale-invariant **enrichment ratio** = default-within-class-frac / uniform-within-class-frac (cancels the class=phylum scale issue P3/spec flagged).

**MAJORs (fixed in v2):**
- **[P3] Decision rule had no thresholds + an unsound refute off-ramp** (frozen-checkpoint `p_j` ≠ training-time `p_j`). → **FIX:** pre-registered numeric thresholds; checkpoint-free enrichment ratio is PRIMARY; explicit mixed-outcome + caveat (v2 Task 3).
- **[P2] numpy distance port diverges up to 0.07 near the boundary; `softmax_pj` multi-row/normalization untested.** → **FIX:** add a closed-form+broadcast test and a multirow/normalization test (v2 Task 1); reword "port" → "re-implementation."
- **[P4] Submodule/branch context unstated** → fixed in the header above; commits use `git -C`.

**MINOR/NIT (folded):** use `pairs.n_nodes` not mapping-row count (drops the off-by-one + the `--mapping` n_nodes role — P4); print per-bucket **n** and flag sparse buckets (P3); within-gp metric meaning shifts with N — rely on the enrichment ratio, not the absolute fraction (P3); `--cov` warnings are benign (P2); no driver E2E test — over-testing (P2). **Cleanup owed (rm blocked):** `scripts/_p4_check_nnodes.py` (P4 review artifact).

## v2 Task 1 — add two tests

Append to `tests/test_negative_hardness.py` (both are pure, deterministic):

```python
def test_poincare_distance_closedform_and_broadcast():
    # closed form: d(0, x) = 2*arctanh(|x|)
    x = np.array([0.3, 0.0, 0.0])
    assert numpy_poincare_distance(np.zeros(3), x) == pytest.approx(2 * np.arctanh(0.3), rel=1e-6)
    # broadcast (B,1,dim) vs (B,n,dim) -> (B,n), the driver's actual path
    anc = np.array([[[0.1, 0, 0]], [[0.2, 0, 0]]])              # (2,1,3)
    negs = np.array([[[0.3, 0, 0], [0.4, 0, 0]],
                     [[0.5, 0, 0], [0.6, 0, 0]]])                # (2,2,3)
    d = numpy_poincare_distance(anc, negs)
    assert d.shape == (2, 2)
    assert d[0, 0] < d[0, 1] and d[1, 0] < d[1, 1]

def test_softmax_pj_multirow_and_normalization():
    d_pos = np.array([0.1, 5.0, 2.0])
    d_neg = np.array([[5.0, 5.0, 5.0], [0.1, 5.0, 0.1], [1.0, 2.0, 3.0]])
    p = softmax_pj(d_pos, d_neg)
    assert p.shape == (3, 3)
    pos_share = 1.0 - p.sum(axis=1)                              # implied positive column
    assert np.all(pos_share > -1e-12) and np.all(pos_share < 1.0 + 1e-12)
    assert p[0].sum() < 0.05 and p[1].sum() > 0.9                # row independence
```
(Reword the `numpy_poincare_distance` docstring "numpy port of" → "numpy re-implementation (broadcasting + arccosh-domain clamp) of"; it is metrically equivalent, not bit-identical.)

## v2 Task 2 — driver corrections

1. **n_nodes from pairs (drop the mapping-count):** replace the `n_nodes = sum(...)` line with
   `n_nodes = pairs.n_nodes`  (authoritative property; retires the off-by-one). `--mapping` is no longer needed for sizing; keep it only if you still pass it elsewhere (you don't — remove it).

2. **Map euclidean-param `z` → ball before distances** (the BLOCKER). Replace the checkpoint-load block with:
```python
    emb = None
    if args.checkpoint:
        from analyze_hierarchy_hyperbolic import load_embeddings
        z = np.asarray(load_embeddings(Path(args.checkpoint)), dtype=np.float64)  # (N,dim); raw z if euclidean-param
        norms = np.linalg.norm(z, axis=1, keepdims=True)
        if norms.max() >= 1.0:   # euclidean-param checkpoint: map tangent z -> Poincaré ball
            emb = np.tanh(norms / 2.0) * z / np.maximum(norms, 1e-8)
        else:
            emb = z
```

3. **Add a `--sampler` arm** (`default` | `uniform` | `tiered`) so the fraction is interpretable:
```python
    ap.add_argument("--sampler", choices=["default", "uniform", "tiered"], default="default")
    ...
    loader = HierarchicalDataLoader(
        training_data=pairs, n_nodes=n_nodes, batch_size=args.batch_size,
        n_negatives=args.n_negatives, depth_stratify=True,
        tiered_negatives=(args.sampler == "tiered"),
    )
    ...
    # inside the batch loop, AFTER getting `negs` from the loader:
    if args.sampler == "uniform":
        negs = np.random.randint(0, n_nodes, size=negs.shape).astype(np.int64)  # chance baseline
```
   (For `uniform` we overwrite the loader's negatives with all-node uniform draws; `default`/`tiered` use the loader's own sampler.)

4. **Print per-bucket `n`** next to each mean, and flag buckets with `n < 200` as noisy (deep-dd buckets are thinly sampled at metazoa).

## v2 Task 3 — confound-free decision protocol

**Primary signal (checkpoint-free, confound-free): within-class ENRICHMENT RATIO across the scale gradient.** For each dataset, run the driver with `--sampler default`, `--sampler uniform`, `--sampler tiered` (fast, no checkpoint). Compute, per dataset, `enrichment = default_within_class_frac / uniform_within_class_frac` and `achievable = tiered_within_class_frac / uniform_within_class_frac`.
- **Premise CONFIRMED** if enrichment **collapses toward ~1 at metazoa** (default sampler no better than chance) while `achievable` stays well >1 (a hard sampler *could* deliver within-clade negatives) — AND echino shows high default enrichment (so the recipe works precisely where the default sampler is already enriched). Pre-registered: confirm if metazoa `enrichment ≤ 1.2` while `achievable ≥ 2×` metazoa enrichment.
- **Premise REFUTED** if metazoa default `enrichment` is already large (default sampler enriches within-clade negatives ~as well as tiered) → the sampler is not the lever → STOP E1c, pivot to E2/E3.

**Secondary / corroboration (checkpoint, with caveats): `p_j` mass at health-matched + collapse milestones.** Measure within-class `p_j` mass on echino `echino_softmax_best.pth` (healthy) and on metazoa **`metazoa_softmax_milestones_milestone_epoch60.pth`** (the dd≤9 PEAK, +0.875/1.14-1.20×) — NOT `_best.pth`. Also sweep ep30/ep60/ep90/ep120 to show `p_j`-mass vs checkpoint health. **Record the caveat:** frozen-checkpoint `p_j` ≠ training-time `p_j`; it corroborates, it does not by itself decide. If metazoa peak-milestone `p_j`-mass is also ≪ echino, the confirmation is stronger; if it's fine at the peak but the enrichment ratio is ~1, trust the (confound-free) enrichment ratio.

**Then** record numbers + verdict in PROJECT_STATE/SESSION_LOG and route: CONFIRMED → write the Phase 2 plan (fold in Experiment-1 ep80 readout); REFUTED → pivot roadmap to E2/E3. This is the ~0-GPU off-ramp.
