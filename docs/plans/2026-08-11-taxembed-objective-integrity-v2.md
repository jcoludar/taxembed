# TaxEmbed Objective Integrity (P1) — Implementation Plan v2

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Supersedes:** `2026-08-11-taxembed-objective-integrity.md` (v1). v1 was reviewed by execution — Tasks 1, 2 and 4 ran clean and are carried over unchanged; Tasks 3, 5, 6, 7 had blocking defects and are rewritten here.

**Goal:** Establish, in the repo and reproducibly, whether the shipped TaxEmbed embedding was trained with a defective objective — and fix it.

**Architecture:** Pure, unit-testable analysis functions land first, each converting a review finding into a repo artifact with a test. Only then does the negative sampler change, because the fix's correctness depends on the zero-pool census the audit produces. Final task retrains fixed-vs-unfixed at clade scale under the **canonical recipe** and reports the delta.

**Tech Stack:** Python 3.13, numpy, torch, pytest. No GPU until Task 8.

**Source spec:** `docs/specs/2026-08-11-taxembed-overfitting-and-plm-showcase-design-v3.md`

## Global Constraints

- **Shell hygiene (CLAUDE.md, zero tolerance):** no compound Bash. No `&&`, `||`, `;`, pipes, heredocs, `python -c`, or `>>` redirection. Every Python call is a single absolute-path invocation. **Every git call uses `git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings`** — never `cd`.
- **Interpreter:** `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python`. This is the venv with `taxembed` editable-installed. The repo-root `.venv` at `SpeciesEmbedding/.venv` does **not** have it and will `ModuleNotFoundError`.
- **Test runner:** `<interpreter> -m pytest <abs-test-path> -v`
- **Never write to `data/`.** New outputs go to `results/` or `docs/`.
- **Rule 16:** before any `sbatch`, run `bash -n`, `python -m py_compile`, and a smoke run.
- **⚠ THE TRAINING ENTRYPOINT IS `train_small.py`.** `src/taxembed/cli/main.py:351` sets `train_script = PROJECT_ROOT / "train_small.py"`, and `artifacts/tags/cellular_canonical/run.json` confirms the shipped model ran it. `train_hierarchical.py`'s `main()` (`:953-1074`) is **dead legacy code** — 11 flags, and it calls `ranking_loss_with_margin` (`:845`), not `softmax_loss`. The model *classes and functions* live in `train_hierarchical.py` and are imported by `train_small.py:30-37`; only the CLI/argparse wiring belongs to `train_small.py`.
- **Canonical closure (read-only):** `data/taxopy/cellular_organisms_131567_clean/taxonomy_edges_cellular_organisms_131567_clean_transitive.npz`
- **Numbers of record** (author-verified; Task 2 must reproduce exactly): overall false-negative rate **47.375240%**; zero-valid-negative pairs **3,026,809 (14.1446%)**; root-anchored **1,102,162 (5.1505%)**; `nodes_per_depth[0:11] = [1, 3, 46, 49584, 1990, 8925, 18866, 20605, 60509, 44289, 43982]`.
- **Radial init floor (author-verified):** with `--radial-schedule log`, max_depth 40, over 1,102,163 nodes, the depth↔norm Pearson r **at initialization** is **0.957151**; linear schedule gives exactly **1.000000**. Task 4 must reproduce these.
- **Commit discipline:** multi-line messages via `Write` to a temp file then `git -C <repo> commit -F <file>` (CLAUDE.md Rule 15).

---

## File Structure

| File | Responsibility |
|---|---|
| `src/taxembed/eval/subtree.py` (new) | Parent array from closure; Euler intervals; vectorized descendant test. |
| `src/taxembed/eval/sampler_audit.py` (new) | Closed-form false-negative rate and valid-pool census. |
| `src/taxembed/eval/nulls.py` (modify) | `null_retrieval_sanity` — guard against a degenerate null. |
| `src/taxembed/eval/radial.py` (new) | Initialization-floor depth↔norm correlation. |
| `scripts/audit_negative_sampling.py` (new) | CLI over a closure npz → JSON in `results/`. |
| `scripts/diagnose_negative_hardness.py` (modify) | Add ancestry-of-**anchor** labelling (spec P1.2). |
| `train_hierarchical.py` (modify) | `seed_everything`; interval-complement negative sampling; bounded counters. |
| `train_small.py` (modify) | **The real entrypoint** — argparse flags, seeding call, loader kwargs. |
| `src/taxembed/cli/main.py` (modify) | Forward the new flags; record them in `run.json`. |
| `tests/eval/test_subtree.py`, `test_sampler_audit.py`, `test_radial.py`, `test_nulls.py`, `tests/test_negative_sampling.py` | Coverage. |

---

## Tasks 1, 2, 4 — carried over unchanged

**These three were executed during v1 review and passed (3/3, 4/4, 3/3 respectively). Task 2 reproduced all three numbers of record exactly against the real 21.4M-pair closure in 3.1 s / 1.76 GB peak.**

Take their code blocks **verbatim** from `docs/plans/2026-08-11-taxembed-objective-integrity.md`:
- **Task 1** (`src/taxembed/eval/subtree.py` + `tests/eval/test_subtree.py`) — plan v1 §Task 1
- **Task 2** (`src/taxembed/eval/sampler_audit.py` + `scripts/audit_negative_sampling.py` + `tests/eval/test_sampler_audit.py`) — plan v1 §Task 2
- **Task 4** (`src/taxembed/eval/radial.py` + `tests/eval/test_radial.py`) — plan v1 §Task 4

One documentation correction to carry: `target_radius`'s third parameter is named **`schedule`**, not `radial_schedule` (`train_hierarchical.py:28`). Positional calls are unaffected.

**Additional acceptance for Task 4** (new — this is a paper-facing result): after the unit tests pass, compute the floor on the real closure and confirm **log = 0.957151**, **linear = 1.000000**. Record both in `results/radial_init_floor.json`. The released model's trained value on the same closure-derived depths is 0.957244 — i.e. **training moves the depth↔norm correlation by ~0.0001**. This closes spec §6's open "0.957 vs 0.952" item: the discrepancy is a depth-source difference, and on any consistent source the correlation is fixed at initialization.

---

## Task 3 (REWRITTEN): Null retrieval sanity check

**Files:**
- Modify: `src/taxembed/eval/nulls.py`
- Modify: `tests/eval/test_nulls.py`

**Interfaces:**
- Produces: `null_retrieval_sanity(retrieved_idx, node_depth, chance_accuracy=None, null_accuracy=None, shallow_max=10, deep_min=25) -> dict` with keys `n_distinct_retrieved`, `modal_share`, `mean_retrieved_depth`, `frac_shallow`, `frac_deep`, `pool_has_deep`, `degenerate`, `reason`.

**v1 defect fixed:** v1's own test failed against v1's own implementation — the fixture's deepest stratum was 20 while `deep_min` was 25, so `frac_deep == 0.0` fired the degeneracy rule. The rule must also be **gated on the pool actually containing deep nodes**, or it flags every shallow-tree null as degenerate.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/eval/test_nulls.py
import numpy as np

from taxembed.eval.nulls import null_retrieval_sanity


def test_shallow_shell_collapse_is_flagged_degenerate():
    # pool has genuinely deep nodes, but retrieval only ever returns shallow ones
    node_depth = np.concatenate([np.full(50, 2), np.full(950, 30)])
    retrieved = np.random.default_rng(0).integers(0, 50, size=1000)
    out = null_retrieval_sanity(retrieved, node_depth)
    assert out["degenerate"] is True
    assert out["pool_has_deep"] is True
    assert out["frac_deep"] == 0.0
    assert out["mean_retrieved_depth"] < 5


def test_healthy_null_spread_over_depths_is_not_degenerate():
    node_depth = np.concatenate([np.full(500, 2), np.full(500, 30)])
    retrieved = np.arange(1000)
    out = null_retrieval_sanity(retrieved, node_depth)
    assert out["degenerate"] is False
    assert out["n_distinct_retrieved"] == 1000


def test_shallow_pool_is_not_flagged_just_for_being_shallow():
    # no node anywhere is deep -- the deep-stratum rule must not fire
    node_depth = np.full(200, 4)
    retrieved = np.arange(200)
    out = null_retrieval_sanity(retrieved, node_depth)
    assert out["pool_has_deep"] is False
    assert out["degenerate"] is False


def test_single_reference_domination_is_flagged():
    node_depth = np.full(100, 30)
    retrieved = np.zeros(100, dtype=int)
    out = null_retrieval_sanity(retrieved, node_depth)
    assert out["degenerate"] is True
    assert out["modal_share"] == 1.0


def test_null_below_chance_is_flagged():
    node_depth = np.concatenate([np.full(50, 2), np.full(50, 30)])
    retrieved = np.arange(100)
    out = null_retrieval_sanity(retrieved, node_depth,
                                chance_accuracy=0.43, null_accuracy=0.0009)
    assert out["degenerate"] is True
    assert "below" in out["reason"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/tests/eval/test_nulls.py -v -k sanity`

Expected: FAIL — `ImportError: cannot import name 'null_retrieval_sanity'`

- [ ] **Step 3: Write the implementation**

```python
# append to src/taxembed/eval/nulls.py

def null_retrieval_sanity(retrieved_idx, node_depth, chance_accuracy=None,
                          null_accuracy=None, shallow_max: int = 10,
                          deep_min: int = 25) -> dict:
    """Guard against a degenerate null (spec v3 §1.4).

    radial_only_null retrieval collapses onto the shallow-norm shell: it returns many
    DISTINCT nodes (so identity concentration looks mild -- which is why an earlier check
    "did not reproduce" the problem) but nearly all of them shallow. A deep majority class
    like Mammalia then becomes unreachable, which is how the null landed ~480x BELOW
    frequency-matched chance. Measure the DEPTH distribution, not identity concentration.

    The deep-stratum rule is gated on the pool actually containing deep nodes, so a
    genuinely shallow tree is not flagged merely for being shallow.
    """
    retrieved_idx = np.asarray(retrieved_idx)
    node_depth = np.asarray(node_depth)
    depths = node_depth[retrieved_idx]
    _, counts = np.unique(retrieved_idx, return_counts=True)

    pool_has_deep = bool((node_depth >= deep_min).any())
    frac_shallow = float((depths <= shallow_max).mean())
    frac_deep = float((depths >= deep_min).mean())
    modal_share = float(counts.max() / counts.sum())

    reasons = []
    if pool_has_deep and frac_deep == 0.0:
        reasons.append(f"never retrieves a node at depth >= {deep_min} despite the pool having them")
    if modal_share > 0.5:
        reasons.append(f"one reference takes {modal_share:.1%} of retrievals")
    if chance_accuracy is not None and null_accuracy is not None and null_accuracy < chance_accuracy:
        reasons.append(
            f"null accuracy {null_accuracy:.5f} is below frequency-matched chance "
            f"{chance_accuracy:.5f} -- a null that cannot reach chance is misspecified"
        )

    return {
        "n_distinct_retrieved": int(len(counts)),
        "modal_share": modal_share,
        "mean_retrieved_depth": float(depths.mean()),
        "frac_shallow": frac_shallow,
        "frac_deep": frac_deep,
        "pool_has_deep": pool_has_deep,
        "chance_accuracy": chance_accuracy,
        "null_accuracy": null_accuracy,
        "degenerate": bool(reasons),
        "reason": "; ".join(reasons) if reasons else "passes depth-spread, concentration and chance checks",
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/tests/eval/test_nulls.py -v`

Expected: PASS — existing null tests plus 5 new.

- [ ] **Step 5: Commit**

```bash
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings add src/taxembed/eval/nulls.py tests/eval/test_nulls.py
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings commit -m "feat(eval): null retrieval sanity check gated on pool depth"
```

---

## Task 5 (REWRITTEN): Seed the training run

**Files:**
- Modify: `train_hierarchical.py` (define `seed_everything`)
- Modify: `train_small.py` (import it; add `--seed`; call it)
- Modify: `src/taxembed/cli/main.py` (add the flag; forward it; record it)
- Test: `tests/test_negative_sampling.py` (create)

**v1 defects fixed:** v1 patched `train_hierarchical.py`'s dead argparse; it must be `train_small.py`. v1's line references were wrong in two places — `train_parser.add_argument("--early-stopping", …)` is at **`main.py:708`**, not 376 (376 is inside the `train_cmd` list literal), and the variable being appended to is **`train_cmd`** (`main.py:353`), not `cmd`.

**Why:** `seed` occurs zero times in `train_hierarchical.py`, `train_small.py`, and `cli/main.py`. Unseeded: direction init (`torch.randn`, `train_hierarchical.py:85`), every negative draw (**legacy global** numpy RNG, so `np.random.default_rng` will not fix it), the per-epoch shuffle and `epoch_fraction` subsample (`:592-623`). The paper design doc's "Reproducible (seeded; repro run matches to ±0.01)" is unsupported.

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

- [ ] **Step 3: Define `seed_everything` in `train_hierarchical.py`**

Insert after the existing imports:

```python
def seed_everything(seed: int) -> None:
    """Seed every RNG the training loop actually uses.

    The negative sampler and epoch subsampler use the LEGACY global numpy RNG
    (np.random.randint / choice / shuffle), so np.random.seed is required --
    np.random.default_rng does not affect them. Direction init uses global torch.

    --amp plus CUDA scatter nondeterminism means this gives run-to-run reproducibility
    on CPU and near-reproducibility on GPU, not bitwise equality.
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

- [ ] **Step 5: Wire it into the real entrypoint**

In `train_small.py`, add `seed_everything` to the existing `from train_hierarchical import (...)` block at `:30-37`. Add to its argparse (near `:1066`, the end of the argument list):

```python
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for init, negative sampling, and epoch subsampling. "
                             "Omit for the historical unseeded behaviour.")
```

Immediately after `args = parser.parse_args()` (`:1068`):

```python
    if args.seed is not None:
        seed_everything(args.seed)
        print(f"  ✓ seeded: {args.seed}")
```

- [ ] **Step 6: Wire it through the CLI**

In `src/taxembed/cli/main.py`, add beside the other `train_parser.add_argument` calls near **`:708`**:

```python
    train_parser.add_argument("--seed", type=int, default=None,
                              help="RNG seed forwarded to the trainer.")
```

Forward it after **`:388`**, before `if args.curriculum:` at `:389`:

```python
    if args.seed is not None:
        train_cmd.extend(["--seed", str(args.seed)])
```

Record it in the metadata dict at **`:457`**, beside `"early_stopping": args.early_stopping,`:

```python
                    "seed": args.seed,
```

- [ ] **Step 7: Verify end to end**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/train_small.py --help`

Expected: `--seed` appears. (v1's plan would have failed here — `train_small.py: error: unrecognized arguments: --seed 0`.)

- [ ] **Step 8: Commit**

```bash
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings add train_hierarchical.py train_small.py src/taxembed/cli/main.py tests/test_negative_sampling.py
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings commit -m "feat(train): --seed across torch, legacy numpy, and epoch subsampling"
```

---

## Task 6 (REWRITTEN): Interval-complement ancestry-aware negative sampling

**Files:**
- Modify: `train_hierarchical.py` (`HierarchicalDataLoader`)
- Modify: `train_small.py`, `src/taxembed/cli/main.py` (two flags)
- Test: `tests/test_negative_sampling.py` (extend)

**Interfaces:**
- Consumes: `parent_from_closure`, `euler_intervals`, `is_descendant` (Task 1); the zero-pool census (Task 2).
- Produces: `HierarchicalDataLoader(..., exclude_descendant_negatives=False, drop_root_anchored=False)`; bounded counters `_realized_sum`, `_realized_count`, `_zero_pool_hits`.

**Four v1 defects fixed:**
1. v1 **rewrote the existing sampler**, collapsing the medium path (`:404-418`, which has self-exclusion and `replace=False`) into a plain `np.random.choice`. That changes the **unfixed** arm, so the A/B was not an A/B. → **Leave `:392-421` byte-for-byte; append the ancestry stage after it.**
2. v1 relaxed depth stratification on **every** rejection with a globally uniform draw. Measured: 166.5 of 300 negatives rejected per row, 255/256 rows per batch — ~55% of every row would become a uniform draw, confounded with anchor depth. Cost 23× (+12.2 h over 200 epochs at cellular scale). → **Interval-complement sampling** below: rejection-free, O(1), same-depth-preserving.
3. v1's mask branch was **dead code**, making Task 7's acceptance criterion unreachable. → replaced by a reachable `_zero_pool_hits` counter.
4. v1 rewrote `self.pairs`, invalidating `_depth_diff_masks` (`:206-209`, positional into pre-drop pairs) → reproduced `IndexError` under `epoch_fraction`, which the canonical recipe uses. → **filter the epoch index instead; never touch `self.pairs`.**

**The key idea.** Sort each depth pool by Euler `tin` once. An anchor's descendants are *contiguous* in `tin` order, so within pool `d` they occupy `[lo, hi)` with `lo = searchsorted(tin[pool_d], tin[a])`, `hi = searchsorted(tin[pool_d], tout[a])`. Sampling uniformly from the complement is then an index remap — draw `j` in `[0, len(pool)-(hi-lo))` and take `j` if `j < lo` else `j + (hi-lo)`. No rejection, no relaxation, same-depth stratification preserved exactly.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_negative_sampling.py
import numpy as np
import pytest

from taxembed.eval.subtree import parent_from_closure, euler_intervals, is_descendant
from taxembed.eval.treedist import TreeDistance


def _fixture_closure():
    """Tree: 0 -> {1,2}; 1 -> {3,4}; 2 -> {5,6}. Depth-2 layer = {3,4,5,6}."""
    anc = np.array([0, 0, 0, 0, 0, 0, 1, 1, 2, 2])
    des = np.array([1, 2, 3, 4, 5, 6, 3, 4, 5, 6])
    dd = np.array([1, 1, 2, 2, 2, 2, 1, 1, 1, 1])
    return anc, des, dd


def test_euler_descendant_test_agrees_with_binary_lifting_lca():
    """Independent oracle (spec v3 §8): Euler intervals must agree with the repo's
    binary-lifting LCA. x descends from a iff LCA(a, x) == a."""
    anc, des, dd = _fixture_closure()
    parent = parent_from_closure(anc, des, dd, n_nodes=7)
    depth = np.array([0, 1, 1, 2, 2, 2, 2])
    tin, tout = euler_intervals(parent)
    td = TreeDistance(parent, depth)

    a_grid, x_grid = np.meshgrid(np.arange(7), np.arange(7), indexing="ij")
    a_flat, x_flat = a_grid.ravel(), x_grid.ravel()
    euler = is_descendant(a_flat, x_flat, tin, tout)
    lca = td.lca(a_flat, x_flat) == a_flat
    assert (euler == lca).all()


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
    for ancestors, _d, negatives, _dep in loader:
        a = ancestors.numpy()[:, None]
        n = negatives.numpy()
        assert not is_descendant(np.broadcast_to(a, n.shape), n, tin, tout).any()


def test_root_anchored_pairs_are_dropped_without_breaking_depth_masks():
    from train_hierarchical import HierarchicalDataLoader

    anc, des, dd = _fixture_closure()
    loader = HierarchicalDataLoader.from_arrays(
        ancestor_idx=anc, descendant_idx=des, depth_diff=dd,
        n_negatives=3, batch_size=16, epoch_fraction=0.5,
        exclude_descendant_negatives=True, drop_root_anchored=True,
    )
    seen = set()
    for ancestors, _d, _n, _dep in loader:      # must not IndexError
        seen.update(ancestors.numpy().tolist())
    assert 0 not in seen


def test_zero_pool_case_is_counted_not_silently_faked():
    """A depth-1 anchor owning its entire depth layer has NO valid negative.
    Tree: 0 -> 1 -> {2,3}. For anchor 1 at depth-2 layer {2,3}, both descend from 1."""
    from train_hierarchical import HierarchicalDataLoader

    anc = np.array([0, 0, 0, 1, 1])
    des = np.array([1, 2, 3, 2, 3])
    dd = np.array([1, 2, 2, 1, 1])
    loader = HierarchicalDataLoader.from_arrays(
        ancestor_idx=anc, descendant_idx=des, depth_diff=dd,
        n_negatives=2, batch_size=8,
        exclude_descendant_negatives=True, drop_root_anchored=True,
    )
    for _a, _d, _n, _dep in loader:
        pass
    assert loader._zero_pool_hits > 0, "the depth-1 zero-pool case must be counted"


def test_counters_are_bounded_not_per_row_lists():
    from train_hierarchical import HierarchicalDataLoader

    anc, des, dd = _fixture_closure()
    loader = HierarchicalDataLoader.from_arrays(
        ancestor_idx=anc, descendant_idx=des, depth_diff=dd,
        n_negatives=3, batch_size=2,
        exclude_descendant_negatives=True, drop_root_anchored=True,
    )
    for _a, _d, _n, _dep in loader:
        pass
    assert isinstance(loader._realized_sum, int)
    assert isinstance(loader._realized_count, int)
    assert loader._realized_count > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/tests/test_negative_sampling.py -v -k "euler or descendant or root_anchored or zero_pool or counters"`

Expected: FAIL — `AttributeError: type object 'HierarchicalDataLoader' has no attribute 'from_arrays'`

- [ ] **Step 3: Add the `from_arrays` test seam**

⚠ The constructor signature is `(self, training_data, n_nodes, batch_size=32, n_negatives=50, depth_stratify=True, epoch_fraction=1.0, tiered_negatives=False, class_balanced=False)`. It takes `training_data`, **not** `pairs`, and `n_nodes` is **required** — v1 got both wrong.

```python
    @classmethod
    def from_arrays(cls, ancestor_idx, descendant_idx, depth_diff, **kwargs):
        """Build a loader directly from arrays (test seam -- no .npz on disk)."""
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

        pairs = TrainingPairs(
            ancestor_idx=anc, descendant_idx=des, depth_diff=dd,
            ancestor_depth=depth[anc], descendant_depth=depth[des],
            ancestor_taxid=anc.copy(), descendant_taxid=des.copy(),
        )
        return cls(training_data=pairs, n_nodes=n_nodes, **kwargs)
```

- [ ] **Step 4: Extend `__init__` — build indices from FULL pairs, never rewrite them**

Add the two kwargs to the signature, then at the **end** of `__init__` (after `_build_depth_index()` at `:202`, `_depth_diff_masks` at `:206-209`, and `_build_class_pair_index()` at `:216` have all run against the complete pair set):

```python
        self.exclude_descendant_negatives = exclude_descendant_negatives
        self.drop_root_anchored = drop_root_anchored
        self._realized_sum = 0
        self._realized_count = 0
        self._zero_pool_hits = 0
        self._tin = self._tout = None
        self._depth_pools_by_tin = {}

        if exclude_descendant_negatives:
            from taxembed.eval.subtree import parent_from_closure, euler_intervals

            parent = parent_from_closure(
                self.pairs.ancestor_idx, self.pairs.descendant_idx,
                self.pairs.depth_diff, self.n_nodes,
            )
            self._tin, self._tout = euler_intervals(parent)
            # Sort each depth pool by tin ONCE, so an anchor's descendants are contiguous.
            for d, pool in self._depth_to_nodes.items():
                self._depth_pools_by_tin[d] = np.asarray(pool)[np.argsort(self._tin[pool])]

        # NOTE: root-anchored pairs are excluded from the EPOCH INDEX (see __iter__), never
        # from self.pairs -- _depth_diff_masks and the class index hold positional indices
        # into the full pair array and would be invalidated by a rewrite.
        if drop_root_anchored:
            n_root = int((self.pairs.ancestor_depth == 0).sum())
            print(f"  ✓ excluding {n_root:,} root-anchored pairs from training "
                  f"(no contrastive signal: every node descends from the root; "
                  f"d(root,x) is covered by the radial regularizer)")
```

In `__iter__`, immediately **after** the epoch index is assembled and before the batch loop at `:625`:

```python
            if self.drop_root_anchored:
                indices = indices[self.pairs.ancestor_depth[indices] != 0]

        self._realized_sum = 0          # reset per epoch
        self._realized_count = 0
        self._zero_pool_hits = 0
```

- [ ] **Step 5: Append the ancestry stage — do NOT edit `:392-421`**

Leave `_sample_negatives_default_vectorized` exactly as it is. Add a new method and call it from `__iter__`:

```python
    def _resample_descendant_negatives(self, negatives, anc_idxs, desc_depths):
        """Replace any negative that is a descendant of its anchor, by sampling uniformly
        from the SAME depth pool minus the anchor's subtree.

        Because each depth pool is pre-sorted by Euler tin, the anchor's descendants occupy
        one contiguous block [lo, hi). Uniform sampling from the complement is an index
        remap -- no rejection loop, and depth stratification is preserved exactly.
        """
        from taxembed.eval.subtree import is_descendant

        anc_idxs = np.asarray(anc_idxs)
        bad = is_descendant(anc_idxs[:, None], negatives, self._tin, self._tout)
        rows = np.flatnonzero(bad.any(axis=1))

        for row in rows:
            a = int(anc_idxs[row])
            pool = self._depth_pools_by_tin.get(int(desc_depths[row]))
            n_bad = int(bad[row].sum())
            if pool is None or len(pool) == 0:
                self._zero_pool_hits += 1
                continue
            pool_tin = self._tin[pool]
            lo = int(np.searchsorted(pool_tin, self._tin[a], side="left"))
            hi = int(np.searchsorted(pool_tin, self._tout[a], side="left"))
            n_valid = len(pool) - (hi - lo)
            if n_valid <= 0:
                # No valid negative exists at this depth (the 3,026,809-pair census).
                # Count it; leave the row as-is rather than fabricating a false negative.
                self._zero_pool_hits += 1
                continue
            j = np.random.randint(0, n_valid, size=n_bad)
            j = np.where(j < lo, j, j + (hi - lo))
            negatives[row, bad[row]] = pool[j]

        self._realized_sum += int(negatives.shape[0] * negatives.shape[1])
        self._realized_count += int(negatives.shape[0])
        return negatives
```

Update the call site at `:638-641`:

```python
            if self.tiered_negatives:
                negatives = self._sample_negatives_tiered_vectorized(desc_depths, desc_idxs, batch_size)
            else:
                negatives = self._sample_negatives_default_vectorized(desc_depths, desc_idxs, batch_size)

            if self.exclude_descendant_negatives:
                negatives = self._resample_descendant_negatives(
                    negatives, self.pairs.ancestor_idx[batch_indices], desc_depths
                )
```

- [ ] **Step 6: Add the two CLI flags**

In `train_small.py`'s argparse near `:1066`:

```python
    parser.add_argument("--exclude-descendant-negatives", action="store_true",
                        help="Reject sampled negatives that are descendants of the anchor "
                             "(restores Nickel-Kiela's {v' : (u,v') not in D}).")
    parser.add_argument("--drop-root-anchored", action="store_true",
                        help="Exclude root-anchored pairs: every node descends from the root, "
                             "so they carry no contrastive signal.")
```

Guard beside the existing validators at `:1070-1085`:

```python
    if args.exclude_descendant_negatives and not args.drop_root_anchored:
        raise SystemExit(
            "--exclude-descendant-negatives requires --drop-root-anchored: all 1,102,162 "
            "root-anchored pairs (5.15%) have NO valid negative at any depth."
        )
    if args.drop_root_anchored and (args.tiered_negatives or args.class_balanced):
        raise SystemExit(
            "--drop-root-anchored is incompatible with --tiered-negatives/--class-balanced: "
            "both derive their indices from observed root-child pairs."
        )
```

Pass both into `HierarchicalDataLoader(...)` at `:1199-1208`. Mirror the flags in `cli/main.py` near `:708`, forward after `:388` via `train_cmd.extend([...])`, and record both in the metadata dict at `:457`.

- [ ] **Step 7: Run tests to verify they pass**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/tests/test_negative_sampling.py -v`

Expected: PASS, 8 tests.

- [ ] **Step 8: Confirm the control arm is genuinely untouched**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m pytest /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/tests -v`

Expected: **105 passed** (the pre-change baseline) plus the new tests. `_sample_negatives_default_vectorized` was not edited, so with both flags off the RNG draw sequence is bit-identical to history — which is what makes Task 8's A/B a real A/B.

- [ ] **Step 9: Commit**

```bash
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings add train_hierarchical.py train_small.py src/taxembed/cli/main.py tests/test_negative_sampling.py
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings commit -m "feat(train): interval-complement ancestry-aware negative sampling"
```

---

## Task 7 (NEW): Ancestry-of-anchor labelling in the hardness diagnostic

**Files:**
- Modify: `scripts/diagnose_negative_hardness.py`

**Why (spec P1.2, dropped from v1):** `diagnose_negative_hardness.py:105` calls `label_negatives(loader._node_class_arr, loader._node_gp_arr, desc, negs)` — labelling each negative against the **descendant**. The defect is that negatives are scored against the **anchor**. The diagnostic currently cannot see the thing it exists to measure.

- [ ] **Step 1: Add anchor-relative labelling**

Extend the diagnostic to report, per batch: the fraction of sampled negatives that are descendants of the **anchor** (using `is_descendant` from Task 1), broken out by anchor depth. This is the *observed* counterpart to Task 2's closed-form rate, and the two must agree.

- [ ] **Step 2: Verify observed matches closed form**

Run the diagnostic on echinodermata and confirm the observed false-negative fraction matches Task 2's closed-form prediction for that closure within sampling error.

- [ ] **Step 3: Commit**

```bash
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings add scripts/diagnose_negative_hardness.py
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings commit -m "fix(diagnostics): label negatives against the anchor, not the descendant"
```

---

## Task 8 (REWRITTEN): Fixed-vs-unfixed retrain under the canonical recipe

**Files:**
- Create: `docs/NUMBERS_OF_RECORD_objective_integrity.md`
- Create: `results/objective_integrity_delta.json`

**v1 defect fixed:** v1's retrain used `--n-negatives 50` and none of the canonical recipe. `run.json` records `curriculum: true`, `curriculum_phases: "auto"`, `epoch_fraction: 0.3`, `radial_schedule: "log"`, `depth_scale_margin: true`, `lr_schedule: "cosine_warmrestart"`, `warm_restart_on_phase: true`, `n_negatives: 300`, `grad_accum_steps: 8`, `amp: true`. Testing a recipe the paper doesn't ship answers nothing — and `epoch_fraction` + `curriculum` are exactly the two consumers of `_depth_diff_masks` that v1's drop implementation broke.

- [ ] **Step 1: Static checks (Rule 16)**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m py_compile /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/train_small.py`

Run the same for `train_hierarchical.py` and `src/taxembed/cli/main.py`. Expected: no output. Confirm every flag the runs pass is a real argparse option in **`train_small.py`**.

- [ ] **Step 2: Audit the echinodermata closure**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/scripts/audit_negative_sampling.py --npz /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/data/taxopy/echinodermata_7586_clean/taxonomy_edges_echinodermata_7586_clean_transitive.npz --out /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/results/negative_sampler_audit_echinodermata.json`

(Paths verified present: 34,938 pairs, 3,965 nodes, matching the mapping file.) Record the overall rate and root-anchored count — they bound the effect size the retrain can show.

- [ ] **Step 3: Smoke run, 2 epochs, canonical recipe**

Run: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/.venv/bin/python -m taxembed.cli.main train --file /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/data/taxopy/echinodermata_7586_clean/taxonomy_edges_echinodermata_7586_clean_transitive.npz --mapping /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/data/taxopy/echinodermata_7586_clean/taxonomy_edges_echinodermata_7586_clean.mapping.tsv -as smoke_fixed_neg --dim 100 --epochs 2 --batch-size 256 --n-negatives 300 --lr 0.001 --loss softmax --euclidean-param --curriculum --curriculum-phases auto --epoch-fraction 0.3 --radial-schedule log --depth-scale-margin --margin-min 0.05 --margin-max 1.0 --lr-schedule cosine_warmrestart --warm-restart-on-phase --seed 0 --exclude-descendant-negatives --drop-root-anchored`

Expected: completes; the exclusion line prints a non-zero count; no `IndexError` (this is the case v1 broke).

- [ ] **Step 4: Paired runs — six total, seeds 0/1/2 per arm**

Same command with `--epochs 200`. **Arm A (unfixed):** omit both new flags, tag `echino_unfixed_s<S>`. **Arm B (fixed):** include both, tag `echino_fixed_s<S>`.

- [ ] **Step 5: Record the delta**

Into `results/objective_integrity_delta.json`, per arm and seed: final training loss; depth-norm r **with the Task 4 initialization floor beside it**; kNN retrieval precision@10; per-epoch mean realized negatives (`_realized_sum / _realized_count`); `_zero_pool_hits`; and the Task 3 null-sanity verdict for any null used. Report mean and spread across seeds.

The seed spread is a deliverable in its own right: the paper's Fig-4 recipe contrast currently rests on n=1 per arm.

- [ ] **Step 6: Second clade (spec P1.4)**

Repeat Steps 3-5 on one mid-size clade from `data/taxopy/` (mollusca or arthropoda). Spec P1.4 asks for two scales; v1 had no step for the second.

- [ ] **Step 7: Write the decision record**

`docs/NUMBERS_OF_RECORD_objective_integrity.md` collects: Task 2's census; Task 4's initialization floor (**log 0.957151 vs trained 0.957244 — training moves the depth↔norm correlation by ~0.0001**); Task 3's null diagnosis; Task 8's deltas. State the verdict as one of:

- *"The fix materially changes the numbers"* ⇒ the shipped 1.1M artifact was trained with a defective objective; Methods must say so and a re-train is scoped.
- *"The fix does not materially change the numbers"* ⇒ a robustness result worth reporting.

**Either way**, two Methods corrections are unconditional: the paper describes a clean Nickel-Kiela softmax while the sampler omits ancestry exclusion; and the reproducibility claim is unsupported until Task 5 lands.

- [ ] **Step 8: Commit**

```bash
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings add docs/NUMBERS_OF_RECORD_objective_integrity.md results/objective_integrity_delta.json results/negative_sampler_audit_echinodermata.json results/radial_init_floor.json
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings commit -m "docs: objective-integrity numbers of record and fixed-vs-unfixed verdict"
```

---

## Task 9 (NEW): Does the recipe claim survive on the unplanted metric?

**Files:**
- Create: `results/recipe_angular_comparison.json`
- Modify: `docs/NUMBERS_OF_RECORD_objective_integrity.md`

**Why this exists.** Task 8 compares *fixed vs unfixed sampler* under one recipe. It does **not** test the claim Figure 4 actually makes, which is *canonical recipe vs prior approach* — and that claim was only ever measured on **depth-norm r**, the metric Task 4 shows is fixed at initialization (log floor **0.957151**, trained 0.957244).

Figure 4 currently reads: prior approach peaks at +0.875 then collapses to +0.669 at the curriculum transition, while canonical stays flat at +0.984. If depth-norm is planted, a *collapse* means the prior recipe **destroyed structure it was handed** — which is a real finding, but a different one from "our recipe learns depth." And nobody has checked whether the two recipes differ at all on the metrics that measure learned structure.

**This is the single result most likely to change what the manuscript says.** It is clade-scale, not a 1.1M retrain.

**Arms.** Per the manuscript's own definition, "prior approach" = the softmax objective **without the scale-aware effective batch** (it retains the curriculum — the collapse curve has one). Take the exact flag set from Task 8 Step 3 for canonical; for prior, change only the effective-batch configuration. Do not vary anything else, and record both flag sets verbatim in the output JSON.

- [ ] **Step 1: Confirm what separates the two arms**

Read the manuscript's Methods and `docs/PROJECT_STATE.md` L227-232 (the documented milestone trajectory Figure 4 is built from) and write down the exact flag delta. If it is more than the effective batch, record every difference — spec v3 §6 already flags the ablation table as confounded (parametrization/loss confounded; effective batch differs in four hyperparameters), so state which of those four are in play here rather than claiming a clean one-factor contrast.

- [ ] **Step 2: Paired runs, seeded**

Echinodermata, seeds 0/1/2 per arm, `--epochs 200`, with `--exclude-descendant-negatives --drop-root-anchored` on **both** arms (so the objective defect is not confounded with the recipe contrast). Tags `echino_canonical_s<S>` and `echino_prior_s<S>`.

- [ ] **Step 3: Score on both metric families**

Into `results/recipe_angular_comparison.json`, per arm and seed:

*Planted family (expected to differ — this is what Fig 4 already shows):*
- depth-norm r, **with the Task 4 initialization floor beside it**

*Learned family (the open question):*
- kNN retrieval precision@10 against cophenetic distance
- family-level kNN purity
- per-rank separation ratio

- [ ] **Step 4: Pre-register the reading before looking**

Commit this interpretation to the JSON before running Step 2:

- **Canonical beats prior on the learned metrics, spreads separated** ⇒ Figure 4's claim holds, and the figure should be **re-plotted on a learned metric** because the current one measures the planted axis.
- **Indistinguishable on the learned metrics, differing only on depth-norm** ⇒ the recipe's demonstrated benefit is **preservation of the radial prior**, not acquisition of structure. Figure 4's caption and the Results sentence must say that, and the "collapse" language needs rewriting.
- **Prior beats canonical on the learned metrics** ⇒ escalate; the recipe claim does not survive and the manuscript needs restructuring before submission.

- [ ] **Step 5: Record the verdict**

Append to `docs/NUMBERS_OF_RECORD_objective_integrity.md` under a "Recipe claim" heading. Whatever the outcome, note that Figure 4's y-axis is a metric with a 0.957 floor and that the floor must appear in the caption.

- [ ] **Step 6: Commit**

```bash
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings add results/recipe_angular_comparison.json docs/NUMBERS_OF_RECORD_objective_integrity.md
git -C /Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings commit -m "feat(eval): recipe contrast scored on learned metrics, not the planted one"
```

---

## Deferred to a follow-up (not blocking)

`train_hierarchical.py:605-606`'s docstring says the epoch sampler draws "equally from each depth_diff level"; `:616` is proportional (`int(len(dd_indices) * epoch_fraction)`). The code is right, the docstring is wrong, and it will mis-model epoch composition for anyone reading it. One-line fix, no behaviour change.

## Follow-on plans

- **P2 — Held-out evaluation.** Ganea-style split (transitive reduction always in training; 5%/5% of non-basic edges; closure visibility swept 0/10/25/50%), the Vendrov trivial-closure baseline, level-stratified holdout at our own quartiles (Q1=11, Q3=28; 593,576 eligible nodes), RandomDAG control. **Downstream of P1.**
- **P3 — Temporal QC.** ~70% built already. **Independent of P1; can run in parallel.**
- **P4 — pLM showcase.** Scoring layer, matched head-to-head, resampling protocol.
- **TimeTree external anchor.** Scheduled, not optional.
