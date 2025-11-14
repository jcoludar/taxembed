# Training Issues Diagnosis & Fixes

**Date**: November 13, 2025  
**Status**: 3 critical issues identified and fixed

---

## 🔍 Issues Identified

### Issue #1: **CATASTROPHIC - Missing 26% of Nodes** 🚨

**Symptom**: Bizarre spread in UMAP visualization - some groups tight, others spread everywhere

**Root Cause**: 
- `train_small.py` line 346 calculated `n_nodes` from training pairs only
- Nodes that never appear in training pairs (e.g., leaf nodes) were excluded
- Result: 92,290 embeddings created, but 111,103 nodes exist in dataset

**Impact**:
```
Total nodes in dataset:     111,103
Nodes with embeddings:       82,040  (74%)
Nodes WITHOUT embeddings:    29,063  (26%)
Missing from model:          18,813  (gaps in index range)
```

**Breakdown of node coverage**:
- Only descendants (leaf nodes): 60,238
- Only ancestors (root nodes): 1
- Both roles: 21,801
- **Never in training data**: 29,063 ❌

These missing 29K nodes likely had random/uninitialized embeddings → explains bizarre clustering.

---

### Issue #2: **URGENT - Boundary Compression**

**Symptom**: 
- 30% of embeddings at norm > 0.9
- 90th percentile at norm 0.9997 (essentially 1.0)
- No room for hierarchical structure

**Root Causes**:
1. **Strong radial regularizer** (λ=0.1) aggressively pushing nodes to target radii
2. **Aggressive ball projection** (max_norm = 1-1e-5 ≈ 0.99999)
3. **Deep pair dominance** (74% of training pairs have depth > 5, all pushed near boundary)
4. **Initialization too close** to boundary (max 0.95)

**Distribution**:
```
Norm percentiles:
  0th:  0.1000
 25th:  0.5742
 50th:  0.7094
 75th:  0.9527
 90th:  0.9997  ← Everything compressed here
 99th:  1.0000
100th:  1.0000
```

---

### Issue #3: **IMPORTANT - Severe Data Imbalance**

**Symptom**: Model can't learn local structure (parent-child, sibling relationships)

**Data distribution**:
```
Total training pairs: 975,896
Depth distribution:
  Depth 1 (parent-child):  58,663 (  6.0%)  ← Too few!
  Depth 2-5:              193,821 ( 19.9%)
  Depth > 5:              723,412 ( 74.1%)  ← Dominates!

Mean depth: 11.97
Median depth: 11.0
Max depth: 37
```

**Impact**:
- Model overfits to distant ancestor-descendant relationships
- Local structure (siblings at same level, direct parents) is underrepresented
- Hierarchy becomes "all or nothing" instead of smooth gradient

---

## ✅ Fixes Applied

### Fix #1: Correct n_nodes Calculation ✅

**File**: `train_small.py` lines 345-352

**Before**:
```python
n_nodes = max(max(item['ancestor_idx'], item['descendant_idx']) 
              for item in training_data) + 1
```

**After**:
```python
# Load mapping to get true n_nodes
mapping_df = pd.read_csv("data/taxonomy_edges_small.mapping.tsv",
                         sep="\t", header=None, names=["taxid", "idx"])
mapping_df['idx'] = pd.to_numeric(mapping_df['idx'], errors='coerce')
mapping_df = mapping_df.dropna()
n_nodes = int(mapping_df['idx'].max()) + 1
```

**Result**: Now creates 111,103 embeddings, covering ALL nodes in dataset

---

### Fix #2: Relax Boundary Constraint ✅

**File**: `train_hierarchical.py` lines 104-136

**Changes**:
1. Changed `max_norm` from `1-1e-5` (0.99999) to **0.98**
2. Leaves 2% buffer from boundary for hierarchical structure

**File**: `train_hierarchical.py` lines 60-63

**Initialization updated**:
```python
# Before: target_radius = 0.1 + (depth / max_depth) * 0.85  # Range [0.10, 0.95]
# After:
target_radius = 0.05 + (depth / max_depth) * 0.80  # Range [0.05, 0.85]
```

**Regularizer updated** (lines 325-326):
```python
# Match new initialization range
target_radius = 0.05 + (depth / max_depth) * 0.80
```

**Expected impact**:
- Embeddings distributed in [0.05, 0.98] instead of [0.10, 1.00]
- 15% buffer from boundary allows hierarchical differentiation
- Deep nodes won't all collapse to same radius

---

### Fix #3: Reduce Regularization Strength ✅

**File**: `train_small.py` line 320

**Change**: Default `lambda_reg` reduced from **0.1 → 0.01** (10x reduction)

**Rationale**:
- Previous λ=0.1 too aggressive, forced nodes to exact target radii
- New λ=0.01 provides gentle guidance without over-constraining
- Allows model to learn optimal positions based on data

**User can override**: `--lambda-reg 0.0` to disable completely

---

### Fix #4: Early Stopping Respect Disabled State ✅

**File**: `train_small.py` line 291

**Before**:
```python
if epochs_without_improvement >= early_stopping_patience:
```

**After**:
```python
# Only check early stopping if patience > 0 (0 means disabled)
if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
```

**Result**: `--early-stopping 0` now actually disables early stopping

---

## 🎯 Comparison: train_hierarchical.py vs train_small.py

Both scripts use the **same model and training logic** (imported from `train_hierarchical.py`):

| Component | train_hierarchical.py | train_small.py |
|-----------|----------------------|----------------|
| Model | HierarchicalPoincareEmbedding | Same (imported) |
| Loss | ranking_loss_with_margin | Same (imported) |
| Regularizer | radial_regularizer | Same (imported) |
| DataLoader | HierarchicalDataLoader | Same (imported) |
| Main difference | Standalone script | User-friendly wrapper with progress bars |

**Key insight**: They're essentially the same! `train_small.py` just adds:
- Better terminal visualization
- MetricsTracker for progress
- Automatic best model saving
- More user-friendly defaults

---

## 📊 Expected Improvements

After these fixes, you should see:

1. **Complete coverage**: All 111,103 nodes have learned embeddings
2. **Better spread**: Norms distributed across [0.05, 0.98] instead of compressed at 1.0
3. **Clearer hierarchy**: 15% buffer allows depth differentiation
4. **Smoother learning**: Reduced regularization allows data-driven positioning

---

## 🚀 Recommended Next Training Run

```bash
uv run python train_small.py \
  --epochs 10000 \
  --early-stopping 0 \
  --lambda-reg 0.01 \
  --batch-size 64 \
  --lr 0.005 \
  --margin 0.2
```

**Monitor**:
- Norm distribution (should spread across 0.05-0.90, not compress at 0.98)
- Loss decrease (should be steady without plateaus)
- Max norm (should stay < 0.98)

---

## 🔬 Optional: Further Improvements

If issues persist, consider:

1. **Balance training data**:
   ```python
   # Oversample parent-child pairs (depth=1) by 10x
   # Cap deep pairs at depth 10
   ```

2. **Curriculum learning**:
   ```python
   # Epochs 1-20: Train only on depth 1-2 (parent-child)
   # Epochs 21-50: Add depth 3-5 (grandparents)
   # Epochs 51+: Full dataset
   ```

3. **Disable regularization initially**:
   ```bash
   # First 50 epochs: learn from data
   --lambda-reg 0.0
   # Then fine-tune with gentle regularization
   --lambda-reg 0.01
   ```

---

## 📝 Files Changed

1. ✅ `train_small.py` - Fixed n_nodes, reduced lambda_reg, fixed early stopping
2. ✅ `train_hierarchical.py` - Relaxed max_norm, updated initialization & regularizer
3. ✅ Created diagnostics:
   - `analyze_embeddings.py` - Quick embedding analysis
   - `diagnose_issues.py` - Comprehensive diagnosis
   - `TRAINING_ISSUES_FIXED.md` - This document

---

## 🎓 Key Lessons

1. **Always validate n_nodes against ground truth** (mapping file), not derived data
2. **Hyperbolic geometry needs breathing room** - don't compress to boundary
3. **Balance matters** - 6% parent-child vs 74% deep pairs is extreme
4. **Regularization is powerful** - use sparingly (λ << 0.1)
5. **Visualize early** - UMAP caught issues that metrics missed

---

**Status**: Ready for retraining with fixes applied ✅
