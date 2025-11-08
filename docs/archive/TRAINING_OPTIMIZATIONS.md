# Training Optimizations Applied

## Pre-Training Efficiency Audit (Nov 8, 2025)

### Critical Fixes

#### 1. **Device Selection** ✅
- **Issue:** Auto-selected MPS (Metal) on M3 Mac → hung during training
- **Fix:** Force CPU when GPU not explicitly requested
- **Impact:** Training can now start and progress
- **Code:** Lines 391-399 in `train_hierarchical.py`

#### 2. **Radial Regularizer** ✅ **MAJOR**
- **Issue:** Python loop over 111,103 nodes per batch
  - 15,248 batches/epoch × 111,103 nodes = **1.7 BILLION operations/epoch**
- **Fix:** Vectorized with precomputed tensors
  - Compute index list and target radii ONCE at start
  - Use PyTorch tensor operations (GPU/vectorized)
- **Impact:** ~100-1000x speedup on regularizer
- **Code:** Lines 238-261, 279-291

#### 3. **Projection Operation** ✅ **MAJOR**
- **Issue:** Projected ALL 111,103 embeddings every batch
  - Only ~64 ancestors + 64 descendants + 3,200 negatives = ~3,328 unique nodes updated per batch
  - Wasted effort on 107,775 unchanged nodes (97% wasted!)
- **Fix:** Only project embeddings that were modified in batch
- **Impact:** ~30x speedup on projection
- **Code:** Lines 103-122, 352-356

#### 4. **Tensor Creation** ✅
- **Issue:** Creating tensors from list of numpy arrays (slow warning)
- **Fix:** Pre-allocate numpy array, fill it, convert once
- **Impact:** ~10-100x faster negative sampling, cleaner output
- **Code:** Lines 187-208

#### 5. **Data Loading** ✅
- **Issue:** List comprehensions creating intermediate Python objects
- **Fix:** Pre-allocate numpy arrays, fill directly, convert to tensors once
- **Impact:** Minor speedup, cleaner code
- **Code:** Lines 171-185

#### 6. **Loop Variable Bug** ✅ **BUGFIX**
- **Issue:** Shadowed loop variable `i` (used in outer and inner loop)
- **Fix:** Renamed inner loop variable to `j`
- **Impact:** Prevents potential indexing bugs
- **Code:** Lines 177-206

---

## Performance Comparison

### Before Optimizations
- **Projection:** 111,103 nodes × 15,248 batches = 1.7B operations/epoch
- **Regularizer:** 111,103 nodes × 15,248 batches = 1.7B operations/epoch
- **Device:** MPS (hangs with hyperbolic ops)
- **Estimated time:** ∞ (hangs)

### After Optimizations
- **Projection:** ~3,328 nodes × 15,248 batches = 50M operations/epoch (97% reduction)
- **Regularizer:** 111,103 nodes × 1 = 111K operations/epoch (99.99% reduction)
- **Device:** CPU (stable)
- **Estimated time:** ~1-2 minutes/epoch

**Overall speedup:** ~100-1000x (from hung to practical)

---

## Training Configuration

### Data
- **Training pairs:** 975,896 (transitive closure)
  - Parent-child: 58,663 (6%)
  - Grandparent: 52,953 (5%)
  - Deep ancestors: 864,280 (89%)
- **Nodes:** 111,103
- **Max depth:** 38

### Hyperparameters
- **Batch size:** 64
- **Batches per epoch:** 15,248
- **Negative samples:** 50 (hard negatives from same depth)
- **Learning rate:** 0.01
- **Margin:** 0.2
- **Radial regularization:** λ=0.01
- **Early stopping:** patience=3 epochs
- **Max epochs:** 10,000

### Expected Training
- **Time per epoch:** ~1-2 minutes (CPU)
- **Total epochs:** ~10-50 (early stopping)
- **Total time:** ~20-100 minutes
- **Memory:** ~500MB

---

## Key Algorithmic Improvements (vs. Old Training)

| Feature | Old Training | New Training |
|---------|-------------|--------------|
| Training pairs | 100K (parent-child only) | 975K (all ancestors) |
| Hierarchy encoding | None | Depth → radius initialization |
| Negative sampling | Random | Hard (same-depth cousins) |
| Depth weighting | None | √depth weighting |
| Radial regularization | None | λ=0.01 penalty |
| Early stopping | Patience=6 | Patience=3 |

---

## Verification Checklist

- [x] CPU device (not MPS)
- [x] Vectorized regularizer
- [x] Selective projection (only modified embeddings)
- [x] Efficient tensor creation
- [x] No loop variable shadowing
- [x] Hard negative sampling
- [x] Depth-aware initialization
- [x] Early stopping (patience=3)

---

## Expected Results

After training, run `analyze_hierarchy_hyperbolic.py`. You should see:

**OLD MODEL (broken):**
- Depth-norm correlation: r = -0.002 ❌
- Phylum separation: 1.00x ❌
- Class separation: 1.01x ❌

**NEW MODEL (expected):**
- Depth-norm correlation: r > 0.5 ✅
- Phylum separation: > 1.5x ✅
- Class separation: > 1.5x ✅

If separation is still low:
1. Increase `--lambda-reg` (0.05 or 0.1)
2. Increase `--margin` (0.3 or 0.5)
3. Train longer (disable early stopping temporarily)
4. Check that depth initialization worked (norms should vary)

---

## Checkpoint Management

### Automatic Saving
- ✅ **Every epoch:** `model_epoch{N}.pth` (keeps last 5, deletes older)
- ✅ **Best model:** `model_best.pth` (updated when loss improves)
- ✅ **Final model:** `model.pth` (at end of training)

### Files Created
```
taxonomy_model_hierarchical_small_epoch1.pth
taxonomy_model_hierarchical_small_epoch2.pth
taxonomy_model_hierarchical_small_epoch3.pth
taxonomy_model_hierarchical_small_epoch4.pth
taxonomy_model_hierarchical_small_epoch5.pth  (oldest 5th deleted after epoch 6)
taxonomy_model_hierarchical_small_best.pth    (always kept - best loss)
taxonomy_model_hierarchical_small.pth         (final model at end)
```

### Why This Matters
- ✅ **No progress loss:** If training crashes, resume from last epoch
- ✅ **Disk space:** Only keeps last 5 + best (not all epochs)
- ✅ **Best model:** Separate file for the best performing epoch

---

## Ready to Train!

```bash
source venv311/bin/activate
python train_hierarchical.py \
    --data data/taxonomy_edges_small_transitive.pkl \
    --checkpoint taxonomy_model_hierarchical_small.pth \
    --dim 10 \
    --epochs 10000 \
    --early-stopping 3 \
    --batch-size 64 \
    --n-negatives 50 \
    --lr 0.01 \
    --margin 0.2 \
    --lambda-reg 0.01 \
    --gpu -1
```
