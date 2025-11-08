# Session Summary - Nov 8, 2025
## Hierarchical Poincaré Embedding Training

---

## 🎯 **What We Accomplished**

### **1. Critical Bug Fixed**
✅ **TaxID as Index Bug** - The transitive closure was using TaxIDs as embedding indices
- Before: 3.4M embeddings created (97% wasted)
- After: 92K embeddings (correct!)
- Impact: Training now uses proper mapped indices

### **2. Ball Constraint Enforced**
✅ **3-Layer Enforcement Strategy** implemented to keep all embeddings inside Poincaré ball
- Before: Max norm = 2.18, 54% outside ball
- After: Max norm = 1.00, 0% outside ball
- **100% compliance achieved!**

### **3. Comprehensive Validation**
✅ **Sanity Check Script** created - validates entire pipeline
- 10/10 checks passed
- Tests: mapping, data, projection, distance, initialization, etc.

---

## 📊 **Training Results**

### **Version Progression:**

| Version | Max Norm | Outside Ball | Status |
|---------|----------|--------------|--------|
| **v1** | 2.18 | 50K (54%) | ❌ Broken |
| **v2** | 1.45 | 2K (2.2%) | ⚠️ Better |
| **v3** | 1.00 | 0 (0%) | ✅ Perfect Constraint |

### **v3 Training (Best - 2 epochs):**
```
Loss: 0.577
Regularization: 0.010
Min norm: 0.087
Mean norm: 0.618
Max norm: 1.000 ✅
Coefficient of variation: 0.386 ✅
```

---

## ❌ **The Problem: Poor Hierarchy Quality**

Despite perfect ball constraints, **hierarchy encoding is still poor**:

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Depth-norm correlation | > 0.5 | **+0.003** | ❌ |
| Phylum separation | > 1.5x | **1.08x** | ❌ |
| Class separation | > 1.5x | **0.99x** | ❌ |
| Order separation | > 1.5x | **0.99x** | ❌ |

**Conclusion:** Embeddings are mathematically valid but **don't encode hierarchy**!

---

## 🔍 **Root Cause Analysis**

### **Why Is Hierarchy Not Learned?**

1. **Training Time:** Only 2 epochs completed
   - Need more epochs for convergence
   - Early stopping patience = 3 (may stop too soon)

2. **Data Imbalance:** 975K pairs, but:
   - Parent-child: 58K (6%)
   - Deep ancestors: 917K (94%)
   - Model may focus on deep pairs, ignore local structure

3. **Regularization Too Strong?** λ=0.1
   - Prevents embeddings from escaping
   - May also prevent learning good separation
   - Trade-off: constraint vs. expressiveness

4. **Hard Negatives Not Effective?**
   - Sibling sampling: avg 31K siblings per node
   - Too many siblings → samples not informative?
   - Need smarter negative sampling

5. **Ranking Loss Margin:** 0.2
   - May be too small for hyperbolic space
   - Need larger margin for deeper hierarchy?

---

## 💡 **Recommendations**

### **Option A: Train Longer (Easiest)**
```bash
# Continue from checkpoint with more patience
python train_hierarchical.py \
    --checkpoint taxonomy_model_hierarchical_small_v3_best.pth \
    --epochs 10000 \
    --early-stopping 10  # Increase patience from 3 to 10
```

**Pros:** Simple, may just need more time  
**Cons:** If fundamental issue, won't help

### **Option B: Stronger Hierarchy Signal**
```bash
# Reduce regularization, increase margin
python train_hierarchical.py \
    --lambda-reg 0.05  # Half as strong (was 0.1)
    --margin 0.5       # Larger separation (was 0.2)
    --lr 0.01          # Faster learning (was 0.005)
```

**Pros:** More freedom to learn hierarchy  
**Cons:** May violate ball constraint again

### **Option C: Balanced Sampling**
Modify dataloader to sample equal amounts from each depth level:
- 10% parent-child
- 10% grandparent
- 80% deeper ancestors (stratified by depth)

**Pros:** Forces model to learn local structure  
**Cons:** Requires code changes

### **Option D: Progressive Training**
1. **Phase 1:** Train on parent-child only (learn local structure)
2. **Phase 2:** Add grandparent pairs
3. **Phase 3:** Add all ancestor pairs

**Pros:** Curriculum learning, builds hierarchy bottom-up  
**Cons:** More complex, 3x training time

### **Option E: Use Existing Working Model**
```bash
# The old simple training worked for 2.7M organisms
# Maybe hierarchical features are overkill?
python embed.py -dset data/taxonomy_edges.mapped.edgelist ...
```

**Pros:** Known to work, simpler  
**Cons:** Misses depth-aware features

---

## 📝 **Files Created**

1. **BUGS_FOUND_AND_FIXED.md** - Documents critical TaxID bug
2. **TRAINING_FIXES_APPLIED.md** - v1→v2 improvements
3. **BALL_CONSTRAINT_ENFORCEMENT.md** - 3-layer strategy
4. **sanity_check.py** - Comprehensive validation script
5. **train_hierarchical.py** - Hierarchical training with all fixes
6. **TRAINING_OPTIMIZATIONS.md** - Performance improvements

---

## 🎓 **Lessons Learned**

### **1. Data Quality Matters More Than Model Complexity**
- TaxID bug wasted hours of debugging
- Always validate: indices, shapes, ranges
- Use sanity checks before training

### **2. Constraints Are Hard**
- Poincaré ball constraint (||x|| < 1) is non-trivial
- Need multiple enforcement layers
- Trade-off: constraint vs. optimization freedom

### **3. Hierarchy Encoding Is Subtle**
- Just because embeddings are "valid" doesn't mean they're "good"
- Need right balance of:
  - Data (what pairs to train on)
  - Loss (how to measure quality)
  - Regularization (what to encourage)

### **4. Start Simple, Add Complexity**
- Old simple model worked
- New complex model has perfect constraints but poor quality
- Sometimes simpler is better

---

## 🚀 **Next Steps**

### **Immediate (Recommended):**
1. ✅ Try **Option A**: Train longer with patience=10
2. ⏭️ If no improvement, try **Option B**: Weaker regularization
3. ⏭️ Monitor depth-norm correlation each epoch

### **If Still Poor:**
1. Investigate data distribution
2. Visualize embeddings (UMAP)
3. Check if ANY pairs are learned correctly
4. Consider going back to simple model

### **Research Questions:**
1. Is transitive closure helping or hurting?
2. Are hard negatives too hard?
3. Is λ=0.1 too constraining?
4. Does ranking loss need different margin for different depths?

---

## 📈 **Current Status**

| Component | Status | Notes |
|-----------|--------|-------|
| Data pipeline | ✅ Fixed | TaxIDs mapped correctly |
| Ball constraints | ✅ Perfect | 100% inside ball |
| Training stability | ✅ Good | No crashes, gradients clipped |
| Hierarchy quality | ❌ Poor | Not encoding depth structure |
| **Overall** | 🟡 | Technically correct, semantically poor |

---

## 💾 **Best Checkpoints**

- `taxonomy_model_hierarchical_small_v3_best.pth` (2 epochs)
  - Loss: 0.577
  - Max norm: 1.000 ✅
  - Hierarchy: Poor ❌

---

## 🔬 **Hypothesis**

The model is learning to satisfy the constraints (ball + regularization) but **NOT** learning hierarchy because:

1. **Regularizer dominates:** λ=0.1 is 10-20% of total loss
2. **Projection resets progress:** Every batch, embeddings pushed back
3. **No room to separate:** All embeddings squeezed near radius ~ 0.6

**Test:** Try λ=0.01 (original) with improved projection, see if hierarchy improves.

---

## 📞 **Summary for User**

**Good News:**
- ✅ Found and fixed critical bug (TaxID mapping)
- ✅ Enforced ball constraints (100% compliance)
- ✅ Training is stable and fast (~3 min/epoch)

**Bad News:**
- ❌ Model doesn't learn hierarchy well
- ❌ Only trained 2 epochs (stopped early)
- ❌ Need to investigate why

**Recommendation:**
Train longer first (Option A), then tune hyperparameters if needed.
