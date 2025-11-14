# Hyperparameter Reversion

## **Problem Identified**

After comparing old (epoch 28) vs current (epoch 36) models:

```
                    OLD         CURRENT       Change
Loss:               0.472       0.531         +12.5% ⬆️ WORSE
Boundary (>0.90):   27.5%       33.5%         +6% more cramped
```

**Root cause:** I over-corrected the hyperparameters when fixing the data coverage issue.

---

## **What I Changed (and shouldn't have)**

### **1. Regularization Strength (λ)**
- ❌ **Changed:** 0.1 → 0.01 (10x weaker)
- ✅ **Reverted:** Back to 0.1
- **Why:** 0.01 was too weak for 111K nodes, not enough structure enforcement

### **2. Initialization Range**
- ❌ **Changed:** [0.1, 0.95] → [0.05, 0.85]
- ✅ **Reverted:** Back to [0.1, 0.95]
- **Why:** Starting too conservative made learning harder

### **3. Hard Projection Boundary**
- ❌ **Changed:** max_norm = 0.99999 → 0.98
- ✅ **Reverted:** Back to 0.999
- **Why:** 0.98 caused artificial compression

---

## **What I Kept (The Real Fix)**

### ✅ **Complete Data Coverage**
- **Before:** 82,040 / 111,103 nodes (73.8%)
- **After:** 109,236 / 111,103 nodes (98.3%)
- **How:** Modified `build_transitive_closure.py` to add parent→node pairs for missing nodes

**This was the ONLY real permanent fix needed.**

---

## **Why This Happened**

I misdiagnosed the original "boundary compression" issue:

1. **Real problem:** 29,063 nodes had no training signal
   - They were initialized at depth=37 (boundary)
   - Never moved because no training pairs
   - Formed noisy cluster at boundary

2. **My incorrect diagnosis:** "Regularization too strong, init too aggressive"
   - Led to over-conservative hyperparameter changes
   - Made training worse, not better

3. **Actual solution needed:** Just fix the data coverage
   - Once all nodes get training signal, original hyperparams work fine
   - More nodes (111K vs 92K) needs STRONGER λ, not weaker

---

## **Files Changed**

### **Reverted:**
1. `train_small.py` line 321:
   ```python
   # Before: default=0.01
   # After:  default=0.1
   ```

2. `train_hierarchical.py` line 63:
   ```python
   # Before: target_radius = 0.05 + (depth / max_depth) * 0.80
   # After:  target_radius = 0.1 + (depth / max_depth) * 0.85
   ```

3. `train_hierarchical.py` line 326:
   ```python
   # Before: target_radius = 0.05 + (depth / max_depth) * 0.80
   # After:  target_radius = 0.1 + (depth / max_depth) * 0.85
   ```

4. `train_hierarchical.py` line 104:
   ```python
   # Before: max_norm=0.98
   # After:  max_norm=0.999
   ```

### **Kept:**
1. `build_transitive_closure.py`: `ensure_complete_coverage()` function
2. Data with 98.3% coverage (1,002,486 pairs)

---

## **Expected Results**

With reverted hyperparameters + complete data:

| Metric | OLD (incomplete data) | CURRENT (should match/beat) |
|--------|----------------------|------------------------------|
| Loss | 0.472 | ~0.45-0.47 (same or better) |
| Coverage | 82,040 nodes (73.8%) | 109,236 nodes (98.3%) ✅ |
| Boundary | 27.5% | ~25-30% (similar) |
| Structure | Good | Better (more nodes trained) |

---

## **Training Command**

```bash
# Now with correct hyperparameters
python train_small.py --epochs 100 --early-stopping 10
```

The default λ=0.1 will now be used (not 0.01).

---

## **Lessons Learned**

1. ✅ **Data fixes are permanent** - coverage fix was right
2. ❌ **Don't over-correct hyperparameters** - original values were fine
3. ✅ **Isolate changes** - should have tested data fix alone first
4. ✅ **Compare rigorously** - user caught this with checkpoint comparison
5. ✅ **Trust the numbers** - loss increase = something wrong

---

## **Summary**

**The only permanent fix needed:**
- ✅ Ensure all nodes appear in training data (98.3% coverage)

**Everything else I changed:**
- ❌ Was over-correction based on misdiagnosis
- ✅ Now reverted to original working values

**Result:**
- Complete data coverage (scales to any size) ✅
- Original proven hyperparameters ✅
- Should now match or beat old model performance ✅
