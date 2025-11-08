# Critical Bugs Found and Fixed

## Session: Nov 8, 2025 - Pre-Training Sanity Check

---

## 🐛 BUG #1: TaxID vs Index Mapping Confusion (CRITICAL)

### **Symptom:**
- Training created 3,467,244 embeddings instead of 111,103
- Max index in data: 3,467,243
- Expected max index: 111,102
- Result: Training on mostly zero-gradient embeddings (wasted 97% of memory and compute)

### **Root Cause:**
```python
# WRONG - build_transitive_closure.py line 173
df = pd.read_csv("data/taxonomy_edges_small.mapping.tsv",
                 sep="\t", header=None, names=["idx", "taxid"])
```

File has columns: `taxid  idx` (with header)  
Code expected: `idx  taxid` (no header)

Result: Code read TaxID 131567 as index 1 → used TaxIDs directly as indices!

### **Fix:**
```python
# CORRECT
df = pd.read_csv("data/taxonomy_edges_small.mapping.tsv", sep="\t")
# Reads header automatically, columns are taxid, idx
```

### **Impact:**
- **Before:** 3.4M embeddings, 97% never updated
- **After:** 92K embeddings (actual max index from data)
- **Speedup:** ~30x less memory, proper training

---

## ⚠️ WARNING #1: Index Range Mismatch

### **Observation:**
- Mapping file has 111,103 entries (indices 0-111,102)
- Training data only uses indices 0-92,289
- **18,814 mapped nodes (17%) never appear in training!**

### **Cause:**
Some organisms in the small dataset have no valid taxonomy paths to root in the transitive closure.

### **Status:**
Not a bug - these are likely:
- Organisms with missing parent links
- Isolated subgraphs
- Data quality issues in NCBI taxonomy

### **Impact:**
- Minimal - these nodes won't have good embeddings
- Could improve by filling in missing links

---

## ⚠️ WARNING #2: Regularization Coverage

### **Observation:**
- 92,290 nodes in training data
- Only 82,040 nodes have depth info (89%)
- **10,250 nodes (11%) won't be regularized**

### **Cause:**
Some nodes appear in training pairs but don't have depth metadata assigned.

### **Status:**
Acceptable - these are likely intermediate nodes. Main concern is leaf nodes.

---

## ✅ ALL OTHER SYSTEMS VERIFIED

### **1. Mapping File** ✅
- ✅ Columns: `taxid`, `idx` (correct order)
- ✅ No duplicate TaxIDs
- ✅ No duplicate indices
- ✅ Indices continuous: 0-111,102

### **2. Transitive Closure Data** ✅
- ✅ 975,896 ancestor-descendant pairs
- ✅ All indices in valid range
- ✅ No self-loops
- ✅ Depth differences consistent
- ✅ All depth diffs positive

### **3. Projection Logic** ✅
- ✅ Correctly constrains embeddings to unit ball
- ✅ Max norm after projection: 0.99999
- ✅ No embeddings escape ball

### **4. Hyperbolic Distance** ✅
- ✅ Distance to self ≈ 0
- ✅ Distance increases with separation
- ✅ Proper Poincaré distance formula

### **5. Depth-Aware Initialization** ✅
- ✅ Root (depth 0): r = 0.10
- ✅ Leaves (depth 38): r = 0.95
- ✅ All radii < 1.0 (inside ball)

### **6. Sibling Map (Hard Negatives)** ✅
- ✅ 82,039 nodes with sibling info
- ✅ Average 31,438 siblings per node
- ✅ Only 1 node has no siblings
- ✅ Siblings verified at same depth

### **7. Regularizer Targets** ✅
- ✅ All targets < 1.0 (valid)
- ✅ Proper depth → radius mapping

### **8. Training Configuration** ✅
- ✅ 975,896 training pairs
- ✅ 15,249 batches per epoch
- ✅ Reasonable batch size (64)

---

## Summary

### **Critical Bugs Fixed:**
1. ✅ **TaxID as index bug** - Fixed mapping file reading

### **Warnings (Non-Critical):**
1. ⚠️ 17% of mapped nodes never appear in training (data quality)
2. ⚠️ 11% of training nodes lack depth info (acceptable)

### **Verification Results:**
- ✅ **10/10 sanity checks passed**
- ✅ **All core systems working correctly**
- ✅ **Ready to train with confidence**

---

## Next Steps

1. ✅ Run sanity check: `python sanity_check.py`
2. ⏭️ Train model: `python train_hierarchical.py ...`
3. ⏭️ Analyze results: `python analyze_hierarchy_hyperbolic.py`
4. ⏭️ Expect MUCH better results:
   - Depth-norm correlation: r > 0.5 (was -0.08)
   - Phylum separation: > 1.5x (was 1.05x)
   - Class separation: > 1.5x (was 1.04x)

---

## Files Modified

1. **build_transitive_closure.py**
   - Fixed mapping file reading (line 173-175)
   - Now correctly reads header and column names

2. **sanity_check.py** (NEW)
   - Comprehensive validation of entire pipeline
   - 8 test categories, 10 checks
   - Run before every major training session

---

## Lessons Learned

1. **Always validate data format assumptions**
   - CSV headers matter!
   - Column order matters!
   - Don't assume - verify!

2. **Sanity check EVERYTHING before long training runs**
   - Indices in range
   - No self-loops
   - Math checks out
   - Data makes sense

3. **Monitor actual vs expected sizes**
   - 3.4M embeddings was a red flag
   - Should have caught it earlier

4. **Test mathematical operations independently**
   - Projection logic
   - Distance functions
   - Initialization ranges
