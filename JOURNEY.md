# Development Journey: From Facebook's Poincaré Embeddings to Hierarchical Taxonomy Embeddings

## Overview

This document chronicles the evolution of this project from Facebook Research's original Poincaré embeddings implementation to a specialized hierarchical taxonomy embedding system for NCBI's biological taxonomy.

---

## Phase 1: Foundation - Facebook's Poincaré Embeddings

### **Starting Point**
We began with Facebook Research's implementation of Poincaré embeddings, as described in their 2017 paper "Poincaré Embeddings for Learning Hierarchical Representations."

**Original Features:**
- Hyperbolic geometry (Poincaré ball model)
- Designed for hierarchical data (e.g., WordNet)
- Pure parent-child edge training
- Simple negative sampling

**Initial Success:**
- Successfully trained 2.7M NCBI taxonomy organisms
- Model converged in ~1 hour on CPU
- Basic hierarchical structure preserved

### **Key Insight**
The Facebook implementation worked but was designed for simpler hierarchies. We needed enhancements for biological taxonomy with 38 depth levels and complex relationships.

---

## Phase 2: Data Quality Issues (Nov 5-7, 2025)

### **Critical Bugs Discovered**

#### **Bug #1: Header Lines in Edgelist Files**
```
Problem: .edgelist files had "id1 id2" header treated as real organism IDs
Result: 111,105 nodes (2 fake + 111,103 real)
Fix: Updated prepare_taxonomy_data.py to skip headers
Impact: Clean datasets with correct node counts
```

#### **Bug #2: Mapping File Inconsistencies**
```
Problem: .mapping.tsv had fake TaxIDs "id1" and "id2" at indices 0 and 1
Fix: Regenerated mappings without header artifacts
Impact: Proper TaxID → index mapping
```

#### **Bug #3: No Data Validation**
```
Solution: Created scripts/validate_data.py
Features:
  - Validates node counts
  - Checks for duplicates
  - Verifies index continuity
  - Reports statistics
```

**Files Created:**
- `scripts/validate_data.py`
- `scripts/regenerate_data.sh`
- `DATA_FIXES_SUMMARY.md`

---

## Phase 3: Hierarchical Training Attempt #1 (Nov 7-8, 2025)

### **Motivation**
Simple parent-child training doesn't capture full hierarchy. Organisms at depth 20 should be "far" from root in hyperbolic space.

### **Strategy: Transitive Closure**
Instead of 100K parent-child edges, train on ALL ancestor-descendant pairs:

```
Parent-child:     58,663 pairs (6%)
Grandparent:      52,953 pairs (5%)
Deep ancestors:  864,280 pairs (89%)
Total:           975,896 pairs (9.8x more data!)
```

### **Implementation**
Created `build_transitive_closure.py`:
- Loads NCBI taxonomy tree
- Computes all ancestor paths
- Adds depth metadata
- Outputs 975K training pairs

### **Critical Bug: TaxID as Index**
```
Problem: Used TaxIDs directly as embedding indices
Result: Created 3,467,244 embeddings (instead of 111,103)
Symptom: 97% of embeddings never updated!
Fix: Properly read mapping file with header
Impact: Reduced to 92,290 embeddings (correct)
```

**Files Created:**
- `build_transitive_closure.py`
- `BUGS_FOUND_AND_FIXED.md`
- `sanity_check.py` (comprehensive validation)

---

## Phase 4: Hierarchical Features (Nov 8, 2025)

### **New Training Features Implemented**

#### **1. Depth-Aware Initialization**
```python
# Root (depth 0): radius ≈ 0.1 (near center)
# Leaves (depth 38): radius ≈ 0.95 (near boundary)
target_radius = 0.1 + (depth / max_depth) * 0.85
```

#### **2. Radial Regularization**
```python
# Soft penalty to encourage depth → radius mapping
reg_loss = λ * (actual_radius - target_radius)²
```

#### **3. Hard Negative Sampling**
```python
# Sample negatives from same depth (cousins, not random)
siblings = nodes_at_same_depth(node)
negatives = random.sample(siblings, n_negatives)
```

#### **4. Depth Weighting**
```python
# Deeper pairs are more informative
weight = sqrt(depth_diff + 1)
loss = loss * weight
```

#### **5. Ranking Loss with Margin**
```python
# Encourage: d(ancestor, descendant) < d(ancestor, negative) + margin
loss = relu(pos_dist - neg_dist + margin)
```

**Files Created:**
- `train_hierarchical.py`
- `run_hierarchical_training.sh`

---

## Phase 5: Ball Constraint Enforcement (Nov 8, 2025)

### **The Problem: Embeddings Escaping**

Training iterations showed embeddings violating the Poincaré ball constraint (||x|| ≥ 1.0):

| Version | Max Norm | Outside Ball | Issue |
|---------|----------|--------------|-------|
| v1 | 2.18 | 50K (54%) | Weak regularization |
| v2 | 1.45 | 2K (2.2%) | Better but not enough |

### **Root Causes**
1. Regularization too weak (λ=0.01)
2. Learning rate too high (0.01)
3. No gradient control
4. Projection insufficient

### **Solution: 3-Layer Enforcement Strategy**

#### **Layer 1: Improved Hyperparameters**
```python
Learning rate: 0.01 → 0.005 (2x slower)
Regularization: 0.01 → 0.1 (10x stronger)
Gradient clipping: None → max_norm=1.0
```

#### **Layer 2: Hard Projection**
```python
# Only scale embeddings that violate constraint
needs_projection = (norms >= 1.0)
scale = where(needs_projection, (1-eps)/norms, 1.0)
```

#### **Layer 3: Periodic Full Projection**
```python
# Every 500 batches: project ALL embeddings
if n_batches % 500 == 0:
    model.project_to_ball(indices=None)

# End of epoch: GUARANTEE all inside ball
model.project_to_ball(indices=None)
```

### **Results**
| Version | Max Norm | Outside Ball | Status |
|---------|----------|--------------|--------|
| v3 | 1.00 | 0 (0%) | ✅ Perfect |

**Files Created:**
- `BALL_CONSTRAINT_ENFORCEMENT.md`
- `TRAINING_OPTIMIZATIONS.md`

---

## Phase 6: Performance Optimizations (Nov 8, 2025)

### **Critical Optimizations**

#### **1. Regularizer Vectorization**
```
Before: Loop over 111K nodes per batch (1.7B ops/epoch)
After: Precompute tensors once (111K ops/epoch)
Speedup: 1000x
```

#### **2. Selective Projection**
```
Before: Project all 92K embeddings every batch
After: Project only updated ~3K embeddings per batch
Speedup: 30x
```

#### **3. Efficient Tensor Creation**
```
Before: List of numpy arrays → tensor (slow)
After: Pre-allocate numpy array → tensor (fast)
Speedup: 10-100x
```

#### **4. Device Selection**
```
Before: Auto-select MPS on M3 Mac → hang
After: Force CPU (stable on macOS)
Result: Training actually runs!
```

**Overall:** From hanging indefinitely to ~3 minutes/epoch

---

## Current State (Nov 8, 2025, 4:00pm)

### **What Works ✅**

1. **Data Pipeline**
   - ✅ Correct TaxID → index mapping
   - ✅ 92,290 embeddings (not 3.4M)
   - ✅ 975,896 training pairs (transitive closure)
   - ✅ Comprehensive validation (sanity_check.py)

2. **Ball Constraints**
   - ✅ 100% embeddings inside ball
   - ✅ Max norm = 1.000 (perfect)
   - ✅ 3-layer enforcement strategy

3. **Training Stability**
   - ✅ No crashes or hangs
   - ✅ Gradient clipping prevents exploding gradients
   - ✅ ~3 minutes per epoch on CPU
   - ✅ Automatic checkpointing (keeps last 5 + best)

### **What Doesn't Work ❌**

**Hierarchy Quality is Poor:**
```
Depth-norm correlation: +0.003 (target: >0.5)
Phylum separation:      1.08x   (target: >1.5x)
Class separation:       0.99x   (target: >1.5x)
Order separation:       0.99x   (target: >1.5x)
```

**Hypothesis:** Only 2 epochs completed before manual stop. Possible issues:
- Need more training time
- Regularization too strong (λ=0.1 may be constraining)
- Data imbalance (94% deep pairs, only 6% parent-child)
- Margin too small (0.2) for hierarchical differences

---

## Key Files & Scripts

### **Core Training**
- `train_hierarchical.py` - Main hierarchical training script
- `build_transitive_closure.py` - Generate ancestor-descendant pairs
- `run_hierarchical_training.sh` - Quick-start training script

### **Analysis**
- `analyze_hierarchy_hyperbolic.py` - Evaluate hierarchy quality
- `sanity_check.py` - Comprehensive validation (10 checks)

### **Data Preparation**
- `prepare_taxonomy_data.py` - Download and prepare NCBI taxonomy
- `remap_edges.py` - Map TaxIDs to continuous indices
- `scripts/validate_data.py` - Validate data integrity

### **Utilities**
- `scripts/visualize_embeddings.py` - UMAP visualization
- `watch_training.sh` - Monitor training progress
- `cleanup_for_release.sh` - Prepare for GitHub push

### **Documentation**
- `README.md` - Main project documentation
- `QUICKSTART.md` - Get started in 5 minutes
- `JOURNEY.md` - This document
- `SESSION_SUMMARY_NOV8.md` - Latest session summary

---

## Lessons Learned

### **1. Data Quality is Critical**
- Always validate input data
- Don't assume file formats
- Use automated checks (sanity_check.py)
- A single wrong mapping can waste hours

### **2. Constraints Are Hard**
- Poincaré ball constraint (||x|| < 1) requires careful handling
- Need multiple enforcement layers
- Trade-off between constraint and optimization freedom
- "Technically correct" ≠ "semantically good"

### **3. Start Simple, Then Add Complexity**
- Facebook's simple model worked for 2.7M organisms
- Our complex model has perfect constraints but poor hierarchy
- Sometimes simpler is better
- Validate each feature independently

### **4. Hyperbolic Space is Different**
- Distances grow exponentially near boundary
- Small norm differences = large distance differences
- Regularization interacts with geometry
- Need domain-specific tuning

### **5. Documentation Matters**
- Intermediate docs helped track decisions
- Checkpoint management saves work
- Reproducibility requires clear instructions
- Future self will thank you

---

## Next Steps & Open Questions

### **Immediate Actions**
1. Train longer (try 20-50 epochs with patience=10)
2. If no improvement, reduce regularization (λ=0.05)
3. Monitor depth-norm correlation each epoch
4. Try balanced sampling (equal parent-child vs deep pairs)

### **Research Questions**
1. Is transitive closure helping or hurting?
2. Are 31K siblings per node too many for hard negatives?
3. What's the optimal regularization strength?
4. Should margin vary by depth level?
5. Is curriculum learning needed (parent-child → grandparent → all)?

### **Alternative Approaches**
1. Go back to simple training (known to work)
2. Use Riemannian optimizer (respects manifold natively)
3. Try different hyperbolic models (Lorentz, Klein)
4. Implement progressive training (build hierarchy bottom-up)

---

## Technical Achievements

### **Performance**
- ✅ 1000x faster regularizer (vectorized)
- ✅ 30x faster projection (selective)
- ✅ 100% ball constraint compliance
- ✅ Stable training on Mac M3 CPU

### **Data Pipeline**
- ✅ Fixed critical TaxID bug
- ✅ Comprehensive validation suite
- ✅ Clean, reproducible data preparation
- ✅ Transitive closure computation

### **Code Quality**
- ✅ Modular, well-documented code
- ✅ Automated testing (sanity_check.py)
- ✅ Professional repository structure
- ✅ Clear error messages and logging

---

## Conclusion

We've successfully transformed Facebook's Poincaré embeddings into a specialized hierarchical taxonomy embedding system with:

1. **Better data quality** (fixed bugs, added validation)
2. **Richer training signal** (975K pairs vs 100K)
3. **Hierarchical features** (depth-aware, hard negatives, weighting)
4. **Perfect constraints** (100% inside ball)
5. **Optimized performance** (1000x speedups)

However, **hierarchy quality is still poor** after only 2 epochs. The mathematical framework is sound, but more experimentation is needed to find the right balance of:
- Training time
- Regularization strength  
- Data sampling strategy
- Loss function parameters

The foundation is solid. Now we need to tune the system to actually learn the hierarchy.

---

## References

- Nickel & Kiela (2017). "Poincaré Embeddings for Learning Hierarchical Representations"
- Facebook Research: https://github.com/facebookresearch/poincare-embeddings
- NCBI Taxonomy: https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/
- This project: https://github.com/jcoludar/taxembed

---

*Last Updated: November 8, 2025*
