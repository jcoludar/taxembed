# Safety Check: Ball Constraint vs. New Fixes

**Date**: November 13, 2025  
**Concern**: Do relaxed constraints reintroduce the ball escape bug?

---

## 🔍 Historical Bug Review

### Original Problem (Phase 5 - Nov 8, 2025)

Embeddings were ESCAPING the Poincaré ball (||x|| > 1.0):

| Version | λ (reg) | Grad Clip | Projection | Max Norm | % Outside | Status |
|---------|---------|-----------|------------|----------|-----------|--------|
| v1 | 0.01 | ❌ | Soft | 2.18 | 54% | 🔴 Broken |
| v2 | 0.1 | ✅ | Soft | 1.45 | 2.2% | 🟡 Better |
| v3 | 0.1 | ✅ | **3-Layer Hard** | 1.00 | 0% | 🟢 Fixed |

**Root cause**: Weak regularization + no hard constraints → gradients pushed embeddings outside

**Solution**: **3-Layer Enforcement Strategy**
1. Gradient clipping (max_norm=1.0)
2. Radial regularizer (λ=0.1) - soft guidance
3. **Hard projection** - per-batch, periodic, epoch-end

---

## 🆕 Our New Changes

### What We Changed:

| Parameter | Old (v3) | New | Change |
|-----------|----------|-----|--------|
| **λ (regularization)** | 0.1 | **0.01** | 10x weaker ⚠️ |
| **max_norm (projection)** | 0.99999 | **0.98** | 2% lower ⚠️ |
| **Init range** | [0.10, 0.95] | **[0.05, 0.85]** | Lower ✅ |

### Concern:
- λ=0.01 is **same as v1** which had 54% escape rate!
- Are we going back to the broken state?

---

## ✅ Safety Analysis: Why We're STILL Safe

### Critical Insight:
The **regularization (λ)** was only ONE layer of defense. The **real fix** was the **3-layer HARD PROJECTION** strategy.

### Defense Layers We KEPT:

#### ✅ **Layer 1: Gradient Clipping**
```python
# train_small.py line 203
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
**Status**: ✅ Still active

#### ✅ **Layer 2: Per-Batch Hard Projection**
```python
# train_small.py lines 207-210
updated_indices = torch.cat([ancestors, descendants, negatives.flatten()])
updated_indices = torch.unique(updated_indices)
model.project_to_ball(updated_indices)  # Uses max_norm=0.98
```
**Status**: ✅ Still active, **enforces max_norm=0.98**

#### ✅ **Layer 3: Periodic Full Projection**
```python
# train_small.py lines 212-214
if n_batches % 500 == 0:
    model.project_to_ball(indices=None)  # Project ALL
```
**Status**: ✅ Still active

#### ✅ **Layer 4: Epoch-End Full Projection**
```python
# train_small.py line 227
model.project_to_ball(indices=None)  # GUARANTEE before checkpoint
```
**Status**: ✅ Still active

---

## 🔐 Key Differences from v1 (Broken Version)

| Component | v1 (Broken) | Our New Version | Status |
|-----------|-------------|-----------------|--------|
| Regularization | λ=0.01 | λ=0.01 | ⚠️ Same |
| Grad clipping | ❌ None | ✅ max=1.0 | 🟢 Protected |
| Per-batch projection | ❌ Soft clamp | ✅ **Hard @ 0.98** | 🟢 Protected |
| Periodic projection | ❌ None | ✅ Every 500 batches | 🟢 Protected |
| Epoch projection | ❌ None | ✅ Every epoch | 🟢 Protected |
| Hard constraint | ❌ None | ✅ **max_norm=0.98** | 🟢 Protected |

**Verdict**: We have **4 additional protection layers** that v1 didn't have!

---

## 🎯 Why max_norm=0.98 is SAFER than 0.99999

### Old constraint (0.99999):
- Allowed embeddings to get extremely close to boundary
- Left only 0.001% buffer
- Numerical instability risk
- Hyperbolic distances explode near boundary: `acosh(1 + ...)` → ∞

### New constraint (0.98):
- **STRICTER** enforcement (further from boundary)
- 2% buffer provides numerical stability
- Distances remain well-behaved
- **Cannot escape** - hard projection enforces it

### Example:
```python
# If gradient pushes embedding to norm=1.05 (outside):

# OLD (max_norm=0.99999):
# - Projection scales it to 0.99999
# - Next gradient could push it outside again
# - Repeat battle

# NEW (max_norm=0.98):
# - Projection scales it to 0.98
# - Has 2% buffer before hitting boundary
# - More stable, less fighting
```

---

## 📊 What Changed and Why

### Problem We Solved:
**Boundary compression** - 90% of embeddings squeezed to norm > 0.9997

### Why it happened:
1. **Strong regularizer** (λ=0.1) aggressively pushed to target radii
2. **Tight projection** (0.99999) allowed clustering at boundary
3. **Deep pairs** (74% at depth > 5) all pushed near boundary
4. Result: **No room for hierarchical differentiation**

### Our Fix:
1. **Weaker regularizer** (λ=0.01) → soft guidance, not forcing
2. **Lower max_norm** (0.98) → keep away from boundary
3. **Lower init** ([0.05, 0.85]) → start with buffer

### Tradeoff:
- **Old**: Strong constraint → perfect containment but over-compressed
- **New**: Gentle guidance → proper spread **still safely contained**

---

## 🔬 Monitoring Plan

To ensure safety, watch these metrics during training:

### ✅ **Safe Indicators:**
```
Max norm: 0.85 - 0.98     ✅ Good spread, within limit
Outside count: 0          ✅ All inside ball
Mean norm: 0.4 - 0.7      ✅ Proper distribution
Loss: decreasing          ✅ Learning
```

### 🚨 **Danger Signs:**
```
Max norm > 0.98           🚨 PROJECTION FAILED - check code
Outside count > 0         🚨 BALL ESCAPE - critical bug
Max norm < 0.5 all epochs 🟡 Under-utilizing space
Loss plateaus early      🟡 May need stronger reg
```

### 🔧 **Emergency Fixes if Escape Detected:**

If you see `Outside count > 0`:

```python
# 1. Check projection is being called
model.project_to_ball(updated_indices)  # Should be in training loop

# 2. Verify max_norm parameter
model.project_to_ball(indices=None, max_norm=0.98)  # Check this value

# 3. Temporarily boost regularization
--lambda-reg 0.05  # Increase from 0.01

# 4. Reduce learning rate
--lr 0.001  # Reduce from 0.005
```

---

## 🎓 Key Insight

**The regularizer (λ) is NOT the constraint enforcer!**

- **Regularizer**: Soft guidance (loss penalty)
  - Encourages embeddings toward target radii
  - Can be ignored by optimizer if other losses are stronger
  - λ=0.01 vs 0.1 affects **preference**, not **enforcement**

- **Projection**: Hard constraint (geometric operation)
  - **GUARANTEES** `||x|| ≤ max_norm`
  - Cannot be violated (it's a post-processing clamp)
  - max_norm=0.98 → **mathematically impossible** to escape

### From BALL_CONSTRAINT_ENFORCEMENT.md:
> "Regularizer guides gradients toward valid solutions. Projection is safety net, not primary mechanism."

We're **reducing the guide** (λ=0.1 → 0.01) but **keeping the safety net** (projection @ 0.98).

---

## ✅ Conclusion: **SAFE TO PROCEED**

### Why we won't see ball escape:

1. ✅ **Hard projection** enforces max_norm=0.98 (STRICTER than before)
2. ✅ **3-layer strategy** still active (batch, periodic, epoch)
3. ✅ **Gradient clipping** prevents exploding updates
4. ✅ **0.98 < 0.99999** → more conservative, not less

### What changed:
- **Softer regularization** → less aggressive pushing toward specific radii
- **Lower max allowed** → actually MORE restrictive boundary
- **Better spread** → avoid compression at 0.9997

### Risk assessment:
- **Ball escape risk**: 🟢 **NEGLIGIBLE** (multiple hard constraints active)
- **Boundary compression**: 🟢 **SOLVED** (lower max_norm, weaker reg)
- **Training stability**: 🟢 **MAINTAINED** (all safety layers intact)

---

## 📝 Recommendations

1. ✅ **Proceed with training** - fixes are safe
2. ✅ **Monitor max_norm** - should stay well below 0.98
3. ✅ **Check outside_count** - should remain 0
4. 🔍 **If issues arise**: See "Emergency Fixes" section above

The 3-layer projection strategy is the **real hero** that prevents ball escape. 
Regularization strength only affects how **smoothly** we learn, not whether we **stay inside**.

---

**Status**: ✅ **SAFE - All critical constraints maintained**
