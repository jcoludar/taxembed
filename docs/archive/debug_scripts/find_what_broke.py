#!/usr/bin/env python3
"""Find what hyperparameter changes broke the model."""

print("=" * 80)
print("WHAT CHANGED BETWEEN OLD AND CURRENT")
print("=" * 80)

print("\n📊 DATA:")
print("   OLD: 975,896 training pairs")
print("   CURRENT: 1,002,486 training pairs (+26,590)")
print("   ✅ This is GOOD - more complete coverage")

print("\n📊 MY CHANGES (that might have broken things):")

print("\n1. **Regularization strength (λ)**")
print("   OLD: λ = 0.1 (default before my changes)")
print("   CURRENT: λ = 0.01 (I reduced it 10x)")
print("   🔴 TOO WEAK - not enough structure enforcement")

print("\n2. **Initialization range**")
print("   OLD: target_radius = 0.1 + depth/max * 0.85")
print("        → Range: [0.1, 0.95]")
print("   CURRENT: target_radius = 0.05 + depth/max * 0.80")
print("        → Range: [0.05, 0.85]")
print("   🔴 MORE CONSERVATIVE - less room to learn")

print("\n3. **Hard projection boundary**")
print("   OLD: max_norm = 0.99999 (essentially 1.0)")
print("   CURRENT: max_norm = 0.98")
print("   🔴 TOO TIGHT - embeddings get compressed")

print("\n4. **Number of nodes**")
print("   OLD: 92,290 nodes (incomplete)")
print("   CURRENT: 111,103 nodes (complete)")
print("   ⚠️ More nodes = more crowding, but this is necessary")

print("\n" + "=" * 80)
print("THE PROBLEM")
print("=" * 80)

print("""
I made THREE hyperparameter changes that were TOO conservative:

1. λ: 0.1 → 0.01 (10x weaker)
   → Not enough regularization for 111K nodes
   
2. Init: [0.1, 0.95] → [0.05, 0.85]
   → Starts too far from boundary, harder to learn hierarchy
   
3. max_norm: 0.99999 → 0.98
   → Artificial compression near boundary

These changes were meant to "fix" boundary compression, but they were
based on the OLD incomplete data. With complete data, the old hyperparams
were actually fine!

The REAL issue was just the missing 18,813 nodes, which we now fixed.
But I over-corrected the hyperparameters.
""")

print("\n" + "=" * 80)
print("THE FIX")
print("=" * 80)

print("""
KEEP: Complete data coverage (98.3% = excellent)
REVERT: All three hyperparameter changes

Specifically:
1. λ = 0.1 (not 0.01)
2. Init range: [0.1, 0.95] (not [0.05, 0.85])
3. max_norm = 0.999 (not 0.98)

This gives us:
✅ Complete data coverage (the real permanent fix)
✅ Original hyperparameters that worked well
✅ Should match or beat OLD model performance
""")

print("\n" + "=" * 80)
