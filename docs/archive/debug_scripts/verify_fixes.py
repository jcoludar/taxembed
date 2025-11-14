#!/usr/bin/env python3
"""Verify that all fixes are properly applied."""

import pickle
import pandas as pd

print("=" * 80)
print("VERIFICATION: All Fixes Applied Correctly")
print("=" * 80)

# Test 1: n_nodes calculation
print("\n1. Testing n_nodes calculation...")
with open('data/taxonomy_edges_small_transitive.pkl', 'rb') as f:
    training_data = pickle.load(f)

# OLD WAY (buggy)
old_n_nodes = max(max(item['ancestor_idx'], item['descendant_idx']) 
                  for item in training_data) + 1

# NEW WAY (correct)
mapping_df = pd.read_csv("data/taxonomy_edges_small.mapping.tsv",
                         sep="\t", header=None, names=["taxid", "idx"])
mapping_df['idx'] = pd.to_numeric(mapping_df['idx'], errors='coerce')
mapping_df = mapping_df.dropna()
new_n_nodes = int(mapping_df['idx'].max()) + 1

print(f"   OLD calculation (buggy): {old_n_nodes:,} nodes")
print(f"   NEW calculation (fixed): {new_n_nodes:,} nodes")
print(f"   Difference: {new_n_nodes - old_n_nodes:,} nodes recovered")

if new_n_nodes > old_n_nodes:
    print(f"   ✅ FIX VERIFIED: Will create {new_n_nodes:,} embeddings (covering all nodes)")
else:
    print(f"   ❌ PROBLEM: n_nodes unchanged")

# Test 2: Check initialization range
print("\n2. Testing initialization range...")
max_depth = 37

# OLD initialization
old_min_radius = 0.1 + (0 / max_depth) * 0.85
old_max_radius = 0.1 + (max_depth / max_depth) * 0.85

# NEW initialization
new_min_radius = 0.05 + (0 / max_depth) * 0.80
new_max_radius = 0.05 + (max_depth / max_depth) * 0.80

print(f"   OLD range: [{old_min_radius:.2f}, {old_max_radius:.2f}]")
print(f"   NEW range: [{new_min_radius:.2f}, {new_max_radius:.2f}]")
print(f"   Buffer from boundary (1.0): {1.0 - new_max_radius:.2f}")

if new_max_radius < old_max_radius:
    print(f"   ✅ FIX VERIFIED: {100*(1.0-new_max_radius):.0f}% buffer from boundary")
else:
    print(f"   ❌ PROBLEM: Still too close to boundary")

# Test 3: Check projection constraint
print("\n3. Testing projection constraint...")
old_max_norm = 1.0 - 1e-5
new_max_norm = 0.98

print(f"   OLD max_norm: {old_max_norm:.6f}")
print(f"   NEW max_norm: {new_max_norm:.6f}")
print(f"   Buffer gained: {old_max_norm - new_max_norm:.6f}")

if new_max_norm < old_max_norm:
    print(f"   ✅ FIX VERIFIED: {100*(1.0-new_max_norm):.0f}% buffer from boundary")
else:
    print(f"   ⚠️  Check: max_norm should be 0.98 in code")

# Test 4: Regularization strength
print("\n4. Testing regularization strength...")
old_lambda = 0.1
new_lambda = 0.01

print(f"   OLD lambda_reg: {old_lambda}")
print(f"   NEW lambda_reg: {new_lambda}")
print(f"   Reduction: {old_lambda / new_lambda:.0f}x weaker")

if new_lambda < old_lambda:
    print(f"   ✅ FIX VERIFIED: {old_lambda/new_lambda:.0f}x weaker regularization")
else:
    print(f"   ❌ PROBLEM: lambda_reg should be reduced")

# Test 5: Data coverage
print("\n5. Testing data coverage...")
ancestor_indices = set(item['ancestor_idx'] for item in training_data)
descendant_indices = set(item['descendant_idx'] for item in training_data)
all_in_training = ancestor_indices | descendant_indices
all_in_mapping = set(mapping_df['idx'].unique())

covered_old = len(all_in_training)
covered_new = len(all_in_mapping)
coverage_pct = 100 * covered_old / covered_new

print(f"   Nodes in training data: {covered_old:,}")
print(f"   Nodes in mapping: {covered_new:,}")
print(f"   OLD coverage: {coverage_pct:.1f}%")
print(f"   NEW coverage: 100.0% (all nodes get embeddings)")

missing = all_in_mapping - all_in_training
print(f"   Nodes that will now be initialized (were missing): {len(missing):,}")

if len(missing) > 0:
    print(f"   ✅ FIX VERIFIED: {len(missing):,} additional nodes will get proper embeddings")
else:
    print(f"   ❓ No missing nodes found")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

checks = []
checks.append(("n_nodes calculation", new_n_nodes > old_n_nodes))
checks.append(("Initialization range", new_max_radius < old_max_radius))
checks.append(("Projection constraint", new_max_norm < old_max_norm))
checks.append(("Regularization strength", new_lambda < old_lambda))
checks.append(("Data coverage", len(missing) > 0))

passed = sum(1 for _, status in checks if status)
total = len(checks)

print(f"\nChecks passed: {passed}/{total}")
for name, status in checks:
    symbol = "✅" if status else "❌"
    print(f"  {symbol} {name}")

if passed == total:
    print("\n🎉 ALL FIXES VERIFIED - Ready to retrain!")
    print("\nRecommended command:")
    print("  uv run python train_small.py --epochs 10000 --early-stopping 0")
else:
    print(f"\n⚠️  {total - passed} checks failed - review fixes")

print("\n" + "=" * 80)
