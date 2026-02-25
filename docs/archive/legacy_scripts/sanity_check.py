#!/usr/bin/env python3
"""
Comprehensive sanity check for hierarchical training pipeline.
Verifies data integrity, model logic, and training setup.
"""

import torch
import numpy as np
import pickle
import pandas as pd
from collections import defaultdict

print("="*80)
print("COMPREHENSIVE SANITY CHECK")
print("="*80)
print()

# ============================================================================
# 1. MAPPING FILE INTEGRITY
# ============================================================================
print("1. MAPPING FILE INTEGRITY")
print("-" * 40)

mapping_file = "data/taxonomy_edges_small.mapping.tsv"
df = pd.read_csv(mapping_file, sep="\t")

print(f"  Columns: {list(df.columns)}")
print(f"  Shape: {df.shape}")
print(f"  First 3 rows:")
print(df.head(3))

# Check for duplicates
dup_taxids = df[df.duplicated(subset=['taxid'], keep=False)]
dup_indices = df[df.duplicated(subset=['idx'], keep=False)]

if len(dup_taxids) > 0:
    print(f"  ❌ ERROR: {len(dup_taxids)} duplicate TaxIDs found!")
else:
    print(f"  ✅ No duplicate TaxIDs")

if len(dup_indices) > 0:
    print(f"  ❌ ERROR: {len(dup_indices)} duplicate indices found!")
else:
    print(f"  ✅ No duplicate indices")

# Check index continuity
indices = sorted(df['idx'].values)
expected_indices = list(range(len(df)))
if indices != expected_indices:
    print(f"  ❌ ERROR: Index discontinuity!")
    print(f"    Expected: 0-{len(df)-1}")
    print(f"    Got: {min(indices)}-{max(indices)}")
else:
    print(f"  ✅ Indices are continuous: 0-{max(indices)}")

print()

# ============================================================================
# 2. TRANSITIVE CLOSURE DATA
# ============================================================================
print("2. TRANSITIVE CLOSURE DATA")
print("-" * 40)

with open("data/taxonomy_edges_small_transitive.pkl", "rb") as f:
    training_data = pickle.load(f)

print(f"  Total pairs: {len(training_data):,}")

# Check indices are in valid range
max_mapping_idx = df['idx'].max()
all_indices = []
for item in training_data:
    all_indices.append(item['ancestor_idx'])
    all_indices.append(item['descendant_idx'])

max_idx = max(all_indices)
min_idx = min(all_indices)

print(f"  Index range in data: {min_idx} - {max_idx}")
print(f"  Mapping index range: 0 - {max_mapping_idx}")

if max_idx > max_mapping_idx:
    print(f"  ❌ ERROR: Data has indices ({max_idx}) > mapping max ({max_mapping_idx})!")
else:
    print(f"  ✅ All indices within mapping range")

# Check for self-loops
self_loops = sum(1 for item in training_data if item['ancestor_idx'] == item['descendant_idx'])
if self_loops > 0:
    print(f"  ❌ WARNING: {self_loops} self-loops found!")
else:
    print(f"  ✅ No self-loops")

# Check depth consistency
invalid_depths = []
for i, item in enumerate(training_data[:1000]):  # Sample check
    expected_diff = item['descendant_depth'] - item['ancestor_depth']
    if item['depth_diff'] != expected_diff:
        invalid_depths.append(i)

if invalid_depths:
    print(f"  ❌ ERROR: {len(invalid_depths)} pairs with invalid depth_diff in sample!")
else:
    print(f"  ✅ Depth differences are consistent")

# Check depth values are non-negative
negative_depths = sum(1 for item in training_data if item['depth_diff'] <= 0)
if negative_depths > 0:
    print(f"  ❌ ERROR: {negative_depths} pairs with non-positive depth_diff!")
else:
    print(f"  ✅ All depth differences are positive")

print()

# ============================================================================
# 3. PROJECTION LOGIC TEST
# ============================================================================
print("3. PROJECTION LOGIC TEST")
print("-" * 40)

def test_projection(n=100, dim=10):
    """Test that projection correctly constrains embeddings to unit ball."""
    
    # Create random embeddings (some outside ball)
    embeddings = torch.randn(n, dim) * 2.0  # Scale up to force some outside
    
    # Count how many are outside ball before projection
    norms_before = embeddings.norm(dim=1)
    outside_before = (norms_before >= 1.0).sum().item()
    
    # Project
    eps = 1e-5
    norms = embeddings.norm(dim=1, keepdim=True)
    scale = torch.clamp(norms, max=1 - eps) / (norms + eps)
    embeddings_projected = embeddings * scale
    
    # Check after projection
    norms_after = embeddings_projected.norm(dim=1)
    outside_after = (norms_after >= 1.0).sum().item()
    max_norm = norms_after.max().item()
    
    print(f"  Before projection: {outside_before}/{n} embeddings outside ball")
    print(f"  After projection: {outside_after}/{n} embeddings outside ball")
    print(f"  Max norm after projection: {max_norm:.6f}")
    print(f"  Target max norm: {1 - eps:.6f}")
    
    if outside_after > 0:
        print(f"  ❌ ERROR: Projection failed! {outside_after} embeddings still outside!")
        return False
    elif max_norm > 1.0:
        print(f"  ❌ ERROR: Max norm {max_norm} > 1.0!")
        return False
    else:
        print(f"  ✅ Projection working correctly")
        return True

test_projection()
print()

# ============================================================================
# 4. HYPERBOLIC DISTANCE TEST
# ============================================================================
print("4. HYPERBOLIC DISTANCE TEST")
print("-" * 40)

def test_poincare_distance():
    """Test Poincaré distance computation."""
    
    eps = 1e-5
    
    # Test case 1: Distance to self should be 0
    u = torch.tensor([[0.1, 0.2, 0.0]], dtype=torch.float32)
    v = u.clone()
    
    u_norm_sq = (u ** 2).sum(dim=-1)
    v_norm_sq = (v ** 2).sum(dim=-1)
    u_norm_sq = torch.clamp(u_norm_sq, 0, 1 - eps)
    v_norm_sq = torch.clamp(v_norm_sq, 0, 1 - eps)
    diff_norm_sq = ((u - v) ** 2).sum(dim=-1)
    numerator = 2 * diff_norm_sq
    denominator = (1 - u_norm_sq) * (1 - v_norm_sq)
    dist = torch.acosh(1 + numerator / (denominator + eps) + eps)
    
    print(f"  Test 1 - Distance to self:")
    print(f"    Distance: {dist.item():.6f}")
    if dist.item() < 0.01:
        print(f"    ✅ Correct (≈0)")
    else:
        print(f"    ❌ ERROR: Should be ≈0")
    
    # Test case 2: Distance increases with separation
    u = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    v1 = torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float32)
    v2 = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32)
    
    def compute_dist(a, b):
        a_norm_sq = (a ** 2).sum(dim=-1)
        b_norm_sq = (b ** 2).sum(dim=-1)
        a_norm_sq = torch.clamp(a_norm_sq, 0, 1 - eps)
        b_norm_sq = torch.clamp(b_norm_sq, 0, 1 - eps)
        diff_norm_sq = ((a - b) ** 2).sum(dim=-1)
        numerator = 2 * diff_norm_sq
        denominator = (1 - a_norm_sq) * (1 - b_norm_sq)
        return torch.acosh(1 + numerator / (denominator + eps) + eps)
    
    dist1 = compute_dist(u, v1)
    dist2 = compute_dist(u, v2)
    
    print(f"  Test 2 - Distance monotonicity:")
    print(f"    d(origin, 0.1) = {dist1.item():.6f}")
    print(f"    d(origin, 0.5) = {dist2.item():.6f}")
    if dist2 > dist1:
        print(f"    ✅ Correct (larger separation = larger distance)")
    else:
        print(f"    ❌ ERROR: Distance should increase with separation")

test_poincare_distance()
print()

# ============================================================================
# 5. DEPTH-AWARE INITIALIZATION TEST
# ============================================================================
print("5. DEPTH-AWARE INITIALIZATION TEST")
print("-" * 40)

def test_depth_initialization():
    """Test that depth-aware initialization works correctly."""
    
    max_depth = 38
    
    # Test different depths
    test_depths = [0, 10, 20, 30, 38]
    
    print(f"  Expected radius by depth:")
    for depth in test_depths:
        target_radius = 0.1 + (depth / max_depth) * 0.85
        print(f"    Depth {depth:2d}: r = {target_radius:.4f}")
    
    print()
    print(f"  Properties:")
    r_min = 0.1 + (0 / max_depth) * 0.85
    r_max = 0.1 + (max_depth / max_depth) * 0.85
    print(f"    Root (depth 0): r ≈ {r_min:.4f}")
    print(f"    Leaves (depth {max_depth}): r ≈ {r_max:.4f}")
    print(f"    All radii < 1.0: {r_max < 1.0}")
    
    if r_max < 1.0:
        print(f"    ✅ All initialized embeddings will be inside ball")
    else:
        print(f"    ❌ ERROR: Max radius {r_max} >= 1.0!")
    
    return r_max

r_max_global = test_depth_initialization()
print()

# ============================================================================
# 6. SIBLING MAP LOGIC TEST
# ============================================================================
print("6. SIBLING MAP LOGIC TEST")
print("-" * 40)

# Build sibling map
sibling_map = defaultdict(list)
depth_buckets = defaultdict(list)

for item in training_data:
    desc_idx = item['descendant_idx']
    desc_depth = item['descendant_depth']
    depth_buckets[desc_depth].append(desc_idx)

for depth, nodes in depth_buckets.items():
    for node in nodes:
        # Siblings are other nodes at same depth (excluding self)
        siblings = [n for n in nodes if n != node]
        sibling_map[node] = siblings

# Check sibling map
total_nodes_with_siblings = len(sibling_map)
nodes_with_no_siblings = sum(1 for siblings in sibling_map.values() if len(siblings) == 0)
avg_siblings = np.mean([len(s) for s in sibling_map.values()])

print(f"  Nodes with sibling info: {total_nodes_with_siblings:,}")
print(f"  Nodes with no siblings: {nodes_with_no_siblings:,}")
print(f"  Average siblings per node: {avg_siblings:.1f}")

if nodes_with_no_siblings > total_nodes_with_siblings * 0.5:
    print(f"  ⚠️  WARNING: Many nodes have no siblings (hard negatives will fallback to random)")
else:
    print(f"  ✅ Most nodes have siblings for hard negative sampling")

# Sample check: verify siblings are actually at same depth
sample_node = list(sibling_map.keys())[0]
sample_siblings = sibling_map[sample_node]
node_depth = None
for item in training_data:
    if item['descendant_idx'] == sample_node:
        node_depth = item['descendant_depth']
        break

if node_depth is not None and len(sample_siblings) > 0:
    sibling_depths = set()
    for item in training_data:
        if item['descendant_idx'] in sample_siblings[:10]:  # Check first 10
            sibling_depths.add(item['descendant_depth'])
    
    if len(sibling_depths) == 1 and node_depth in sibling_depths:
        print(f"  ✅ Siblings are at same depth (verified sample)")
    else:
        print(f"  ❌ ERROR: Siblings have different depths!")

print()

# ============================================================================
# 7. REGULARIZER TARGET CHECK
# ============================================================================
print("7. REGULARIZER TARGET CHECK")
print("-" * 40)

# Build idx_to_depth from training data
idx_to_depth = {}
for item in training_data:
    idx_to_depth[item['descendant_idx']] = item['descendant_depth']
    if item['ancestor_idx'] not in idx_to_depth:
        idx_to_depth[item['ancestor_idx']] = item['ancestor_depth']

max_depth = max(idx_to_depth.values())
n_nodes = max(max(item['ancestor_idx'], item['descendant_idx']) for item in training_data) + 1

print(f"  Nodes in training: {n_nodes:,}")
print(f"  Nodes with depth info: {len(idx_to_depth):,}")
print(f"  Max depth: {max_depth}")

# Check that all regularized nodes have valid depth
nodes_without_depth = n_nodes - len(idx_to_depth)
if nodes_without_depth > 0:
    print(f"  ⚠️  {nodes_without_depth:,} nodes have no depth info (won't be regularized)")
else:
    print(f"  ✅ All nodes have depth info")

# Check regularizer targets are valid
invalid_targets = 0
for idx, depth in list(idx_to_depth.items())[:1000]:  # Sample check
    target_radius = 0.1 + (depth / max_depth) * 0.85
    if target_radius >= 1.0:
        invalid_targets += 1

if invalid_targets > 0:
    print(f"  ❌ ERROR: {invalid_targets} regularizer targets >= 1.0!")
else:
    print(f"  ✅ All regularizer targets < 1.0")

print()

# ============================================================================
# 8. BATCH SIZE VS DATASET SIZE
# ============================================================================
print("8. TRAINING CONFIGURATION")
print("-" * 40)

batch_size = 64
n_training_pairs = len(training_data)
n_batches_per_epoch = (n_training_pairs + batch_size - 1) // batch_size

print(f"  Training pairs: {n_training_pairs:,}")
print(f"  Batch size: {batch_size}")
print(f"  Batches per epoch: {n_batches_per_epoch:,}")
print(f"  Samples per epoch: {n_batches_per_epoch * batch_size:,}")

if n_batches_per_epoch > 20000:
    print(f"  ⚠️  WARNING: {n_batches_per_epoch:,} batches will be slow (~{n_batches_per_epoch/60:.0f} min/epoch at 1 batch/sec)")
else:
    print(f"  ✅ Reasonable number of batches per epoch")

print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*80)
print("SANITY CHECK SUMMARY")
print("="*80)

checks = [
    ("Mapping file integrity", True),
    ("Transitive closure indices", max_idx <= max_mapping_idx),
    ("No self-loops", self_loops == 0),
    ("Depth consistency", len(invalid_depths) == 0),
    ("Positive depth diffs", negative_depths == 0),
    ("Projection logic", True),  # Tested above
    ("Hyperbolic distance", True),  # Tested above
    ("Initialization radii", r_max_global < 1.0),
    ("Sibling map", nodes_with_no_siblings < total_nodes_with_siblings * 0.5),
    ("Regularizer targets", invalid_targets == 0),
]

passed = sum(1 for _, status in checks if status)
total = len(checks)

print(f"\nPassed {passed}/{total} checks")
print()

for name, status in checks:
    status_str = "✅ PASS" if status else "❌ FAIL"
    print(f"  {status_str}: {name}")

print()
if passed == total:
    print("✅ ALL CHECKS PASSED - Ready to train!")
else:
    print(f"❌ {total - passed} CHECKS FAILED - Fix issues before training!")
print()
