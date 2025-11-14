#!/usr/bin/env python3
"""Compare old checkpoints (before fixes) vs new (after fixes)."""

import torch
import numpy as np

print("=" * 80)
print("OLD vs NEW CHECKPOINT COMPARISON")
print("=" * 80)

# OLD checkpoint (before fixes - smaller file, epoch 57)
old_ckpt = "taxonomy_model_small_epoch57.pth"
# NEW checkpoint (after fixes - larger file, epoch 36)  
new_ckpt = "taxonomy_model_small_epoch36.pth"

print("\n📦 OLD CHECKPOINT (before n_nodes fix)")
print(f"   File: {old_ckpt}")
print(f"   Date: Nov 13, 11:12 (epoch 57)")
print(f"   Size: 3.5M")

model_old = torch.load(old_ckpt, map_location='cpu')
if isinstance(model_old, dict) and 'embeddings' in model_old:
    embs_old = model_old['embeddings'].detach().numpy()
elif 'embeddings.weight' in model_old:
    embs_old = model_old['embeddings.weight'].numpy()
else:
    embs_old = model_old

norms_old = np.linalg.norm(embs_old, axis=1)

print(f"\n   Shape: {embs_old.shape}")
print(f"   Nodes: {embs_old.shape[0]:,} (should be 111,103)")
print(f"   Missing nodes: {111103 - embs_old.shape[0]:,}")
print(f"\n   Norm statistics:")
print(f"     Min:  {norms_old.min():.4f}")
print(f"     Mean: {norms_old.mean():.4f}")
print(f"     Max:  {norms_old.max():.4f}")
print(f"\n   Distribution:")
print(f"     Outside ball (>1.0): {(norms_old > 1.0).sum():,} ({100*(norms_old > 1.0).sum()/len(norms_old):.2f}%)")
print(f"     Near boundary (>0.95): {(norms_old > 0.95).sum():,} ({100*(norms_old > 0.95).sum()/len(norms_old):.2f}%)")
print(f"     Compressed (>0.90): {(norms_old > 0.90).sum():,} ({100*(norms_old > 0.90).sum()/len(norms_old):.2f}%)")
print(f"\n   Percentiles:")
print(f"     25%: {np.percentile(norms_old, 25):.4f}")
print(f"     50%: {np.percentile(norms_old, 50):.4f}")
print(f"     75%: {np.percentile(norms_old, 75):.4f}")
print(f"     90%: {np.percentile(norms_old, 90):.4f}")
print(f"     95%: {np.percentile(norms_old, 95):.4f}")
print(f"     99%: {np.percentile(norms_old, 99):.4f}")

print("\n" + "=" * 80)

print("\n📦 NEW CHECKPOINT (after n_nodes fix)")
print(f"   File: {new_ckpt}")
print(f"   Date: Nov 13, 13:41 (epoch 36)")
print(f"   Size: 4.2M")

model_new = torch.load(new_ckpt, map_location='cpu')
if isinstance(model_new, dict) and 'embeddings' in model_new:
    embs_new = model_new['embeddings'].detach().numpy()
elif 'embeddings.weight' in model_new:
    embs_new = model_new['embeddings.weight'].numpy()
else:
    embs_new = model_new

norms_new = np.linalg.norm(embs_new, axis=1)

print(f"\n   Shape: {embs_new.shape}")
print(f"   Nodes: {embs_new.shape[0]:,} (should be 111,103)")
print(f"   Missing nodes: {111103 - embs_new.shape[0]:,}")
print(f"\n   Norm statistics:")
print(f"     Min:  {norms_new.min():.4f}")
print(f"     Mean: {norms_new.mean():.4f}")
print(f"     Max:  {norms_new.max():.4f}")
print(f"\n   Distribution:")
print(f"     Outside ball (>1.0): {(norms_new > 1.0).sum():,} ({100*(norms_new > 1.0).sum()/len(norms_new):.2f}%)")
print(f"     Near boundary (>0.95): {(norms_new > 0.95).sum():,} ({100*(norms_new > 0.95).sum()/len(norms_new):.2f}%)")
print(f"     Compressed (>0.90): {(norms_new > 0.90).sum():,} ({100*(norms_new > 0.90).sum()/len(norms_new):.2f}%)")
print(f"\n   Percentiles:")
print(f"     25%: {np.percentile(norms_new, 25):.4f}")
print(f"     50%: {np.percentile(norms_new, 50):.4f}")
print(f"     75%: {np.percentile(norms_new, 75):.4f}")
print(f"     90%: {np.percentile(norms_new, 90):.4f}")
print(f"     95%: {np.percentile(norms_new, 95):.4f}")
print(f"     99%: {np.percentile(norms_new, 99):.4f}")

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

print(f"\n📊 Size:")
print(f"   OLD: {embs_old.shape[0]:,} nodes (missing {111103 - embs_old.shape[0]:,})")
print(f"   NEW: {embs_new.shape[0]:,} nodes (missing {111103 - embs_new.shape[0]:,})")
if embs_new.shape[0] > embs_old.shape[0]:
    print(f"   ✅ NEW has {embs_new.shape[0] - embs_old.shape[0]:,} more nodes")

print(f"\n📊 Boundary Compression:")
print(f"   OLD: {100*(norms_old > 0.90).sum()/len(norms_old):.1f}% at norm > 0.90")
print(f"   NEW: {100*(norms_new > 0.90).sum()/len(norms_new):.1f}% at norm > 0.90")
if (norms_new > 0.90).sum()/len(norms_new) < (norms_old > 0.90).sum()/len(norms_old):
    improvement = (norms_old > 0.90).sum()/len(norms_old) - (norms_new > 0.90).sum()/len(norms_new)
    print(f"   ✅ NEW has {100*improvement:.1f}% LESS boundary compression")
else:
    worsening = (norms_new > 0.90).sum()/len(norms_new) - (norms_old > 0.90).sum()/len(norms_old)
    print(f"   ❌ NEW has {100*worsening:.1f}% MORE boundary compression")

print(f"\n📊 Spread (90th percentile):")
print(f"   OLD: {np.percentile(norms_old, 90):.4f}")
print(f"   NEW: {np.percentile(norms_new, 90):.4f}")
if np.percentile(norms_new, 90) < np.percentile(norms_old, 90):
    print(f"   ✅ NEW has BETTER spread (lower 90th percentile)")
else:
    print(f"   ❌ NEW has WORSE spread (higher 90th percentile)")

print(f"\n📊 Mean norm:")
print(f"   OLD: {norms_old.mean():.4f}")
print(f"   NEW: {norms_new.mean():.4f}")
diff = norms_new.mean() - norms_old.mean()
print(f"   Δ: {diff:+.4f}")

print(f"\n📊 Max norm (boundary adherence):")
print(f"   OLD: {norms_old.max():.6f}")
print(f"   NEW: {norms_new.max():.6f}")
if norms_old.max() > 1.0:
    print(f"   ⚠️  OLD violated ball constraint!")
if norms_new.max() > 1.0:
    print(f"   ⚠️  NEW violated ball constraint!")
if norms_new.max() <= 0.98:
    print(f"   ✅ NEW respects max_norm=0.98 constraint")

print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)

issues = []
improvements = []

if embs_new.shape[0] > embs_old.shape[0]:
    improvements.append(f"✅ Complete coverage: +{embs_new.shape[0] - embs_old.shape[0]:,} nodes")
else:
    issues.append(f"❌ Still missing nodes")

if (norms_new > 0.90).sum()/len(norms_new) < (norms_old > 0.90).sum()/len(norms_old):
    improvements.append("✅ Reduced boundary compression")
else:
    issues.append("❌ Increased boundary compression")

if norms_new.max() <= 0.98:
    improvements.append("✅ Respects max_norm=0.98")
elif norms_new.max() < norms_old.max():
    improvements.append("✅ Better boundary adherence")

if np.percentile(norms_new, 90) < np.percentile(norms_old, 90):
    improvements.append("✅ Better spread")
else:
    issues.append("❌ Worse spread (more compression)")

if improvements:
    print("\n🟢 Improvements:")
    for imp in improvements:
        print(f"   {imp}")

if issues:
    print("\n🔴 Issues:")
    for issue in issues:
        print(f"   {issue}")

if not issues:
    print("\n🎉 NEW checkpoint is BETTER in all metrics!")
elif len(improvements) > len(issues):
    print(f"\n⚖️  NEW checkpoint is BETTER overall ({len(improvements)} improvements vs {len(issues)} issues)")
else:
    print(f"\n⚠️  NEW checkpoint has concerns ({len(issues)} issues vs {len(improvements)} improvements)")

print("\n" + "=" * 80)
