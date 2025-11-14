#!/usr/bin/env python3
"""Inspect checkpoint structure."""

import torch

old_ckpt = "taxonomy_model_small_epoch57.pth"
new_ckpt = "taxonomy_model_small_epoch36.pth"

print("=" * 80)
print("OLD CHECKPOINT STRUCTURE")
print("=" * 80)

model_old = torch.load(old_ckpt, map_location='cpu')
print(f"\nType: {type(model_old)}")
if isinstance(model_old, dict):
    print(f"Keys: {list(model_old.keys())}")
    for key in model_old.keys():
        val = model_old[key]
        print(f"\n  '{key}': {type(val)}")
        if isinstance(val, dict):
            print(f"    Sub-keys: {list(val.keys())[:10]}")
        elif hasattr(val, 'shape'):
            print(f"    Shape: {val.shape}")

print("\n" + "=" * 80)
print("NEW CHECKPOINT STRUCTURE")
print("=" * 80)

model_new = torch.load(new_ckpt, map_location='cpu')
print(f"\nType: {type(model_new)}")
if isinstance(model_new, dict):
    print(f"Keys: {list(model_new.keys())}")
    for key in model_new.keys():
        val = model_new[key]
        print(f"\n  '{key}': {type(val)}")
        if isinstance(val, dict):
            print(f"    Sub-keys: {list(val.keys())[:10]}")
        elif hasattr(val, 'shape'):
            print(f"    Shape: {val.shape}")
