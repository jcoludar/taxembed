import sys

import numpy as np
import pandas as pd
import torch

ckpt_path, map_path, query = sys.argv[1], sys.argv[2], sys.argv[3]
ckpt = torch.load(ckpt_path, map_location="cpu")
sd = ckpt["state_dict"]

# Get embedding tensor
for k in ["embeddings.weight", "emb.weight", "E.weight", "lt.weight", "weight"]:
    if k in sd:
        emb = sd[k]
        print(f"Using embedding from '{k}': shape {emb.shape}")
        break

E = emb.detach().cpu().numpy()

# Load mapping
m = pd.read_csv(map_path, sep="\t")
tax2idx = dict(zip(m["taxid"].astype(str), m["idx"], strict=False))
idx2tax = dict(zip(m["idx"], m["taxid"].astype(str), strict=False))

# Query
i = tax2idx[str(query)]
x = E[i]

# Compute distances (Euclidean for sanity check)
d = np.linalg.norm(E - x, axis=1)
nbrs = np.argsort(d)[:11]
nbrs = [j for j in nbrs if j != i][:10]

print(f"\nNearest neighbors to TaxID {query} (idx {i}):")
for rank, j in enumerate(nbrs, 1):
    print(f"  {rank}. TaxID {idx2tax[j]} (distance: {d[j]:.4f})")
