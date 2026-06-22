"""P4 review probe (read-only): does mapping-row-count == TrainingPairs.n_nodes == checkpoint rows?

Checks the load-bearing n_nodes assumption in the E1c Phase-1 plan's driver, which derives
n_nodes from `sum(1 for _ in open(mapping)) - 1` instead of `pairs.n_nodes`.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from taxembed.utils.training_pairs import TrainingPairs

DS = {
    "echino": "echinodermata_7586_clean",
    "metazoa": "metazoa_33208_clean",
}
CKPT = {
    "echino": "artifacts/tags/echino_softmax/echino_softmax_best.pth",
    "metazoa": "artifacts/tags/metazoa_softmax_milestones/metazoa_softmax_milestones_best.pth",
}

for name, ds in DS.items():
    base = ROOT / "data" / "taxopy" / ds
    npz = base / f"taxonomy_edges_{ds}_transitive.npz"
    mapping = base / f"taxonomy_edges_{ds}.mapping.tsv"

    pairs = TrainingPairs.load(npz)
    pairs_n = pairs.n_nodes

    with open(mapping) as fh:
        rows = sum(1 for _ in fh)
    mapping_n = rows - 1  # minus header, as the plan's driver does

    # checkpoint embedding rows (only load echino; metazoa is 400MB — load lazily/skip)
    ckpt_n = None
    if name == "echino":
        import torch
        ck = torch.load(ROOT / CKPT[name], map_location="cpu", weights_only=False)
        sd = ck.get("state_dict", ck)
        if "lt.weight" in sd:
            ckpt_n = sd["lt.weight"].shape[0]
        elif "embeddings" in ck:
            ckpt_n = ck["embeddings"].shape[0]

    print(f"[{name}] pairs.n_nodes={pairs_n}  mapping_rows-1={mapping_n}  ckpt_rows={ckpt_n}  "
          f"match_pairs_vs_mapping={pairs_n == mapping_n}  "
          f"match_pairs_vs_ckpt={None if ckpt_n is None else pairs_n == ckpt_n}")
