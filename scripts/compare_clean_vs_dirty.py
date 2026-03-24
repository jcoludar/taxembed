#!/usr/bin/env python3
"""
Compare training on clean vs original (dirty) Mollusca dataset.

Trains a few epochs on each dataset with identical hyperparameters and reports
per-epoch loss, convergence speed, and produces a comparison plot.

Usage (from repo root):
    uv run python scripts/compare_clean_vs_dirty.py --epochs 10
"""

import argparse
import gc
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train_hierarchical import (
    HierarchicalDataLoader,
    HierarchicalPoincareEmbedding,
    TrainingPairs,
    ranking_loss_with_margin,
    radial_regularizer,
    target_radius,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "taxopy"

DATASETS = {
    "original": {
        "data": DATA_DIR / "mollusca_6447" / "taxonomy_edges_mollusca_6447_transitive.npz",
        "mapping": DATA_DIR / "mollusca_6447" / "taxonomy_edges_mollusca_6447.mapping.tsv",
        "label": "Original (53.7K nodes, 481K pairs)",
        "color": "#e74c3c",
    },
    "clean": {
        "data": DATA_DIR / "mollusca_clean" / "taxonomy_edges_mollusca_clean_transitive.npz",
        "mapping": DATA_DIR / "mollusca_clean" / "taxonomy_edges_mollusca_clean.mapping.tsv",
        "label": "Clean (26.3K nodes, 229K pairs)",
        "color": "#2ecc71",
    },
}

# Match mollusca_v11 hyperparams
DIM = 100
BATCH_SIZE = 128
N_NEGATIVES = 50
LR = 0.005
MARGIN = 0.2
LAMBDA_REG = 0.1


def load_data(data_path, mapping_path):
    pairs = TrainingPairs.load(data_path)
    mapping_df = pd.read_csv(mapping_path, sep="\t")
    if "taxid" not in mapping_df.columns or "idx" not in mapping_df.columns:
        mapping_df = pd.read_csv(mapping_path, sep="\t", header=None, names=["taxid", "idx"])
    mapping_df["idx"] = pd.to_numeric(mapping_df["idx"], errors="coerce").dropna().astype(int)
    n_nodes = int(mapping_df["idx"].max()) + 1
    max_depth = pairs.max_depth
    idx_to_depth = pairs.idx_to_depth_dict()
    return pairs, n_nodes, max_depth, idx_to_depth


def train_epochs(name, pairs, n_nodes, max_depth, idx_to_depth, n_epochs, device):
    """Train for n_epochs, return list of (epoch, loss, epoch_time_sec)."""
    torch.manual_seed(42)
    np.random.seed(42)

    model = HierarchicalPoincareEmbedding(
        n_nodes=n_nodes, dim=DIM, max_depth=max_depth,
        init_depth_data=idx_to_depth, euclidean_param=True,
    )
    model.to(device)

    dataloader = HierarchicalDataLoader(
        training_data=pairs, n_nodes=n_nodes,
        batch_size=BATCH_SIZE, n_negatives=N_NEGATIVES,
        depth_stratify=True, tiered_negatives=True,
    )

    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Precompute regularization
    reg_indices = []
    reg_target_radii = []
    for idx, depth in idx_to_depth.items():
        if idx < n_nodes:
            reg_indices.append(idx)
            reg_target_radii.append(target_radius(depth, max_depth, "linear"))
    reg_idx = torch.LongTensor(reg_indices).to(device)
    reg_radii = torch.FloatTensor(reg_target_radii).to(device)

    results = []
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        if device.type == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()

        for ancestors, descendants, negatives, depths in dataloader:
            ancestors = ancestors.to(device)
            descendants = descendants.to(device)
            negatives = negatives.to(device)
            depths = depths.to(device)

            optimizer.zero_grad()
            loss = ranking_loss_with_margin(
                model, ancestors, descendants, negatives, depths,
                margin=MARGIN, depth_weight=True,
            )
            reg_loss = radial_regularizer(model, reg_idx, reg_radii, LAMBDA_REG)
            total_loss = loss + reg_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            updated = torch.cat([ancestors, descendants, negatives.flatten()])
            model.project_to_ball(torch.unique(updated))
            if n_batches % 500 == 0:
                model.project_to_ball(indices=None)

            epoch_loss += loss.item()
            n_batches += 1

        if device.type == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - t0

        avg_loss = epoch_loss / n_batches
        throughput = len(pairs) / elapsed
        results.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "time_sec": elapsed,
            "throughput": throughput,
        })
        print(f"  [{name}] Epoch {epoch+1:2d}: loss={avg_loss:.4f}  "
              f"{elapsed:.1f}s  ({throughput:,.0f} pairs/s)")

    del model, optimizer, dataloader
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare clean vs dirty Mollusca training")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    parser.add_argument("--out", type=Path, default=Path("artifacts/compare_clean_vs_dirty.png"))
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print()

    all_results = {}
    for key, ds in DATASETS.items():
        print(f"Loading {key} dataset: {ds['data']}")
        pairs, n_nodes, max_depth, idx_to_depth = load_data(ds["data"], ds["mapping"])
        print(f"  Pairs: {len(pairs):,}  Nodes: {n_nodes:,}  Max depth: {max_depth}")

        print(f"Training {key}...")
        results = train_epochs(
            key, pairs, n_nodes, max_depth, idx_to_depth, args.epochs, device
        )
        all_results[key] = results
        print()

    # ── Comparison table ──────────────────────────────────────────────────
    print("=" * 70)
    print("  COMPARISON")
    print("=" * 70)
    print(f"  {'':>20} {'Original':>14} {'Clean':>14} {'Ratio':>10}")
    print(f"  {'':>20} {'─'*14} {'─'*14} {'─'*10}")

    orig = all_results["original"]
    clean = all_results["clean"]

    orig_final = orig[-1]["loss"]
    clean_final = clean[-1]["loss"]
    print(f"  {'Final loss':>20} {orig_final:>14.4f} {clean_final:>14.4f}")

    orig_time = np.mean([r["time_sec"] for r in orig])
    clean_time = np.mean([r["time_sec"] for r in clean])
    speedup = orig_time / clean_time
    print(f"  {'Mean epoch time (s)':>20} {orig_time:>14.1f} {clean_time:>14.1f} {speedup:>9.2f}x")

    orig_throughput = np.mean([r["throughput"] for r in orig])
    clean_throughput = np.mean([r["throughput"] for r in clean])
    print(f"  {'Throughput (pairs/s)':>20} {orig_throughput:>14,.0f} {clean_throughput:>14,.0f}")

    # ── Plot ──────────────────────────────────────────────────────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Mollusca Training: Clean vs Original ({args.device.upper()}, {args.epochs} epochs)",
                 fontsize=14, fontweight="bold")

    # Loss curves
    ax = axes[0]
    for key, ds in DATASETS.items():
        results = all_results[key]
        epochs = [r["epoch"] for r in results]
        losses = [r["loss"] for r in results]
        ax.plot(epochs, losses, marker="o", label=ds["label"], color=ds["color"], linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss convergence")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Epoch time
    ax = axes[1]
    for key, ds in DATASETS.items():
        results = all_results[key]
        epochs = [r["epoch"] for r in results]
        times = [r["time_sec"] for r in results]
        ax.plot(epochs, times, marker="s", label=ds["label"], color=ds["color"], linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Time (seconds)")
    ax.set_title(f"Epoch wall time (clean {speedup:.1f}x faster)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Throughput
    ax = axes[2]
    for key, ds in DATASETS.items():
        results = all_results[key]
        epochs = [r["epoch"] for r in results]
        tp = [r["throughput"] for r in results]
        ax.plot(epochs, tp, marker="^", label=ds["label"], color=ds["color"], linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Pairs/second")
    ax.set_title("Training throughput")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved to {args.out}")


if __name__ == "__main__":
    main()
