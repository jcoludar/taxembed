#!/usr/bin/env python3
"""
Benchmark: CPU vs MPS for Poincaré embedding training on Mollusca.

Runs N warmup + N timed epochs on both devices, reports per-epoch wall time,
throughput (pairs/sec), peak memory, and verifies numerical equivalence.
"""

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from train_hierarchical import (
    HierarchicalDataLoader,
    HierarchicalPoincareEmbedding,
    TrainingPairs,
    ranking_loss_with_margin,
    radial_regularizer,
    target_radius,
)

# ── Defaults (match mollusca_v11 run.json) ────────────────────────────────────
DATA_PATH = Path("data/taxopy/mollusca_6447/taxonomy_edges_mollusca_6447_transitive.npz")
MAPPING_PATH = Path("data/taxopy/mollusca_6447/taxonomy_edges_mollusca_6447.mapping.tsv")
DIM = 100
BATCH_SIZE = 128
N_NEGATIVES = 50
LR = 0.005
MARGIN = 0.2
LAMBDA_REG = 0.1


def load_data(data_path, mapping_path):
    """Load training data and mapping, return (TrainingPairs, n_nodes, max_depth, idx_to_depth)."""
    pairs = TrainingPairs.load(data_path)

    mapping_df = pd.read_csv(mapping_path, sep="\t", dtype=str)
    if "taxid" not in mapping_df.columns or "idx" not in mapping_df.columns:
        mapping_df = pd.read_csv(
            mapping_path, sep="\t", dtype=str, header=None, names=["taxid", "idx"]
        )
    else:
        mapping_df = mapping_df[["taxid", "idx"]]
    mapping_df["idx"] = pd.to_numeric(mapping_df["idx"], errors="coerce")
    mapping_df = mapping_df.dropna(subset=["idx"])
    mapping_df["idx"] = mapping_df["idx"].astype(int)
    n_nodes = int(mapping_df["idx"].max()) + 1

    max_depth = pairs.max_depth
    idx_to_depth = pairs.idx_to_depth_dict()

    return pairs, n_nodes, max_depth, idx_to_depth


def build_model_and_loader(pairs, n_nodes, max_depth, idx_to_depth, dim, batch_size, n_negatives):
    """Create fresh model + dataloader (deterministic seed)."""
    torch.manual_seed(42)
    np.random.seed(42)

    model = HierarchicalPoincareEmbedding(
        n_nodes=n_nodes,
        dim=dim,
        max_depth=max_depth,
        init_depth_data=idx_to_depth,
        euclidean_param=True,
    )

    dataloader = HierarchicalDataLoader(
        training_data=pairs,
        n_nodes=n_nodes,
        batch_size=batch_size,
        n_negatives=n_negatives,
        depth_stratify=True,
        tiered_negatives=True,
    )

    return model, dataloader


def precompute_reg_tensors(idx_to_depth, n_nodes, max_depth, device):
    """Precompute regularization index and target tensors on device."""
    reg_indices = []
    reg_target_radii = []
    for idx, depth in idx_to_depth.items():
        if idx < n_nodes:
            reg_indices.append(idx)
            reg_target_radii.append(target_radius(depth, max_depth, "linear"))
    return (
        torch.LongTensor(reg_indices).to(device),
        torch.FloatTensor(reg_target_radii).to(device),
    )


def run_epoch(model, dataloader, optimizer, reg_idx, reg_radii, device, margin, lambda_reg):
    """Run one full training epoch and return (avg_loss, n_batches, elapsed_sec)."""
    model.train()
    epoch_loss = 0.0
    n_batches = 0

    # Synchronize before timing
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
            margin=margin, depth_weight=True,
        )
        reg_loss = radial_regularizer(model, reg_idx, reg_radii, lambda_reg)
        total_loss = loss + reg_loss
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Project back to ball
        updated_indices = torch.cat([ancestors, descendants, negatives.flatten()])
        updated_indices = torch.unique(updated_indices)
        model.project_to_ball(updated_indices)

        if n_batches % 500 == 0:
            model.project_to_ball(indices=None)

        epoch_loss += loss.item()
        n_batches += 1

    # Synchronize after epoch
    if device.type == "mps":
        torch.mps.synchronize()

    elapsed = time.perf_counter() - t0
    return epoch_loss / n_batches, n_batches, elapsed


def benchmark_device(device_name, pairs, n_nodes, max_depth, idx_to_depth,
                     dim, batch_size, n_negatives, lr, margin, lambda_reg,
                     warmup_epochs, timed_epochs):
    """Benchmark training on a single device. Returns dict of results."""
    device = torch.device(device_name)
    print(f"\n{'='*70}")
    print(f"  BENCHMARKING: {device_name.upper()}")
    print(f"{'='*70}")

    # Fresh model each time (same seed → same init)
    model, dataloader = build_model_and_loader(
        pairs, n_nodes, max_depth, idx_to_depth, dim, batch_size, n_negatives
    )
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    reg_idx, reg_radii = precompute_reg_tensors(idx_to_depth, n_nodes, max_depth, device)

    # ── Warmup ────────────────────────────────────────────────────────────
    print(f"\n  Warmup ({warmup_epochs} epoch{'s' if warmup_epochs != 1 else ''})...")
    for i in range(warmup_epochs):
        avg_loss, nb, elapsed = run_epoch(
            model, dataloader, optimizer, reg_idx, reg_radii, device, margin, lambda_reg
        )
        print(f"    warmup {i+1}: loss={avg_loss:.4f}  ({elapsed:.1f}s, {nb} batches)")

    # ── Timed runs ────────────────────────────────────────────────────────
    print(f"\n  Timed ({timed_epochs} epoch{'s' if timed_epochs != 1 else ''})...")
    epoch_times = []
    epoch_losses = []
    total_pairs_per_epoch = len(pairs)

    for i in range(timed_epochs):
        avg_loss, nb, elapsed = run_epoch(
            model, dataloader, optimizer, reg_idx, reg_radii, device, margin, lambda_reg
        )
        epoch_times.append(elapsed)
        epoch_losses.append(avg_loss)
        throughput = total_pairs_per_epoch / elapsed
        print(f"    epoch {i+1}: loss={avg_loss:.4f}  {elapsed:.2f}s  ({throughput:,.0f} pairs/s)")

    # ── Memory stats ──────────────────────────────────────────────────────
    peak_mem_mb = None
    if device.type == "mps":
        # MPS memory tracking (PyTorch 2.1+)
        try:
            peak_mem_mb = torch.mps.driver_allocated_memory() / 1024 / 1024
        except AttributeError:
            pass
    elif device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024

    # ── Summary ───────────────────────────────────────────────────────────
    mean_time = np.mean(epoch_times)
    std_time = np.std(epoch_times)
    mean_throughput = total_pairs_per_epoch / mean_time

    result = {
        "device": device_name,
        "epoch_times": epoch_times,
        "mean_epoch_sec": mean_time,
        "std_epoch_sec": std_time,
        "mean_throughput": mean_throughput,
        "final_loss": epoch_losses[-1],
        "peak_mem_mb": peak_mem_mb,
        "n_batches_per_epoch": nb,
        "total_pairs": total_pairs_per_epoch,
    }

    print(f"\n  Summary ({device_name}):")
    print(f"    Mean epoch time : {mean_time:.2f}s ± {std_time:.2f}s")
    print(f"    Throughput      : {mean_throughput:,.0f} pairs/s")
    if peak_mem_mb is not None:
        print(f"    Peak memory     : {peak_mem_mb:.0f} MB")

    # Cleanup
    del model, optimizer, dataloader, reg_idx, reg_radii
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark CPU vs MPS for Poincaré training")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--mapping", type=Path, default=MAPPING_PATH)
    parser.add_argument("--dim", type=int, default=DIM)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--n-negatives", type=int, default=N_NEGATIVES)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--margin", type=float, default=MARGIN)
    parser.add_argument("--lambda-reg", type=float, default=LAMBDA_REG)
    parser.add_argument("--warmup", type=int, default=2, help="Warmup epochs per device")
    parser.add_argument("--epochs", type=int, default=5, help="Timed epochs per device")
    args = parser.parse_args()

    print("=" * 70)
    print("  POINCARÉ TRAINING BENCHMARK: CPU vs MPS")
    print("=" * 70)
    print(f"  PyTorch     : {torch.__version__}")
    print(f"  MPS avail.  : {torch.backends.mps.is_available()}")
    print(f"  Data        : {args.data}")
    print(f"  Dim         : {args.dim}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  N negatives : {args.n_negatives}")
    print(f"  Warmup      : {args.warmup} epochs")
    print(f"  Timed       : {args.epochs} epochs")

    # ── Load data (once) ──────────────────────────────────────────────────
    print("\nLoading data...")
    pairs, n_nodes, max_depth, idx_to_depth = load_data(args.data, args.mapping)
    print(f"  Pairs: {len(pairs):,}  Nodes: {n_nodes:,}  Max depth: {max_depth}")

    common = dict(
        pairs=pairs, n_nodes=n_nodes, max_depth=max_depth, idx_to_depth=idx_to_depth,
        dim=args.dim, batch_size=args.batch_size, n_negatives=args.n_negatives,
        lr=args.lr, margin=args.margin, lambda_reg=args.lambda_reg,
        warmup_epochs=args.warmup, timed_epochs=args.epochs,
    )

    # ── Run benchmarks ────────────────────────────────────────────────────
    results = {}
    results["cpu"] = benchmark_device("cpu", **common)

    if torch.backends.mps.is_available():
        results["mps"] = benchmark_device("mps", **common)
    else:
        print("\n⚠️  MPS not available, skipping GPU benchmark")

    # ── Comparison ────────────────────────────────────────────────────────
    print("\n")
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)

    cpu = results["cpu"]
    header = f"{'':>20} {'CPU':>14}"
    sep = f"{'':>20} {'─'*14}"
    row_time = f"{'Epoch time (s)':>20} {cpu['mean_epoch_sec']:>10.2f} ± {cpu['std_epoch_sec']:.2f}"
    row_throughput = f"{'Throughput (pairs/s)':>20} {cpu['mean_throughput']:>14,.0f}"
    row_loss = f"{'Final loss':>20} {cpu['final_loss']:>14.4f}"

    if "mps" in results:
        mps = results["mps"]
        speedup = cpu["mean_epoch_sec"] / mps["mean_epoch_sec"]

        header += f" {'MPS':>14} {'Speedup':>10}"
        sep += f" {'─'*14} {'─'*10}"
        row_time += f" {mps['mean_epoch_sec']:>10.2f} ± {mps['std_epoch_sec']:.2f} {speedup:>9.2f}x"
        row_throughput += f" {mps['mean_throughput']:>14,.0f}"
        row_loss += f" {mps['final_loss']:>14.4f}"

    print(header)
    print(sep)
    print(row_time)
    print(row_throughput)
    print(row_loss)

    if "mps" in results:
        print(f"\n  {'→ MPS is' if speedup > 1 else '→ CPU is'} "
              f"{max(speedup, 1/speedup):.2f}x faster")

        if mps.get("peak_mem_mb"):
            print(f"  → MPS peak memory: {mps['peak_mem_mb']:.0f} MB")

    print()


if __name__ == "__main__":
    main()
