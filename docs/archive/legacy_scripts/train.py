#!/usr/bin/env python3
"""Training script for Poincaré embeddings.

This script trains hierarchical embeddings on graph data using Poincaré geometry.
"""

import os
import sys

# Suppress PyTorch verbose logging on startup
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import logging
import shutil
import torch as th
import torch.multiprocessing as mp
import numpy as np

try:
    from hype.adjacency_matrix_dataset import AdjacencyDataset
except Exception:
    AdjacencyDataset = None

from hype import MANIFOLDS, MODELS, build_model, train
from hype.checkpoint import LocalCheckpoint
from hype.graph import eval_reconstruction, load_adjacency_matrix, load_edge_list
from hype.graph_dataset import BatchedDataset
from hype.rsgd import RiemannianSGD

# Optional import for hypernymy evaluation
try:
    from hype.hypernymy_eval import main as hype_eval
except ImportError:
    hype_eval = None

th.manual_seed(42)
np.random.seed(42)


def reconstruction_eval(adj, opt, epoch, elapsed, loss, pth, best):
    """Evaluate reconstruction metrics."""
    chkpnt = th.load(pth, map_location="cpu")
    model = build_model(opt, chkpnt["embeddings"].size(0))
    model.load_state_dict(chkpnt["model"])

    meanrank, maprank = eval_reconstruction(adj, model)
    sqnorms = model.manifold.norm(model.lt)
    return {
        "epoch": epoch,
        "elapsed": elapsed,
        "loss": loss,
        "mean_rank": meanrank.item(),
        "map": maprank.item(),
        "sqnorm_min": sqnorms.min().item(),
        "sqnorm_max": sqnorms.max().item(),
        "sqnorm_mean": sqnorms.mean().item(),
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Poincaré embeddings")
    parser.add_argument(
        "-dset",
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset file (edgelist format)",
    )
    parser.add_argument(
        "-checkpoint",
        "--checkpoint",
        type=str,
        required=True,
        help="Path to save checkpoint",
    )
    parser.add_argument(
        "-dim", "--dim", type=int, default=10, help="Embedding dimension"
    )
    parser.add_argument(
        "-epochs", "--epochs", type=int, default=50, help="Number of epochs"
    )
    parser.add_argument(
        "-negs",
        "--negs",
        type=int,
        default=50,
        help="Number of negative samples",
    )
    parser.add_argument(
        "-burnin",
        "--burnin",
        type=int,
        default=10,
        help="Burn-in period",
    )
    parser.add_argument(
        "-batchsize",
        "--batchsize",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "-model",
        "--model",
        type=str,
        default="distance",
        choices=MODELS.keys(),
        help="Model type",
    )
    parser.add_argument(
        "-manifold",
        "--manifold",
        type=str,
        default="poincare",
        choices=MANIFOLDS.keys(),
        help="Manifold type",
    )
    parser.add_argument(
        "-lr", "--lr", type=float, default=0.1, help="Learning rate"
    )
    parser.add_argument(
        "-gpu", "--gpu", type=int, default=-1, help="GPU ID (-1 for CPU)"
    )
    parser.add_argument(
        "-ndproc",
        "--ndproc",
        type=int,
        default=1,
        help="Number of data loading processes",
    )
    parser.add_argument(
        "-train_threads",
        "--train_threads",
        type=int,
        default=1,
        help="Number of training threads",
    )
    parser.add_argument(
        "-eval_each",
        "--eval_each",
        type=int,
        default=999999,
        help="Evaluate every N epochs",
    )
    parser.add_argument(
        "-fresh",
        "--fresh",
        action="store_true",
        help="Start fresh training",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Loading dataset from {args.dataset}")

    # Load data
    if args.dataset.endswith(".csv"):
        adj = load_adjacency_matrix(args.dataset, "csv")
    else:
        adj = load_edge_list(args.dataset)

    logger.info(f"Dataset loaded: {adj.shape[0]} nodes, {adj.nnz} edges")

    # Build model
    opt = {
        "model": args.model,
        "manifold": args.manifold,
        "dim": args.dim,
        "epochs": args.epochs,
        "negs": args.negs,
        "burnin": args.burnin,
        "batchsize": args.batchsize,
        "lr": args.lr,
        "gpu": args.gpu,
        "ndproc": args.ndproc,
        "train_threads": args.train_threads,
        "eval_each": args.eval_each,
    }

    model = build_model(opt, adj.shape[0])
    logger.info(f"Model built: {args.model} on {args.manifold}")

    # Setup optimizer
    optimizer = RiemannianSGD(model.parameters(), lr=args.lr)

    # Setup checkpoint
    checkpoint = LocalCheckpoint(args.checkpoint, include_in_all=["model"])

    # Load checkpoint if exists and not fresh
    if not args.fresh and os.path.exists(args.checkpoint):
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        chkpnt = th.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(chkpnt["model"])
        optimizer.load_state_dict(chkpnt["optimizer"])

    # Setup dataset
    dataset = BatchedDataset(adj, opt["batchsize"], opt["negs"])

    # Train
    logger.info("Starting training...")
    train(
        model,
        optimizer,
        dataset,
        opt,
        checkpoint,
        reconstruction_eval if args.eval_each < 999999 else None,
        adj if args.eval_each < 999999 else None,
    )

    logger.info(f"Training complete. Model saved to {args.checkpoint}")


if __name__ == "__main__":
    main()
