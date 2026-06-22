#!/usr/bin/env python3
"""Unit test for the Nickel-Kiela softmax/NLL Poincaré loss (softmax_loss_from_dists).

Tests the pure distance->loss function (no model needed). Run directly:
    .venv/bin/python tests/test_softmax_loss.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from train_hierarchical import softmax_loss_from_dists


def test_softmax_loss_from_dists():
    # Positive much closer than negatives -> near-zero loss
    loss_close = softmax_loss_from_dists(
        torch.tensor([0.1]), torch.tensor([[5.0, 5.0, 5.0]])
    ).item()
    assert loss_close < 0.05, f"expected ~0, got {loss_close}"

    # Positive much farther than negatives -> large loss
    loss_far = softmax_loss_from_dists(
        torch.tensor([5.0]), torch.tensor([[0.1, 0.1, 0.1]])
    ).item()
    assert loss_far > 3.0, f"expected large, got {loss_far}"

    # Monotonic: increasing positive distance (negatives fixed) raises the loss
    neg = torch.tensor([[1.0, 1.0]])
    losses = [
        softmax_loss_from_dists(torch.tensor([p]), neg).item()
        for p in [0.0, 0.5, 1.0, 2.0, 4.0]
    ]
    assert all(losses[i] < losses[i + 1] for i in range(len(losses) - 1)), losses

    # Gradient w.r.t. positive distance is positive (so optimizer pulls positive closer)
    pos = torch.tensor([1.0], requires_grad=True)
    softmax_loss_from_dists(pos, torch.tensor([[1.0, 1.0]])).sum().backward()
    assert pos.grad.item() > 0, f"expected positive grad, got {pos.grad.item()}"

    print(
        "softmax_loss self-test OK:",
        {
            "close": round(loss_close, 4),
            "far": round(loss_far, 3),
            "monotonic": [round(x, 3) for x in losses],
            "grad": round(pos.grad.item(), 4),
        },
    )


if __name__ == "__main__":
    test_softmax_loss_from_dists()
