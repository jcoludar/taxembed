"""CUDA/MPS/CPU device management for embedding extraction."""

import torch


def get_device() -> torch.device:
    """Return best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_device(tensor: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
    """Move tensor to device, cast to float32."""
    if device is None:
        device = get_device()
    return tensor.to(device=device, dtype=torch.float32)
