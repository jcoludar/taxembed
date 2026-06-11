"""Device-aware (CUDA / MPS / CPU) batched kNN observed-purity + vectorized matched-null for App #2.

The per-node anomaly score needs EVERY pool node as a query against the full pool — an exact
O(P^2) kNN. The numpy/single-core path (`knn_purity_hyperbolic._batch_distances` + argpartition)
is fine for the SUBSAMPLED queries the original anti-Goodhart check used, but for FULL-pool scoring
at 877k / 1.1M it is the bottleneck (single core, ~15 GB, no ETA). This module runs the identical
Poincaré neighbour ordering on GPU via batched `torch.topk`, and vectorizes the matched-null
sampling (the former per-node Python loop over ~877k nodes is replaced by one draw per bin).

Distance semantics are byte-compatible with `knn_purity_hyperbolic`:
    arg = 1 + 2*||u-v||^2 / ((1-clip(||u||^2))(1-clip(||v||^2))) ,  d = arccosh(arg)
arccosh is monotone, so for kNN we rank by `arg` directly (skip arccosh). The numpy kernel already
returns float32 distances, so an all-float32 torch path reproduces its neighbour ordering to within
the same tolerance `_validate_distance` accepts — and float32 is portable to MPS (no float64 there).
Neighbour ordering is cross-checked against the numpy kernel in tests/eval/test_anomaly_knn.py.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-5


def pick_device(prefer: str | None = None) -> str:
    """Resolve the compute device: explicit > cuda > mps > cpu."""
    import torch

    if prefer and prefer != "auto":
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _safe_batch(P, requested, budget_bytes=4 * 1024 ** 3, n_buffers=8, bytes_per_elem=4):
    """Cap the query-block size so the live (batch x P) float32 intermediates fit a memory budget.

    The hot loop holds ~`n_buffers` simultaneous (batch x P) float32 tensors at peak (dot, the
    broadcast sum, sq_diff, denom, arg, plus topk workspace). At P=864k a `batch` of 2048 needs
    ~6.6 GiB *per* buffer and OOM'd a 16 GiB V100 (job 5674199). This shrinks the effective block so
    `n_buffers * batch * P * bytes_per_elem <= budget_bytes`, leaving `requested` untouched whenever
    the pool is small enough that the cap doesn't bind (e.g. the echino 4k regime). Never returns 0
    (a zero step would stall the loop). Result ordering is batch-invariant, so capping is lossless.
    """
    requested = max(1, int(requested))
    if P <= 0:
        return requested
    per_row = n_buffers * int(P) * bytes_per_elem
    cap = max(1, int(budget_bytes) // per_row)
    return max(1, min(requested, cap))


def observed_purity(emb, pool_idx, pool_lab, k, device=None, batch=1024, eps=EPS,
                    mem_budget_bytes=4 * 1024 ** 3):
    """Fraction of each pool node's k nearest neighbours (Poincaré) sharing its label.

    Scores EVERY pool node as a query against the full pool (self excluded). Returns a float64
    numpy array aligned to `pool_idx`. Runs on `device` (auto-resolved if None) in float32 blocks of
    queries; memory is dominated by the (block x P) distance matrix, so the requested `batch` is
    capped via `_safe_batch` to keep the working set under `mem_budget_bytes` (default 4 GiB). The
    cap only shrinks the block — results are batch-invariant — so it never changes the scores.
    """
    import torch

    dev = pick_device(device)
    emb = np.asarray(emb, dtype=np.float32)
    pool_idx = np.asarray(pool_idx, dtype=np.int64)
    pool_lab = np.asarray(pool_lab, dtype=np.int64)

    pe = torch.as_tensor(emb[pool_idx], dtype=torch.float32, device=dev)        # (P, D)
    lab = torch.as_tensor(pool_lab, dtype=torch.long, device=dev)               # (P,)
    P = pe.shape[0]
    if P < 2:
        return np.zeros(P, dtype=np.float64)

    raw = (pe * pe).sum(dim=1)                                                  # (P,) ||v||^2
    one_minus_clip = 1.0 - raw.clamp(0.0, 1.0 - eps)                            # conformal factor
    keff = min(int(k), P - 1)
    peT = pe.T.contiguous()
    out = torch.empty(P, dtype=torch.float32, device=dev)        # float32: MPS has no float64

    batch = _safe_batch(P, batch, budget_bytes=mem_budget_bytes)
    for s in range(0, P, batch):
        e = min(s + batch, P)
        q = pe[s:e]                                                            # (b, D)
        q_raw = raw[s:e]                                                       # (b,)
        q_omc = one_minus_clip[s:e]                                            # (b,)
        dot = q @ peT                                                         # (b, P) — heavy op
        sq_diff = (q_raw[:, None] + raw[None, :] - 2.0 * dot).clamp_min(0.0)
        denom = q_omc[:, None] * one_minus_clip[None, :]
        arg = 1.0 + 2.0 * sq_diff / denom                                     # monotone in distance
        rows = torch.arange(e - s, device=dev)
        arg[rows, s + rows] = float("inf")                                    # exclude self
        nn = arg.topk(keff, dim=1, largest=False).indices                     # (b, keff)
        match = (lab[nn] == lab[s:e][:, None]).float().mean(dim=1)
        out[s:e] = match

    return out.detach().cpu().numpy().astype(np.float64)


def _qbin(x, n_bins):
    x = np.asarray(x, dtype=np.float64)
    qs = np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1]) if n_bins > 1 else np.array([])
    return np.digitize(x, qs)


def matched_null(observed, pool_idx, depth, clade_size, n_null, n_bins, seed):
    """Per-query null observed-purity drawn from the SAME depth x clade-size stratum (vectorized).

    Replaces the former per-node Python loop: for each (small) depth x size bin we draw an
    (m, n_null) index matrix in one call and gather, so the only loop is over the <= (n_bins+1)^2
    bins. Returns (Q, n_null) float64 aligned to `pool_idx`.
    """
    rng = np.random.default_rng(seed)
    observed = np.asarray(observed, dtype=np.float64)
    depth = np.asarray(depth)[np.asarray(pool_idx, dtype=np.int64)]
    csize = np.asarray(clade_size)[np.asarray(pool_idx, dtype=np.int64)]
    bins = _qbin(depth, n_bins) * (n_bins + 1) + _qbin(csize, n_bins)
    Q = len(observed)
    null = np.empty((Q, n_null), dtype=np.float64)
    for b in np.unique(bins):
        members = np.flatnonzero(bins == b)                                   # (m,)
        m = len(members)
        draw = rng.integers(0, m, size=(m, n_null))                          # (m, n_null) -> members
        null[members] = observed[members[draw]]
    return null
