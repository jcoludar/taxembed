"""The initialization floor for the depth<->norm correlation (spec v3 §1.3).

_initialize_by_depth sets ||x|| = target_radius(depth) at step 0 and radial_regularizer
penalizes deviation from the same target, so the headline depth-norm r is not an outcome
of training. This computes what the correlation is BEFORE any gradient step, which is the
floor the trained value must be reported against.

The random direction does not affect the norm, so this is deterministic -- no seed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train_hierarchical import target_radius  # noqa: E402


def initialization_depth_norm_r(depths, max_depth: int, radial_schedule: str) -> float:
    """Pearson r between initialization Poincare norm and node depth.

    Returns nan when depth has zero variance (correlation undefined).
    """
    depths = np.asarray(depths, dtype=np.float64)
    if depths.std() == 0:
        return float("nan")
    norms = np.array(
        [float(target_radius(int(d), max_depth, radial_schedule)) for d in depths],
        dtype=np.float64,
    )
    if norms.std() == 0:
        return float("nan")
    return float(np.corrcoef(norms, depths)[0, 1])
