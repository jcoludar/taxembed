"""How large do the hyperbolic radii actually get, and is the exact-distance path safe on them?

WHY (2026-09-24). P2's exact Poincare distance uses the hyperbolic law of cosines,
`cosh d = cosh(r_u)cosh(r_v) - sinh(r_u)sinh(r_v)cos(theta)`, with radii taken from `z_embeddings`
as `|z|` (the convention in scripts/score_recipe_checkpoints.py:61-75).

A review raised that this is unbounded in principle: the canonical recipe trains with
`--euclidean-param`, under which `project_to_ball` is a NO-OP and `radial_regularizer` penalises
`tanh(|z|/2)`, which SATURATES -- so its gradient on `|z|` vanishes once `|z|` is a few units out
and nothing hard-bounds the norm. cosh/sinh overflow float64 between r=709 and r=710, and
catastrophic cancellation bites well before that when two radii are similar and cos(theta) ~ 1.

Radii derived instead from an in-ball point as `2*artanh(|x|)` are structurally capped near 37.4,
because float64 cannot represent |x| closer to 1 than ~2.2e-16.

So: MEASURE the real distribution rather than argue about it. Reports, per checkpoint, the
|z|-derived and |x|-derived radii side by side, their disagreement, and the headroom to overflow.

Read-only. Usage: <venv-python> helpers/p2_measure_hyperbolic_radii.py <checkpoint.pth> [...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

OVERFLOW_R = 709.78   # cosh/sinh overflow float64 just above this
ARTANH_CAP = 37.43    # 2*artanh(1 - 2.2e-16): the hardest cap an in-ball float64 point can reach


def q(a: np.ndarray, name: str) -> None:
    ps = np.percentile(a, [0, 50, 90, 99, 99.9, 100])
    print(f"    {name:<22} min {ps[0]:9.4f} | p50 {ps[1]:8.4f} | p90 {ps[2]:8.4f} | "
          f"p99 {ps[3]:8.4f} | p99.9 {ps[4]:8.4f} | MAX {ps[5]:9.4f}")


def report(path: Path) -> None:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    print(f"\n=== {path.name} ===")
    if not isinstance(ck, dict):
        print("  not a dict checkpoint")
        return

    x = ck.get("embeddings")
    z = ck.get("z_embeddings")
    if x is None and "state_dict" in ck:
        x = ck["state_dict"].get("lt.weight")

    if x is not None:
        xn = np.linalg.norm(x.detach().double().numpy(), axis=1)
        print(f"  ball norms |x|: n={len(xn):,}")
        q(xn, "|x|")
        n_at_one = int((xn >= 1.0).sum())
        print(f"    |x| >= 1.0 (outside ball): {n_at_one:,}")
        safe = np.clip(xn, 0.0, 1.0 - 1e-16)
        r_from_x = 2.0 * np.arctanh(safe)
        q(r_from_x, "r = 2*artanh(|x|)")

    if z is None:
        print("  NO z_embeddings -- the exact path would fall back to ball coordinates")
        return

    zn = np.linalg.norm(z.detach().double().numpy(), axis=1)
    print(f"  euclidean-param norms |z| (the canonical radius source): n={len(zn):,}")
    q(zn, "r = |z|")

    rmax = float(zn.max())
    print(f"\n  overflow headroom: max r = {rmax:.4f} vs float64 cosh/sinh limit {OVERFLOW_R:.2f} "
          f"=> factor {OVERFLOW_R / max(rmax, 1e-9):.1f}x")
    print(f"  cosh(max r) = {np.cosh(np.clip(rmax, 0, 700)):.6g}"
          f"{'  [CLIPPED FOR DISPLAY]' if rmax > 700 else ''}")
    if rmax > OVERFLOW_R:
        print("  🛑 OVERFLOW: the exact path WILL produce inf/nan on this checkpoint")
    elif rmax > 100:
        print("  ⚠ large radii: check catastrophic cancellation for similar radii at cos(theta)~1")
    else:
        print("  ✅ safe: far below the overflow bound")

    if x is not None:
        d = np.abs(r_from_x - zn)
        print(f"\n  |z| vs 2*artanh(|x|) disagreement: median {np.median(d):.6g}, max {d.max():.6g}")
        print(f"  (the two agree only where |x| = tanh(|z|/2) holds numerically; near the boundary")
        print(f"   artanh saturates at ~{ARTANH_CAP:.2f} while |z| keeps going -- that gap IS the")
        print(f"   precision the exact path buys.)")
        n_saturated = int((r_from_x >= ARTANH_CAP - 0.5).sum())
        print(f"  nodes where 2*artanh(|x|) is at its float64 ceiling: {n_saturated:,}")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    for p in sys.argv[1:]:
        report(Path(p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
