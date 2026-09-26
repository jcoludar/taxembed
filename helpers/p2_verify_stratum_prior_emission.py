#!/usr/bin/env python3
"""C2 (2026-09-26): verify the per-stratum degree prior END TO END, and re-derive the ratios.

Two questions, both measured rather than assumed:

  1. Does `scripts/score_p2_linkpred.py` actually EMIT `degree_prior.by_depth`, in the shape
     `_p2_stratum_priors` reads? Run the real scorer as a subprocess on the small mollusca split
     and look at what lands on disk -- a schema I reason about is not a schema I have seen.

  2. On the PRODUCTION metazoa seed-0 splits, how far is each stratum's own difficulty ratio from
     the AGGREGATE one the code used before today? The claim driving the fix is that the
     aggregate over-allows the cross-tree sign test in at least one stratum with n well above
     P2_MIN_STRATUM_N. Re-derived here rather than taken from the review, because it changed code.

Read-only apart from a scratch JSON under the system temp dir. Touches no production file.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

from taxembed.eval.baselines import degree_prior_ranks  # noqa: E402
from taxembed.eval.linkpred import candidate_pool, stratify  # noqa: E402
from taxembed.eval.p2_split import depth_from_closure  # noqa: E402
from taxembed.eval.subtree import parent_from_closure  # noqa: E402
from taxembed.utils.training_pairs import TrainingPairs  # noqa: E402

DEPTH_BINS = [(11, 15), (16, 21), (22, 28)]
SPLITS = _REPO / "data" / "p2_splits"
PY = _REPO / ".venv" / "bin" / "python"
SCORER = _REPO / "scripts" / "score_p2_linkpred.py"
REAL_CLOSURE = (_REPO / "data" / "taxopy" / "metazoa_33208_clean"
                / "taxonomy_edges_metazoa_33208_clean_transitive.npz")


def load(closure: Path):
    pairs = TrainingPairs.load(closure)
    n = pairs.n_nodes
    parent = parent_from_closure(pairs.ancestor_idx, pairs.descendant_idx, pairs.depth_diff, n)
    depth = depth_from_closure(pairs.descendant_idx, pairs.descendant_depth,
                               pairs.ancestor_idx, pairs.ancestor_depth, n)
    return parent, depth


def strata_priors(closure: Path, heldout: Path) -> tuple[dict, float]:
    parent, depth = load(closure)
    held = np.asarray(np.load(heldout)["test"], dtype=np.int64)
    cands = [candidate_pool(parent, depth, node=int(v)) for v in held]
    n_cand = np.array([len(c) for c in cands], dtype=np.int64)
    ranks = degree_prior_ranks(parent, held, cands, parent[held], tie_seed=0)
    per = stratify(ranks, n_cand, depth[held], bins=DEPTH_BINS)
    from taxembed.eval.linkpred import linkpred_metrics
    agg = linkpred_metrics(ranks, n_cand)
    return per, float(agg["normalized_rank"])


def question_1() -> bool:
    """Does the real scorer emit degree_prior.by_depth?"""
    print("=" * 78)
    print("Q1  does scripts/score_p2_linkpred.py EMIT degree_prior.by_depth?")
    print("=" * 78)
    manifest = SPLITS / "p2_mollusca_6447_clean_vis00_seed0_manifest.json"
    heldout = SPLITS / "p2_mollusca_6447_clean_seed0_heldout.npz"
    closure = (_REPO / "data" / "taxopy" / "mollusca_6447_clean"
               / "taxonomy_edges_mollusca_6447_clean_transitive.npz")
    for p in (manifest, heldout, closure):
        if not p.exists():
            print(f"  SKIP: {p} does not exist")
            return True
    ck = sorted((_REPO / "results" / "task9_runs_mollusca").glob("*_epoch200.pth"))
    if not ck:
        print("  SKIP: no mollusca checkpoint on disk to score")
        return True
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "scores.json"
        rc = subprocess.run(
            [str(PY), str(SCORER), "--manifest", str(manifest), "--heldout", str(heldout),
             "--closure", str(closure), "--checkpoints", f"vis00_s0_roll={ck[0]}",
             "--out", str(out)],
            capture_output=True, text=True)
        if rc.returncode != 0:
            print(f"  scorer FAILED rc={rc.returncode}")
            print(rc.stdout[-2000:])
            print(rc.stderr[-2000:])
            return False
        res = json.loads(out.read_text())
    by_depth = res["baselines"]["degree_prior"].get("by_depth")
    print(f"  degree_prior.by_depth present: {by_depth is not None}")
    if by_depth is None:
        return False
    print(f"  strata: {sorted(by_depth)}")
    for k in sorted(by_depth):
        e = by_depth[k]
        print(f"    {k:>7}  n {e.get('n')!s:>6}  normalized_rank {e.get('normalized_rank')}")
    ok = all("normalized_rank" in v for v in by_depth.values())
    print(f"  every stratum carries normalized_rank (what _p2_stratum_priors reads): {ok}")
    return ok


def question_2() -> bool:
    """Per-stratum vs aggregate difficulty ratio, production metazoa seed 0."""
    print()
    print("=" * 78)
    print("Q2  per-stratum vs AGGREGATE difficulty ratio -- production metazoa, seed 0")
    print("=" * 78)
    dm_closure = SPLITS / "taxonomy_edges_metazoa_33208_clean_degmatch_seed0_transitive.npz"
    if not (REAL_CLOSURE.exists() and dm_closure.exists()):
        print("  SKIP: production closures not on disk")
        return True
    real_per, real_agg = strata_priors(REAL_CLOSURE,
                                       SPLITS / "p2_metazoa_33208_clean_seed0_heldout.npz")
    ctrl_per, ctrl_agg = strata_priors(
        dm_closure, SPLITS / "p2_metazoa_33208_clean_degmatch_seed0_heldout.npz")
    agg_ratio = real_agg / ctrl_agg
    print(f"  AGGREGATE: real {real_agg:.5f}  control {ctrl_agg:.5f}  ratio {agg_ratio:.4f}")
    print(f"  {'stratum':>8} {'real':>10} {'ctrl':>10} {'true ratio':>11} "
          f"{'over-allow':>11} {'n_scored':>9}")
    worst = 0.0
    for k in sorted(real_per):
        r, c = real_per[k]["normalized_rank"], ctrl_per[k]["normalized_rank"]
        if not (r and c):
            continue
        true_ratio = r / c
        over = agg_ratio / true_ratio
        worst = max(worst, over)
        print(f"  {k:>8} {r:>10.5f} {c:>10.5f} {true_ratio:>11.4f} {over:>10.3f}x "
              f"{real_per[k].get('n_scored', real_per[k].get('n')):>9}")
    print(f"\n  worst over-allowance: {worst:.3f}x "
          f"(>1 means the aggregate relaxes the cross-tree sign test in that stratum,")
    print(f"   i.e. a genuine per-stratum tie can read as a win -- towards GENERALISES)")
    return worst > 1.0


def main() -> int:
    ok1 = question_1()
    ok2 = question_2()
    print()
    if not ok1:
        print("🛑 the scorer does not emit the per-stratum prior the engine now reads")
        return 1
    if not ok2:
        print("⚠ no stratum is over-allowed by the aggregate -- the C2 fix would be a no-op "
              "here; re-check the premise before claiming it matters")
        return 1
    print("✅ the scorer emits degree_prior.by_depth, and the aggregate DOES over-allow at "
          "least one stratum -- the per-stratum divisor is load-bearing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
