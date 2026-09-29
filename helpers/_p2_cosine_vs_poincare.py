#!/usr/bin/env python3
"""P2 below-chance diagnosis: does the POINCARE metric behave where COSINE does not?

The banked scoring artifacts record `primary_metric = cosine`, but every checkpoint also
carries `metrics_poincare`. If the Poincare reading sits above its own chance floor while
cosine sits below, the UNINFORMATIVE verdict is a metric-selection artifact, not a model
failure.

Read-only over artifacts/p2_scoring/*.json. No retrain, no LRZ.

Written 2026-09-29 for the P2 below-chance-MRR diagnosis.
"""
import json
from pathlib import Path

ROOT = Path("/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings")
ARTDIR = ROOT / "artifacts" / "p2_scoring"


def row(cp):
    """Pull the two metric readings off one checkpoint record."""
    out = {"epoch": cp.get("epoch")}
    for tag in ("metrics_cosine", "metrics_poincare"):
        m = cp.get(tag) or {}
        out[tag] = {
            "mrr": m.get("mrr"),
            "normalized_rank": m.get("normalized_rank"),
            "hits_at_1": m.get("hits_at_1"),
            "n_scored": m.get("n_scored"),
        }
    out["radius_source"] = cp.get("poincare_radius_source")
    out["max_radius"] = cp.get("max_radius")
    out["radius_guard"] = cp.get("radius_guard")
    out["radius_overflow_risk"] = cp.get("radius_overflow_risk")
    return out


def fmt(v, nd=4):
    return "  None " if v is None else f"{v:.{nd}f}"


def main():
    for path in sorted(ARTDIR.glob("*.json")):
        d = json.loads(path.read_text())
        chance = (d.get("baselines_s0") or {}).get("chance_mrr_mean")
        if chance is None:
            for k, v in d.items():
                if k.startswith("baselines_") and isinstance(v, dict):
                    chance = v.get("chance_mrr_mean")
                    break
        print("=" * 100)
        print(f"{path.name}   primary_metric={d.get('primary_metric')}   "
              f"chance_mrr={fmt(chance)}   n_held={d.get('n_held')}")
        print("=" * 100)
        for arm_name, arm in (d.get("arms") or {}).items():
            cps = arm.get("checkpoints") or []
            if not cps:
                continue
            print(f"\n  --- {arm_name} ({len(cps)} checkpoints) ---")
            print(f"  {'epoch':>6} | {'COSINE mrr':>11} {'norm_rank':>10} | "
                  f"{'POINCARE mrr':>13} {'norm_rank':>10} | {'radius_src':>14} {'max_r':>8}")
            for cp in cps:
                r = row(cp)
                c, p = r["metrics_cosine"], r["metrics_poincare"]
                mr = r["max_radius"]
                print(f"  {str(r['epoch']):>6} | {fmt(c['mrr']):>11} {fmt(c['normalized_rank']):>10} | "
                      f"{fmt(p['mrr']):>13} {fmt(p['normalized_rank']):>10} | "
                      f"{str(r['radius_source']):>14} {fmt(mr, 3) if isinstance(mr, (int, float)) else str(mr):>8}")
            guards = {str(cp.get("radius_guard")) for cp in cps}
            risks = {str(cp.get("radius_overflow_risk")) for cp in cps}
            print(f"  radius_guard values: {guards}   radius_overflow_risk: {risks}")
        print()


if __name__ == "__main__":
    main()
