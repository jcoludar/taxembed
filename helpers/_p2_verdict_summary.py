#!/usr/bin/env python3
"""Summarise results/p2_verdict_20260929.json without dumping 55 KB into context.

Read-only. Prints the fields needed for the below-chance-MRR diagnosis: what the run
scored from, the gate records, and per-arm rank statistics against each arm's own
chance baseline.

Written 2026-09-29 for the P2 below-chance-MRR diagnosis.
"""
import json
from pathlib import Path

ROOT = Path("/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings")
VERDICT = ROOT / "results" / "p2_verdict_20260929.json"


def main():
    d = json.loads(VERDICT.read_text())

    print("=" * 78)
    print("PROVENANCE")
    print("=" * 78)
    print("scored_from:", d.get("scored_from"))
    print("verdict    :", d.get("verdict"))
    print("meaning    :", d.get("meaning"))
    print("seeds      :", d.get("seeds"))

    print()
    print("=" * 78)
    print("GATES + INVALID RUNS (per arm)")
    print("=" * 78)
    arms = {}
    arms.update(d.get("groups", {}))
    arms.update(d.get("controls", {}))
    for name, g in arms.items():
        print(f"\n--- {name} ---")
        print("  per_run_values (MRR)   :", g.get("per_run_values"))
        print("  chance_mrr_mean        :", g.get("chance_mrr_mean"))
        print("  norm_rank per run      :", g.get("per_run_values_normalized_rank"))
        print("  invalid_runs           :", json.dumps(g.get("invalid_runs"))[:400])
        for i, gate in enumerate(g.get("gates", [])):
            print(f"  gate[{i}]               :", json.dumps(gate)[:600])

    print()
    print("=" * 78)
    print("DEGREE PRIOR (non-embedding baseline, same harness)")
    print("=" * 78)
    print(json.dumps(d.get("degree_prior"), indent=2)[:1500])

    print()
    print("=" * 78)
    print("PREREGISTRATION")
    print("=" * 78)
    print(json.dumps(d.get("preregistration"), indent=2)[:2500])


if __name__ == "__main__":
    main()
