#!/usr/bin/env python3
"""Which closure and which model does every recorded S_angle belong to? Provenance audit.

WHY. helpers/_c9_masked_sangle.py measured S_angle = 0.9653 on the SHIPPED cellular release, and I
flagged a "0.0073 unexplained gap" against the manuscript headline 0.9726. That flag was wrong: the
headline's source, results/fig4_runs_20260922_104857.json, records

    closure  /data/taxonomy_edges_metazoa_33208_clean_transitive.npz
    n_nodes  498,246        max_depth 37
    runs     prior, canonical, prior_roll, canonical_roll

-- METAZOA (498k nodes), scored on Task-9 RECIPE-CONTRAST checkpoints. The masking run is
CELLULAR (1,102,163 nodes), scored on release/taxembed-cellular-v1/cellular_embedding.safetensors.
The two were never the same quantity, so there was no gap to reconcile
([[feedback_two_numbers_are_comparable_only_if_their_definitions_are]]).

That raises the real question, which this audit answers: does ANY recorded S_angle belong to the
cellular closure and the shipped artifact? If not, a Metazoa recipe-contrast number is standing in
the manuscript as the released model's score.

Index alignment for the masking run is separately confirmed: md5 of
release/taxembed-cellular-v1/taxid_to_index.tsv equals md5 of the cellular closure's mapping.tsv
(a8ef06b048e03613230a0a14908f515e), so release row i IS closure node i.

Counts and provenance only; computes no new score.
Written 2026-09-29.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings")
RESULTS = ROOT / "results"


def walk(obj, path=""):
    """Yield (path, value) for every S_angle-ish scalar anywhere in a nested structure."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("S_angle", "S") and isinstance(v, (int, float)):
                yield f"{path}.{k}", float(v)
            else:
                yield from walk(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:200]):
            yield from walk(v, f"{path}[{i}]")


def main() -> None:
    print("=" * 96)
    print("S_angle PROVENANCE AUDIT — which closure, which model, for every recorded score")
    print("=" * 96)

    rows = []
    for p in sorted(RESULTS.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:                                    # noqa: BLE001
            continue
        if not isinstance(d, dict):
            continue
        closure = d.get("closure")
        n_nodes = d.get("n_nodes")
        if closure is None and n_nodes is None:
            continue
        scores = [(k, v) for k, v in walk(d) if 0.5 < v <= 1.0]
        top = max((v for _, v in scores), default=None)
        rows.append({"file": p.name, "closure": closure, "n_nodes": n_nodes,
                     "n_scores": len(scores), "max_S": top,
                     "runs": list(d.get("runs", {}).keys()) if isinstance(d.get("runs"), dict)
                     else None})

    print(f"\n  {'file':<46}{'n_nodes':>10}  closure / runs")
    print("  " + "-" * 92)
    cellular, metazoa, other = [], [], []
    for r in rows:
        c = (r["closure"] or "")
        tag = ("CELLULAR" if "cellular" in c else
               "METAZOA" if "metazoa" in c else
               "mollusca" if "mollusca" in c else "?")
        (cellular if tag == "CELLULAR" else metazoa if tag == "METAZOA" else other).append(r)
        mx = f"{r['max_S']:.4f}" if r["max_S"] is not None else "   -  "
        nn = f"{r['n_nodes']:,}" if isinstance(r["n_nodes"], int) else "?"
        print(f"  {r['file']:<46}{nn:>10}  {tag:<9} maxS={mx}  runs={r['runs']}")

    print("\n" + "=" * 96)
    print(f"  files on a CELLULAR closure : {len(cellular)}")
    print(f"  files on a METAZOA closure  : {len(metazoa)}")
    print(f"  other / unlabelled          : {len(other)}")
    if not cellular:
        print("\n  🛑 NO recorded S_angle belongs to the CELLULAR closure.")
        print("     ⇒ the manuscript's headline S_angle is a METAZOA number, and the shipped")
        print("       artifact is the CELLULAR model. Those are different models on different")
        print("       trees. This must be resolved before the headline is used as the released")
        print("       model's score.")
    else:
        for r in cellular:
            print(f"     cellular: {r['file']}  maxS={r['max_S']}")

    outp = RESULTS / "sangle_provenance_audit_20260929.json"
    json.dump({"purpose": "which closure/model each recorded S_angle belongs to",
               "rows": rows,
               "n_cellular": len(cellular), "n_metazoa": len(metazoa), "n_other": len(other),
               "mapping_md5_release_equals_closure": "a8ef06b048e03613230a0a14908f515e"},
              outp.open("w"), indent=2)
    print(f"\nwritten: {outp}")


if __name__ == "__main__":
    main()
