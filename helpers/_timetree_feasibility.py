#!/usr/bin/env python3
"""Spec §4.5 TimeTree — FEASIBILITY COUNTS ONLY. No distance, no correlation, no statistic.

🛑 DELIBERATELY COUNTS ONLY. §4.5's endpoint is a pair of Spearman correlations, and a
pre-registration is worth nothing if it is written after its own answer is visible. This helper
establishes only (a) can the data be joined at all, and (b) is there enough of it -- exactly the
role helpers/_p3_placement_feasibility.py played before the P3 pre-registration was frozen. NOTHING
here touches an embedding coordinate or a divergence time value.

WHAT THE PROBE ESTABLISHED (helpers/_timetree_fetch_probe.py, results/timetree_fetch_probe_*.json):
  - TimetreeOfLife2015.nwk: HTTP 200, 1.9 MB, 0.70 s. ONE request for the whole dated tree.
  - the pairwise API returns CSV keyed on taxon_a_id / taxon_b_id, and 9606/10090 resolve to
    Homo sapiens / Mus musculus -> 🎯 THOSE ARE NCBI TAXIDS, the identifier the embedding is already
    keyed on. Name reconciliation, normally the expensive half of this job, is only needed for the
    bulk newick route and is a straight names.dmp lookup.
  ⇒ "days" was wrong by orders of magnitude, again
     ([[feedback_sequence_dont_cut_and_dont_overblow_time]]).

BULK ROUTE PREFERRED, and not only for speed: one 1.9 MB request against a public academic service
is kinder than tens of thousands of API calls. The API is then used only to spot-check a handful of
newick-derived ages.

WHY §4.5 SURVIVES FINDING 4. Finding 4 (§4.7) is that outside-the-tree information must enter the
ANSWER, not merely the QUESTION -- P3's labels were unseen but predictable from candidate subtree
size, so it measured subtree size. §4.5 is the one remaining protocol that names its cheap
tree-internal baseline BY CONSTRUCTION: it reports Spearman(embedded distance, divergence time)
beside Spearman(NCBI path length, divergence time). The baseline is the comparison, not an
afterthought.

Written 2026-09-29.
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path("/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings")
sys.path.insert(0, str(ROOT / "helpers"))
sys.path.insert(0, str(ROOT / "src"))

from _p3_placement_score import (  # noqa: E402
    MAPPING, OLD_DATE, build_tree, load_mapping,
)
from taxembed.eval.release_diff import parse_parents  # noqa: E402

NWK_URL = "https://timetree.org/public/data/TimetreeOfLife2015.nwk"
NWK_PATH = ROOT / "data" / "timetree" / "TimetreeOfLife2015.nwk"
NAMES = ROOT / "data" / f"taxdump_archive_{OLD_DATE}" / "names.dmp"
CLADES = {"Vertebrata": 7742, "Insecta": 50557, "Mammalia": 40674,
          "Aves": 8782, "Actinopterygii": 7898}


def fetch_newick() -> float:
    if NWK_PATH.exists():
        print(f"  newick already on disk: {NWK_PATH} ({NWK_PATH.stat().st_size/1e6:.1f} MB)")
        return 0.0
    NWK_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    req = urllib.request.Request(NWK_URL, headers={"User-Agent": "Mozilla/5.0 (research)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        NWK_PATH.write_bytes(r.read())
    dt = time.time() - t0
    print(f"  downloaded {NWK_PATH.stat().st_size/1e6:.1f} MB in {dt:.2f} s -> {NWK_PATH}")
    return dt


def newick_leaves(path: Path) -> list[str]:
    """Leaf labels of a newick: tokens after '(' or ',' and before ':'."""
    text = path.read_text()
    return re.findall(r"[(,]\s*([A-Za-z][A-Za-z0-9_.\-']*)\s*:", text)


def scientific_name_to_taxid(path: Path) -> dict[str, int]:
    """names.dmp scientific names, underscored to match newick labels."""
    out: dict[str, int] = {}
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "scientific name" not in line:
                continue
            parts = line.split("\t|\t")
            if len(parts) < 2:
                continue
            try:
                tid = int(parts[0].strip())
            except ValueError:
                continue
            out[parts[1].strip().replace(" ", "_")] = tid
    return out


def main() -> None:
    print("=" * 92)
    print("§4.5 TimeTree — FEASIBILITY COUNTS ONLY (no distance, no correlation)")
    print("=" * 92)

    dt = fetch_newick()
    leaves = newick_leaves(NWK_PATH)
    uniq = sorted(set(leaves))
    print(f"\n  TTOL leaf labels: {len(leaves):,} ({len(uniq):,} unique)")
    print(f"  examples: {uniq[:3]}")

    print(f"\n  parsing {NAMES.name} ({NAMES.stat().st_size/1e6:.0f} MB) ...")
    t0 = time.time()
    name2tax = scientific_name_to_taxid(NAMES)
    print(f"  {len(name2tax):,} scientific names in {time.time()-t0:.1f} s")

    resolved = {n: name2tax[n] for n in uniq if n in name2tax}
    print(f"  TTOL leaves resolved to an NCBI taxid: {len(resolved):,} / {len(uniq):,} "
          f"({100*len(resolved)/max(len(uniq),1):.1f} %)")

    taxid2row = load_mapping(MAPPING)
    in_emb = {n: t for n, t in resolved.items() if t in taxid2row}
    print(f"  ...and carrying an EMBEDDING coordinate: {len(in_emb):,} "
          f"({100*len(in_emb)/max(len(uniq),1):.1f} % of TTOL leaves)")

    # clade breakdown, via the old tree's euler intervals
    print(f"\n  parsing nodes.dmp for the clade breakdown ...")
    parent_map = parse_parents(ROOT / "data" / f"taxdump_archive_{OLD_DATE}" / "nodes.dmp")
    taxids, idx, parent, depth, tin, tout = build_tree(parent_map)
    usable = np.array(sorted(set(in_emb.values())), dtype=np.int64)
    rows = np.array([idx[int(t)] for t in usable if int(t) in idx], dtype=np.int64)
    print(f"  usable taxa located in the {OLD_DATE} tree: {len(rows):,}")

    breakdown = {}
    for cname, ctax in CLADES.items():
        ci = idx.get(ctax)
        if ci is None:
            breakdown[cname] = None
            continue
        n = int(np.sum((tin[rows] >= tin[ci]) & (tin[rows] < tout[ci])))
        breakdown[cname] = n
        pairs = n * (n - 1) // 2
        print(f"    {cname:<16} {n:>6,} usable taxa   ({pairs:,} pairs available)")

    total_pairs = len(rows) * (len(rows) - 1) // 2
    print(f"\n  TOTAL usable taxa {len(rows):,}  ⇒  {total_pairs:,} pairs available")
    print("  (§4.5 asks for 'a few hundred well-sampled vertebrates and insects')")

    outp = ROOT / "results" / "timetree_feasibility_20260929.json"
    json.dump({
        "purpose": "spec §4.5 feasibility; COUNTS ONLY, no statistic computed",
        "newick": {"url": NWK_URL, "path": str(NWK_PATH),
                   "bytes": NWK_PATH.stat().st_size, "download_seconds": round(dt, 2)},
        "ttol_leaf_labels": len(leaves), "ttol_unique": len(uniq),
        "resolved_to_taxid": len(resolved),
        "resolved_and_embedded": len(in_emb),
        "located_in_tree": int(len(rows)),
        "pairs_available": int(total_pairs),
        "clade_breakdown": breakdown,
        "names_source": str(NAMES),
    }, outp.open("w"), indent=2)
    print(f"\nwritten: {outp}")


if __name__ == "__main__":
    main()
