#!/usr/bin/env python3
"""C9 — count placeholder-named leaves in TaxEmbed's OWN tree. Discharges an owed measurement.

WHY. MANUSCRIPT_CORRECTIONS_PENDING.md's C9 candidate says the `--clean` noise filter
(`taxopy_clade.py:26-33`: sp., cf./aff./nr., environmental, uncultured, unidentified, hybrid) misses
the dominant PROKARYOTE placeholder form, "<clade> bacterium|archaeon <id>" -- e.g.
*Verrucomicrobiia bacterium DG1235*, *Thermoplasmatales archaeon BRNA1* -- whose NCBI placement is
just the clade the submitter typed. On the UU all-cellular panel that form was 3,144 of 4,820
prokaryotes, and it is the LEAST reliable: NCBI-vs-GTDB order agreement 69.3 % vs 81.6 % for
properly named taxa.

🛑 But C9 carries "⚠ NOT yet measured on TaxEmbed's own tree. Owed first: count such leaves in
cellular_canonical." The UU panel is 5,962 PROTEOMES; TaxEmbed's tree is ~1M embedded TAXA. A
proportion measured on one is not a proportion on the other, and the retrain decision has been
deferred three times on the unmeasured version
([[feedback_a_blocker_you_did_not_test_is_not_a_blocker]]).

This counts the actual embedded taxa, by name form, split Bacteria / Archaea / Eukaryota. Counts
only; it decides nothing by itself.

Written 2026-09-29.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings")
sys.path.insert(0, str(ROOT / "helpers"))
sys.path.insert(0, str(ROOT / "src"))

from _p3_placement_score import MAPPING, OLD_DATE, build_tree, load_mapping  # noqa: E402
from taxembed.eval.release_diff import parse_parents  # noqa: E402

NAMES = ROOT / "data" / f"taxdump_archive_{OLD_DATE}" / "names.dmp"
NODES = ROOT / "data" / f"taxdump_archive_{OLD_DATE}" / "nodes.dmp"
DOMAINS = {"Bacteria": 2, "Archaea": 2157, "Eukaryota": 2759, "Viruses": 10239}

# the form C9 says is UNFILTERED and dominant
PLACEHOLDER = re.compile(r"\b(?:bacterium|archaeon)\b", re.IGNORECASE)
# what taxopy_clade.py:26-33 DOES catch, reproduced here to size the current filter's reach
CAUGHT = re.compile(
    r"(\bsp\.|\bcf\.|\baff\.|\bnr\.|environmental|uncultured|unidentified|hybrid)", re.IGNORECASE)


def scientific_names(path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "scientific name" not in line:
                continue
            p = line.split("\t|\t")
            if len(p) < 2:
                continue
            try:
                out[int(p[0].strip())] = p[1].strip()
            except ValueError:
                continue
    return out


def main() -> None:
    print("=" * 94)
    print("C9 — placeholder-name census on TaxEmbed's OWN embedded taxa (counts only)")
    print("=" * 94)

    taxid2row = load_mapping(MAPPING)
    print(f"  embedded taxa in the release: {len(taxid2row):,}")

    t0 = time.time()
    names = scientific_names(NAMES)
    print(f"  scientific names parsed: {len(names):,} in {time.time()-t0:.1f} s")

    parent_map = parse_parents(NODES)
    taxids, idx, parent, depth, tin, tout = build_tree(parent_map)

    embedded = np.array([t for t in taxid2row if t in idx], dtype=np.int64)
    rows = np.array([idx[int(t)] for t in embedded], dtype=np.int64)
    print(f"  embedded taxa located in the {OLD_DATE} tree: {len(rows):,}\n")

    results = {}
    hdr = f"  {'domain':<12}{'embedded':>10}{'placeholder':>14}{'already caught':>16}{'clean':>10}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for dname, dtax in DOMAINS.items():
        di = idx.get(dtax)
        if di is None:
            continue
        mask = (tin[rows] >= tin[di]) & (tin[rows] < tout[di])
        sub = embedded[mask]
        n = len(sub)
        if n == 0:
            continue
        ph = caught = 0
        for t in sub:
            nm = names.get(int(t), "")
            if PLACEHOLDER.search(nm):
                ph += 1
            elif CAUGHT.search(nm):
                caught += 1
        clean = n - ph - caught
        results[dname] = {"embedded": n, "placeholder_unfiltered": ph,
                          "already_caught_by_filter": caught, "clean": clean,
                          "placeholder_pct": round(100 * ph / n, 2)}
        print(f"  {dname:<12}{n:>10,}{ph:>10,} ({100*ph/n:>4.1f}%){caught:>11,} "
              f"({100*caught/n:>4.1f}%){clean:>10,}")

    prok = {k: results[k] for k in ("Bacteria", "Archaea") if k in results}
    pn = sum(v["embedded"] for v in prok.values())
    pp = sum(v["placeholder_unfiltered"] for v in prok.values())
    pc = sum(v["already_caught_by_filter"] for v in prok.values())
    tot = len(rows)
    print(f"\n  PROKARYOTES combined: {pn:,} embedded, "
          f"{pp:,} placeholder ({100*pp/max(pn,1):.1f} %), {pc:,} already caught")
    print(f"  Placeholder taxa as a share of the WHOLE embedding: "
          f"{pp:,} / {tot:,} = {100*pp/max(tot,1):.2f} %")
    print(f"  The filter's reach would go from {pc:,} to {pc+pp:,} prokaryote taxa "
          f"(x{(pc+pp)/max(pc,1):.1f}) if the pattern were added.")

    outp = ROOT / "results" / "c9_placeholder_census_20260929.json"
    json.dump({
        "purpose": "C9 owed measurement: placeholder-named leaves in TaxEmbed's own embedded set",
        "taxdump": OLD_DATE, "embedded_total": int(tot),
        "by_domain": results,
        "prokaryote_total": int(pn), "prokaryote_placeholder": int(pp),
        "prokaryote_already_caught": int(pc),
        "placeholder_share_of_whole_embedding_pct": round(100 * pp / max(tot, 1), 3),
    }, outp.open("w"), indent=2)
    print(f"\nwritten: {outp}")


if __name__ == "__main__":
    main()
