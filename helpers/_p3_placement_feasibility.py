#!/usr/bin/env python3
"""P3 placement arm — FEASIBILITY ONLY. How much signal is even available?

This is a POWER measurement, not a result. It counts, and deliberately scores nothing:

  1. How many taxa in the shipped embedding changed DIRECT PARENT between two post-training
     NCBI releases (release_diff.reclassified_taxa, canonicalized via merged/delnodes).
  2. Of those, how many have BOTH their old and new parent present in the embedding --
     the subset on which a placement comparison is even defined.
  3. The transductive-limit number the spec asks for: taxa present in the new release but
     absent from the embedding (no coordinate, cannot be placed).

🛑 It prints NO distance, NO ranking and NO placement statistic. The placement statistic must be
pre-registered BEFORE it is computed -- this session's own Finding 3 is what happens when a test
is designed against data it has already seen. This helper exists to answer one question: is the
2-month clean window (training 2026-06-09 => old_date >= 2026-06-09) big enough to bother, or
does the distribution need retraining on an older snapshot?

Written 2026-09-29 for the P3 temporal-QC placement arm.
"""
import sys
from pathlib import Path

ROOT = Path("/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from taxembed.eval.release_diff import (  # noqa: E402
    canonicalize_taxid, parse_delnodes, parse_merged, parse_parents,
    reclassified_taxa,
)

MAPPING = ROOT / "release" / "taxembed-cellular-v1" / "taxid_to_index.tsv"
PAIRS = [("2026-07-01", "2026-08-01"), ("2026-08-01", "2026-09-01"), ("2026-07-01", "2026-09-01")]


def snap(date):
    d = ROOT / "data" / f"taxdump_archive_{date}"
    return (parse_parents(d / "nodes.dmp"), parse_merged(d / "merged.dmp"),
            parse_delnodes(d / "delnodes.dmp"))


def load_embedded_taxids():
    """taxid -> row index, from the SHIPPED release mapping."""
    taxids = []
    with MAPPING.open() as fh:
        header = fh.readline()
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 1 and parts[0].isdigit():
                taxids.append(int(parts[0]))
    return set(taxids), header.rstrip("\n")


def main():
    embedded, header = load_embedded_taxids()
    print(f"mapping header : {header!r}")
    print(f"embedded taxa  : {len(embedded):,}")
    print()

    cache = {}
    for date in {d for pair in PAIRS for d in pair}:
        print(f"parsing {date} ...", flush=True)
        cache[date] = snap(date)

    print()
    for old_date, new_date in PAIRS:
        old_parent, _old_merged, _old_del = cache[old_date]
        new_parent, new_merged, new_del = cache[new_date]

        moved = reclassified_taxa(embedded, old_parent, new_parent, new_merged, new_del)

        # Of the moved set, how many have BOTH parents embedded (comparison well-defined)?
        both = 0
        old_missing = new_missing = 0
        for t in moved:
            op = canonicalize_taxid(old_parent[t], new_merged, new_del)
            npar = new_parent[t]
            o_in, n_in = op in embedded, npar in embedded
            if o_in and n_in:
                both += 1
            if not o_in:
                old_missing += 1
            if not n_in:
                new_missing += 1

        # Transductive limit: taxa in the NEW release with no coordinate at all.
        new_only = len(set(new_parent) - embedded)

        print("=" * 70)
        print(f"{old_date}  ->  {new_date}")
        print("=" * 70)
        print(f"  nodes in old release            : {len(old_parent):,}")
        print(f"  nodes in new release            : {len(new_parent):,}")
        print(f"  MOVED (direct parent changed)   : {len(moved):,}")
        print(f"    both parents embedded         : {both:,}   <- usable sample")
        print(f"    old parent not embedded       : {old_missing:,}")
        print(f"    new parent not embedded       : {new_missing:,}")
        print(f"  new-release taxa with NO coord  : {new_only:,}  (transductive limit)")
        print()


if __name__ == "__main__":
    main()
