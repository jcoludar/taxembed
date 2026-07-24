"""Freeze a {species -> taxid -> embedding idx} table for a testbed (spec §3, §8: 85/85 must resolve;
unresolved are SURFACED, never silently dropped). Run once; output is a committed artifact."""
import sys
from pathlib import Path
import pandas as pd
from . import config  # noqa
from .taxdump import TaxonResolver
from .core import TaxonomyEmbedding

def main():
    resolver = TaxonResolver(config.TAXDUMP_DIR)
    te = TaxonomyEmbedding(config.CKPT, config.TAXMAP, config.EDGELIST)
    species = sorted(pd.read_csv(config.PLA2_CSV)["Species"].dropna().unique())
    rows, unresolved = [], []
    for sp in species:
        taxid = resolver.resolve(sp, aliases=config.ALIAS_MAP)
        idx = te.idx_of_taxid(taxid) if taxid else None
        rows.append({"species": sp, "taxid": taxid or "NA", "idx": idx if idx is not None else "NA",
                     "via": "alias" if sp in config.ALIAS_MAP else "direct"})
        if taxid is None or idx is None:
            unresolved.append((sp, taxid, idx))
    pd.DataFrame(rows).to_csv(config.PLA2_RESOLUTION, sep="\t", index=False)
    print(f"wrote {len(rows)} rows; unresolved={len(unresolved)}")
    for u in unresolved:
        print("  UNRESOLVED", u)
    if unresolved:
        raise SystemExit(f"{len(unresolved)} species unresolved — fix ALIAS_MAP before freezing")

if __name__ == "__main__":
    main()
