#!/usr/bin/env python3
"""Fetch the post-training NCBI taxdump snapshots the P3 placement arm needs.

Training closure was built 2026-06-09, so every snapshot here is strictly AFTER training and
satisfies the release-diff CLI's no-leakage ordering (training_date <= old_date < new_date).

Offline-safe: ensure_taxdump_archive skips the network if the dmp files are already extracted.
Each snapshot lands in its own directory so old/new can never be confused.

Written 2026-09-29 for the P3 temporal-QC placement arm.
"""
import sys
import time
from pathlib import Path

ROOT = Path("/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings")
sys.path.insert(0, str(ROOT / "src"))

from taxembed.utils.taxdump import ensure_taxdump_archive  # noqa: E402

DATES = ["2026-07-01", "2026-08-01", "2026-09-01"]

# Strong variant (spec §P3 "Strong variant"): train on an OLD snapshot, score on the next, across
# many release pairs, turning n=1 into a distribution. Pass dates on argv to fetch those instead.
#
# names.dmp is ~300 MB per snapshot and is NOT used by the placement arm (it needs nodes/merged/
# delnodes only), so it is deleted after extraction to keep the archive set to ~230 MB per date.
DROP_AFTER_EXTRACT = ("names.dmp",)


def main():
    dates = sys.argv[1:] or DATES
    for date in dates:
        outdir = ROOT / "data" / f"taxdump_archive_{date}"
        archive = f"taxdmp_{date}.zip"
        t0 = time.time()
        print(f"[{date}] -> {outdir}", flush=True)
        try:
            nodes, names, merged, delnodes = ensure_taxdump_archive(outdir, archive)
        except Exception as exc:                       # noqa: BLE001
            print(f"[{date}] FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue
        dt = time.time() - t0
        for label, p in (("nodes", nodes), ("names", names),
                         ("merged", merged), ("delnodes", delnodes)):
            size = f"{p.stat().st_size / 1e6:.1f} MB" if p and p.exists() else "MISSING"
            print(f"[{date}]   {label:<9} {size}", flush=True)
        for junk in DROP_AFTER_EXTRACT:
            jp = outdir / junk
            if jp.exists():
                jp.unlink()
                print(f"[{date}]   dropped {junk} (unused by the placement arm)", flush=True)
        zp = outdir / archive
        if zp.exists():
            zp.unlink()
            print(f"[{date}]   dropped {archive} (extracted)", flush=True)
        print(f"[{date}] done in {dt:.1f}s", flush=True)

    print("ALL SNAPSHOTS DONE", flush=True)


if __name__ == "__main__":
    main()
