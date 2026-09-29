#!/usr/bin/env python3
"""List the NCBI taxdump_archive snapshots available AFTER our training dump.

Our closure was built 2026-06-09 (data/new_taxdump.tar.gz mtime). The release-diff CLI
enforces training_date <= old_date < new_date, so we need snapshots dated after that.

Read-only: lists names and sizes, downloads nothing.

Written 2026-09-29 for the P3 temporal-QC placement arm.
"""
import re
import urllib.request

BASE = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump_archive/"


def main():
    print(f"listing {BASE}")
    with urllib.request.urlopen(BASE, timeout=60) as resp:
        html = resp.read().decode("utf-8", errors="replace")

    names = sorted(set(re.findall(r'href="([^"]+\.(?:zip|tar\.gz))"', html)))
    print(f"  {len(names)} archive files total\n")

    # Anything dated 2026-06 or later is a candidate for old_date/new_date.
    recent = [n for n in names if re.search(r"202[56]-(0[6-9]|1[0-2])-", n)]
    print("CANDIDATES dated 2026-06 or later:")
    for n in sorted(recent):
        print(f"    {n}")
    if not recent:
        print("    (none matched; showing the last 15 names by sort order)")
        for n in names[-15:]:
            print(f"    {n}")

    # Which prefixes exist at all?
    prefixes = sorted({re.sub(r"_?\d{4}-\d{2}-\d{2}.*$", "", n) for n in names})
    print(f"\nprefixes present: {prefixes}")


if __name__ == "__main__":
    main()
