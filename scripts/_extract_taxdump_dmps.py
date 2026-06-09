"""Extract the four taxdump .dmp files (names/nodes/merged/delnodes) from a taxdump tarball into a
target dir, without clobbering. Used to stage names.dmp for App #2 leg C (incertae-sedis enrichment)
and nodes/merged/delnodes for the leg-B release-diff. Reproducible artifact (no ad-hoc tar pipes)."""
import argparse
import tarfile
from pathlib import Path

MEMBERS = ("names.dmp", "nodes.dmp", "merged.dmp", "delnodes.dmp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tarball", default="data/new_taxdump.tar.gz")
    ap.add_argument("--out-dir", default="data/taxdump_current")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.tarball, "r:gz") as tar:
        for m in MEMBERS:
            dest = out / m
            if dest.exists() and not args.force:
                print(f"  skip (exists): {dest}")
                continue
            try:
                member = tar.getmember(m)
            except KeyError:
                print(f"  not in tarball: {m}")
                continue
            tar.extract(member, path=out)
            print(f"  extracted: {dest}  ({dest.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
