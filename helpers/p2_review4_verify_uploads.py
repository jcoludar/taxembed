"""READ-ONLY review helper (adversarial pre-submit review, 2026-09-26).

Rule 10: the split files the GPU array will train on must be BYTE-IDENTICAL to the ones
reviewed locally, and every path the two job scripts name must resolve on the host.

Runs `md5sum` over ssh (argv list, no shell, no pipes) and compares to local md5s.
Also `test -e`s every container path the scripts reference, mapped through the bind mounts
declared in the #SBATCH --container-mounts line.

Writes nothing.
"""
from __future__ import annotations

import hashlib
import os
import subprocess

LOCAL_REPO = "/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings"
LRZ_ROOT = "/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz"

# bind mounts from both job scripts' --container-mounts
MOUNTS = {
    "/data": f"{LRZ_ROOT}/data",
    "/app/artifacts": f"{LRZ_ROOT}/artifacts",
    "/app": f"{LRZ_ROOT}/src",
}

ARMS = ("vis00", "vis50", "degmatch_vis00", "degmatch_vis50")
SEEDS = (0, 1, 2)

# (container path, local path or None)
def container_paths():
    out = []
    out.append(("/data/taxonomy_edges_metazoa_33208_clean.mapping.tsv", None))
    for arm in ARMS:
        for s in SEEDS:
            out.append((f"/data/p2_splits/p2_metazoa_33208_clean_{arm}_seed{s}_train.npz",
                        f"{LOCAL_REPO}/data/p2_splits/p2_metazoa_33208_clean_{arm}_seed{s}_train.npz"))
    for s in SEEDS:
        out.append((f"/data/p2_splits/p2_metazoa_33208_clean_vis00_seed{s}_manifest.json",
                    f"{LOCAL_REPO}/data/p2_splits/p2_metazoa_33208_clean_vis00_seed{s}_manifest.json"))
        out.append((f"/data/p2_splits/p2_metazoa_33208_clean_seed{s}_heldout.npz",
                    f"{LOCAL_REPO}/data/p2_splits/p2_metazoa_33208_clean_seed{s}_heldout.npz"))
        out.append((f"/data/p2_splits/p2_metazoa_33208_clean_degmatch_seed{s}_heldout.npz",
                    f"{LOCAL_REPO}/data/p2_splits/p2_metazoa_33208_clean_degmatch_seed{s}_heldout.npz"))
        out.append((f"/data/p2_splits/taxonomy_edges_metazoa_33208_clean_degmatch_seed{s}_transitive.npz",
                    f"{LOCAL_REPO}/data/p2_splits/taxonomy_edges_metazoa_33208_clean_degmatch_seed{s}_transitive.npz"))
        for arm in ("degmatch_vis00", "degmatch_vis50"):
            out.append((f"/data/p2_splits/p2_metazoa_33208_clean_{arm}_seed{s}_manifest.json",
                        f"{LOCAL_REPO}/data/p2_splits/p2_metazoa_33208_clean_{arm}_seed{s}_manifest.json"))
    # the scorer's --closure for the REAL tree
    out.append(("/data/taxopy/metazoa_33208_clean/taxonomy_edges_metazoa_33208_clean_transitive.npz",
                f"{LOCAL_REPO}/data/taxopy/metazoa_33208_clean/taxonomy_edges_metazoa_33208_clean_transitive.npz"))
    # the smoke's inputs
    out.append(("/data/taxopy/mollusca_6447_clean/taxonomy_edges_mollusca_6447_clean.mapping.tsv", None))
    out.append(("/data/p2_splits/p2_mollusca_6447_clean_vis00_seed0_train.npz", None))
    # entrypoints inside the mounted src
    out.append(("/app/train_small.py", f"{LOCAL_REPO}/train_small.py"))
    out.append(("/app/scripts/score_p2_linkpred.py", f"{LOCAL_REPO}/scripts/score_p2_linkpred.py"))
    out.append(("/app/src/taxembed/eval/subtree.py", f"{LOCAL_REPO}/src/taxembed/eval/subtree.py"))
    return out


def to_host(cpath: str) -> str:
    for prefix in sorted(MOUNTS, key=len, reverse=True):
        if cpath == prefix or cpath.startswith(prefix + "/"):
            return MOUNTS[prefix] + cpath[len(prefix):]
    raise ValueError(f"no mount covers {cpath}")


def local_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    entries = container_paths()
    host_paths = [to_host(c) for c, _ in entries]

    # existence, in one ssh call, no shell metacharacters
    proc = subprocess.run(["ssh", "ai", "ls", "-1", *host_paths],
                          capture_output=True, text=True)
    present = set(proc.stdout.split())
    missing_msgs = [ln for ln in proc.stderr.splitlines() if ln.strip()]

    print("=== EXISTENCE on the HOST (through the bind mounts) ===")
    n_missing = 0
    for (cpath, _), hpath in zip(entries, host_paths):
        ok = hpath in present
        if not ok:
            n_missing += 1
            print(f"  MISSING  {cpath}")
            print(f"           -> host {hpath}")
    print(f"  {len(entries) - n_missing}/{len(entries)} present")
    if missing_msgs:
        print("  ssh stderr:")
        for m in missing_msgs:
            print(f"    {m}")

    # md5 comparison for everything that exists on both sides
    both = [(c, l, h) for (c, l), h in zip(entries, host_paths)
            if l and os.path.exists(l) and h in present]
    print()
    print(f"=== MD5 local vs LRZ ({len(both)} file pairs) ===")
    proc = subprocess.run(["ssh", "ai", "md5sum", *[h for _, _, h in both]],
                          capture_output=True, text=True)
    remote = {}
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            remote[parts[1]] = parts[0]
    n_bad = 0
    for cpath, lpath, hpath in both:
        lm = local_md5(lpath)
        rm = remote.get(hpath)
        if rm != lm:
            n_bad += 1
            print(f"  MISMATCH {cpath}")
            print(f"    local  {lm}  {lpath}")
            print(f"    lrz    {rm}  {hpath}")
    print(f"  {len(both) - n_bad}/{len(both)} byte-identical")
    if proc.stderr.strip():
        print("  ssh stderr:", proc.stderr.strip())


if __name__ == "__main__":
    main()
