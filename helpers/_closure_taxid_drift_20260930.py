#!/usr/bin/env python3
"""Why 76,766 closure taxids have no name in data/new_taxdump.tar.gz, and the census completed.

_owed_numbers_20260930.py found that the 2026-06-09 tarball on disk resolves scientific names for
only 1,025,397 of the closure's 1,102,163 taxids, and that 76,946 closure taxids are absent from
the 2026-07-01 nodes.dmp (76,939 of them listed in that snapshot's merged.dmp). Two questions:

  A. Are those taxids present in the tarball's OWN nodes.dmp / merged.dmp? If they are in its
     merged.dmp, the closure was built from an OLDER taxdump than the tarball on disk, and the
     "downloaded 2026-06-09" handle in the manuscript names the wrong file.
  B. Do the merge TARGETS sit in the closure too? If so the embedding carries the same organism
     under two taxids (a duplicate), which the Data section must say.

Then the placeholder census is completed with ONE denominator: unresolved names are canonicalised
through the 2026-07-01 merged.dmp and looked up in that snapshot's names.dmp, so every closure
taxid gets a name from the nearest record that has one.

Counts only. Output: results/closure_taxid_drift_20260930.json (new file).
Written 2026-09-30.
"""
from __future__ import annotations

import json
import re
import sys
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path("/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings")
sys.path.insert(0, str(ROOT / "helpers"))
from _owed_numbers_20260930 import (  # noqa: E402
    CAUGHT, DOMAINS, LATER_SNAPSHOT, PLACEHOLDER, TRAIN_TAXDUMP, load_mapping, load_parent,
    parse_two_col, scientific_names_from_tar, subtree_mask,
)

OUT = ROOT / "results" / "closure_taxid_drift_20260930.json"


def tar_member_lines(tar_path: Path, basename: str):
    # exact basename: endswith("nodes.dmp") also matches delnodes.dmp (bug in the first run,
    # which reported 778,645 "nodes" -- the deleted-node count)
    with tarfile.open(tar_path, "r:gz") as tf:
        member = next(m for m in tf.getmembers() if m.name.split("/")[-1] == basename)
        fh = tf.extractfile(member)
        assert fh is not None
        for raw in fh:
            yield raw.decode("utf-8", errors="replace")


def scientific_names_from_dmp(path: Path, wanted: set[int]) -> dict[int, str]:
    out: dict[int, str] = {}
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "scientific name" not in line:
                continue
            p = line.split("\t|\t")
            try:
                t = int(p[0].strip())
            except ValueError:
                continue
            if t in wanted:
                out[t] = p[1].strip()
    return out


def main() -> None:
    idx2taxid, taxid2idx = load_mapping()
    n = len(idx2taxid)
    parent, _ = load_parent(n)
    closure = set(int(t) for t in idx2taxid)

    # ---- A. the tarball's own nodes.dmp and merged.dmp -------------------------------------
    tar_nodes: set[int] = set()
    for line in tar_member_lines(TRAIN_TAXDUMP, "nodes.dmp"):
        a, _, _ = line.partition("\t")
        if a.isdigit():
            tar_nodes.add(int(a))
    tar_merged: dict[int, int] = {}
    for line in tar_member_lines(TRAIN_TAXDUMP, "merged.dmp"):
        p = [x.strip() for x in line.split("|")]
        if len(p) >= 2 and p[0].isdigit() and p[1].isdigit():
            tar_merged[int(p[0])] = int(p[1])
    missing_in_tar = sorted(t for t in closure if t not in tar_nodes)
    in_tar_merged = [t for t in missing_in_tar if t in tar_merged]
    print(f"A. tarball nodes.dmp: {len(tar_nodes):,} taxids; closure taxids absent from it: "
          f"{len(missing_in_tar):,}; of those in the tarball's merged.dmp: {len(in_tar_merged):,}")

    # ---- B. merge targets in the closure? --------------------------------------------------
    later_merged = parse_two_col(LATER_SNAPSHOT / "merged.dmp")
    targets_in_closure = sum(1 for t in missing_in_tar if later_merged.get(t, -1) in closure)
    targets_in_closure_tar = sum(1 for t in in_tar_merged if tar_merged[t] in closure)
    print(f"B. merge targets (2026-07-01 merged.dmp) that are THEMSELVES in the closure: "
          f"{targets_in_closure:,} of {len(missing_in_tar):,}")
    print(f"   same, via the tarball's merged.dmp: {targets_in_closure_tar:,} of {len(in_tar_merged):,}")

    # depth / leaf profile of the missing taxids
    miss_idx = np.array([taxid2idx[t] for t in missing_in_tar], dtype=np.int64)
    is_parent = np.zeros(n, dtype=bool)
    is_parent[parent] = True
    is_parent[np.arange(n)[parent == np.arange(n)]] = True
    leaves_among_missing = int((~is_parent[miss_idx]).sum())
    print(f"   leaves among the missing: {leaves_among_missing:,} of {len(miss_idx):,}")

    # ---- census completed with one denominator ----------------------------------------------
    names = scientific_names_from_tar(TRAIN_TAXDUMP, closure)
    unresolved = [t for t in closure if t not in names]
    canon = {t: later_merged.get(t, t) for t in unresolved}
    later_names = scientific_names_from_dmp(LATER_SNAPSHOT / "names.dmp", set(canon.values()))
    filled = 0
    for t, c in canon.items():
        nm = later_names.get(c)
        if nm is not None:
            names[t] = nm
            filled += 1
    still = len(unresolved) - filled
    print(f"census: {len(unresolved):,} names unresolved in the tarball; {filled:,} filled via "
          f"2026-07-01 merged.dmp + names.dmp; {still:,} still unresolved")

    census = {}
    tot_ph = 0
    for dname, dtax in DOMAINS.items():
        m = subtree_mask(parent, taxid2idx[dtax])
        ids = idx2taxid[m]
        ph = caught = unres = 0
        for t in ids:
            nm = names.get(int(t))
            if nm is None:
                unres += 1
            elif PLACEHOLDER.search(nm):
                ph += 1
            elif CAUGHT.search(nm):
                caught += 1
        nn = int(m.sum())
        census[dname] = {"nodes": nn, "placeholder": ph, "placeholder_pct": round(100 * ph / nn, 2),
                         "already_caught_by_filter": caught, "unresolved_name": unres}
        tot_ph += ph
        print(f"   {dname:<10} {nn:>9,}  placeholder {ph:>7,} ({100*ph/nn:5.2f} %)  "
              f"caught {caught:,}  unresolved {unres}")
    print(f"   whole closure: {tot_ph:,} / {n:,} = {100*tot_ph/n:.3f} %")

    # ---- alias geometry: does an alias row sit where its target sits? ------------------------
    from _p3_placement_score import EMB, load_safetensors  # noqa: E402
    E = load_safetensors(EMB)
    pairs = [(t, tar_merged[t]) for t in in_tar_merged if tar_merged[t] in closure]
    old_i = np.array([taxid2idx[o] for o, _ in pairs], dtype=np.int64)
    new_i = np.array([taxid2idx[m] for _, m in pairs], dtype=np.int64)
    same_parent = int((parent[old_i] == parent[new_i]).sum())
    # sibling control: for each target, another child of the same parent (if any), else skip
    rng = np.random.default_rng(0)
    children: dict[int, list[int]] = {}
    for c in range(n):
        p = int(parent[c])
        if p != c:
            children.setdefault(p, []).append(c)
    sib_i, keep = [], []
    for k, (o, m) in enumerate(zip(old_i, new_i)):
        sibs = [c for c in children.get(int(parent[m]), []) if c != int(m) and c != int(o)]
        if sibs:
            sib_i.append(int(rng.choice(sibs))); keep.append(k)
    keep = np.array(keep, dtype=np.int64); sib_i = np.array(sib_i, dtype=np.int64)
    U = E / np.linalg.norm(E, axis=1, keepdims=True)
    cos_alias = np.einsum("ij,ij->i", U[old_i], U[new_i])
    cos_sib = np.einsum("ij,ij->i", U[old_i[keep]], U[sib_i])
    rand_i = rng.integers(0, n, len(old_i))
    cos_rand = np.einsum("ij,ij->i", U[old_i], U[rand_i])
    alias_geom = {
        "alias_pairs_both_in_closure": len(pairs),
        "same_parent_in_closure": same_parent,
        "cos_alias_to_target_median": float(np.median(cos_alias)),
        "cos_alias_to_target_mean": float(cos_alias.mean()),
        "cos_alias_to_random_sibling_of_target_median": float(np.median(cos_sib)),
        "cos_alias_to_random_sibling_of_target_mean": float(cos_sib.mean()),
        "n_with_sibling_control": int(len(keep)),
        "cos_alias_to_random_node_median": float(np.median(cos_rand)),
        "note": ("An alias (old, merged taxid) and its target were both nodes of the training tree "
                 "with the same parent, so they are trained as siblings. If the alias is no nearer "
                 "its target than another sibling, it is a second, independent point for the same "
                 "organism, not a copy."),
    }
    print("alias geometry:", json.dumps(alias_geom, indent=1))

    example = []
    for t in missing_in_tar[:8]:
        example.append({"taxid": t, "name": names.get(t), "merged_into_2026_07_01": later_merged.get(t),
                        "in_tarball_merged": tar_merged.get(t)})

    rec = {
        "purpose": "closure taxids vs the tarball on disk; merge targets; placeholder census, one denominator",
        "closure_nodes": n,
        "tarball": str(TRAIN_TAXDUMP.relative_to(ROOT)),
        "tarball_nodes": len(tar_nodes),
        "closure_taxids_absent_from_tarball_nodes": len(missing_in_tar),
        "of_which_in_tarball_merged": len(in_tar_merged),
        "merge_targets_in_closure_via_2026_07_01": targets_in_closure,
        "merge_targets_in_closure_via_tarball": targets_in_closure_tar,
        "leaves_among_missing": leaves_among_missing,
        "alias_geometry": alias_geom,
        "census_one_denominator": {**census, "total_placeholder": tot_ph,
                                   "share_of_closure_pct": round(100 * tot_ph / n, 3),
                                   "names_unresolved_after_fill": still},
        "examples": example,
    }
    assert not OUT.exists(), f"refusing to overwrite {OUT}"
    json.dump(rec, OUT.open("w"), indent=2)
    print(f"written: {OUT}")
    for e in example:
        print("  ", e)


if __name__ == "__main__":
    main()
