#!/usr/bin/env python3
"""The numbers the v4 manuscript quotes without an on-disk record. Counts and timings only.

manuscript.v4_BUILD_AND_FIGURES.md §3 lists five numbers owed before submission. This helper
produces four of them from the TRAINING closure itself (data/taxopy/cellular_organisms_131567_clean)
and the released artifact, so that every figure it prints has a closure-keyed provenance:

  1. per-domain counts of the 1,102,163-node closure (Bacteria / Archaea / Eukaryota / root),
     computed as subtree sizes on the closure's own parent array -- no taxdump involved;
  2. the placeholder-binomial census ("<clade> bacterium|archaeon <id>") over the SAME closure,
     with names taken from the taxdump the closure was built from (data/new_taxdump.tar.gz,
     downloaded 2026-06-09), so the percentages have ONE denominator. C9's census
     (results/c9_placeholder_census_20260929.json) used the 2026-07-01 snapshot and therefore a
     1,025,217-node name-resolvable population; the regexes here are IDENTICAL to C9's so the two
     are comparable;
  4. query cost: Poincare distance between two rows of the released tensor (a fixed 100-d
     operation) against NCBI path length by parent-chain traversal on the same tree (a walk whose
     length is the two depths), both per pair, one core, vectorised and single-call;
  5. the 14-node gap in the 2026-07-21 regen_analysis (n_embedded 1,102,163 vs
     n_taxonomy_nodes 1,102,149): how many closure taxids are absent from a LATER nodes.dmp
     (2026-07-01 archive, the nearest snapshot on disk), and whether they sit in that snapshot's
     merged.dmp / delnodes.dmp.

Item 3 of that list (a Euclidean-geometry control) is an experiment, not a count, and is not here.

Rule 18: closed-form audit over an npz + one pass over names.dmp; one core; minutes.
Output: results/owed_numbers_20260930.json (new file; nothing overwritten).

Written 2026-09-30.
"""
from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

ROOT = Path("/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings")
sys.path.insert(0, str(ROOT / "helpers"))
sys.path.insert(0, str(ROOT / "src"))

from _p3_placement_score import load_safetensors  # noqa: E402

CLOSURE = ROOT / "data" / "taxopy" / "cellular_organisms_131567_clean"
STEM = "taxonomy_edges_cellular_organisms_131567_clean"
MAPPING = CLOSURE / f"{STEM}.mapping.tsv"
MAPPED_EDGES = CLOSURE / f"{STEM}.mapped.edgelist"
TRAIN_TAXDUMP = ROOT / "data" / "new_taxdump.tar.gz"          # downloaded 2026-06-09
LATER_SNAPSHOT = ROOT / "data" / "taxdump_archive_2026-07-01"   # nearest later snapshot on disk
EMB = ROOT / "release" / "taxembed-cellular-v1" / "cellular_embedding.safetensors"
OUT = ROOT / "results" / "owed_numbers_20260930.json"

DOMAINS = {"Bacteria": 2, "Archaea": 2157, "Eukaryota": 2759}
ROOT_TAXID = 131567
# identical to helpers/_c9_placeholder_census.py
PLACEHOLDER = re.compile(r"\b(?:bacterium|archaeon)\b", re.IGNORECASE)
CAUGHT = re.compile(
    r"(\bsp\.|\bcf\.|\baff\.|\bnr\.|environmental|uncultured|unidentified|hybrid)", re.IGNORECASE)

N_PAIRS_VEC = 1_000_000
N_PAIRS_LOOP = 20_000
SEED = 0


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_mapping() -> tuple[np.ndarray, dict[int, int]]:
    """idx -> taxid array, and taxid -> idx dict."""
    taxid2idx: dict[int, int] = {}
    with MAPPING.open() as fh:
        fh.readline()
        for line in fh:
            a, _, b = line.partition("\t")
            if a.isdigit():
                taxid2idx[int(a)] = int(b)
    n = max(taxid2idx.values()) + 1
    idx2taxid = np.zeros(n, dtype=np.int64)
    for t, i in taxid2idx.items():
        idx2taxid[i] = t
    return idx2taxid, taxid2idx


def load_parent(n: int) -> np.ndarray:
    """parent[i] for every node; the root is its own parent. Edgelist is 'parent child'."""
    parent = np.arange(n, dtype=np.int64)
    seen = 0
    with MAPPED_EDGES.open() as fh:
        for line in fh:
            p, c = line.split()[:2]
            parent[int(c)] = int(p)
            seen += 1
    return parent, seen


def depths(parent: np.ndarray) -> np.ndarray:
    cur = np.arange(len(parent), dtype=np.int64)
    d = np.zeros(len(parent), dtype=np.int32)
    while True:
        nxt = parent[cur]
        moving = nxt != cur
        if not moving.any():
            break
        d[moving] += 1
        cur = nxt
    return d


def subtree_mask(parent: np.ndarray, top: int) -> np.ndarray:
    """Boolean mask of nodes whose ancestor chain contains `top` (top included)."""
    n = len(parent)
    mask = np.zeros(n, dtype=bool)
    mask[top] = True
    cur = np.arange(n, dtype=np.int64)
    while True:
        hit = mask[cur]
        mask |= hit
        nxt = parent[cur]
        moving = nxt != cur
        if not moving.any():
            break
        cur = np.where(moving, nxt, cur)
    return mask


def scientific_names_from_tar(tar_path: Path, wanted: set[int]) -> dict[int, str]:
    out: dict[int, str] = {}
    with tarfile.open(tar_path, "r:gz") as tf:
        member = next(m for m in tf.getmembers() if m.name.endswith("names.dmp"))
        fh = tf.extractfile(member)
        assert fh is not None
        for raw in fh:
            line = raw.decode("utf-8", errors="replace")
            if "scientific name" not in line:
                continue
            p = line.split("\t|\t")
            if len(p) < 2:
                continue
            try:
                t = int(p[0].strip())
            except ValueError:
                continue
            if t in wanted:
                out[t] = p[1].strip()
    return out


def parse_nodes_taxids(path: Path) -> set[int]:
    out: set[int] = set()
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            a, _, _ = line.partition("\t")
            if a.isdigit():
                out.add(int(a))
    return out


def parse_two_col(path: Path) -> dict[int, int]:
    out: dict[int, int] = {}
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            p = [x.strip() for x in line.split("|")]
            if len(p) >= 2 and p[0].isdigit() and p[1].isdigit():
                out[int(p[0])] = int(p[1])
            elif p and p[0].isdigit():
                out[int(p[0])] = -1
    return out


def poincare_pairs(E: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    U, V = E[a], E[b]
    su = np.einsum("ij,ij->i", U, U)
    sv = np.einsum("ij,ij->i", V, V)
    sd = np.einsum("ij,ij->i", U - V, U - V)
    denom = np.clip((1.0 - su) * (1.0 - sv), 1e-12, None)
    return np.arccosh(np.clip(1.0 + 2.0 * sd / denom, 1.0, None))


def poincare_one(u: np.ndarray, v: np.ndarray) -> float:
    su = float(u @ u)
    sv = float(v @ v)
    d = u - v
    sd = float(d @ d)
    return float(np.arccosh(max(1.0, 1.0 + 2.0 * sd / max((1.0 - su) * (1.0 - sv), 1e-12))))


def path_len_one(parent: np.ndarray, depth: np.ndarray, a: int, b: int) -> int:
    da, db = int(depth[a]), int(depth[b])
    steps = 0
    while da > db:
        a = int(parent[a]); da -= 1; steps += 1
    while db > da:
        b = int(parent[b]); db -= 1; steps += 1
    while a != b:
        a = int(parent[a]); b = int(parent[b]); steps += 2
    return steps


def path_len_vec(parent: np.ndarray, depth: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.copy(); b = b.copy()
    da, db = depth[a].astype(np.int64), depth[b].astype(np.int64)
    steps = np.zeros(len(a), dtype=np.int64)
    while True:
        m = da > db
        if not m.any():
            break
        a[m] = parent[a[m]]; da[m] -= 1; steps[m] += 1
    while True:
        m = db > da
        if not m.any():
            break
        b[m] = parent[b[m]]; db[m] -= 1; steps[m] += 1
    while True:
        m = a != b
        if not m.any():
            break
        a[m] = parent[a[m]]; b[m] = parent[b[m]]; steps[m] += 2
    return steps


def main() -> None:
    t_start = time.time()
    rec: dict = {
        "purpose": "owed numbers for the TaxEmbed manuscript, computed on the training closure",
        "closure": str(CLOSURE.relative_to(ROOT)),
        "mapping_md5": md5(MAPPING),
        "embedding": str(EMB.relative_to(ROOT)),
        "training_taxdump": str(TRAIN_TAXDUMP.relative_to(ROOT)),
        "later_snapshot": str(LATER_SNAPSHOT.relative_to(ROOT)),
        "hardware": {"machine": platform.machine(), "processor": platform.processor(),
                     "python": platform.python_version(), "numpy": np.__version__},
    }
    print("=" * 96)
    print("owed numbers (2026-09-30) — training closure, one denominator")
    print("=" * 96)

    idx2taxid, taxid2idx = load_mapping()
    n = len(idx2taxid)
    parent, n_edges = load_parent(n)
    depth = depths(parent)
    roots = np.flatnonzero(parent == np.arange(n))
    assert len(roots) == 1 and idx2taxid[roots[0]] == ROOT_TAXID, roots
    print(f"  closure nodes {n:,}  edges {n_edges:,}  root taxid {ROOT_TAXID}  max depth {int(depth.max())}")
    rec["closure_nodes"] = int(n)
    rec["closure_edges"] = int(n_edges)
    rec["max_depth"] = int(depth.max())
    rec["mean_depth"] = float(depth.mean())

    # ---- 1. per-domain counts -------------------------------------------------------------
    per_domain = {}
    masks = {}
    for name, t in DOMAINS.items():
        m = subtree_mask(parent, taxid2idx[t])
        masks[name] = m
        per_domain[name] = int(m.sum())
    accounted = sum(per_domain.values()) + 1  # + root
    print("\n  1. per-domain counts (subtree incl. the domain node):")
    for k, v in per_domain.items():
        print(f"     {k:<10}{v:>10,}  ({100*v/n:5.2f} %)")
    print(f"     root       {1:>10,}")
    print(f"     sum        {accounted:>10,}   {'== closure' if accounted == n else '!= closure'}")
    rec["per_domain"] = {**per_domain, "root": 1, "sum": accounted,
                         "pct": {k: round(100 * v / n, 3) for k, v in per_domain.items()}}
    assert accounted == n

    # ---- 2. placeholder census on the training names ------------------------------------------
    t0 = time.time()
    wanted = set(int(t) for t in idx2taxid)
    names = scientific_names_from_tar(TRAIN_TAXDUMP, wanted)
    print(f"\n  2. names from the training taxdump: {len(names):,} of {n:,} resolved "
          f"in {time.time()-t0:.0f} s")
    census = {}
    tot_ph = 0
    for name, m in masks.items():
        ids = idx2taxid[m]
        ph = caught = unresolved = 0
        for t in ids:
            nm = names.get(int(t))
            if nm is None:
                unresolved += 1
                continue
            if PLACEHOLDER.search(nm):
                ph += 1
            elif CAUGHT.search(nm):
                caught += 1
        census[name] = {"nodes": int(m.sum()), "placeholder": ph,
                        "placeholder_pct": round(100 * ph / int(m.sum()), 3),
                        "already_caught_by_filter": caught, "unresolved_name": unresolved}
        tot_ph += ph
        print(f"     {name:<10} nodes {int(m.sum()):>9,}  placeholder {ph:>7,} "
              f"({100*ph/int(m.sum()):5.2f} %)  caught-by-filter {caught:,}  unresolved {unresolved}")
    print(f"     whole closure: {tot_ph:,} / {n:,} = {100*tot_ph/n:.3f} %")
    rec["placeholder_census"] = {**census, "total_placeholder": tot_ph,
                                 "share_of_closure_pct": round(100 * tot_ph / n, 3),
                                 "names_resolved": len(names)}

    # ---- 5. the 14-node gap: closure taxids vs a later snapshot --------------------------------
    later_nodes = parse_nodes_taxids(LATER_SNAPSHOT / "nodes.dmp")
    missing = [int(t) for t in idx2taxid if int(t) not in later_nodes]
    merged = parse_two_col(LATER_SNAPSHOT / "merged.dmp")
    deleted = parse_two_col(LATER_SNAPSHOT / "delnodes.dmp")
    n_merged = sum(1 for t in missing if t in merged)
    n_deleted = sum(1 for t in missing if t in deleted)
    print(f"\n  5. closure taxids absent from nodes.dmp of {LATER_SNAPSHOT.name}: {len(missing):,}"
          f"  (merged.dmp {n_merged}, delnodes.dmp {n_deleted}, neither {len(missing)-n_merged-n_deleted})")
    rec["later_snapshot_drift"] = {
        "closure_taxids_absent_from_later_nodes_dmp": len(missing),
        "in_merged": n_merged, "in_delnodes": n_deleted,
        "note": ("regen_analysis_20260721 reported n_taxonomy_nodes 1,102,149 = 14 fewer than the "
                 "closure; that analysis rebuilt its taxonomy from the taxdump current on 2026-07-21, "
                 "not from the closure. This block measures the same drift against the nearest later "
                 "snapshot on disk (2026-07-01)."),
        "example_missing_taxids": missing[:20],
    }

    # ---- 4. query cost ------------------------------------------------------------------------
    E = load_safetensors(EMB)
    assert E.shape[0] == n, E.shape
    rng = np.random.default_rng(SEED)
    a = rng.integers(0, n, N_PAIRS_VEC)
    b = rng.integers(0, n, N_PAIRS_VEC)
    # warm-up touches the rows once so page-cache effects do not land on the timed call
    _ = poincare_pairs(E, a[:1000], b[:1000])
    t0 = time.perf_counter(); dvec = poincare_pairs(E, a, b); t_vec_emb = time.perf_counter() - t0
    _ = path_len_vec(parent, depth, a[:1000], b[:1000])
    t0 = time.perf_counter(); pvec = path_len_vec(parent, depth, a, b); t_vec_tree = time.perf_counter() - t0

    al, bl = a[:N_PAIRS_LOOP], b[:N_PAIRS_LOOP]
    t0 = time.perf_counter()
    for i in range(N_PAIRS_LOOP):
        poincare_one(E[al[i]], E[bl[i]])
    t_one_emb = time.perf_counter() - t0
    t0 = time.perf_counter()
    for i in range(N_PAIRS_LOOP):
        path_len_one(parent, depth, int(al[i]), int(bl[i]))
    t_one_tree = time.perf_counter() - t0

    cost = {
        "n_pairs_vectorised": N_PAIRS_VEC, "n_pairs_single_call": N_PAIRS_LOOP,
        "embedding_vectorised_us_per_pair": round(1e6 * t_vec_emb / N_PAIRS_VEC, 4),
        "tree_vectorised_us_per_pair": round(1e6 * t_vec_tree / N_PAIRS_VEC, 4),
        "embedding_single_call_us_per_pair": round(1e6 * t_one_emb / N_PAIRS_LOOP, 3),
        "tree_single_call_us_per_pair": round(1e6 * t_one_tree / N_PAIRS_LOOP, 3),
        "mean_path_length_random_pairs": float(pvec.mean()),
        "mean_poincare_distance_random_pairs": float(dvec.mean()),
        "embedding_dim": int(E.shape[1]),
        "note": ("One core, numpy, released F32 tensor in RAM. Vectorised = one call over all pairs; "
                 "single-call = a Python call per pair. The tree walk is on the closure's own "
                 "parent array with precomputed depths, i.e. the cheapest traversal, not a taxdump "
                 "parse. Ratios are hardware-specific; the fixed-size vs depth-dependent shape is not."),
    }
    print("\n  4. query cost per pair (µs), one core:")
    print(f"     {'':<22}{'vectorised':>12}{'single call':>14}")
    print(f"     {'embedding (100-d)':<22}{cost['embedding_vectorised_us_per_pair']:>12.3f}"
          f"{cost['embedding_single_call_us_per_pair']:>14.2f}")
    print(f"     {'tree walk':<22}{cost['tree_vectorised_us_per_pair']:>12.3f}"
          f"{cost['tree_single_call_us_per_pair']:>14.2f}")
    print(f"     mean path length of a random pair {pvec.mean():.2f} edges; mean depth {depth.mean():.2f}")
    rec["query_cost"] = cost

    rec["wall_seconds"] = round(time.time() - t_start, 1)
    OUT.parent.mkdir(exist_ok=True)
    assert not OUT.exists(), f"refusing to overwrite {OUT}"
    json.dump(rec, OUT.open("w"), indent=2)
    print(f"\nwritten: {OUT}  ({rec['wall_seconds']} s)")


if __name__ == "__main__":
    main()
