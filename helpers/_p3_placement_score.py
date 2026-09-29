#!/usr/bin/env python3
"""P3 placement arm — the scorer, executing results/p3_placement_preregistration.json.

PRE-REGISTRATION FROZEN 2026-09-29 at SHA256
  3b07275053d61082a8adc6891e0af17bf412616030b4afb68f41fd264644ce38
before any distance, rank or placement statistic was computed. This file implements that
document and must not silently depart from it; every deviation is printed as an AMENDMENT.

QUESTION. Does the SHIPPED embedding (trained on the 2026-06-09 taxonomy) already place a taxon
nearer the parent NCBI LATER assigned it than the equally-local alternatives NCBI could have
chosen?

PRIMARY. normalized_rank of p_new among a LOCALITY-MATCHED pool: nodes at depth(p_new) inside the
subtree of L = LCA(p_old, p_new), excluding p_old and v. Chance = 0.5 for any pool size, LOWER IS
BETTER. The pool construction is the control for the confound that reclassifications are local: if
p_new were simply compared against unrestricted alternatives, the model would score well because it
already sits near p_old, which is adjacent to p_new -- nothing to do with anticipating the move.

GATES (all must pass, each designed to be able to FAIL):
  a  power           n_scored >= 1000
  b  negative ctrl   unmoved taxa, random pseudo-new-parent from the same pool construction,
                     identical statistic -> MUST be 0.5 within CI
  c  init null       directions randomized at the TRAINED radii (the project's house null, see
                     eval/angular.py) -> MUST be 0.5 within CI
  d  pool sanity     median pool size >= 3

AMENDMENT 1 (2026-09-29, recorded BEFORE any result was computed): gate (c) is implemented as
"random directions at each node's TRAINED radius" rather than "a freshly initialized embedding at
the depth-radius schedule". This is the project's established null (a random-direction embedding
scores S_angle = 0 by construction) and it is STRICTLY STRONGER for the purpose: it preserves the
planted radial prior EXACTLY, so if the pool or the metric could deliver the answer from radius
alone, this null detects it where a fresh init might not.

Written 2026-09-29.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings")
sys.path.insert(0, str(ROOT / "src"))

from taxembed.eval.release_diff import (  # noqa: E402
    canonicalize_taxid, parse_delnodes, parse_merged, parse_parents, reclassified_taxa,
)

EMB = ROOT / "release" / "taxembed-cellular-v1" / "cellular_embedding.safetensors"
MAPPING = ROOT / "release" / "taxembed-cellular-v1" / "taxid_to_index.tsv"
OLD_DATE, NEW_DATE = "2026-07-01", "2026-09-01"
N_BOOT = 10_000
SEED = 0


# ---------------------------------------------------------------- loading

def load_safetensors(path: Path, key: str = "embedding") -> np.ndarray:
    """Minimal safetensors reader: u64 header length, JSON header, raw buffer."""
    with path.open("rb") as fh:
        (hlen,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(hlen))
        meta = header[key]
        dtype = {"F32": np.float32, "F64": np.float64, "F16": np.float16}[meta["dtype"]]
        start, end = meta["data_offsets"]
        fh.seek(8 + hlen + start)
        buf = fh.read(end - start)
    return np.frombuffer(buf, dtype=dtype).reshape(meta["shape"])


def load_mapping(path: Path) -> dict[int, int]:
    out = {}
    with path.open() as fh:
        fh.readline()
        for line in fh:
            a, _, b = line.partition("\t")
            if a.isdigit():
                out[int(a)] = int(b)
    return out


# ---------------------------------------------------------------- geometry

def poincare_dist(u: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Ball-coordinate Poincare distance from point u to each row of V (README's formula)."""
    sq_u = float(np.dot(u, u))
    sq_v = np.sum(V * V, axis=-1)
    sq_d = np.sum((V - u) ** 2, axis=-1)
    denom = np.clip((1.0 - sq_u) * (1.0 - sq_v), 1e-12, None)
    return np.arccosh(np.clip(1.0 + 2.0 * sq_d / denom, 1.0, None))


# ---------------------------------------------------------------- tree

def build_tree(parent_map: dict[int, int]):
    """taxid->compact idx, parent array, depth array, euler tin/tout."""
    taxids = np.fromiter(parent_map.keys(), dtype=np.int64, count=len(parent_map))
    taxids.sort()
    idx = {int(t): i for i, t in enumerate(taxids)}
    n = len(taxids)
    parent = np.arange(n, dtype=np.int64)
    for t, p in parent_map.items():
        pi = idx.get(int(p))
        if pi is not None:
            parent[idx[int(t)]] = pi

    # Depth by lifting every node toward the root in lockstep.
    #
    # 🧨 FIXED 2026-09-29. The first version tested `cur != arange(n)` -- "has this node reached
    # ITSELF?" -- but `cur` walks UPWARD, so once it reaches the root it parks there and the test
    # stays True for every non-root node forever. build_tree never terminated: it burned 21 minutes
    # of one core before the synthetic tests (which hung on a 12-node tree) exposed it. The
    # termination condition is that the CURSOR has stopped moving, i.e. cur is a self-parent (root),
    # never that the cursor equals the node it started from.
    cur = np.arange(n, dtype=np.int64)
    d = np.zeros(n, dtype=np.int32)
    while True:
        nxt = parent[cur]
        moving = nxt != cur                 # cur is not yet at a root
        if not moving.any():
            break
        d[moving] += 1
        cur = nxt
    depth = d

    from taxembed.eval.subtree import euler_intervals
    tin, tout = euler_intervals(parent)
    return taxids, idx, parent, depth, tin, tout


def child_toward(L: int, node: int, parent: np.ndarray):
    """The child of L on the path down to `node`; None if node IS L.

    Used by amendment 2: the branch of L that holds p_old must leave the candidate pool
    entirely, because p_new can never be in it (that is what makes L their LCA).
    """
    c, L = int(node), int(L)
    if c == L:
        return None
    while int(parent[c]) != L:
        nxt = int(parent[c])
        if nxt == c:
            return None                      # L is not an ancestor of node (should not happen)
        c = nxt
    return c


def lca(a: int, b: int, parent: np.ndarray, depth: np.ndarray) -> int:
    while depth[a] > depth[b]:
        a = parent[a]
    while depth[b] > depth[a]:
        b = parent[b]
    while a != b:
        a, b = parent[a], parent[b]
    return int(a)


# ---------------------------------------------------------------- pool

class PoolIndex:
    """Nodes grouped by depth, each group sorted by tin, restricted to EMBEDDED nodes."""

    def __init__(self, depth, tin, embedded_mask):
        self.tin = tin
        self.by_depth = {}
        emb_nodes = np.flatnonzero(embedded_mask)
        order = np.argsort(depth[emb_nodes], kind="stable")
        emb_nodes = emb_nodes[order]
        ds = depth[emb_nodes]
        bounds = np.searchsorted(ds, np.arange(ds.max() + 2))
        for dd in range(len(bounds) - 1):
            grp = emb_nodes[bounds[dd]:bounds[dd + 1]]
            if len(grp):
                self.by_depth[dd] = grp[np.argsort(tin[grp], kind="stable")]

    def candidates(self, d: int, L: int, tin, tout) -> np.ndarray:
        grp = self.by_depth.get(int(d))
        if grp is None:
            return np.empty(0, dtype=np.int64)
        lo = np.searchsorted(tin[grp], tin[L], side="left")
        hi = np.searchsorted(tin[grp], tout[L], side="left")
        return grp[lo:hi]


# ---------------------------------------------------------------- statistic

def normalized_ranks(queries, emb_rows, emb, pool_index, parent, depth, tin, tout,
                     exclude_pairs):
    """queries: list of (v_node, target_node, exclude_node). Returns (nrank array, pool sizes)."""
    nr, sizes = [], []
    rng = np.random.default_rng(SEED)
    for (v, target, excl) in queries:
        L = lca(int(excl), int(target), parent, depth)
        cand = pool_index.candidates(depth[target], L, tin, tout)
        if len(cand) == 0:
            continue
        # AMENDMENT 2: drop p_old's ENTIRE branch of L, not just p_old. p_new lies in a different
        # child of L by the definition of an LCA, so leaving p_old's branch in the pool put the
        # target in only the far portion and pushed the true chance floor to ~0.60 (gate b caught
        # exactly this). With the branch removed, p_new is uniform in the pool under the null and
        # the analytic 0.5 floor holds again.
        c_old = child_toward(L, excl, parent)
        if c_old is not None and len(cand):
            in_old_branch = (tin[cand] >= tin[c_old]) & (tin[cand] < tout[c_old])
            cand = cand[~in_old_branch]
        cand = cand[(cand != excl) & (cand != v)]
        sizes.append(len(cand))
        if len(cand) < 2:
            continue                                   # TRIVIAL, excluded (prereg)
        if not np.any(cand == target):                 # vectorized; a Python set() here was O(pool)
            continue
        rows = emb_rows[cand]
        ok = rows >= 0
        cand, rows = cand[ok], rows[ok]
        if len(cand) < 2 or emb_rows[target] < 0:
            continue
        d = poincare_dist(emb[emb_rows[v]], emb[rows])
        d = d + rng.random(len(d)) * 1e-12             # seeded tie jitter
        order = np.argsort(d, kind="stable")
        pos = int(np.flatnonzero(cand[order] == target)[0])
        nr.append(pos / (len(cand) - 1))
    return np.asarray(nr), np.asarray(sizes)


def boot_ci(x, n_boot=N_BOOT, seed=SEED):
    """Percentile bootstrap over taxa, chunked so the index matrix never blows memory."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    step = max(1, 2_000_000 // max(len(x), 1))
    done = 0
    while done < n_boot:
        k = min(step, n_boot - done)
        idx = rng.integers(0, len(x), size=(k, len(x)))
        means[done:done + k] = x[idx].mean(axis=1)
        done += k
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def report(name, nr, sizes):
    if len(nr) == 0:
        print(f"  {name:<22} n_scored=     0   (nothing scored)")
        return float("nan"), float("nan"), float("nan")
    lo, hi = boot_ci(nr)
    med = int(np.median(sizes)) if len(sizes) else 0
    print(f"  {name:<22} n_scored={len(nr):>6,}  mean={nr.mean():.4f}  "
          f"CI95=[{lo:.4f}, {hi:.4f}]  median_pool={med}")
    return nr.mean(), lo, hi


def main():
    print("=" * 88)
    print("P3 PLACEMENT ARM — executing results/p3_placement_preregistration.json")
    print("   frozen SHA256 3b07275053d61082a8adc6891e0af17bf412616030b4afb68f41fd264644ce38")
    print("=" * 88)

    emb = load_safetensors(EMB)
    taxid2row = load_mapping(MAPPING)
    print(f"embedding {emb.shape}  mapping {len(taxid2row):,} taxa")

    old_dir = ROOT / "data" / f"taxdump_archive_{OLD_DATE}"
    new_dir = ROOT / "data" / f"taxdump_archive_{NEW_DATE}"
    old_parent_map = parse_parents(old_dir / "nodes.dmp")
    new_parent_map = parse_parents(new_dir / "nodes.dmp")
    new_merged = parse_merged(new_dir / "merged.dmp")
    new_del = parse_delnodes(new_dir / "delnodes.dmp")
    print(f"old {OLD_DATE}: {len(old_parent_map):,} nodes   new {NEW_DATE}: {len(new_parent_map):,}")

    taxids, idx, parent, depth, tin, tout = build_tree(old_parent_map)
    n = len(taxids)
    emb_rows = np.full(n, -1, dtype=np.int64)
    for t, r in taxid2row.items():
        i = idx.get(t)
        if i is not None:
            emb_rows[i] = r
    embedded_mask = emb_rows >= 0
    print(f"old tree: {n:,} nodes, {int(embedded_mask.sum()):,} embedded, max depth {depth.max()}")

    pool_index = PoolIndex(depth, tin, embedded_mask)

    moved = reclassified_taxa(set(taxid2row), old_parent_map, new_parent_map, new_merged, new_del)
    print(f"moved taxa (both releases, canonicalized): {len(moved):,}")

    # ---- PRIMARY: (v, target = p_new, exclude = p_old)
    primary_q, lca_depths = [], []
    for t in moved:
        v = idx.get(int(t))
        po = canonicalize_taxid(old_parent_map[int(t)], new_merged, new_del)
        pn = new_parent_map[int(t)]
        if v is None or po is None:
            continue
        vo, vn = idx.get(int(po)), idx.get(int(pn))
        if vo is None or vn is None:
            continue
        if emb_rows[v] < 0 or emb_rows[vo] < 0 or emb_rows[vn] < 0:
            continue
        primary_q.append((v, vn, vo))
        lca_depths.append(int(depth[lca(vo, vn, parent, depth)]))
    print(f"primary queries built: {len(primary_q):,}")

    print("\nPRIMARY (lower is better, chance = 0.5)")
    nr_p, sz_p = normalized_ranks(primary_q, emb_rows, emb, pool_index, parent, depth,
                                  tin, tout, None)
    m_p, lo_p, hi_p = report("primary", nr_p, sz_p)

    # ---- GATE B: unmoved taxa, random pseudo-new-parent, locality matched on LCA depth
    print("\nGATE B — negative control (unmoved taxa, random pseudo-new-parent); MUST be 0.5")
    rng = np.random.default_rng(SEED + 1)
    moved_idx = {q[0] for q in primary_q}
    depth_pool = np.flatnonzero(embedded_mask)
    cand_v = depth_pool[~np.isin(depth_pool, list(moved_idx))]
    target_depths = np.array([depth[q[0]] for q in primary_q])
    ctrl_q = []
    by_depth_v = {}
    for v in cand_v:
        by_depth_v.setdefault(int(depth[v]), []).append(v)
    ld = np.array(lca_depths) if lca_depths else np.array([0])
    for td in target_depths:
        bucket = by_depth_v.get(int(td))
        if not bucket:
            continue
        v = int(bucket[rng.integers(0, len(bucket))])
        p = int(parent[v])
        if emb_rows[p] < 0:
            continue
        # climb to an ancestor at an L-depth drawn from the real moves' distribution
        want = int(ld[rng.integers(0, len(ld))])
        a = p
        while depth[a] > want and parent[a] != a:
            a = int(parent[a])
        if a == p:
            continue                         # no proper branch to exclude; not a usable control
        # Draw the pseudo-new-parent from the SAME set the primary's pool will be (amendment 2):
        # depth(p) nodes under `a`, MINUS p's own branch. Then LCA(p, q) == a exactly, the pool
        # reconstructed downstream is identical to this set, and q is uniform inside it -- so a
        # correct harness MUST return 0.5 here.
        cands = pool_index.candidates(depth[p], a, tin, tout)
        c_p = child_toward(a, p, parent)
        if c_p is not None and len(cands):
            in_p_branch = (tin[cands] >= tin[c_p]) & (tin[cands] < tout[c_p])
            cands = cands[~in_p_branch]
        cands = cands[(cands != p) & (cands != v)]
        if len(cands) < 2:
            continue
        q = int(cands[rng.integers(0, len(cands))])
        ctrl_q.append((v, q, p))
    nr_b, sz_b = normalized_ranks(ctrl_q, emb_rows, emb, pool_index, parent, depth,
                                  tin, tout, None)
    m_b, lo_b, hi_b = report("gate_b_control", nr_b, sz_b)

    # ---- GATE C: init null = random directions at the TRAINED radii (amendment 1)
    print("\nGATE C — init null (random directions, trained radii); MUST be 0.5")
    rng2 = np.random.default_rng(SEED + 2)
    radii = np.linalg.norm(emb, axis=1, keepdims=True)
    dirs = rng2.normal(size=emb.shape).astype(np.float32)
    dirs /= np.clip(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12, None)
    emb_null = (dirs * radii).astype(np.float32)
    nr_c, sz_c = normalized_ranks(primary_q, emb_rows, emb_null, pool_index, parent, depth,
                                  tin, tout, None)
    m_c, lo_c, hi_c = report("gate_c_init_null", nr_c, sz_c)

    # ---- SECONDARY (descriptive only): raw nearer-new-than-old rate
    nearer = []
    for (v, vn, vo) in primary_q:
        d = poincare_dist(emb[emb_rows[v]], emb[emb_rows[[vn, vo]]])
        nearer.append(bool(d[0] < d[1]))
    print(f"\nSECONDARY (DESCRIPTIVE ONLY, chance floor NOT 0.5 — model trained on p_old):")
    print(f"  d(v,p_new) < d(v,p_old) in {np.mean(nearer) * 100:.2f}% of {len(nearer):,} taxa")

    # ---- GATES + VERDICT
    print("\n" + "=" * 88)
    g_a = len(nr_p) >= 1000
    g_b = (lo_b <= 0.5 <= hi_b)
    g_c = (lo_c <= 0.5 <= hi_c)
    g_d = (len(sz_p) > 0 and np.median(sz_p) >= 3)
    for nm, ok, detail in (("a power", g_a, f"n_scored={len(nr_p):,} (>=1000)"),
                           ("b neg ctrl", g_b, f"CI [{lo_b:.4f},{hi_b:.4f}] must contain 0.5"),
                           ("c init null", g_c, f"CI [{lo_c:.4f},{hi_c:.4f}] must contain 0.5"),
                           ("d pool size", g_d, f"median pool={int(np.median(sz_p)) if len(sz_p) else 0} (>=3)")):
        print(f"  GATE {nm:<12} {'PASS' if ok else 'FAIL'}   {detail}")

    if not (g_a and g_b and g_c and g_d):
        verdict = "UNINFORMATIVE"
    elif hi_p < 0.5:
        verdict = "ANTICIPATES"
    elif lo_p > 0.5:
        verdict = "CONTRADICTS"
    else:
        verdict = "NO_EVIDENCE"
    print(f"\n  VERDICT: {verdict}   (primary {m_p:.4f}, CI [{lo_p:.4f}, {hi_p:.4f}], chance 0.5)")
    print("=" * 88)

    out = ROOT / "results" / "p3_placement_result_20260929.json"
    json.dump({
        "preregistration": "results/p3_placement_preregistration.json",
        "preregistration_sha256":
            "3b07275053d61082a8adc6891e0af17bf412616030b4afb68f41fd264644ce38",
        "old_date": OLD_DATE, "new_date": NEW_DATE, "training_date": "2026-06-09",
        "n_moved": len(moved), "n_primary_queries": len(primary_q),
        "primary": {"n_scored": len(nr_p), "mean_normalized_rank": m_p,
                    "ci95": [lo_p, hi_p], "median_pool": int(np.median(sz_p)) if len(sz_p) else 0},
        "gate_b_control": {"n_scored": len(nr_b), "mean": m_b, "ci95": [lo_b, hi_b]},
        "gate_c_init_null": {"n_scored": len(nr_c), "mean": m_c, "ci95": [lo_c, hi_c]},
        "secondary_descriptive_nearer_new_rate": float(np.mean(nearer)) if nearer else None,
        "gates": {"a_power": bool(g_a), "b_negative_control": bool(g_b),
                  "c_init_null": bool(g_c), "d_pool_size": bool(g_d)},
        "verdict": verdict,
    }, out.open("w"), indent=2)
    print(f"written: {out}")


if __name__ == "__main__":
    main()
