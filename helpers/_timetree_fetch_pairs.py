#!/usr/bin/env python3
"""Spec §4.5 — fetch divergence times for the PRE-REGISTERED pair sample. Ages only.

EXECUTES results/timetree_preregistration.json, SHA256
  a3e733c781914a80046a7e116f491951269422d46ff31bf88bd2929521eb7fba

Draws the pair sample exactly as pre-registered -- N_PAIRS = 3,000 per stratum (Vertebrata 7742,
Insecta 50557), uniform without replacement, seed 0, within-stratum only -- and fetches each pair's
age from the TimeTree pairwise API.

🛑 THIS SCRIPT TOUCHES NO EMBEDDING. It fetches ages and study counts and nothing else, so the
pair draw and the label collection stay strictly separated from any distance computation. The
correlation is a later, separate script.

BEING A GOOD CITIZEN of a public academic service:
  - 3 workers, no burst, ~3 req/s. 6,000 calls ~ 30-35 min.
  - RESUMABLE via a JSONL cache: an interrupted run re-reads what it already has and fetches only
    the remainder, so a restart costs the service nothing.
  - the bulk alternative (one 1.9 MB newick) is kinder still and is used for the at-scale
    robustness surface, but it carries NO all_total study count -- and gate (c), the interpolation
    guard, needs it. That gate is why the API route is the pre-registered primary.

Only taxid pairs leave this machine. No credentials, no local data.

Written 2026-09-29.
"""
from __future__ import annotations

import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path("/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings")
sys.path.insert(0, str(ROOT / "helpers"))
sys.path.insert(0, str(ROOT / "src"))

from _p3_placement_score import MAPPING, OLD_DATE, build_tree, load_mapping  # noqa: E402
from taxembed.eval.release_diff import parse_parents  # noqa: E402

NWK = ROOT / "data" / "timetree" / "TimetreeOfLife2015.nwk"
NAMES = ROOT / "data" / f"taxdump_archive_{OLD_DATE}" / "names.dmp"
NODES = ROOT / "data" / f"taxdump_archive_{OLD_DATE}" / "nodes.dmp"
CACHE = ROOT / "data" / "timetree" / "pairwise_cache.jsonl"
OUT = ROOT / "results" / "timetree_pairs_20260929.json"

STRATA = {"Vertebrata": 7742, "Insecta": 50557}
N_PAIRS, SEED, WORKERS = 3000, 0, 3
UA = {"User-Agent": "Mozilla/5.0 (academic research; TaxEmbed spec 4.5)"}
API = "https://www.timetree.org/api/pairwise/{}/{}"

_lock = threading.Lock()


def newick_leaves(path: Path) -> list[str]:
    return re.findall(r"[(,]\s*([A-Za-z][A-Za-z0-9_.\-']*)\s*:", path.read_text())


def sci_to_taxid(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "scientific name" not in line:
                continue
            p = line.split("\t|\t")
            if len(p) < 2:
                continue
            try:
                out[p[1].strip().replace(" ", "_")] = int(p[0].strip())
            except ValueError:
                continue
    return out


def fetch_pair(a: int, b: int) -> dict:
    url = API.format(a, b)
    rec = {"a": a, "b": b}
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=30) as r:
            txt = r.read().decode("utf-8", "replace").strip()
        lines = [ln for ln in txt.split("\n") if ln.strip()]
        if len(lines) < 2:
            rec["error"] = "no data row"
            return rec
        hdr = [h.strip() for h in lines[0].split(",")]
        val = lines[1].split(",")
        d = dict(zip(hdr, val))
        rec["all_total"] = int(d["all_total"]) if d.get("all_total", "").strip().isdigit() else 0
        for key, out_key in (("precomputed_age", "age"),
                             ("precomputed_ci_low", "ci_low"),
                             ("precomputed_ci_high", "ci_high")):
            try:
                rec[out_key] = float(d.get(key, ""))
            except (TypeError, ValueError):
                rec[out_key] = None
    except urllib.error.HTTPError as e:
        rec["error"] = f"HTTP {e.code}"
    except Exception as e:                                   # noqa: BLE001
        rec["error"] = f"{type(e).__name__}"
    return rec


def main() -> None:
    print("=" * 92)
    print("§4.5 — fetch divergence times for the pre-registered pair sample (ages only)")
    print("=" * 92)

    taxid2row = load_mapping(MAPPING)
    name2tax = sci_to_taxid(NAMES)
    ttol = {name2tax[n] for n in set(newick_leaves(NWK)) if n in name2tax}
    usable = ttol & set(taxid2row)
    print(f"  TTOL leaves with an embedding coordinate: {len(usable):,}")

    parent_map = parse_parents(NODES)
    taxids, idx, parent, depth, tin, tout = build_tree(parent_map)

    rng = np.random.default_rng(SEED)
    want: list[tuple[int, int, str]] = []
    for sname, stax in STRATA.items():
        si = idx.get(stax)
        pool = np.array(sorted(t for t in usable
                               if t in idx and tin[si] <= tin[idx[t]] < tout[si]), dtype=np.int64)
        print(f"  {sname:<12} eligible taxa {len(pool):,}")
        seen = set()
        while len(seen) < N_PAIRS:
            i, j = rng.integers(0, len(pool), size=2)
            if i == j:
                continue
            key = (int(min(pool[i], pool[j])), int(max(pool[i], pool[j])))
            if key in seen:
                continue
            seen.add(key)
        want += [(a, b, sname) for a, b in sorted(seen)]
    print(f"  pairs drawn (seed {SEED}, before any distance): {len(want):,}")

    done: dict[tuple[int, int], dict] = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            try:
                r = json.loads(line)
                done[(r["a"], r["b"])] = r
            except Exception:                                # noqa: BLE001
                continue
        print(f"  cache: {len(done):,} pairs already fetched — resuming")
    CACHE.parent.mkdir(parents=True, exist_ok=True)

    todo = [(a, b, s) for a, b, s in want if (a, b) not in done]
    print(f"  to fetch: {len(todo):,}  with {WORKERS} workers\n")
    t0 = time.time()
    n_done = [0]
    fh = CACHE.open("a")

    def work(item):
        a, b, s = item
        rec = fetch_pair(a, b)
        rec["stratum"] = s
        with _lock:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n_done[0] += 1
            k = n_done[0]
            if k % 250 == 0 or k == len(todo):
                el = time.time() - t0
                rate = k / max(el, 1e-9)
                print(f"    {k:,}/{len(todo):,}  {rate:.1f} req/s  "
                      f"eta {(len(todo)-k)/max(rate,1e-9)/60:.0f} min", flush=True)
        return rec

    if todo:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for rec in ex.map(work, todo):
                done[(rec["a"], rec["b"])] = rec
    fh.close()

    rows = [done[(a, b)] | {"stratum": s} for a, b, s in want if (a, b) in done]
    ok = [r for r in rows if r.get("age") is not None and "error" not in r]
    dated = [r for r in ok if r.get("all_total", 0) >= 1]
    print(f"\n  fetched {len(rows):,}; with an age {len(ok):,}; "
          f"with all_total >= 1 (gate c) {len(dated):,}")
    for sname in STRATA:
        d = [r for r in dated if r["stratum"] == sname]
        print(f"    {sname:<12} dated pairs {len(d):,}  "
              f"{'GATE C OK' if len(d) >= 1000 else '⚠ BELOW the 1,000 floor'}")

    json.dump({
        "preregistration": "results/timetree_preregistration.json",
        "preregistration_sha256":
            "a3e733c781914a80046a7e116f491951269422d46ff31bf88bd2929521eb7fba",
        "note": "AGES ONLY — no embedding was read by this script",
        "n_pairs_requested": len(want), "n_fetched": len(rows),
        "n_with_age": len(ok), "n_all_total_ge_1": len(dated),
        "seed": SEED, "n_pairs_per_stratum": N_PAIRS,
        "pairs": rows,
    }, OUT.open("w"), indent=2)
    print(f"\nwritten: {OUT}   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
