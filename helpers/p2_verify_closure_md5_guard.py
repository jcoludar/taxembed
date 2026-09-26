#!/usr/bin/env python3
"""R5 (2026-09-26): does every production manifest's `source_md5` match the closure the job
scripts actually point `--closure` at?

The new guard in `scripts/score_p2_linkpred.py` refuses when they disagree. A guard that refuses
the CORRECT inputs is worse than no guard, so this checks the healthy direction first -- all 12
production manifests against their intended closure -- and then confirms the guard would fire on
a deliberately crossed pair.

Read-only. Writes nothing except stdout.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
SPLITS = _REPO / "data" / "p2_splits"
REAL_CLOSURE = (_REPO / "data" / "taxopy" / "metazoa_33208_clean"
                / "taxonomy_edges_metazoa_33208_clean_transitive.npz")


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main() -> int:
    cache: dict[Path, str] = {}

    def digest(p: Path) -> str:
        if p not in cache:
            cache[p] = md5(p)
        return cache[p]

    rows, bad = [], []
    for seed in (0, 1, 2):
        for vis in ("vis00", "vis50"):
            # the real arms are scored against the SHARED real closure
            m = SPLITS / f"p2_metazoa_33208_clean_{vis}_seed{seed}_manifest.json"
            rows.append((m, REAL_CLOSURE))
            # each degmatch arm is scored against ITS OWN SEED's shuffled closure
            dm = SPLITS / f"p2_metazoa_33208_clean_degmatch_{vis}_seed{seed}_manifest.json"
            dmc = (SPLITS
                   / f"taxonomy_edges_metazoa_33208_clean_degmatch_seed{seed}_transitive.npz")
            rows.append((dm, dmc))

    print("HEALTHY DIRECTION -- every production manifest vs the closure its job points at")
    for manifest_path, closure in rows:
        man = json.loads(manifest_path.read_text())
        want = man.get("source_md5")
        got = digest(closure)
        ok = want == got
        if not ok:
            bad.append((manifest_path.name, closure.name, want, got))
        print(f"  {'ok  ' if ok else 'BAD '} {manifest_path.name}")
        print(f"        -> {closure.name}  want {want}  got {got}")

    print("\nCROSSED DIRECTION -- would the guard actually fire?")
    # seed 0's degmatch manifest against seed 1's degmatch closure: same node count, different tree
    m0 = json.loads(
        (SPLITS / "p2_metazoa_33208_clean_degmatch_vis00_seed0_manifest.json").read_text())
    c1 = SPLITS / "taxonomy_edges_metazoa_33208_clean_degmatch_seed1_transitive.npz"
    crossed_detectable = m0.get("source_md5") != digest(c1)
    print(f"  seed0 manifest vs seed1 closure -> md5 differs? {crossed_detectable}  "
          f"(True == the guard fires; the node-count check cannot, both are 498,246 nodes)")

    print(f"\n{'=' * 78}")
    if bad:
        print(f"🛑 {len(bad)} manifest/closure pair(s) DISAGREE -- the new guard would refuse a "
              f"CORRECT input. Do not submit:")
        for name, closure, want, got in bad:
            print(f"  {name} -> {closure}: want {want}, got {got}")
        return 1
    if not crossed_detectable:
        print("🛑 the guard cannot distinguish a crossed seed -- it is not evidence")
        return 1
    print(f"✅ all {len(rows)} production manifest/closure pairs match, and a crossed seed IS "
          f"detectable by md5 where the node count is blind to it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
