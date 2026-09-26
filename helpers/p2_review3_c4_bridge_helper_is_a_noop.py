#!/usr/bin/env python3
"""REVIEW WAVE 3 (read-only): is `helpers/check_p2_prereg_parses_real_scorer_output.py` -- the
designated C4 scorer->verdict bridge check -- able to fail on the JSON shape production now writes?

`scripts/p2_lrz_score.sh` (after the amendment_5 / Part 2 fix) passes SEED-TAGGED
`--baselines-key` values: `baselines_s0`, `baselines_degmatch_vis00_s1`, ... There is no bare
`"baselines"` key anywhere in its 9 output files. The bridge helper does:

    baselines = result.get("baselines")
    ...
    if baselines is not None:   <- gates the gate checks
    ...
    if baselines is not None:   <- gates the p2_group_stats call

so on production-shaped input every gate check and the whole `p2_group_stats` pass are SKIPPED, and
the script still prints "PARSER OK on real scorer output". And the one call it would make,
`p2_group_stats(result, arm, (0,1,2), baselines)`, passes a single baselines DICT where the Part 2
signature now expects a `{seed: dict}` MAPPING -- so had the key been present it would raise KeyError.

This script runs the bridge helper twice: once on PRE-Part-2-shaped input (bare `baselines`) and
once on PRODUCTION-shaped input (seed-tagged), and reports which checks each run actually performed.

Writes only into a temp dir. Touches no production file.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = REPO / ".venv" / "bin" / "python"
BRIDGE = REPO / "helpers" / "check_p2_prereg_parses_real_scorer_output.py"

MS_EPOCHS = list(range(10, 201, 10))
ROLL_EPOCHS = [196, 197, 198, 199, 200]


def ck(epoch, mrr, nr, jit=0.0):
    m = {"n": 500, "n_scored": 500, "n_trivial": 0, "mean_rank": 3.0, "mrr": mrr + jit,
         "hits_at_1": 0.7, "hits_at_10": 0.95, "normalized_rank": nr}
    depth = {k: {"n": 2000, "mrr": mrr + jit, "normalized_rank": nr}
             for k in ("11-15", "16-21", "22-28")}
    return {"epoch": epoch, "path": f"f{epoch}.pth", "trainer": {"loss": 4.0},
            "metrics": m, "metrics_cosine": m, "metrics_poincare": m,
            "by_pool_size": {"cosine": {}, "poincare": {}},
            "by_depth": {"cosine": depth, "poincare": depth}}


def baselines():
    return {"sibling_chance_mean": 0.13126, "chance_hits_at_1_mean": 0.13126,
            "chance_mrr_mean": 0.27107, "vendrov_recall": 0.0, "majority_parent_rate": 0.002,
            "degree_prior": {"mrr": 0.56, "hits_at_1": 0.40, "hits_at_10": 0.86,
                             "normalized_rank": 0.1538, "n": 28418, "n_scored": 27446,
                             "n_trivial": 972}}


def scorer_file(seed: int, seed_tagged: bool) -> dict:
    arms = {}
    for arm in ("vis00", "vis50"):
        arms[f"{arm}_s{seed}_ms"] = {"checkpoints": [ck(e, 0.82 * min(1, e / 100), 0.04)
                                                     for e in MS_EPOCHS]}
        arms[f"{arm}_s{seed}_roll"] = {"checkpoints": [ck(e, 0.82, 0.04, (i - 2) * 1e-4)
                                                      for i, e in enumerate(ROLL_EPOCHS)]}
    key = f"baselines_s{seed}" if seed_tagged else "baselines"
    return {"arms": arms, key: baselines()}


def run(label: str, seed_tagged: bool, tmp: Path) -> None:
    paths = []
    for s in (0, 1, 2):
        p = tmp / f"{'tagged' if seed_tagged else 'bare'}_s{s}.json"
        p.write_text(json.dumps(scorer_file(s, seed_tagged)))
        paths.append(str(p))
    r = subprocess.run([str(PY), str(BRIDGE), *paths], capture_output=True, text=True)
    print(f"\n========== {label} ==========")
    print(f"exit code: {r.returncode}")
    print(r.stdout.strip())
    if r.stderr.strip():
        print("STDERR:\n" + r.stderr.strip()[-2000:])
    did_gates = "gate_a" in r.stdout
    did_group = "group stats" in r.stdout
    said_ok = "PARSER OK" in r.stdout
    print(f"\n  -> ran per-run gate checks? {did_gates}")
    print(f"  -> ran the p2_group_stats pass? {did_group}")
    print(f"  -> printed 'PARSER OK'? {said_ok}")
    if said_ok and not (did_gates and did_group):
        print("  -> 🛑 the bridge check reported OK while performing NEITHER of the two checks it "
              "exists for.")


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        run("PRE-Part-2 shape: bare 'baselines' key", False, tmp)
        run("PRODUCTION shape (what p2_lrz_score.sh writes): seed-tagged 'baselines_s{n}'",
            True, tmp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
