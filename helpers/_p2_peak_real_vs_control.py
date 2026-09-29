#!/usr/bin/env python3
"""P2 below-chance diagnosis: at its BEST epoch, does a real arm beat its matched control?

The curriculum confound means epoch 200 (the pre-registered read point) is deep in a decay.
The obvious next question -- and the one that decides whether anything is being masked -- is
whether the real taxonomy separates from its degree-matched control at ANY epoch.

Cross-tree comparisons use normalized_rank (p2_amendment_1_20260924): chance is 0.5 for any
pool size, LOWER is better. Reported here per seed, for the arm's best (minimum) value over
all 20 milestone checkpoints.

⚠ This is a DIAGNOSTIC read, not a result. Reading an arm at its best epoch is exactly the
post-hoc rescue the 2026-09-29 handoff forbids. Its only legitimate use is the negative one:
if the real arm does NOT beat its control even when both are read at their own best epoch,
then the epoch-200 collapse is not hiding a positive finding.

Written 2026-09-29 for the P2 below-chance-MRR diagnosis.
"""
import json
from pathlib import Path

ROOT = Path("/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings")
ARTDIR = ROOT / "artifacts" / "p2_scoring"


def best_of(arm, metric="metrics_cosine"):
    """(best normalized_rank, epoch, value at epoch 200) over the milestone checkpoints."""
    rows = []
    for cp in arm.get("checkpoints") or []:
        m = cp.get(metric) or {}
        nr = m.get("normalized_rank")
        if nr is not None:
            rows.append((nr, cp.get("epoch")))
    if not rows:
        return None, None, None
    best_nr, best_ep = min(rows, key=lambda t: t[0])
    at200 = next((nr for nr, ep in rows if ep == 200), None)
    return best_nr, best_ep, at200


def collect():
    """arm_name -> (best_nr, best_epoch, nr_at_200), from the *_ms milestone series."""
    out = {}
    for path in sorted(ARTDIR.glob("*.json")):
        d = json.loads(path.read_text())
        for arm_name, arm in (d.get("arms") or {}).items():
            if not arm_name.endswith("_ms"):
                continue
            out[arm_name[:-3]] = best_of(arm)
    return out


def main():
    got = collect()
    print("normalized_rank -- LOWER IS BETTER, 0.5 = chance for any pool size")
    print("(cross-tree metric per p2_amendment_1_20260924)\n")
    header = (f"{'pair (real vs matched control)':<34} {'real best':>10} {'@ep':>5} | "
              f"{'ctrl best':>10} {'@ep':>5} | {'real-ctrl':>10} | {'real@200':>9} {'ctrl@200':>9}")
    print(header)
    print("-" * len(header))
    for vis in ("vis00", "vis50"):
        for seed in (0, 1, 2):
            real, ctrl = f"{vis}_s{seed}", f"degmatch_{vis}_s{seed}"
            if real not in got or ctrl not in got:
                continue
            rb, rep, r200 = got[real]
            cb, cep, c200 = got[ctrl]
            delta = rb - cb
            verdict = "real better" if delta < 0 else "CONTROL better"
            print(f"{real + ' vs ' + ctrl:<34} {rb:>10.4f} {str(rep):>5} | "
                  f"{cb:>10.4f} {str(cep):>5} | {delta:>+10.4f} | {r200:>9.4f} {c200:>9.4f}"
                  f"   {verdict}")
    print()
    print("A real arm that does not beat its control AT ITS OWN BEST EPOCH is not a finding")
    print("hidden by the epoch-200 collapse.")


if __name__ == "__main__":
    main()
