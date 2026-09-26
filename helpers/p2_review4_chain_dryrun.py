"""READ-ONLY review helper (adversarial pre-submit review, 2026-09-26).

Exercises the ENTIRE post-GPU read chain -- 9 scorer JSONs -> merge -> gates -> verdict --
BEFORE 130-230 GPU-hours are spent, using the JSON SCHEMA of a REAL
`scripts/score_p2_linkpred.py` run (produced in this review on the tiny mollusca split) rather
than a hand-written fixture, and driving it through the exact production entrypoint
`scripts/apply_preregistration.py`.

No P2 scorer output has ever existed before this review, so `merge_p2_scorer_outputs`,
`assert_control_baselines_exist`, `p2_run_value`'s roll-window assertion and the amendment_6
arm reading have only ever run on fixtures their own author wrote.

Scenarios:
  A  the 9-file production shape, healthy  -> verdict with and without --amendment-6
  B  one control's baselines block missing -> must RAISE (C-B guard)
  C  one arm's tag dir polluted, 10 rolling checkpoints -> must RAISE (C-C guard)

Writes only into the session scratchpad.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys

REPO = "/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings"
SCRATCH = ("/private/tmp/claude-501/-Users-jcoludar-CascadeProjects-SpeciesEmbedding-"
           "TaxPointCare/30c206db-80e2-441f-aa57-185551d95d1f/scratchpad")
REAL = os.path.join(SCRATCH, "p2_dryrun_real_s0.json")
PY = os.path.join(REPO, ".venv", "bin", "python")
APPLY = os.path.join(REPO, "scripts", "apply_preregistration.py")

# measured on the 12 production metazoa splits (helpers/p2_amendment6_reachability.py)
REAL_PRIOR_NR = 0.15381
DEG_PRIOR_NR = 0.04041
CHANCE_MRR = 0.27107
SIB_CHANCE = 0.13126
STRATA = ["11-15", "16-21", "22-28"]

MS_EPOCHS = list(range(10, 201, 10))       # --save-every 10
ROLL_EPOCHS = [196, 197, 198, 199, 200]    # trainer's deque(maxlen=5)


def ckpt(epoch: int, mrr: float, nr: float) -> dict:
    m = {"n": 28418, "n_scored": 28000, "n_trivial": 418, "mean_rank": 2.0,
         "mrr": mrr, "hits_at_1": mrr * 0.8, "hits_at_10": 0.99, "normalized_rank": nr}
    by_depth = {s: {"n": 5000, "n_scored": 5000, "n_trivial": 0, "mean_rank": 2.0,
                    "mrr": mrr, "hits_at_1": mrr * 0.8, "hits_at_10": 0.99,
                    "normalized_rank": nr} for s in STRATA}
    return {"epoch": epoch, "path": f"/fake/_epoch{epoch}.pth",
            "trainer": {"loss": 4.0, "epoch": epoch},
            "metrics": m, "metrics_cosine": m, "metrics_poincare": m,
            "by_pool_size": {"cosine": {}, "poincare": {}},
            "by_depth": {"cosine": by_depth, "poincare": by_depth},
            "poincare_radius_source": "z_embeddings", "max_radius": 5.0,
            "radius_guard": "z_norm_gt_100", "radius_overflow_risk": False,
            "n_norm_saturated": None, "clip_count_total": 0}


def group(mrr_final: float, nr: float, n_roll: int = 5) -> dict:
    ms = [ckpt(e, CHANCE_MRR + (mrr_final - CHANCE_MRR) * (e / 200.0), nr) for e in MS_EPOCHS]
    roll_eps = ROLL_EPOCHS if n_roll == 5 else list(range(201 - n_roll, 201))
    roll = [ckpt(e, mrr_final + (i - 2) * 1e-4, nr) for i, e in enumerate(roll_eps)]
    return {"ms": {"checkpoints": ms}, "roll": {"checkpoints": roll}}


def baselines(prior_nr: float) -> dict:
    return {"sibling_chance_mean": SIB_CHANCE, "chance_hits_at_1_mean": SIB_CHANCE,
            "chance_mrr_mean": CHANCE_MRR, "vendrov_recall": 0.0,
            "majority_parent_rate": 0.02,
            "degree_prior": {"mrr": 0.40, "hits_at_1": 0.30, "normalized_rank": prior_nr}}


def base_doc() -> dict:
    with open(REAL) as fh:
        real = json.load(fh)
    d = {k: copy.deepcopy(v) for k, v in real.items()
         if k not in ("arms",) and not k.startswith("baselines")}
    d["arms"] = {}
    return d


def build_files(outdir: str, drop_control_baselines: bool = False,
                pollute_roll_for: tuple[str, int] | None = None) -> list[str]:
    os.makedirs(outdir, exist_ok=True)
    paths = []
    # R1 shape: real arm improves to 0.25x its own prior, control only to 0.75x of its own
    real_nr = 0.25 * REAL_PRIOR_NR
    ctrl_nr = 0.75 * DEG_PRIOR_NR
    for s in (0, 1, 2):
        doc = base_doc()
        doc[f"baselines_s{s}"] = baselines(REAL_PRIOR_NR)
        for arm in ("vis00", "vis50"):
            n_roll = 10 if pollute_roll_for == (arm, s) else 5
            g = group(0.82002, real_nr, n_roll=n_roll)
            doc["arms"][f"{arm}_s{s}_ms"] = g["ms"]
            doc["arms"][f"{arm}_s{s}_roll"] = g["roll"]
        p = os.path.join(outdir, f"p2_real_s{s}.json")
        with open(p, "w") as fh:
            json.dump(doc, fh)
        paths.append(p)

        for arm in ("degmatch_vis00", "degmatch_vis50"):
            doc = base_doc()
            if not drop_control_baselines:
                doc[f"baselines_{arm}_s{s}"] = baselines(DEG_PRIOR_NR)
            doc[f"baselines_s{s}"] = baselines(REAL_PRIOR_NR)
            g = group(0.80, ctrl_nr)
            doc["arms"][f"{arm}_s{s}_ms"] = g["ms"]
            doc["arms"][f"{arm}_s{s}_roll"] = g["roll"]
            p = os.path.join(outdir, f"p2_{arm}_s{s}.json")
            with open(p, "w") as fh:
                json.dump(doc, fh)
            paths.append(p)
    return paths


def run_apply(paths: list[str], amendments: list[str], label: str) -> None:
    cmd = [PY, APPLY, "--task", "p2", "--json", *paths, *amendments]
    out = subprocess.run(cmd, capture_output=True, text=True)
    print(f"--- {label}  (exit {out.returncode}) ---")
    tail = (out.stdout or "").strip().splitlines()
    for line in tail[-14:]:
        print("   ", line)
    if out.returncode != 0:
        err = (out.stderr or "").strip().splitlines()
        for line in err[-6:]:
            print("  E ", line)
    print()


def main() -> None:
    if not os.path.exists(REAL):
        sys.exit(f"missing {REAL} -- run score_p2_linkpred.py on the mollusca split first")

    print("=" * 78)
    print("A. healthy 9-file production shape (schema from a REAL scorer run)")
    print("=" * 78)
    paths = build_files(os.path.join(SCRATCH, "chainA"))
    print(f"   {len(paths)} scorer JSONs written")
    run_apply(paths, ["--amendment-1", "--amendment-2", "--amendment-4"],
              "PRESCRIBED by both job-script headers: --amendment-1 --amendment-2 --amendment-4")
    run_apply(paths, ["--amendment-1", "--amendment-2", "--amendment-4", "--amendment-6"],
              "with --amendment-6")

    print("=" * 78)
    print("B. C-B guard: every control's own baselines block ABSENT from the merge")
    print("=" * 78)
    paths_b = build_files(os.path.join(SCRATCH, "chainB"), drop_control_baselines=True)
    run_apply(paths_b, ["--amendment-1", "--amendment-2", "--amendment-4", "--amendment-6"],
              "must RAISE, not silently use the real tree's floor")

    print("=" * 78)
    print("C. C-C guard: one arm's tag dir polluted -> 10 rolling checkpoints")
    print("=" * 78)
    paths_c = build_files(os.path.join(SCRATCH, "chainC"), pollute_roll_for=("vis50", 1))
    run_apply(paths_c, ["--amendment-1", "--amendment-2", "--amendment-4", "--amendment-6"],
              "must RAISE naming vis50 seed 1")


if __name__ == "__main__":
    main()
