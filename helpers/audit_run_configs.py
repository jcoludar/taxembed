"""Verify the nine LRZ runs ARE the inputs the pre-registrations declare, before any verdict.

Rule 10 validates inputs before expensive compute; this is the same move one step later — validate
the inputs before an expensive CLAIM. The pre-registrations make checkable assertions about the
runs:

  recipe_angular_comparison.json :: arm_definition_step1
      canonical vs prior differ in EXACTLY four factors (effective batch, n_negatives, lr,
      lr_schedule) and are identical in ~18 others.
  objective_integrity_delta_preregistration.json :: arms
      task8_fixed_* is IDENTICAL to task9_canonical_* plus --exclude-descendant-negatives
      --drop-root-anchored, with --save-every 20 instead of 10.

If either claim is false on the recorded configs, the contrast measures something other than what
the pre-registration says, and no amount of scoring fixes it. Reads run.json over ssh; writes
nothing remote.

  <python> helpers/audit_run_configs.py [--json <out>]
"""
from __future__ import annotations

import argparse
import json
import subprocess

R = "/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz"
TAGS = {
    "canonical": [f"task9_canonical_s{s}" for s in (0, 1, 2)],
    "prior": [f"task9_prior_s{s}" for s in (0, 1, 2)],
    "fixed": [f"task8_fixed_canonical_s{s}" for s in (0, 1, 2)],
}

# The four factors the Task 9 pre-registration declares, and the two flags Task 8 adds.
_T9 = {"batch_size", "n_negatives", "lr", "lr_schedule", "grad_accum_steps",
       "warm_restart_on_phase", "lr_min_multiplier"}
_T8 = {"exclude_descendant_negatives", "drop_root_anchored", "save_every"}
DECLARED_T9 = _T9 | {"cli:" + k for k in _T9}
DECLARED_T8 = _T8 | {"cli:" + k for k in _T8}
# Identity/provenance keys, and the argv paths that merely restate the tag.
IGNORE = {"seed", "cli:seed", "tag", "slug", "created_at", "identifier", "run_id", "started_at",
          "finished_at", "hostname", "job_id", "output_dir", "timestamp", "date", "elapsed",
          "duration", "slurm_job_id", "cli:as", "cli:tag",
          # Output paths, derived from the tag (already ignored). Provenance, not configuration.
          "cli:checkpoint", "cli:save_every_dir", "cli:out"}


def fetch(tag: str) -> dict:
    out = subprocess.run(["ssh", "ai", "cat", f"{R}/artifacts/tags/{tag}/run.json"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"{tag}: {out.stderr.strip()}")
    return json.loads(out.stdout)


def flatten(d: dict) -> dict:
    """The recorded config is `training.args`, plus flags recoverable only from `training.command`.

    🧨 `training.args` DOES NOT CONTAIN `grad_accum_steps`, and effective batch
    (batch_size x grad_accum_steps) is one of the four factors the Task 9 pre-registration
    attributes the contrast to. Nor does it contain Task 8's `--exclude-descendant-negatives` /
    `--drop-root-anchored`. Auditing `args` alone would silently report those factors as IDENTICAL
    across arms — a check that could not have failed. So the argv is parsed too, and every
    command-line flag is recorded under a `cli:` prefix.
    """
    flat = dict(d.get("training", {}).get("args", {}))
    argv = d.get("training", {}).get("command", []) or []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if isinstance(tok, str) and tok.startswith("--"):
            name = "cli:" + tok[2:].replace("-", "_")
            nxt = argv[i + 1] if i + 1 < len(argv) else None
            if isinstance(nxt, str) and not nxt.startswith("--"):
                flat[name] = nxt
                i += 2
                continue
            flat[name] = True          # bare switch
        i += 1
    return flat


def differing(a: dict, b: dict) -> dict:
    keys = (set(a) | set(b)) - IGNORE
    return {k: (a.get(k, "<absent>"), b.get(k, "<absent>")) for k in sorted(keys)
            if a.get(k, "<absent>") != b.get(k, "<absent>")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    args = ap.parse_args()

    cfg = {arm: [flatten(fetch(t)) for t in tags] for arm, tags in TAGS.items()}
    report: dict = {"within_arm": {}, "between": {}}

    print("=== 1. WITHIN each arm, only the seed may differ ===")
    for arm, runs in cfg.items():
        diffs = {}
        for i in range(1, len(runs)):
            diffs[f"s0_vs_s{i}"] = differing(runs[0], runs[i])
        bad = {k: v for k, v in diffs.items() if v}
        report["within_arm"][arm] = bad
        print(f"  {arm:>10}: {'CLEAN' if not bad else 'DIFFERS -> ' + json.dumps(bad)}")
        for runs_i, tag in zip(runs, TAGS[arm]):
            print(f"             {tag}: seed={runs_i.get('seed')} epochs={runs_i.get('epochs')} "
                  f"batch={runs_i.get('batch_size')} accum={runs_i.get('cli:grad_accum_steps')} "
                  f"lr={runs_i.get('lr')} neg={runs_i.get('n_negatives')}")

    print("\n=== 2. canonical vs prior — the pre-registered FOUR factors ===")
    d = differing(cfg["canonical"][0], cfg["prior"][0])
    report["between"]["canonical_vs_prior"] = d
    for k, (x, y) in d.items():
        mark = "declared" if k in DECLARED_T9 else "*** UNDECLARED ***"
        print(f"  {k:<32} canonical={x!r:<28} prior={y!r:<20} [{mark}]")
    undeclared = sorted(set(d) - DECLARED_T9)
    print(f"  -> {len(d)} differing keys; UNDECLARED: {undeclared or 'none'}")

    print("\n=== 3. fixed vs unfixed — identical except the two sampler flags + save_every ===")
    d8 = differing(cfg["fixed"][0], cfg["canonical"][0])
    report["between"]["fixed_vs_unfixed"] = d8
    for k, (x, y) in d8.items():
        mark = "declared" if k in DECLARED_T8 else "*** UNDECLARED ***"
        print(f"  {k:<32} fixed={x!r:<28} unfixed={y!r:<20} [{mark}]")
    undeclared8 = sorted(set(d8) - DECLARED_T8)
    print(f"  -> {len(d8)} differing keys; UNDECLARED: {undeclared8 or 'none'}")

    verdict = not undeclared and not undeclared8 and not any(report["within_arm"].values())
    print(f"\n>>> INPUTS {'MATCH' if verdict else 'DO NOT MATCH'} the pre-registered arm definitions")
    if args.json:
        report["inputs_match_preregistration"] = verdict
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
