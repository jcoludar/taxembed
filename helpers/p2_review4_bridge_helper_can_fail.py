"""READ-ONLY PROBE (review 4, 2026-09-26).

C-B's third bullet rewrote `helpers/check_p2_prereg_parses_real_scorer_output.py` so it "resolves
per seed exactly as the engine does and exits 1 rather than skipping to success". This drives the
REAL `scripts/score_p2_linkpred.py` to produce production-shaped JSON (seed-tagged
`--baselines-key`, `_ms` + a 5-checkpoint `_roll` group per arm) and then runs the bridge helper
on four inputs:

  GOOD   the full production shape (9 files, 3 seeds x {real pair, 2 degmatch controls})
  BAD-1  a control's own baselines family absent from the merge (review scenario S5)
  BAD-2  a 10-checkpoint roll group (the C-C case)
  BAD-3  the same S5 defect delivered as ONE pre-merged file (`apply_preregistration.py` skips
         `merge_p2_scorer_outputs` entirely when only one --json is given)

Writes only into a throwaway temp directory. Touches nothing in the repo.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "score_p2_linkpred.py"
HELPER = REPO / "helpers" / "check_p2_prereg_parses_real_scorer_output.py"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

spec = importlib.util.spec_from_file_location(
    "tsc", REPO / "tests" / "eval" / "test_score_p2_cli.py")
tsc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tsc)


def score(tmp: Path, manifest, held, specs, baselines_key, out_name) -> Path:
    out = tmp / out_name
    cmd = [sys.executable, str(SCRIPT), "--manifest", str(manifest), "--heldout", str(held)]
    for s in specs:
        cmd += ["--checkpoints", s]
    cmd += ["--baselines-key", baselines_key, "--out", str(out)]
    rc = subprocess.run(cmd, capture_output=True, text=True)
    if rc.returncode != 0:
        raise SystemExit(f"scorer failed: {rc.stderr[-2000:]}")
    return out


def run_helper(paths) -> tuple[int, str]:
    rc = subprocess.run([sys.executable, str(HELPER)] + [str(p) for p in paths],
                        capture_output=True, text=True)
    return rc.returncode, (rc.stdout + rc.stderr)


def build(tmp: Path, n_roll: int = 5):
    manifest, held = tsc._write_small_tree(tmp, test=(3, 5))
    ms = tmp / "ms_milestone_epoch200.pth"
    torch.save({"embeddings": tsc._SMALL_TREE_BALL_COORDS, "epoch": 200}, ms)
    epochs = list(range(201 - n_roll, 201)) if n_roll <= 5 else (
        list(range(116, 121)) + list(range(196, 201)))
    for e in epochs:
        torch.save({"embeddings": tsc._SMALL_TREE_BALL_COORDS, "epoch": e},
                   tmp / f"roll_epoch{e}.pth")
    roll = str(tmp / "roll_epoch*.pth")
    files = []
    for s in (0, 1, 2):
        files.append(score(
            tmp, manifest, held,
            [f"vis00_s{s}_ms={ms}", f"vis00_s{s}_roll={roll}",
             f"vis50_s{s}_ms={ms}", f"vis50_s{s}_roll={roll}"],
            f"baselines_s{s}", f"real_s{s}.json"))
        for arm in ("degmatch_vis00", "degmatch_vis50"):
            files.append(score(
                tmp, manifest, held,
                [f"{arm}_s{s}_ms={ms}", f"{arm}_s{s}_roll={roll}"],
                f"baselines_{arm}_s{s}", f"{arm}_s{s}.json"))
    return files


def strip_control_baselines(files, tmp: Path):
    """review scenario S5: the control scorer JSONs reach the merge WITHOUT their own key."""
    out = []
    for i, f in enumerate(files):
        d = json.loads(Path(f).read_text())
        for k in [k for k in d if k.startswith("baselines_degmatch")]:
            del d[k]
        p = tmp / f"stripped_{i}.json"
        p.write_text(json.dumps(d))
        out.append(p)
    return out


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        files = build(tmp, n_roll=5)
        print(f"built {len(files)} production-shaped scorer JSONs\n")

        rc, out = run_helper(files)
        print(f"GOOD  (full production shape)            -> exit {rc}   "
              f"PARSER OK printed: {'PARSER OK' in out}")
        for line in out.splitlines():
            if line.startswith(("parsing", "resolved", "PARSER OK", "  vis", "  degmatch")):
                print(f"        | {line[:110]}")

        stripped = strip_control_baselines(files, tmp)
        rc, out = run_helper(stripped)
        print(f"\nBAD-1 (control baselines absent, S5)      -> exit {rc}   "
              f"PARSER OK printed: {'PARSER OK' in out}")
        print(f"        | {[l for l in out.splitlines() if 'NO baselines family' in l or 'ValueError' in l][:1]}")

        merged_single = tmp / "single.json"
        merged = {"arms": {}}
        for f in stripped:
            d = json.loads(Path(f).read_text())
            merged["arms"].update(d["arms"])
            for k, v in d.items():
                if k.startswith("baselines") and k not in merged:
                    merged[k] = v
        merged_single.write_text(json.dumps(merged))
        rc, out = run_helper([merged_single])
        print(f"\nBAD-3 (same defect, ONE pre-merged file)  -> exit {rc}   "
              f"PARSER OK printed: {'PARSER OK' in out}")
        print(f"        | control arms in 'arms': "
              f"{sorted({k.rsplit('_s', 1)[0] for k in merged['arms']})}")
        print(f"        | baselines keys present : "
              f"{sorted(k for k in merged if k.startswith('baselines'))}")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        files = build(tmp, n_roll=10)
        rc, out = run_helper(files)
        print(f"\nBAD-2 (10-checkpoint roll group, C-C)     -> exit {rc}   "
              f"PARSER OK printed: {'PARSER OK' in out}")
        print(f"        | {[l for l in out.splitlines() if 'rolling checkpoints' in l][:1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
