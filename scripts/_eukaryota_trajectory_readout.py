"""Run the seeded separation analyzer across eukaryota milestone checkpoints to
read the ep80->180 trajectory (stall-signal check for the all-Life run).

One venv-python invocation (shell-hygiene safe); subprocess-calls the analyzer
once per milestone with --repeats 1 (point estimate is enough for a trajectory).
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TAG = ROOT / "artifacts" / "tags" / "eukaryota_canonical"
MAPPING = ROOT / "data" / "taxopy" / "eukaryota_2759_clean" / "taxonomy_edges_eukaryota_2759_clean.mapping.tsv"
PY = ROOT / ".venv" / "bin" / "python"
ANALYZER = ROOT / "scripts" / "analyze_hierarchy_hyperbolic.py"

MILESTONES = [80, 100, 120, 150, 180]
RANK_RE = re.compile(r"^(Phylum|Class|Order|Family)\s+([0-9.]+)x", re.M)

rows = []
for ep in MILESTONES:
    ckpt = TAG / f"eukaryota_canonical_milestone_epoch{ep}.pth"
    if not ckpt.exists():
        print(f"[skip] ep{ep}: checkpoint not present")
        continue
    out = TAG / f"analysis_milestone_ep{ep:03d}"
    cmd = [str(PY), str(ANALYZER), "--checkpoint", str(ckpt), "--mapping",
           str(MAPPING), "--ranks", "phylum", "class", "order", "family",
           "--seed", "0", "--repeats", "1", "-o", str(out)]
    print(f"[run ] ep{ep} ...", flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[FAIL] ep{ep} rc={res.returncode}\n{res.stderr[-1500:]}")
        continue
    seps = {m.group(1): float(m.group(2)) for m in RANK_RE.finditer(res.stdout)}
    rows.append((ep, seps))
    print(f"[done] ep{ep}: phylum={seps.get('Phylum')} class={seps.get('Class')} "
          f"order={seps.get('Order')} family={seps.get('Family')}", flush=True)

print("\n==== EUKARYOTA SEPARATION TRAJECTORY (seed 0, repeats 1) ====")
print(f"{'epoch':>6} {'phylum':>8} {'class':>8} {'order':>8} {'family':>8}")
for ep, s in rows:
    print(f"{ep:>6} {s.get('Phylum',float('nan')):>8.2f} {s.get('Class',float('nan')):>8.2f} "
          f"{s.get('Order',float('nan')):>8.2f} {s.get('Family',float('nan')):>8.2f}")
# ep200 (final) for the tail, from the already-run seeded analysis
print(f"{200:>6} {'3.12':>8} {'5.09':>8} {'7.13':>8} {'8.23':>8}  (final, 5-seed)")

(TAG / "trajectory_readout.json").write_text(json.dumps(
    {"milestones": [{"epoch": ep, **s} for ep, s in rows]}, indent=2))
print(f"\nSaved: {TAG / 'trajectory_readout.json'}")
