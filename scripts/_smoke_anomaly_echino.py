"""S0274 local end-to-end gate for the anomaly OOM fix (commit cdae1ff).

Runs the FULL anomaly pipeline (scorer + leg A ROC + leg C enrichment) on the small local
echino_softmax embedding via MPS, mirroring analyze_lrz_anomaly.sh's exact flag combo, to confirm
the modified observed_purity kernel runs end-to-end before the LRZ resubmit. Small pool (no big
files, no disk pressure); the _safe_batch cap is a no-op at this scale — its scale behaviour is
covered by tests/eval/test_anomaly_knn.py.

taxopy.TaxDb CONSUMES (deletes) names.dmp/nodes.dmp from its taxdb_dir on load, so — exactly like
analyze_lrz_anomaly.sh ($WORK_TAXDUMP copy for the scorer vs the pristine ro $NAMES_DMP for leg C) —
the scorer/leg-A run against a throwaway WORK copy while leg C reads the pristine source names.dmp.
Point TAXDUMP_DIR at a freshly-extracted taxdump (data/taxdump_current is mutated/Dropbox-evicted).
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
CKPT = ROOT / "artifacts/tags/echino_softmax/echino_softmax.pth"
MAP = ROOT / "data/taxopy/echinodermata_7586_clean/taxonomy_edges_echinodermata_7586_clean.mapping.tsv"
SRC = Path(os.environ.get("TAXDUMP_DIR", str(ROOT / "data/taxdump_current")))  # pristine taxdump
WORK = Path("/tmp/_anomaly_gate_taxwork")          # mutable copy — taxopy consumes names/nodes here
NAMES = SRC / "names.dmp"                           # leg C reads the PRISTINE names (mirrors LRZ ro $NAMES_DMP)
OUT = ROOT / "artifacts/tags/echino_softmax/anomaly_s0274_gate"

for f in (CKPT, MAP, NAMES, SRC / "nodes.dmp"):
    if not f.exists():
        print(f"PREREQ MISSING: {f}  (set TAXDUMP_DIR to a freshly-extracted taxdump)")
        sys.exit(2)

# fresh mutable work-copy for the scorer/leg-A taxopy loads (taxopy deletes names/nodes.dmp on load)
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
for f in ("names.dmp", "nodes.dmp", "merged.dmp"):
    if (SRC / f).exists():
        shutil.copy(SRC / f, WORK / f)

steps = [
    ("scorer", [PY, str(ROOT / "scripts/taxonomy_anomaly.py"),
                "--checkpoint", str(CKPT), "--mapping", str(MAP), "--data-dir", str(WORK),
                "--rank", "family", "--k", "10", "--n-null", "200", "--n-bins", "5", "--seed", "0",
                "--device", "mps", "--knn-batch", "2048", "-o", str(OUT)]),
    ("legA_roc", [PY, str(ROOT / "scripts/_anomaly_validation.py"), "roc",
                  "--checkpoint", str(CKPT), "--mapping", str(MAP), "--data-dir", str(WORK),
                  "--rank", "family", "--k", "10", "--n-relocate", "3000", "--n-null", "200", "--seed", "0",
                  "--device", "mps", "--knn-batch", "2048", "-o", str(OUT)]),
    ("legC_enrich", [PY, str(ROOT / "scripts/_anomaly_validation.py"), "enrichment",
                     "--pool-npz", str(OUT / "anomaly_pool.npz"), "--mapping", str(MAP),
                     "--names-dmp", str(NAMES), "--top-frac", "0.1", "-o", str(OUT)]),
]

for name, cmd in steps:
    t0 = time.time()
    print(f"\n=== {name} ===", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"FAIL: {name} exited {r.returncode}")
        sys.exit(1)
    print(f"OK: {name} ({time.time() - t0:.1f}s)")

print("\nS0274 GATE PASSED: scorer + leg A + leg C all completed end-to-end on echino.")
