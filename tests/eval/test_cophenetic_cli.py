import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / ".venv" / "bin" / "python"


def _make_fixture(tmp_path):
    # 6-node tree from Task 1; embed each node at radius=depth/4 along a random-but-clade-coherent dir
    parent = {1: 131567, 2: 131567, 3: 1, 4: 1, 5: 3, 6: 3}  # taxids; 131567 root
    # mapping idx->taxid
    taxids = [131567, 1, 2, 3, 4, 5]
    emb = np.zeros((6, 4), np.float32)
    rng = np.random.default_rng(0)
    for i, t in enumerate(taxids):
        d = {131567: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3}[t]
        v = rng.standard_normal(4); v /= np.linalg.norm(v)
        emb[i] = (v * (d / 5.0)).astype(np.float32)
    ckpt = tmp_path / "fix.pth"
    torch.save({"embeddings": torch.tensor(emb)}, ckpt)
    mp = tmp_path / "map.tsv"
    parent_of = {131567: 131567, 1: 131567, 2: 131567, 3: 1, 4: 1, 5: 3}
    mp.write_text("taxid\tidx\tparent\n" +
                  "\n".join(f"{t}\t{i}\t{parent_of[t]}" for i, t in enumerate(taxids)) + "\n")
    return ckpt, mp


def test_cli_runs_and_emits_model_vs_null(tmp_path):
    ckpt, mp = _make_fixture(tmp_path)
    out = tmp_path / "out"
    r = subprocess.run(
        [str(PY), str(ROOT / "scripts" / "cophenetic_fidelity.py"),
         "--checkpoint", str(ckpt), "--mapping", str(mp),
         "--parent-from-mapping",          # test mode: read parent from a sidecar instead of taxdump
         "--k", "1", "--n-queries", "6", "--n-pairs", "10", "-o", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    res = json.loads((out / "cophenetic_fidelity.json").read_text())
    assert "model" in res and "radial_only_null" in res
    assert "knn_precision" in res["model"]
    # model should be at least as faithful as the radial-only null on this clade-coherent fixture
    assert res["model"]["knn_precision"]["mean"] >= res["radial_only_null"]["knn_precision"]["mean"]
