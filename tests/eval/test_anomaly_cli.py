import json
import subprocess
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / ".venv" / "bin" / "python"


def _make_fixture(tmp_path):
    rng = np.random.default_rng(0)
    dim = 8
    root = 0
    fam_parents = [1, 2, 3]
    emb = np.zeros((30, dim), np.float32)
    taxids = list(range(30))
    parent_of = {t: 0 for t in range(30)}
    parent_of.update({0: 0, 1: 0, 2: 0, 3: 0})
    fam_dir = rng.standard_normal((3, dim)); fam_dir /= np.linalg.norm(fam_dir, axis=1, keepdims=True)
    leaf = 4
    leaf_family = {}
    for f in range(3):
        for _ in range(8):
            v = fam_dir[f] + 0.05 * rng.standard_normal(dim)
            v /= np.linalg.norm(v)
            emb[leaf] = (v * 0.8).astype(np.float32)
            parent_of[leaf] = fam_parents[f]
            leaf_family[leaf] = f
            leaf += 1
    v = fam_dir[0] + 0.05 * rng.standard_normal(dim); v /= np.linalg.norm(v)
    emb[29] = (v * 0.8).astype(np.float32)
    parent_of[29] = fam_parents[2]
    leaf_family[29] = 2
    for f in range(3):
        emb[fam_parents[f]] = (fam_dir[f] * 0.3).astype(np.float32)
    ckpt = tmp_path / "fix.pth"
    torch.save({"embeddings": torch.tensor(emb)}, ckpt)
    parent_line = lambda t: parent_of[t]
    mp = tmp_path / "map.tsv"
    rows = ["taxid\tidx\tparent\tfamlabel"]
    for i, t in enumerate(taxids):
        fam = leaf_family.get(t, -1)
        rows.append(f"{t}\t{i}\t{parent_line(t)}\t{fam}")
    mp.write_text("\n".join(rows) + "\n")
    return ckpt, mp


def test_cli_ranks_planted_anomaly_high(tmp_path):
    ckpt, mp = _make_fixture(tmp_path)
    out = tmp_path / "out"
    r = subprocess.run(
        [str(PY), str(ROOT / "scripts" / "taxonomy_anomaly.py"),
         "--checkpoint", str(ckpt), "--mapping", str(mp),
         "--parent-from-mapping", "--rank-from-mapping", "famlabel",
         "--k", "3", "--n-null", "20", "--n-bins", "2", "--seed", "0",
         "-o", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    summary = json.loads((out / "anomaly_summary.json").read_text())
    assert summary["k"] == 3 and summary["rank"] == "famlabel"
    import csv
    with (out / "anomaly_ranked.tsv").open() as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    order = [int(row["taxid"]) for row in rows]
    assert 29 in order[: len(order) // 4]
    assert "q_value" in rows[0]
