import json
import subprocess
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / ".venv" / "bin" / "python"


def _fixture(tmp_path):
    rng = np.random.default_rng(1)
    dim = 8
    fam_parents = [1, 2, 3]
    emb = np.zeros((30, dim), np.float32)
    parent_of = {0: 0, 1: 0, 2: 0, 3: 0}
    fam_dir = rng.standard_normal((3, dim)); fam_dir /= np.linalg.norm(fam_dir, axis=1, keepdims=True)
    leaf, leaf_family = 4, {}
    for f in range(3):
        for _ in range(8):
            v = fam_dir[f] + 0.05 * rng.standard_normal(dim); v /= np.linalg.norm(v)
            emb[leaf] = (v * 0.8).astype(np.float32)
            parent_of[leaf] = fam_parents[f]; leaf_family[leaf] = f; leaf += 1
    parent_of[28] = fam_parents[0]; leaf_family[28] = 0
    parent_of[29] = fam_parents[1]; leaf_family[29] = 1
    for f in range(3):
        emb[fam_parents[f]] = (fam_dir[f] * 0.3).astype(np.float32)
    ckpt = tmp_path / "fix.pth"; torch.save({"embeddings": torch.tensor(emb)}, ckpt)
    rows = ["taxid\tidx\tparent\tfamlabel"]
    for i in range(30):
        rows.append(f"{i}\t{i}\t{parent_of[i]}\t{leaf_family.get(i, -1)}")
    mp = tmp_path / "map.tsv"; mp.write_text("\n".join(rows) + "\n")
    return ckpt, mp


def test_roc_emits_auc_curve(tmp_path):
    ckpt, mp = _fixture(tmp_path)
    out = tmp_path / "out"
    r = subprocess.run(
        [str(PY), str(ROOT / "scripts" / "_anomaly_validation.py"), "roc",
         "--checkpoint", str(ckpt), "--mapping", str(mp),
         "--parent-from-mapping", "--rank-from-mapping", "famlabel",
         "--k", "3", "--n-relocate", "12", "--seed", "0", "-o", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    res = json.loads((out / "roc_by_displacement.json").read_text())
    assert "auc_by_displacement" in res
    assert "baseline_auc_by_displacement" in res
    aucs = [v["score_z"] for v in res["auc_by_displacement"].values()]
    assert max(aucs) > 0.5


def test_enrichment_runs_on_precomputed_pool(tmp_path):
    import numpy as np
    pool_idx = np.arange(20)
    taxids = np.arange(100, 120)
    score_z = np.linspace(-1, 5, 20)
    np.savez(tmp_path / "anomaly_pool.npz", pool_idx=pool_idx, pool_lab=np.zeros(20, int),
             observed=np.zeros(20), score_z=score_z, score_excess=np.zeros(20),
             pvals=np.ones(20), qvals=np.ones(20), clade_size=np.ones(20),
             depth=np.ones(20), degree=np.ones(20), dist_to_parent_centroid=np.zeros(20))
    mp = tmp_path / "map.tsv"
    mp.write_text("taxid\tidx\n" + "\n".join(f"{t}\t{i}" for i, t in enumerate(taxids)) + "\n")
    names = tmp_path / "names.dmp"
    lines = []
    for t in taxids:
        nm = "incertae sedis sp." if t >= 118 else f"Genus species{t}"
        lines.append(f"{t}\t|\t{nm}\t|\t\t|\tscientific name\t|")
    names.write_text("\n".join(lines) + "\n")

    from pathlib import Path
    import subprocess, json
    ROOT = Path(__file__).resolve().parents[2]
    PY = ROOT / ".venv" / "bin" / "python"
    out = tmp_path / "out"
    r = subprocess.run(
        [str(PY), str(ROOT / "scripts" / "_anomaly_validation.py"), "enrichment",
         "--pool-npz", str(tmp_path / "anomaly_pool.npz"), "--mapping", str(mp),
         "--names-dmp", str(names), "--top-frac", "0.2", "-o", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    res = json.loads((out / "enrichment.json").read_text())
    assert res["odds_ratio"] >= 1.0
    assert res["n_uncertain"] >= 2
