# tests/test_stage2_invariant.py
import types
import numpy as np, pandas as pd, pytest
from taxembed.bridge import clean_eval


def _fake_te(n_nodes=10, dim=4, taxids=None):
    te = types.SimpleNamespace()
    te.n_nodes = n_nodes
    te.dim = dim
    te.positions = np.zeros((n_nodes, dim), dtype=np.float64)
    te.idx2taxid = np.array(taxids if taxids is not None else list(range(1000, 1000 + n_nodes)),
                            dtype=np.int64)
    return te


def _ann(idx, resolved_taxid):
    return pd.DataFrame({"idx": idx, "resolved_taxid": resolved_taxid})


def test_invariant_passes_on_matching_roundtrip():
    te = _fake_te()
    ann = _ann([2, 5], [1002, 1005])           # idx2taxid[2]=1002, [5]=1005
    stamp = {"target_name": "cellular", "n_nodes": 10, "ckpt_sha256": "abc"}
    clean_eval._assert_target_invariant(te, ann, "cellular", {**stamp, "te_ckpt_sha256": "abc"})


def test_invariant_raises_on_idx_taxid_mismatch():
    te = _fake_te()
    ann = _ann([2, 5], [1002, 9999])           # 9999 != idx2taxid[5]=1005  (wrong-taxon, in range)
    stamp = {"target_name": "cellular", "n_nodes": 10, "te_ckpt_sha256": "abc"}
    with pytest.raises(SystemExit):
        clean_eval._assert_target_invariant(te, ann, "cellular", stamp)


def test_invariant_raises_on_n_nodes_mismatch():
    te = _fake_te(n_nodes=10)
    ann = _ann([2], [1002])
    stamp = {"target_name": "cellular", "n_nodes": 999, "te_ckpt_sha256": "abc"}  # different embedding
    with pytest.raises(SystemExit):
        clean_eval._assert_target_invariant(te, ann, "cellular", stamp)


def test_invariant_raises_on_target_name_mismatch():
    te = _fake_te()
    ann = _ann([2], [1002])
    stamp = {"target_name": "metazoa", "n_nodes": 10, "te_ckpt_sha256": "abc"}   # panel built under metazoa
    with pytest.raises(SystemExit):
        clean_eval._assert_target_invariant(te, ann, "cellular", stamp)
