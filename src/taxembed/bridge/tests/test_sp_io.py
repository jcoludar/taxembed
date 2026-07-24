import numpy as np, h5py, pytest
from taxembed.bridge import build_sp_panel as B


def test_collapse_ec_to_third_level():
    assert B.collapse_ec("3.4.24.-", 3) == "3.4.24"
    assert B.collapse_ec("3.4.24.1", 3) == "3.4.24"


def test_collapse_ec_too_shallow_is_none():
    assert B.collapse_ec("2.7", 3) is None
    assert B.collapse_ec("3.4.-.-", 3) is None


def test_subset_h5_casts_float16_and_partitions(tmp_path):
    h5_path = tmp_path / "emb.h5"
    with h5py.File(h5_path, "w") as h:
        h.create_dataset("A0", data=np.ones(8, dtype=np.float16))
        h.create_dataset("A1", data=(np.ones(8, dtype=np.float16) * 2))
    reps, present, missing = B.subset_h5(str(h5_path), ["A0", "MISSING", "A1"])
    assert present == ["A0", "A1"]
    assert missing == ["MISSING"]
    assert reps.dtype == np.float32             # float16 -> float32 cast
    assert reps.shape == (2, 8)
    assert np.allclose(reps[0], 1.0) and np.allclose(reps[1], 2.0)


def test_subset_h5_all_missing_returns_empty(tmp_path):
    h5_path = tmp_path / "emb.h5"
    with h5py.File(h5_path, "w") as h:
        h.create_dataset("A0", data=np.ones(8, dtype=np.float16))
    reps, present, missing = B.subset_h5(str(h5_path), ["X", "Y"])
    assert present == [] and missing == ["X", "Y"]
    assert reps.shape == (0, 1024)   # ProtT5 width fallback on the empty set (Task 10; (0,0) breaks PCA)


def test_gunzip_all_peels_single_and_double_layer():
    import gzip
    payload = b"Entry\tOrganism (ID)\nP12345\t9606\n"
    one = gzip.compress(payload)
    two = gzip.compress(one)                          # double-wrap (compressed=true + transfer gzip)
    assert B._gunzip_all(one) == payload              # single layer
    assert B._gunzip_all(two) == payload              # double layer (the real-server failure mode)
    assert B._gunzip_all(payload) == payload          # already-plain text is a no-op


def test_freeze_cluster_ids_raises_when_mmseqs_absent(tmp_path, monkeypatch):
    # build_sp_panel import primed tools/ onto sys.path, so the REAL module is `embeddings.cluster_ids`
    # (the test tree shadows only `tools.embeddings`, never the bare `embeddings` package).
    from taxembed.bridge import clusters as CI
    monkeypatch.setattr(CI, "have_mmseqs", lambda: False)
    with pytest.raises(RuntimeError):
        B.freeze_cluster_ids(["A", "B"], ["MKV", "MKW"],
                             str(tmp_path / "clust.tsv"), str(tmp_path / "manifest.json"))
