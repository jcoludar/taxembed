import numpy as np, h5py
from taxembed.bridge import build_sp_panel


def test_subset_h5_empty_is_1024_wide(tmp_path):
    p = tmp_path / "e.h5"
    with h5py.File(p, "w") as h:
        h.create_dataset("A0", data=np.zeros(1024, np.float16))
    reps, present, missing = build_sp_panel.subset_h5(str(p), ["NOPE1", "NOPE2"])
    assert reps.shape == (0, 1024) and present == [] and missing == ["NOPE1", "NOPE2"]


def test_subset_h5_present_preserves_order(tmp_path):
    p = tmp_path / "f.h5"
    with h5py.File(p, "w") as h:
        for a in ("A0", "A1"):
            h.create_dataset(a, data=np.arange(1024, dtype=np.float16))
    reps, present, missing = build_sp_panel.subset_h5(str(p), ["A1", "X", "A0"])
    assert present == ["A1", "A0"] and missing == ["X"] and reps.shape == (2, 1024)


def test_load_sp_clean_raises_on_low_join_coverage(tmp_path, monkeypatch):
    """The Task-10 runtime guard: load_sp_clean REFUSES to run the battery on a biased subset when h5
    join coverage drops below config.SP_JOIN_COVERAGE_MIN (the default 0.98 floor). 1-of-3 missing = 0.67."""
    import pandas as pd, pytest
    from taxembed.bridge import clean_eval
    panel = pd.DataFrame({"accession": ["A", "B", "C"], "pfam_family": ["PF1", "PF1", "PF2"],
                          "ec": ["1.1.1", "1.1.1", None], "taxid": [9606, 7227, 10090], "idx": [1, 2, 3],
                          "tax_class": ["Mammalia", "Insecta", "Mammalia"],
                          "tax_order": ["Primates", "Diptera", "Rodentia"],
                          "tax_phylum": ["Chordata", "Arthropoda", "Chordata"],
                          "cluster_id": ["A", "B", "C"], "length": [100, 100, 100]})
    pp = tmp_path / "p.tsv"; panel.to_csv(pp, sep="\t", index=False)
    h5 = tmp_path / "e.h5"
    with h5py.File(h5, "w") as h:                  # only A,B present -> C missing -> 0.67 < 0.98
        for a in ["A", "B"]: h.create_dataset(a, data=np.ones(1024, np.float16))

    class FakeTE:
        positions = np.random.rand(10, 100)
    monkeypatch.setattr(clean_eval, "_load_sp_panel_path", lambda tb: str(pp))
    monkeypatch.setattr(clean_eval, "_load_sp_h5_path", lambda tb=None: str(h5))
    monkeypatch.setattr(clean_eval, "_load_te", lambda target="metazoa": FakeTE())
    with pytest.raises(SystemExit, match="join coverage"):
        clean_eval.load_sp_clean("sp_metazoa_pfam")
