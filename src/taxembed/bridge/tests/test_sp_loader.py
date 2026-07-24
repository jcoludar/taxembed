"""Task C1 — load_sp_clean() loader contract (row-aligned to present accessions).

Synthetic fixture only (no real ckpt / h5 / download): the three indirections
(_load_sp_panel_path, _load_sp_h5_path, _load_te) are monkeypatched so the loader
reads a tmp_path panel + a tmp_path h5 and never touches production artifacts.
"""


def test_load_sp_clean_drops_missing_h5_rows(tmp_path, monkeypatch):
    import numpy as np, pandas as pd, h5py
    from taxembed.bridge import clean_eval
    panel = pd.DataFrame({"accession": ["A", "B", "C"], "pfam_family": ["PF1", "PF1", "PF2"],
                          "ec": ["1.1.1", "1.1.1", None], "taxid": [9606, 7227, 10090], "idx": [1, 2, 3],
                          "tax_class": ["Mammalia", "Insecta", "Mammalia"],
                          "tax_order": ["Primates", "Diptera", "Rodentia"],
                          "tax_phylum": ["Chordata", "Arthropoda", "Chordata"],
                          "cluster_id": ["A", "B", "C"], "length": [100, 100, 100]})
    pp = tmp_path / "p.tsv"; panel.to_csv(pp, sep="\t", index=False)
    h5 = tmp_path / "e.h5"
    with h5py.File(h5, "w") as h:                  # C is MISSING from the h5
        for a in ["A", "B"]: h.create_dataset(a, data=np.ones(1024, np.float16))
    class FakeTE: positions = np.random.rand(10, 100)
    monkeypatch.setattr(clean_eval, "_load_sp_panel_path", lambda tb: str(pp))
    monkeypatch.setattr(clean_eval, "_load_sp_h5_path", lambda tb=None: str(h5))
    monkeypatch.setattr(clean_eval, "_load_te", lambda target="metazoa": FakeTE())
    # this fixture tests DROP-AND-ALIGN mechanics on a 1-of-3-missing toy set (67% coverage), which the
    # Task-10 join-coverage guard would refuse at the production 0.98 floor — relax it here so the test
    # exercises the alignment path it is about. The guard's own refusal is covered in test_stage2_join_guard.
    monkeypatch.setattr(clean_eval.config, "SP_JOIN_COVERAGE_MIN", 0.5)   # clean_eval does `import config`
    ann, reps, positions, pca_dim = clean_eval.load_sp_clean("sp_metazoa_pfam")
    assert list(ann.accession) == ["A", "B"]         # C dropped, ann/reps/positions aligned
    assert reps.shape == (2, 1024) and positions.shape == (2, 100)
