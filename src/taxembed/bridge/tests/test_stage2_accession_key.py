import pandas as pd
from taxembed.bridge import build_sp_panel


def test_write_panel_fasta_bare_accession_headers(tmp_path):
    panel = pd.DataFrame({"accession": ["P12345", "Q9XYZ1"]})
    seqs = {"P12345": "MKV", "Q9XYZ1": "AAC"}
    out = tmp_path / "panel.fasta"
    build_sp_panel.write_panel_fasta(panel, seqs, str(out))
    lines = out.read_text().splitlines()
    assert lines[0] == ">P12345" and lines[1] == "MKV"
    assert lines[2] == ">Q9XYZ1" and lines[3] == "AAC"
    # headers must be the bare panel accession — no sp|...| or version suffix
    assert all(not h.startswith(">sp|") and "." not in h for h in lines[::2])


def test_all_life_aborts_on_accession_key_mismatch(tmp_path, monkeypatch):
    """The Task-11 all-life accession-key contract: even when join coverage clears the floor, an
    all_life_* run aborts (SystemExit) if ANY panel accession is absent from the self-embed h5 — that
    is a key-format mismatch (sp|..| / version suffix), not an acceptable drop. Floor relaxed so the
    EXACT-containment check (not the coverage guard) is what fires."""
    import numpy as np, pandas as pd, h5py, pytest
    from taxembed.bridge import clean_eval
    panel = pd.DataFrame({"accession": ["A", "B", "C"], "pfam_family": ["PF1", "PF1", "PF2"],
                          "ec": ["1.1.1", "1.1.1", None], "taxid": [9606, 7227, 10090], "idx": [1, 2, 3],
                          "resolved_taxid": [9606, 7227, 10090],
                          "tax_class": ["Mammalia", "Insecta", "Mammalia"],
                          "tax_order": ["Primates", "Diptera", "Rodentia"],
                          "tax_phylum": ["Chordata", "Arthropoda", "Chordata"],
                          "cluster_id": ["A", "B", "C"], "length": [100, 100, 100]})
    pp = tmp_path / "p.tsv"; panel.to_csv(pp, sep="\t", index=False)
    h5 = tmp_path / "e.h5"
    with h5py.File(h5, "w") as h:                  # C missing -> exact containment violated
        for a in ["A", "B"]: h.create_dataset(a, data=np.ones(1024, np.float16))

    class FakeTE:
        positions = np.random.rand(10, 100)
    monkeypatch.setattr(clean_eval, "_load_sp_panel_path", lambda tb: str(pp))
    monkeypatch.setattr(clean_eval, "_load_sp_h5_path", lambda tb=None: str(h5))
    monkeypatch.setattr(clean_eval, "_load_te", lambda target="metazoa": FakeTE())
    monkeypatch.setattr(clean_eval.config, "SP_JOIN_COVERAGE_MIN", 0.5)   # so containment, not coverage, fires
    with pytest.raises(SystemExit, match="accession-key mismatch"):
        clean_eval.load_sp_clean("all_life_pfam")
