"""Task D1 — read_eval `--testbed` dispatcher contract.

Asserts the argparse parser exposes the three READ testbeds and that select_loader routes
`pla2` -> the existing load_pla2 and the SP testbeds -> load_sp_read. Pure introspection: no
ckpt / h5 / network touched (we never CALL the loaders, only check their identity).
"""
from taxembed.bridge import read_eval


def test_testbed_choices_and_routing():
    p = read_eval.build_parser()
    choices = p._option_string_actions["--testbed"].choices
    assert {"pla2", "sp_metazoa_pfam", "sp_metazoa_ec"} <= set(choices)
    assert read_eval.select_loader("pla2").__name__.startswith("load_pla2")
    assert read_eval.select_loader("sp_metazoa_pfam").__name__ == "load_sp_read"


def test_load_sp_read_adds_clade_species(tmp_path, monkeypatch):
    """load_sp_read reuses load_sp_clean then aliases Clade=tax_class, Species=taxid and returns
    (ann, reps). Synthetic fixture (same C1 seam) — no real ckpt / h5 / network touched."""
    import numpy as np, pandas as pd, h5py
    from taxembed.bridge import read_eval
    # read_eval does a BARE `from clean_eval import load_sp_clean` (it primes sys.path), so the live
    # module is sys.modules["clean_eval"], NOT the dotted taxembed.bridge.clean_eval. Patch the
    # one read_eval actually resolves — the same three indirections the C1 fixture monkeypatches.
    from taxembed.bridge import clean_eval
    panel = pd.DataFrame({"accession": ["A", "B"], "pfam_family": ["PF1", "PF2"],
                          "ec": ["1.1.1", "2.2.2"], "taxid": [9606, 7227], "idx": [1, 2],
                          "tax_class": ["Mammalia", "Insecta"],
                          "tax_order": ["Primates", "Diptera"],
                          "tax_phylum": ["Chordata", "Arthropoda"],
                          "cluster_id": ["A", "B"], "length": [100, 120]})
    pp = tmp_path / "p.tsv"; panel.to_csv(pp, sep="\t", index=False)
    h5 = tmp_path / "e.h5"
    with h5py.File(h5, "w") as h:
        for a in ["A", "B"]:
            h.create_dataset(a, data=np.ones(1024, np.float16))

    class FakeTE:
        positions = np.random.rand(10, 100)

    monkeypatch.setattr(clean_eval, "_load_sp_panel_path", lambda tb: str(pp))
    monkeypatch.setattr(clean_eval, "_load_sp_h5_path", lambda tb=None: str(h5))
    monkeypatch.setattr(clean_eval, "_load_te", lambda target="metazoa": FakeTE())

    ann, reps = read_eval.load_sp_read("sp_metazoa_pfam")
    assert list(ann["Clade"]) == list(ann["tax_class"])      # Clade aliases tax_class
    assert list(ann["Species"]) == list(ann["taxid"])        # Species aliases taxid
    assert {"func", "taxid", "idx", "tax_class", "tax_order", "tax_phylum",
            "cluster_id"} <= set(ann.columns)
    assert reps.shape == (2, 1024)
