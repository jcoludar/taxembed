import pandas as pd
from taxembed.bridge import build_sp_panel


class _Resolver:
    LINS = {
        9606: [("superkingdom", 2759, "Eukaryota"), ("class", 40674, "Mammalia"),
               ("order", 9443, "Primates"), ("phylum", 7711, "Chordata")],
        562:  [("superkingdom", 2, "Bacteria"), ("phylum", 1224, "Pseudomonadota")],  # no class/order
        2157: [],  # archaeon with no resolvable ranks at all -> all None
    }
    def lineage(self, t):
        return self.LINS.get(int(t), [])


def test_attach_ranks_adds_superkingdom():
    df = pd.DataFrame({"resolved_taxid": [9606, 562, 2157]})
    out = build_sp_panel.attach_ranks(df, _Resolver())
    assert list(out["tax_superkingdom"]) == ["Eukaryota", "Bacteria", None]
    # the prokaryote keeps phylum but loses class/order; the archaeon is all-None
    assert out.loc[1, "tax_class"] is None and out.loc[1, "tax_phylum"] == "Pseudomonadota"
    assert out.loc[2, "tax_superkingdom"] is None


def test_panel_cols_include_superkingdom():
    assert "tax_superkingdom" in build_sp_panel._PANEL_COLS
