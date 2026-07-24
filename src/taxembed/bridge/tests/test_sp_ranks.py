import pandas as pd
from taxembed.bridge import build_sp_panel as B

class FakeResolver:
    # resolved_taxid -> lineage as [(rank, taxid, name), ...]
    _lin = {
        9606: [("species", 9606, "Homo sapiens"), ("order", 9443, "Primates"),
               ("class", 40674, "Mammalia"), ("phylum", 7711, "Chordata")],
        7227: [("species", 7227, "Drosophila melanogaster"), ("order", 7147, "Diptera"),
               ("phylum", 6656, "Arthropoda")],   # no class node
    }
    def lineage(self, t): return self._lin.get(int(t), [])

def test_attach_ranks_names_and_nulls():
    df = pd.DataFrame({"accession": ["A", "B"], "resolved_taxid": [9606, 7227]})
    out = B.attach_ranks(df, FakeResolver())
    assert list(out.tax_class) == ["Mammalia", None]
    assert list(out.tax_order) == ["Primates", "Diptera"]
    assert list(out.tax_phylum) == ["Chordata", "Arthropoda"]


class _DomainResolver:
    """The real 2026 NCBI taxdump labels the top rank `domain` (NCBI renamed `superkingdom`→`domain`
    in 2024), e.g. ('domain', 2, 'Bacteria'). attach_ranks must read it into tax_superkingdom."""
    _lin = {
        562: [("species", 562, "Escherichia coli"), ("order", 91347, "Enterobacterales"),
              ("class", 1236, "Gammaproteobacteria"), ("phylum", 1224, "Pseudomonadota"),
              ("domain", 2, "Bacteria")],
        2157: [("order", 2158, "Methanobacteriales"), ("domain", 2157, "Archaea")],
    }
    def lineage(self, t): return self._lin.get(int(t), [])


def test_attach_ranks_reads_domain_into_superkingdom():
    df = pd.DataFrame({"accession": ["A", "B"], "resolved_taxid": [562, 2157]})
    out = B.attach_ranks(df, _DomainResolver())
    assert list(out.tax_superkingdom) == ["Bacteria", "Archaea"]


class _SuperkingdomResolver:
    """Older dumps use `superkingdom` — must still work (back-compat fallback)."""
    _lin = {562: [("superkingdom", 2, "Bacteria")]}
    def lineage(self, t): return self._lin.get(int(t), [])


def test_attach_ranks_superkingdom_fallback():
    df = pd.DataFrame({"accession": ["A"], "resolved_taxid": [562]})
    out = B.attach_ranks(df, _SuperkingdomResolver())
    assert list(out.tax_superkingdom) == ["Bacteria"]
