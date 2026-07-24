import pandas as pd
from taxembed.bridge import build_sp_panel as B

def test_apply_drops_only_flagged_and_sorts():
    panel = pd.DataFrame({"pfam_family":["PF3","PF1","PF2","PF1"],
                          "accession":["Z","B","Y","A"]})
    excl = pd.DataFrame({"pfam_family":["PF2"],"action":["drop"],"reason":["degenerate cluster"]})
    out = B.apply_exclusions(panel, excl)
    assert list(out.pfam_family) == ["PF1","PF1","PF3"]      # PF2 gone; sorted by (family,accession)
    assert list(out.accession) == ["A","B","Z"]
