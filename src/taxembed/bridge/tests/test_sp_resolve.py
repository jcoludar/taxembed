import pandas as pd
from taxembed.bridge import build_sp_panel as B
class FakeEmb:
    _idx={9606:11,10090:22}
    def idx_of_taxid(self,t): return self._idx.get(int(t))
class FakeResolver:
    _canon={99999:10090}; _parent={123456:9606}
    def canonical(self,t): return self._canon.get(int(t),int(t))
    def species_parent(self,t): return self._parent.get(int(t))
def test_resolution_direct_merged_subspecies_and_drop():
    df=pd.DataFrame({"accession":["A","B","C","D"],"taxid":[9606,99999,123456,7777]})
    out,dropped=B.resolve_taxids(df,FakeEmb(),FakeResolver())
    assert dict(zip(out.accession,out["via"]))=={"A":"direct","B":"merged","C":"subspecies_parent"}
    assert dict(zip(out.accession,out["idx"]))=={"A":11,"B":22,"C":11}
    assert list(dropped.accession)==["D"]
