import pandas as pd
from taxembed.bridge import build_sp_panel as B

def test_coverage_filter_drops_low_coverage():
    df = pd.DataFrame({"accession":["P0","P1"],"pfam_family":["PF1","PF1"],
                       "length":[100,100],"domain_aa":[80,30]})
    out = B.filter_coverage(df, min_cov=0.5)
    assert list(out.accession)==["P0"] and abs(out.coverage.iloc[0]-0.8) < 1e-9

def test_coverage_nan_is_passthrough():
    df = pd.DataFrame({"accession":["P0","P1"],"pfam_family":["PF1","PF1"],
                       "length":[100,100],"domain_aa":[float("nan"),float("nan")]})
    out = B.filter_coverage(df, min_cov=0.5)
    assert list(out.accession)==["P0","P1"] and out["coverage"].isna().all()
