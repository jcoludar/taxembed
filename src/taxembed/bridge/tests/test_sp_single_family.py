def test_single_family_keeps_exactly_one_distinct():
    import pandas as pd; from taxembed.bridge import build_sp_panel as B
    df = pd.DataFrame({"accession":["P0","P1","P2","P3"],
                       "pfam_set":[("PF1",),("PF1","PF2"),(),("PF9",)]})
    out = B.filter_single_family(df)
    assert list(out.accession)==["P0","P3"] and list(out.pfam_family)==["PF1","PF9"]
