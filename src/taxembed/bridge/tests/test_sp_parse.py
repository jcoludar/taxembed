import pandas as pd
from taxembed.bridge import build_sp_panel as B

RAW = pd.DataFrame({
    "Entry": ["P1","P2","P3","P4"], "Organism (ID)": [9606,7227,9606,10090],
    "Organism": ["Homo sapiens","Drosophila melanogaster","Homo sapiens","Mus musculus"],
    "Length": [150,300,120,400],
    "Pfam": ["PF00042;","PF16746;PF00169;","","PF00042;"],
    "EC number": ["","3.4.24.-","","1.1.1.1;2.7.11.1"],
    "Protein names": ["Hemoglobin","X","Y","Hemoglobin"], "Keywords": ["","","",""]})

def test_parse_splits_pfam_and_ec_sets():
    df = B.parse_annotations(RAW)
    assert df.loc[df.accession=="P1","pfam_set"].iloc[0] == ("PF00042",)
    assert df.loc[df.accession=="P2","pfam_set"].iloc[0] == ("PF00169","PF16746")
    assert df.loc[df.accession=="P3","pfam_set"].iloc[0] == ()
    assert df.loc[df.accession=="P4","ec_set"].iloc[0] == ("1.1.1.1","2.7.11.1")
    assert df.loc[df.accession=="P1","taxid"].iloc[0] == 9606
