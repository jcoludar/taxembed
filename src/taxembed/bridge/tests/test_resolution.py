import csv
from pathlib import Path
import pytest
TABLE = Path(__file__).resolve().parents[1] / "data" / "pla2_species_resolution.tsv"
MF_TABLE = Path(__file__).resolve().parents[1] / "data" / "multifamily_species_resolution.tsv"

@pytest.mark.skipif(not TABLE.exists(), reason="run resolve_species.py first")
def test_all_85_resolve_and_present():
    rows = list(csv.DictReader(TABLE.open(), delimiter="\t"))
    assert len(rows) == 85
    assert all(r["taxid"] and r["taxid"] != "NA" for r in rows)      # every species resolved
    assert all(r["idx"] and r["idx"] != "NA" for r in rows)          # every taxid in the metazoa embedding


@pytest.mark.skipif(not MF_TABLE.exists(), reason="run resolve_multifamily.py first")
def test_multifamily_84_rows_and_columns():
    rows = list(csv.DictReader(MF_TABLE.open(), delimiter="\t"))
    assert len(rows) == 84                                           # 84 ToxFam accessions
    assert set(rows[0].keys()) == {"identifier", "family", "taxid", "idx", "organism", "via"}
    # every row carries a via verdict; resolved rows have a taxid + idx in the embedding
    assert all(r["via"] in {"direct", "unresolved_taxid", "not_in_embedding"} for r in rows)
    direct = [r for r in rows if r["via"] == "direct"]
    assert all(r["taxid"] != "NA" and r["idx"] != "NA" for r in direct)
    assert len({r["family"] for r in direct}) >= 10                  # multiple gene families represented
