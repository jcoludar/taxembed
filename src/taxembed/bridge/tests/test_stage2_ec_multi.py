import pandas as pd
from taxembed.bridge import build_sp_panel


def test_multi_ec_dropped_and_reported_per_superkingdom():
    frame = pd.DataFrame({
        "ec_set": [("1.1.1.1",), ("2.7.1.1", "3.1.3.1"), ("1.1.1.5",), ("4.1.1.1", "5.3.1.9")],
        "tax_superkingdom": ["Bacteria", "Bacteria", "Eukaryota", "Eukaryota"],
    })
    out, report = build_sp_panel._ec_relabel(frame)
    # rows 1 and 3 are multi-EC3 -> dropped; rows 0 and 2 survive
    assert len(out) == 2
    assert set(out["ec"]) == {"1.1.1"}             # EC3 collapse of the single-EC survivors
    assert report["Bacteria"] == 1 and report["Eukaryota"] == 1
    assert report["_total"] == 2


def test_same_ec3_multiple_ecs_is_not_multi():
    # two EC numbers that collapse to the SAME EC3 -> 1 distinct EC3 -> kept (not dropped)
    frame = pd.DataFrame({"ec_set": [("2.7.11.1", "2.7.11.2")], "tax_superkingdom": ["Eukaryota"]})
    out, report = build_sp_panel._ec_relabel(frame)
    assert len(out) == 1 and list(out["ec"]) == ["2.7.11"] and report["_total"] == 0
