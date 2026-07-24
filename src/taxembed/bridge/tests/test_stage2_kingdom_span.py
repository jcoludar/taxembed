import pandas as pd
from taxembed.bridge import build_sp_panel


def _members(fam, orders, kingdoms):
    return pd.DataFrame({"pfam_family": [fam] * len(orders), "tax_order": orders,
                         "tax_superkingdom": kingdoms})


def _kept(out, fam):
    # gate_scan returns ONE row per family with a `keep` flag — it does NOT filter.
    row = out[out["pfam_family"] == fam]
    return bool(row["keep"].iloc[0]) if len(row) else False


def test_kingdom_span_gate_rejects_single_kingdom_family():
    # 6 distinct orders but ALL Eukaryota -> passes order gate, FAILS kingdom-span (>=2 of 3)
    df = _members("PF1", [f"o{i}" for i in range(6)], ["Eukaryota"] * 6)
    out = build_sp_panel.gate_scan(df, member_floor=5, min_orders=4, min_eff_orders=3,
                                   span_superkingdoms=True, min_superkingdoms=2)
    assert _kept(out, "PF1") is False
    assert out.loc[out["pfam_family"] == "PF1", "n_superkingdoms"].iloc[0] == 1


def test_kingdom_span_gate_admits_cross_kingdom_family():
    df = _members("PF2", [f"o{i}" for i in range(6)],
                  ["Bacteria", "Bacteria", "Archaea", "Eukaryota", "Eukaryota", "Archaea"])
    out = build_sp_panel.gate_scan(df, member_floor=5, min_orders=4, min_eff_orders=3,
                                   span_superkingdoms=True, min_superkingdoms=2)
    assert _kept(out, "PF2") is True


def test_span_off_preserves_metazoa_behavior():
    df = _members("PF3", [f"o{i}" for i in range(6)], ["Eukaryota"] * 6)
    out = build_sp_panel.gate_scan(df, member_floor=5, min_orders=4, min_eff_orders=3)
    assert _kept(out, "PF3") is True            # default span_superkingdoms=False == Stage-1
