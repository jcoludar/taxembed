"""B11 pure-helper unit tests: aa-composition, generalized exclusions-apply, EC relabel."""
import numpy as np
import pandas as pd
from taxembed.bridge import build_sp_panel as B


def test_aa_composition_renormalizes_over_standard_only():
    # 4 standard residues + 1 non-standard (X) -> denominator is 4, X ignored.
    comp = B._aa_composition("AACGX")
    assert set(comp) == {f"aa_{c}" for c in B.STD_AA}     # all 20 keys present
    assert comp["aa_A"] == 0.5 and comp["aa_C"] == 0.25 and comp["aa_G"] == 0.25
    assert abs(sum(comp.values()) - 1.0) < 1e-9            # sums to 1 over standard residues
    assert comp["aa_X"] if "aa_X" in comp else True        # (no aa_X column — X is non-standard)


def test_aa_composition_empty_is_all_zero():
    comp = B._aa_composition("")
    assert all(v == 0.0 for v in comp.values())
    comp2 = B._aa_composition("XUBZ")                      # all non-standard -> all zero, no div0
    assert all(v == 0.0 for v in comp2.values())


def test_apply_exclusions_on_drops_by_funccol_and_sorts_by_accession():
    panel = pd.DataFrame({"ec": ["E3", "E1", "E2", "E1"], "accession": ["Z", "B", "Y", "A"]})
    excl = pd.DataFrame({"ec": ["E2"], "action": ["drop"], "reason": ["degenerate"]})
    out = B._apply_exclusions_on(panel, excl, "ec")
    assert list(out.accession) == ["A", "B", "Z"]          # sorted by accession; E2 ('Y') gone
    assert "E2" not in set(out.ec)


def test_apply_exclusions_on_empty_keeps_all():
    panel = pd.DataFrame({"pfam_family": ["PF1", "PF2"], "accession": ["B", "A"]})
    out = B._apply_exclusions_on(panel, None, "pfam_family")
    assert list(out.accession) == ["A", "B"] and set(out.pfam_family) == {"PF1", "PF2"}


def test_ec_relabel_collapses_to_ec3_and_drops_nulls():
    frame = pd.DataFrame({
        "accession": ["A", "B", "C", "D"],
        "ec_set": [("3.4.24.1",), ("2.7.11.1", "2.7.11.2"), (), ("1.1",)],  # D too shallow -> dropped
    })
    out, report = B._ec_relabel(frame)
    assert list(out.accession) == ["A", "B"]               # C (no EC) + D (shallow) dropped
    assert list(out.ec) == ["3.4.24", "2.7.11"]            # B's 2 ECs share EC3 2.7.11 -> 1 distinct -> kept
    assert report["_total"] == 0                            # no row has >1 DISTINCT EC3
