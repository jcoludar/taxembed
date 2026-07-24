"""Task E1 — NMI(Pfam, EC) redundancy gate (spec §6.4).

`nmi_pfam_ec` is the TESTED deliverable: it must flag a bijective Pfam<->EC mapping
(NMI ~ 1) and stay ~0 on an independent mapping. `build_ec_panel` is a real-run
composition (untestable here — needs the locked metazoa embedding + the SwissProt h5).
"""
from taxembed.bridge import build_sp_panel as B


def test_nmi_bijective_mapping_is_high():
    # Pfam family <-> EC class are in bijection: knowing one fully determines the other.
    pfam = ["PF1", "PF1", "PF2", "PF2", "PF3", "PF3"]
    ec = ["3.4.24", "3.4.24", "2.7.11", "2.7.11", "1.1.1", "1.1.1"]
    assert B.nmi_pfam_ec(pfam, ec) > 0.9


def test_nmi_independent_mapping_is_near_zero():
    # Pfam and EC vary independently: each EC class appears under every Pfam family equally.
    pfam = ["PF1", "PF1", "PF2", "PF2", "PF3", "PF3"]
    ec = ["3.4.24", "2.7.11", "3.4.24", "2.7.11", "3.4.24", "2.7.11"]
    assert B.nmi_pfam_ec(pfam, ec) < 0.1
