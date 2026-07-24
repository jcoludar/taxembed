from taxembed.bridge import build_sp_panel as B
LOW = {"tax_given_func": 0.10, "func_given_tax": 0.20}

def test_accept_passes_when_enough_and_crossed():
    out = B.panel_acceptance(20, LOW, min_families=15, nesting_max=0.30)
    assert out["accepted"] is True

def test_accept_fails_too_few_families():
    assert B.panel_acceptance(10, LOW, 15, 0.30)["accepted"] is False

def test_accept_fails_high_nesting():
    hi = {"tax_given_func": 0.50, "func_given_tax": 0.20}
    out = B.panel_acceptance(20, hi, 15, 0.30)
    assert out["accepted"] is False and out["nesting_ok"] is False
