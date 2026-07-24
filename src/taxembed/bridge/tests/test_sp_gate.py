import numpy as np, pandas as pd
from taxembed.bridge import build_sp_panel as B

def test_entropy_balanced_and_degenerate():
    assert abs(B._entropy([1,1,1,1]) - np.log(4)) < 1e-9     # ln 4 nats
    assert B._entropy([5]) == 0.0
    assert B._entropy([]) == 0.0

def test_effective_orders_balanced_and_skewed():
    assert abs(B.effective_orders(["a","b","c","d"]) - 4.0) < 1e-9
    skew = ["a"]*9 + ["b"]                                    # exp(H[0.9,0.1]) = 1.3842
    assert abs(B.effective_orders(skew) - 1.3842) < 1e-2
    assert B.effective_orders([]) == 0.0

def test_cond_reduction_nested_and_crossed():
    func = ["A","A","B","B"]
    assert abs(B._cond_reduction(func, ["x","x","y","y"]) - 1.0) < 1e-9   # nested -> 1
    assert abs(B._cond_reduction(func, ["x","y","x","y"]) - 0.0) < 1e-9   # crossed -> 0

def test_panel_nesting_bidirectional():
    func = ["A","A","B","B"]
    biject = B.panel_nesting(func, ["x","x","y","y"])         # bijection: both ~1
    assert biject["tax_given_func"] > 0.99 and biject["func_given_tax"] > 0.99
    crossed = B.panel_nesting(func, ["x","y","x","y"])        # both ~0
    assert crossed["tax_given_func"] < 0.01 and crossed["func_given_tax"] < 0.01

def test_gate_scan_keeps_broad_drops_narrow():
    df = pd.DataFrame({
        "pfam_family": ["PF1"]*4 + ["PF2"]*2,
        "tax_order":   ["o1","o2","o3","o4","o1","o1"]})
    out = B.gate_scan(df, member_floor=3, min_orders=2, min_eff_orders=2)
    keep = dict(zip(out.pfam_family, out.keep))
    assert keep["PF1"] is True and keep["PF2"] is False      # PF1: 4 members/4 orders; PF2: 2/1
    assert list(out.pfam_family) == ["PF1","PF2"]            # sorted, deterministic
