import numpy as np
from taxembed.bridge import eval_utils


def test_label_nesting_perfectly_nested():
    func = np.array(["A", "A", "B", "B"]); tax = np.array(["x", "x", "y", "y"])
    out = eval_utils.label_nesting(func, tax)              # public name; alias keeps _label_nesting
    assert out["uncertainty_reduction"] > 0.99            # 1 - H(tax|func)/H(tax); nested -> 1
    assert out["n_function_labels_mapping_to_one_tax"] == 2


def test_label_nesting_crossed():
    func = np.array(["A", "A", "B", "B"]); tax = np.array(["x", "y", "x", "y"])
    out = eval_utils.label_nesting(func, tax)
    assert out["uncertainty_reduction"] < 0.05
    assert out["n_function_labels_mapping_to_one_tax"] == 0


def test_rank_lookup_real_contract():
    class FakeResolver:
        def lineage(self, t):
            return [("species", 9606, "Homo sapiens"), ("order", 9443, "Primates")]
    rl = eval_utils.RankLookup(FakeResolver(), ranks=["order", "class"])
    rm = rl.rank_map(9606)
    assert rm["order"] == 9443
    assert "class" not in rm                                # missing rank absent
