import numpy as np, pandas as pd
from taxembed.bridge import clean_eval


def test_run_battery_sp_returns_real_keys():
    rng = np.random.default_rng(0); n = 600
    fam = rng.integers(0, 20, n); order = rng.integers(0, 8, n)
    reps = rng.standard_normal((n, 1024)).astype(np.float32)
    positions = rng.standard_normal((n, 100))
    ann = pd.DataFrame({"func": [f"PF{f}" for f in fam], "accession": [f"A{i}" for i in range(n)],
        "taxid": np.arange(n), "idx": np.arange(n), "cluster_id": [f"c{i}" for i in range(n)],
        "tax_class": [f"cl{o % 3}" for o in order], "tax_order": [f"o{o}" for o in order],
        "tax_phylum": ["p"] * n, "length": rng.integers(80, 400, n)})
    out = clean_eval.run_battery_sp(ann, reps, positions, pca_dim=64, stratum_col="tax_class")
    for key in ["leace", "step2_erasure_worked", "step3_function_preserved", "step4_specificity",
                "coherence", "label_nesting", "disentanglement_matrix", "stratum_grain", "headline_stratum"]:
        assert key in out
    assert "energy_match_ok_half_to_2x" in out["step4_specificity"]
    assert "pass" in out["step4_specificity"]                       # named §6.1 specificity gate
    assert out["headline_stratum"] in ("tax_class", "tax_order")  # finest-feasible SP rank (order finer than class)
    assert set(out["label_nesting"]) == {"tax_given_func", "func_given_tax"}  # bidirectional
