"""Task C2c — amino-acid composition-erasure confound control (STRONGER confound gate, spec §6.1 S1).

Composition (20-D AA fraction, frozen as columns aa_A..aa_Y) is the stronger taxonomy proxy than length.
On a SMALL synthetic panel where composition ⟂ the function/taxonomy signal, erasing the 20-D composition
must NOT collapse the headline: within-stratum function purity is preserved AND taxonomy stays recoverable
on the composition-erased reps. The fixture is small (n≈120, 1024-d) so the full stack runs fast
(pca_dim = min(PCA_DIM, n//3) ≈ 40). The control RAISES if the aa_A..aa_Y columns are absent (never a
silent skip — that would disable the S1 gate)."""
import numpy as np
import pandas as pd
import pytest
from taxembed.bridge import clean_eval

_AA = "ACDEFGHIKLMNPQRSTVWY"                                    # 20 canonical AAs, fixed order


def _composition_independent_panel(n=180, d=1024, seed=0, with_aa=True):
    """Panel where AA composition is a SEPARABLE nuisance: reps are the SUM of a function block (driven by
    func), a taxonomy block (driven by tax class), and — when aa is present — a small composition block
    that lives along its OWN random directions, ⟂ in expectation to the function/taxonomy blocks. So a
    20-D LEACE erasure of composition removes the composition block and leaves the function/taxonomy
    structure (hence the headline purity + taxonomy probe) intact: the headline SURVIVES.

    n=180 -> pca_dim = min(PCA_DIM=64, 60) = 60, so a 20-D composition erasure is ~1/3 of the basis (not
    half), giving the function subspace headroom to survive."""
    rng = np.random.default_rng(seed)
    n_fam, n_class = 6, 3
    fam = rng.integers(0, n_fam, n)
    cls = rng.integers(0, n_class, n)
    fam_centers = rng.standard_normal((n_fam, d))
    cls_centers = rng.standard_normal((n_class, d))
    reps = (1.6 * fam_centers[fam] + 1.4 * cls_centers[cls]
            + 0.5 * rng.standard_normal((n, d)))
    comp = rng.dirichlet(np.ones(20), size=n)                   # (n,20) AA fraction, ⟂ fam/cls (Dirichlet)
    if with_aa:
        # embed composition into reps along its OWN 20 random directions (separable confound block). A
        # 20-D LEACE then removes THIS block; the function/taxonomy blocks above are untouched in expectation.
        comp_dirs = rng.standard_normal((20, d))
        reps = reps + 1.2 * (comp @ comp_dirs)
    reps = reps.astype(np.float32)
    # LOW-RANK taxonomy tangent target so the tax-LEACE erases a PROPER subspace (full-rank trips the
    # coherence 0<k<pca_dim guard). The CATEGORICAL tax_class probe carries taxonomy recoverability.
    r_lat = 12
    positions = (rng.standard_normal((n, r_lat)) @ rng.standard_normal((r_lat, 100))
                 + 0.01 * rng.standard_normal((n, 100)))
    length = rng.integers(80, 400, n)
    ann = pd.DataFrame({
        "func": [f"PF{f}" for f in fam],
        "accession": [f"A{i}" for i in range(n)],
        "taxid": np.arange(n), "idx": np.arange(n),
        "cluster_id": [f"c{i}" for i in range(n)],
        "tax_class": [f"cl{c}" for c in cls],
        "tax_order": [f"o{c}_{i % 2}" for i, c in enumerate(cls)],
        "tax_phylum": ["p"] * n, "length": length,
    })
    if with_aa:
        for j, aa in enumerate(_AA):
            ann[f"aa_{aa}"] = comp[:, j]
    return ann, reps, positions


def test_composition_control_keys_and_survives():
    ann, reps, positions = _composition_independent_panel()
    pca_dim = min(clean_eval.config.PCA_DIM, len(ann) // 3)
    out = clean_eval.run_battery_sp(ann, reps, positions, pca_dim,
                                    stratum_col="tax_class", composition_control=True)
    assert "composition_control" in out
    cc = out["composition_control"]
    for key in ("purity_after_comp_erase", "tax_after_comp_erase", "comp_dim",
                "geometry_change", "headline_survives"):
        assert key in cc, key
    assert cc["comp_dim"] == 20
    assert isinstance(cc["headline_survives"], bool)
    # composition ⟂ signal => erasing composition does not collapse the headline
    assert cc["headline_survives"] is True

    # verdict folds the composition gate in (S1: survives BOTH confounds run -> here only comp ran)
    v = clean_eval._disentanglement_verdict_sp(out)
    assert "composition_control_headline_survives" in v
    assert v["composition_control_headline_survives"] is True
    assert v["confound_control_gates_pass"] is True             # S1 over the controls that ran


def test_composition_control_raises_when_aa_columns_absent():
    """If aa_A..aa_Y are absent the control RAISES (never silently skips — a silent skip disables S1)."""
    ann, reps, positions = _composition_independent_panel(seed=2, with_aa=False)
    pca_dim = min(clean_eval.config.PCA_DIM, len(ann) // 3)
    with pytest.raises((ValueError, KeyError)):
        clean_eval.run_battery_sp(ann, reps, positions, pca_dim,
                                  stratum_col="tax_class", composition_control=True)
