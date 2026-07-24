"""Task C2b — length-erasure confound control (the ENFORCED confound gate, spec §6.2).

On a SMALL synthetic panel where sequence LENGTH ⟂ the function/taxonomy signal, erasing length must
NOT collapse the headline: within-stratum function purity is preserved AND taxonomy stays recoverable on
the length-erased reps. The fixture is small (n≈120, 1024-d) so the full LEACE/probe stack runs in well
under a minute (pca_dim = min(PCA_DIM, n//3) ≈ 40)."""
import numpy as np
import pandas as pd
from taxembed.bridge import clean_eval


def _length_independent_panel(n=120, d=1024, seed=0):
    """Build a panel where length is an INDEPENDENT nuisance: reps carry a function block (driven by func)
    and a taxonomy block (driven by tax position), and `length` is drawn independently of both. So a
    LEACE erasure of length should barely move the function purity or the taxonomy probe."""
    rng = np.random.default_rng(seed)
    n_fam, n_class = 6, 3
    fam = rng.integers(0, n_fam, n)
    cls = rng.integers(0, n_class, n)
    # function-structured + taxonomy-structured signal blocks; length drawn INDEPENDENTLY.
    fam_centers = rng.standard_normal((n_fam, d))
    cls_centers = rng.standard_normal((n_class, d))
    reps = (1.6 * fam_centers[fam] + 1.4 * cls_centers[cls]
            + 0.5 * rng.standard_normal((n, d))).astype(np.float32)
    # positions: a LOW-RANK continuous taxonomy tangent target (rank ~r_lat << pca_dim) so the tax-LEACE
    # erases a PROPER subspace, not the full PCA basis (a full-rank erasure trips the coherence 0<k<pca_dim
    # guard). Built from an r_lat-dim latent projected to 100-d. Independent of length. The CATEGORICAL
    # tax_class probe (driven by the cls structure in reps) is what carries taxonomy recoverability here.
    r_lat = 12
    latent = rng.standard_normal((n, r_lat))
    proj = rng.standard_normal((r_lat, 100))
    positions = latent @ proj + 0.01 * rng.standard_normal((n, 100))
    length = rng.integers(80, 400, n)                          # ⟂ fam, cls, reps
    ann = pd.DataFrame({
        "func": [f"PF{f}" for f in fam],
        "accession": [f"A{i}" for i in range(n)],
        "taxid": np.arange(n), "idx": np.arange(n),
        "cluster_id": [f"c{i}" for i in range(n)],
        "tax_class": [f"cl{c}" for c in cls],
        "tax_order": [f"o{c}_{i % 2}" for i, c in enumerate(cls)],
        "tax_phylum": ["p"] * n, "length": length,
    })
    return ann, reps, positions


def test_length_control_keys_and_survives():
    ann, reps, positions = _length_independent_panel()
    pca_dim = min(clean_eval.config.PCA_DIM, len(ann) // 3)
    out = clean_eval.run_battery_sp(ann, reps, positions, pca_dim,
                                    stratum_col="tax_class", length_control=True)
    assert "length_control" in out
    lc = out["length_control"]
    for key in ("purity_after_length_erase", "tax_after_length_erase", "headline_survives"):
        assert key in lc, key
    assert isinstance(lc["headline_survives"], bool)
    # length ⟂ signal => erasing length does not collapse the headline
    assert lc["headline_survives"] is True

    # verdict folds the length gate in
    v = clean_eval._disentanglement_verdict_sp(out)
    assert "length_control_headline_survives" in v
    assert v["length_control_headline_survives"] is True


def test_length_control_default_off():
    """Default behaviour unchanged: without length_control=True there is no length_control block."""
    ann, reps, positions = _length_independent_panel(seed=1)
    pca_dim = min(clean_eval.config.PCA_DIM, len(ann) // 3)
    out = clean_eval.run_battery_sp(ann, reps, positions, pca_dim, stratum_col="tax_class")
    assert "length_control" not in out
