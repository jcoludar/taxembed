"""Tasks D2a + D2b — SP-Metazoa READ harness + family-composition confound controls.

Small synthetic fixture (n=400, no real ckpt / h5 / network). `run_sp_read` takes (ann, reps, te)
DIRECTLY — the fixture supplies a FakeTE whose only attribute is `.positions` (an (n,100) array the
harness indexes by ann['idx']). The PLA2 READ path is untouched; these tests exercise only the SP clone.
"""
import numpy as np
import pandas as pd

from taxembed.bridge import read_eval


def _panel(n=400, seed=0):
    rng = np.random.default_rng(seed)
    order = rng.integers(0, 6, n)
    ann = pd.DataFrame({"func": [f"PF{rng.integers(0, 15)}" for _ in range(n)],
                        "taxid": np.arange(n) % 20, "idx": np.arange(n),  # 20 species -> 20 LOSO folds (fast)
                        "cluster_id": [f"c{i}" for i in range(n)],
                        "tax_class": [f"cl{o % 4}" for o in order],
                        "tax_order": [f"o{o}" for o in order],
                        "tax_phylum": ["p"] * n})
    ann["Clade"] = ann["tax_class"]
    ann["Species"] = ann["taxid"]
    reps = rng.standard_normal((n, 1024)).astype(np.float32)
    return ann, reps


class FakeTE:
    positions = np.random.default_rng(1).standard_normal((400, 100))


# --------------------------------------------------------------------------- D2a
def test_run_sp_read_keys():
    ann, reps = _panel()
    out = read_eval.run_sp_read(ann, reps, FakeTE())
    assert "class_rank_acc" in out and "leave_clade_out" in out
    assert "per_fold_family_overlap" in out["leave_clade_out"]
    assert abs(sum(out["leave_clade_out"]["fold_weights"]) - 1.0) < 1e-6


# --------------------------------------------------------------------------- D2b
def test_run_sp_read_confound_controls():
    ann, reps = _panel()
    out = read_eval.run_sp_read(ann, reps, FakeTE())
    lco = out["leave_clade_out"]
    # (a) per-fold family-overlap is a list of Jaccards (0..1)
    overlaps = lco["per_fold_family_overlap"]
    assert isinstance(overlaps, list)
    assert all(0.0 <= j <= 1.0 for j in overlaps)
    # (b) family-balanced variant exists with an explicit informative bool (S3 power floor)
    assert "family_balanced" in lco
    assert isinstance(lco["family_balanced"]["informative"], bool)
    # (c) leave-species-AND-clade leak diagnostic exists
    assert "leave_species_and_clade" in out
