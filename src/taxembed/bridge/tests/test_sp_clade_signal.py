import numpy as np
from taxembed.bridge import build_sp_panel as B

def test_clade_signal_high_when_reps_track_clade():
    rng = np.random.default_rng(0)
    a = rng.normal(0,0.1,(40,8)) + 10; b = rng.normal(0,0.1,(40,8)) - 10
    reps = np.vstack([a,b]); clades = ["c0"]*40 + ["c1"]*40
    assert B.within_family_clade_mi(reps, clades, n_clusters=2) > 0.6     # near H(clade)=ln2

def test_clade_signal_low_when_reps_random():
    rng = np.random.default_rng(1)
    reps = rng.normal(0,1,(80,8)); clades = ["c0"]*40 + ["c1"]*40
    assert B.within_family_clade_mi(reps, clades, n_clusters=2) < 0.2
