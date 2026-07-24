import pandas as pd

from taxembed.bridge import build_sp_panel


def _members(fam, n):
    return pd.DataFrame({"accession": [f"{fam}_{i:04d}" for i in range(n)],
                         "pfam_family": [fam] * n,
                         "tax_order": [f"o{i % 5}" for i in range(n)]})


# NOTE: build_sp_panel does top-level `import config` (a different module object than
# taxembed.bridge.config), so we patch build_sp_panel.config — patching the package
# config would have no effect on apply_family_cap.

def test_per_family_cap_caps_at_C(monkeypatch):
    monkeypatch.setattr(build_sp_panel.config, "SP_CAP_C", 10)
    monkeypatch.setattr(build_sp_panel.config, "SP_TOTAL_EMBED_CEILING", 1_000_000)
    df = pd.concat([_members("PF1", 50), _members("PF2", 7)], ignore_index=True)
    out = build_sp_panel.apply_family_cap(df)
    assert (out["pfam_family"] == "PF1").sum() == 10       # capped
    assert (out["pfam_family"] == "PF2").sum() == 7        # under cap, untouched
    assert set(out["pfam_family"]) == {"PF1", "PF2"}       # no family dropped (cap-after-gate)


def test_ceiling_proportional_downsample(monkeypatch):
    monkeypatch.setattr(build_sp_panel.config, "SP_CAP_C", 1000)
    monkeypatch.setattr(build_sp_panel.config, "SP_TOTAL_EMBED_CEILING", 30)
    df = pd.concat([_members("PF1", 40), _members("PF2", 20)], ignore_index=True)  # 60 -> 30
    out = build_sp_panel.apply_family_cap(df)
    assert len(out) <= 30
    assert set(out["pfam_family"]) == {"PF1", "PF2"}       # proportional, no family wiped


def test_cap_is_deterministic(monkeypatch):
    monkeypatch.setattr(build_sp_panel.config, "SP_CAP_C", 10)
    monkeypatch.setattr(build_sp_panel.config, "SP_TOTAL_EMBED_CEILING", 1_000_000)
    df = _members("PF1", 50)
    a = build_sp_panel.apply_family_cap(df)
    b = build_sp_panel.apply_family_cap(df)
    assert list(a["accession"]) == list(b["accession"])    # SP_CAP_SEED fixed


def test_ceiling_overshoot_trimmed_to_exact(monkeypatch):
    """M-7: the per-family max(1) floor lifts every rounded-to-0 family above its proportional
    share, so Σ can exceed the hard ceiling. A deterministic surplus trim must bring it to <= ceiling
    while keeping every admitted family (ceiling here is >> #families)."""
    monkeypatch.setattr(build_sp_panel.config, "SP_CAP_C", 1000)
    monkeypatch.setattr(build_sp_panel.config, "SP_TOTAL_EMBED_CEILING", 50)
    df = pd.concat([_members("PF_big", 100), _members("PFa", 1), _members("PFb", 1),
                    _members("PFc", 1)], ignore_index=True)   # 103 -> floor gives 48+1+1+1=51 > 50
    out = build_sp_panel.apply_family_cap(df)
    assert len(out) <= 50                                   # trimmed to the hard ceiling
    assert set(out["pfam_family"]) == {"PF_big", "PFa", "PFb", "PFc"}  # no family wiped


def test_cap_is_stratified_by_order(monkeypatch):
    """R-4: the per-family cap samples stratified-by-order, so every populated order survives and
    the per-order distribution tracks the pre-cap one (no order-grain feasibility loss from the cap)."""
    monkeypatch.setattr(build_sp_panel.config, "SP_CAP_C", 10)
    monkeypatch.setattr(build_sp_panel.config, "SP_TOTAL_EMBED_CEILING", 1_000_000)
    df = _members("PF1", 100)                              # 20 per order o0..o4
    out = build_sp_panel.apply_family_cap(df)
    counts = out["tax_order"].value_counts()
    assert len(out) == 10
    assert set(counts.index) == {f"o{i}" for i in range(5)}   # every order represented
    assert all(1 <= c <= 3 for c in counts)                   # ~2 each (stratified, not 10-from-one-order)
