import numpy as np
import pandas as pd
import pytest

from taxembed.bridge import twin_batch_eval as tb, config, clean_eval, build_sp_panel


# --- source column carried in the panel (B-2: makes the twin + batch guard buildable) ---

def test_panel_cols_carry_source():
    assert "source" in build_sp_panel._PANEL_COLS


# --- §4.4 SP-only twin: source_subset + dispatch routing ---

def test_source_subset_filters_by_source():
    ann = pd.DataFrame({"accession": list("ABCD"), "source": ["sp", "trembl", "sp", "trembl"]})
    assert list(tb.source_subset(ann, "sp")["accession"]) == ["A", "C"]
    assert list(tb.source_subset(ann, "trembl")["accession"]) == ["B", "D"]


def test_source_filter_of_testbed_routing():
    assert clean_eval._source_filter_of("all_life_pfam_sp") == "sp"
    assert clean_eval._source_filter_of("all_life_pfam_trembl") == "trembl"
    assert clean_eval._source_filter_of("all_life_pfam") is None
    assert clean_eval._source_filter_of("sp_metazoa_pfam") is None


def test_source_twin_panel_path_resolves_to_base_panel():
    # the _sp/_trembl twin reads the SAME base all-life panel, then row-filters by source
    assert clean_eval._load_sp_panel_path("all_life_pfam_sp") == str(config.ALL_LIFE_PANEL)
    assert clean_eval._load_sp_panel_path("all_life_pfam_trembl") == str(config.ALL_LIFE_PANEL)
    assert clean_eval._load_sp_panel_path("all_life_ec_sp") == str(config.ALL_LIFE_EC_PANEL)


# --- §7 SP-vs-TrEMBL batch-effect guard ---

def test_batch_effect_auc_near_chance_on_random():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 16))
    src = ["sp" if i % 2 == 0 else "trembl" for i in range(200)]
    auc = tb.batch_effect_auc(X, src, cv=3)
    assert 0.35 <= auc <= 0.65                              # near chance — no batch effect


def test_batch_effect_auc_high_when_separable():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 16))
    src = ["sp" if i < 100 else "trembl" for i in range(200)]
    X[100:] += 5.0                                          # shift the TrEMBL cloud
    auc = tb.batch_effect_auc(X, src, cv=3)
    assert auc > 0.9


# --- M-5 §4.2.1 span-denominator floor ---

def test_assert_span_denominator_ok_above_floor():
    sk = ["Bacteria", "Eukaryota", None, "Archaea", "Bacteria"]   # 4/5 non-null = 0.8
    out = tb.assert_span_denominator(sk, min_frac=0.5)
    assert out["ok"] is True and out["frac"] == 0.8


def test_assert_span_denominator_raises_when_mostly_null():
    sk = [None] * 997 + ["Bacteria", "Archaea", "Eukaryota"]      # 3/1000 = 0.003
    with pytest.raises(SystemExit):
        tb.assert_span_denominator(sk, min_frac=0.5)


# --- R-SC-8 per-superkingdom retention for result-JSON denominators ---

def test_per_superkingdom_retention_counts_nulls():
    sk = ["Bacteria", "Bacteria", "Eukaryota", None]
    out = tb.per_superkingdom_retention(sk)
    assert out["Bacteria"] == 2 and out["Eukaryota"] == 1 and out["null"] == 1


# --- #9 TrEMBL annotation-completeness relabel bound ---

def test_trembl_relabel_check_flags_domain_over_bound():
    out = tb.trembl_relabel_check({"Bacteria": 0.10, "Archaea": 0.70, "Eukaryota": 0.05}, bound=0.50)
    assert out["relabel"] is True and out["excluded_domains"] == ["Archaea"]
    assert out["retained_domains"] == ["Bacteria", "Eukaryota"]


def test_trembl_relabel_check_clean_when_all_under_bound():
    out = tb.trembl_relabel_check({"Bacteria": 0.10, "Archaea": 0.20}, bound=0.50)
    assert out["relabel"] is False and out["excluded_domains"] == []


def test_config_span_and_relabel_constants_exist():
    assert 0.0 < config.ALL_LIFE_SPAN_DENOM_MIN <= 1.0
    assert 0.0 < config.ALL_LIFE_TREMBL_RELABEL_MAX <= 1.0
