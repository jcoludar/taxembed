from taxembed.bridge import config


def test_all_life_constants_present_and_typed():
    # cellular target paths are siblings of the metazoa ones under the poincare repo
    assert config.CELLULAR_CKPT.name.endswith(".pth")
    assert "cellular" in str(config.CELLULAR_CKPT).lower()
    assert config.CELLULAR_ROOT_TAXID == 131567
    # frozen all-life artifacts live under data/, distinct from sp_metazoa_*
    for p in (config.ALL_LIFE_PANEL, config.ALL_LIFE_EC_PANEL, config.ALL_LIFE_H5):
        assert "all_life" in p.name
    # cap + ceiling committed numerically (no deferral)
    assert config.SP_CAP_C == 1500
    assert config.SP_TOTAL_EMBED_CEILING == 250_000
    assert config.SP_CAP_SEED == 0
    # headroom criteria for BOTH legs
    assert 0.0 < config.ALL_LIFE_PURITY_REL_DROP_MAX <= 0.10
    assert config.ALL_LIFE_MIN_PURITY_HEADROOM > 0.0
    assert config.ALL_LIFE_MIN_TAX_HEADROOM > 0.0
    assert 0.0 < config.ALL_LIFE_TAX_COLLAPSE_FRAC_MIN <= 1.0
    # grain literals promoted from clean_eval.py:1397
    assert config.GRAIN_MIN_FRAC_STRATA == 0.50
    assert config.GRAIN_MIN_COVERAGE == 0.80
    # acquisition field set (narrow; EC for the corroborative panel; reviewed for the §4.4 twin / §7 batch guard)
    assert config.SP2_METADATA_FIELDS == "accession,organism_id,xref_pfam,length,ec,reviewed"
