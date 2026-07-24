from taxembed.bridge import config

def test_sp_constants_present():
    assert str(config.SP_METAZOA_H5).endswith("per-protein.h5")
    assert config.SP_PFAM_MEMBER_FLOOR == 50
    assert config.SP_CROSS_MIN_ORDERS == 4
    assert config.SP_CROSS_MIN_EFF_ORDERS == 3
    assert config.SP_NESTING_MAX == 0.30
    assert config.SP_NMI_MAX == 0.70
    assert config.SP_COVERAGE_MIN == 0.50
    assert config.SP_JOIN_COVERAGE_MIN == 0.98
    assert config.SP_MIN_CROSSED_FAMILIES == 15
    assert config.SP_EC_LEVEL == 3
    assert config.SP_TAXONOMY_QUERY == "reviewed:true AND taxonomy_id:33208"
