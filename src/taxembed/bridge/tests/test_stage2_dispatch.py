from taxembed.bridge import clean_eval, config


def test_all_life_pfam_routes_to_all_life_panel_not_metazoa():
    # all_life_pfam CONTAINS '_pfam' — must NOT fall through to SP_METAZOA_PANEL
    assert clean_eval._load_sp_panel_path("all_life_pfam") == str(config.ALL_LIFE_PANEL)
    assert clean_eval._load_sp_panel_path("all_life_pfam_full") == str(config.ALL_LIFE_PANEL_FULL)


def test_all_life_ec_routes_to_all_life_ec_panel_not_metazoa():
    # all_life_ec CONTAINS '_ec' — must NOT fall through to SP_METAZOA_EC_PANEL
    assert clean_eval._load_sp_panel_path("all_life_ec") == str(config.ALL_LIFE_EC_PANEL)
    assert clean_eval._load_sp_panel_path("all_life_ec_full") == str(config.ALL_LIFE_EC_PANEL_FULL)


def test_metazoa_routing_unchanged():
    assert clean_eval._load_sp_panel_path("sp_metazoa_pfam") == str(config.SP_METAZOA_PANEL)
    assert clean_eval._load_sp_panel_path("sp_metazoa_ec") == str(config.SP_METAZOA_EC_PANEL)
    assert clean_eval._load_sp_panel_path("sp_metazoa_pfam_full") == str(config.SP_METAZOA_PANEL_FULL)


def test_h5_path_routes_all_life():
    assert clean_eval._load_sp_h5_path("all_life_pfam") == str(config.ALL_LIFE_H5)
    assert clean_eval._load_sp_h5_path("all_life_ec") == str(config.ALL_LIFE_H5)
    assert clean_eval._load_sp_h5_path("sp_metazoa_pfam") == str(config.SP_METAZOA_H5)
    assert clean_eval._load_sp_h5_path() == str(config.SP_METAZOA_H5)        # argless default → metazoa
