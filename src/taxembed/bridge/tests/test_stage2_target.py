from unittest import mock
from taxembed.bridge import clean_eval, config, build_sp_panel


def test_load_te_routes_metazoa_vs_cellular():
    with mock.patch("embeddings.taxonomy_bridge.TaxonomyEmbedding") as TE:
        clean_eval._load_te("metazoa")
        TE.assert_called_once_with(config.CKPT, config.TAXMAP, config.EDGELIST)
    with mock.patch("embeddings.taxonomy_bridge.TaxonomyEmbedding") as TE:
        clean_eval._load_te("cellular")
        TE.assert_called_once_with(config.CELLULAR_CKPT, config.CELLULAR_TAXMAP, config.CELLULAR_EDGELIST)


def test_load_te_default_is_metazoa():
    with mock.patch("embeddings.taxonomy_bridge.TaxonomyEmbedding") as TE:
        clean_eval._load_te()
        TE.assert_called_once_with(config.CKPT, config.TAXMAP, config.EDGELIST)


def test_load_te_rejects_unknown_target():
    import pytest
    with pytest.raises(ValueError):
        clean_eval._load_te("eukaryota")


def test_load_emb_routes_metazoa_vs_cellular():
    with mock.patch("embeddings.taxonomy_bridge.TaxonomyEmbedding") as TE:
        build_sp_panel._load_emb("metazoa")
        TE.assert_called_once_with(config.CKPT, config.TAXMAP, config.EDGELIST)
    with mock.patch("embeddings.taxonomy_bridge.TaxonomyEmbedding") as TE:
        build_sp_panel._load_emb("cellular")
        TE.assert_called_once_with(config.CELLULAR_CKPT, config.CELLULAR_TAXMAP, config.CELLULAR_EDGELIST)


def test_load_emb_default_is_metazoa():
    with mock.patch("embeddings.taxonomy_bridge.TaxonomyEmbedding") as TE:
        build_sp_panel._load_emb()
        TE.assert_called_once_with(config.CKPT, config.TAXMAP, config.EDGELIST)


def test_load_emb_rejects_unknown_target():
    import pytest
    with pytest.raises(ValueError):
        build_sp_panel._load_emb("eukaryota")


def test_cellular_taxmap_and_edgelist_resolve_on_disk():
    """B-3 regression: the cellular taxmap/edgelist paths must point at files that EXIST
    (the mapping/edgelist ship locally under poincare; only the .pth is GPU cluster-only, spec §3).
    Plan-1's mock-only target tests never resolved these strings → the
    `cellular_131567_clean` vs real `cellular_organisms_131567_clean` typo went uncaught and
    would FileNotFoundError in TaxonomyEmbedding.__init__ on GPU cluster too."""
    assert config.CELLULAR_TAXMAP.exists(), f"CELLULAR_TAXMAP missing: {config.CELLULAR_TAXMAP}"
    assert config.CELLULAR_EDGELIST.exists(), f"CELLULAR_EDGELIST missing: {config.CELLULAR_EDGELIST}"
