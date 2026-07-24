from taxembed.bridge import clean_eval


def test_provenance_block_shape(tmp_path):
    panel = tmp_path / "p.tsv"; panel.write_text("accession\tx\nA0\t1\n")
    prov = clean_eval._build_provenance(testbed="all_life_pfam", panel_path=str(panel),
                                        h5_path="/nonexistent.h5", cap_seed=0)
    for key in ("config_git_sha", "panel_path", "panel_sha256", "h5_path", "h5_sha256",
                "mmseqs_version", "cap_seed", "poincare_ckpt_sha256"):
        assert key in prov
    assert prov["panel_sha256"] is not None              # panel exists -> hashed
    assert prov["h5_sha256"] is None                     # missing file -> None, not a crash
    assert isinstance(prov["cap_seed"], int)
