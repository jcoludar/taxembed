import pytest

from taxembed.bridge import step0_taxonomy_gate as s0, config


def test_assert_ckpt_present_halts_when_absent(tmp_path):
    with pytest.raises(SystemExit):
        s0.assert_ckpt_present(str(tmp_path / "missing.pth"))


def test_assert_ckpt_present_ok_when_exists(tmp_path):
    p = tmp_path / "cellular_canonical.pth"
    p.write_bytes(b"x")
    assert s0.assert_ckpt_present(str(p)) == str(p)


def test_node_counts_pass_when_all_superkingdoms_above_floor():
    counts = {"Bacteria": 2_000_000, "Archaea": 80_000, "Eukaryota": 1_500_000}
    floors = {"Bacteria": 10_000, "Archaea": 1_000, "Eukaryota": 10_000}
    assert s0.check_superkingdom_node_counts(counts, floors)["ok"] is True


def test_node_counts_raise_on_eukaryote_subset_dump():
    """The exact silent mis-scope §3 exists to prevent: a eukaryote-subset dump that still resolves
    one bacterial taxid (Bacteria count ~ 0) must HALT, not pass."""
    counts = {"Bacteria": 3, "Archaea": 0, "Eukaryota": 1_500_000}
    floors = {"Bacteria": 10_000, "Archaea": 1_000, "Eukaryota": 10_000}
    with pytest.raises(SystemExit):
        s0.check_superkingdom_node_counts(counts, floors)


def test_superkingdom_rank_labels_required():
    ok = {9606: "Eukaryota", 562: "Bacteria", 2157: "Archaea"}
    assert s0.assert_superkingdom_rank_labels(ok)["ok"] is True
    with pytest.raises(SystemExit):
        s0.assert_superkingdom_rank_labels({9606: "Eukaryota", 562: None})


def test_record_dump_provenance(tmp_path):
    (tmp_path / "nodes.dmp").write_text("1\t|\tno rank\t|\n")
    out = s0.record_dump_provenance(str(tmp_path), release="2026-06-01")
    assert out["release"] == "2026-06-01"
    assert len(out["sha256"]) == 64


def test_config_step0_floors_exist():
    f = config.SP_STEP0_SUPERKINGDOM_NODE_MIN
    assert {"Bacteria", "Archaea", "Eukaryota"} <= set(f)
    assert all(isinstance(v, int) and v > 0 for v in f.values())
