from pathlib import Path

from taxembed.eval.release_diff import (
    parse_merged,
    parse_delnodes,
    parse_parents,
    canonicalize_taxid,
    reclassified_taxa,
)


def _write(p, text):
    p.write_text(text)
    return p


def test_parse_merged_and_delnodes(tmp_path):
    merged = _write(tmp_path / "merged.dmp", "12\t|\t34\t|\n56\t|\t78\t|\n")
    deln = _write(tmp_path / "delnodes.dmp", "99\t|\n100\t|\n")
    m = parse_merged(merged)
    d = parse_delnodes(deln)
    assert m == {12: 34, 56: 78}
    assert d == {99, 100}


def test_canonicalize_follows_merge_chain():
    merged = {12: 34, 34: 56}
    assert canonicalize_taxid(12, merged, set()) == 56
    assert canonicalize_taxid(34, merged, set()) == 56
    assert canonicalize_taxid(56, merged, set()) == 56
    assert canonicalize_taxid(99, {}, {99}) is None


def test_parse_parents(tmp_path):
    nodes = _write(tmp_path / "nodes.dmp",
                   "2\t|\t1\t|\tsuperkingdom\t|\n"
                   "1\t|\t1\t|\tno rank\t|\n"
                   "9\t|\t2\t|\tgenus\t|\n")
    par = parse_parents(nodes)
    assert par == {2: 1, 1: 1, 9: 2}


def test_reclassified_excludes_merges_and_rankonly():
    old_parent = {9: 2, 10: 2, 11: 3, 12: 4}
    new_parent = {9: 5, 10: 2, 11: 3, 99: 7}
    new_merged = {12: 99}
    new_delnodes = set()
    scored = [9, 10, 11, 12]
    reclass = reclassified_taxa(scored, old_parent, new_parent, new_merged, new_delnodes)
    assert reclass == {9}
