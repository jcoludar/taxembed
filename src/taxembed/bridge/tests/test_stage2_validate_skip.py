import hashlib
import json

from taxembed.bridge import build_sp_panel


def _write(p, text):
    p.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def test_outputs_valid_true_when_rows_and_sha_match(tmp_path):
    out = tmp_path / "panel.tsv"
    man = tmp_path / "panel.manifest.json"
    text = "accession\tx\nP1\t1\nP2\t2\n"                     # 2 data rows
    sha = _write(out, text)
    man.write_text(json.dumps({"rows": 2, "sha256": sha}))
    assert build_sp_panel.outputs_valid(str(out), str(man), n_expected=2) is True


def test_outputs_valid_false_on_rowcount_mismatch(tmp_path):
    out = tmp_path / "panel.tsv"
    man = tmp_path / "panel.manifest.json"
    sha = _write(out, "accession\tx\nP1\t1\n")               # 1 data row
    man.write_text(json.dumps({"rows": 99, "sha256": sha}))
    assert build_sp_panel.outputs_valid(str(out), str(man), n_expected=99) is False


def test_outputs_valid_false_when_absent(tmp_path):
    assert build_sp_panel.outputs_valid(str(tmp_path / "nope.tsv"),
                                        str(tmp_path / "nope.json"), n_expected=1) is False
