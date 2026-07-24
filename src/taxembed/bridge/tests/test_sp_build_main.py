"""B11 orchestration integration test (hermetic): main() on a synthetic snapshot + fakes.

Exercises the full deterministic glue — resolve -> h5-availability -> ranks -> gate-scan -> clade-MI ->
cluster-freeze -> panel/EC freeze (+ full twins) -> build log — WITHOUT the network / real ckpt / mmseqs.
Then proves the round-trip: clean_eval.load_sp_clean reads BOTH the Pfam panel and the INDEPENDENT EC
panel the build froze (the loader-routing fix), with all columns aligned. Thresholds are patched DOWN
so a tiny snapshot crosses the gate (the thresholds themselves are tested in test_sp_gate/accept).
"""
import json
import numpy as np
import pandas as pd
import h5py
import pytest
from taxembed.bridge import build_sp_panel as B
from taxembed.bridge import clean_eval


# --- fakes ---------------------------------------------------------------------------------------
class FakeEmb:
    _idx = {10: 1, 11: 2, 12: 3, 13: 4, 14: 5}
    n_nodes = 20
    positions = np.random.default_rng(0).standard_normal((20, 100))
    def idx_of_taxid(self, t): return self._idx.get(int(t))

_LINE = {
    10: [("species", 10, "sp10"), ("order", 110, "OrA"), ("class", 210, "ClA"), ("phylum", 310, "PhX")],
    11: [("species", 11, "sp11"), ("order", 111, "OrB"), ("class", 210, "ClA"), ("phylum", 310, "PhX")],
    12: [("species", 12, "sp12"), ("order", 112, "OrC"), ("class", 211, "ClB"), ("phylum", 310, "PhX")],
    13: [("species", 13, "sp13"), ("order", 113, "OrD"), ("class", 211, "ClB"), ("phylum", 311, "PhY")],
    14: [("species", 14, "sp14"), ("order", 114, "OrE"), ("class", 212, "ClC"), ("phylum", 311, "PhY")],
}
class FakeResolver:
    def canonical(self, t): return int(t)
    def species_parent(self, t): return None
    def lineage(self, t): return _LINE.get(int(t), [])

class FakeTE:
    n_nodes = 20
    positions = np.random.default_rng(1).standard_normal((20, 100))
    # idx2taxid mirrors FakeEmb._idx: indices 1-5 -> taxids 10-14 (used by _assert_target_invariant
    # when the stamp file exists after a `B.main("all")` call in integration tests)
    idx2taxid = np.array([0, 10, 11, 12, 13, 14] + [0] * 14, dtype=np.int64)


def _snapshot():
    """Human-readable UniProt-TSV-shaped snapshot. PF1/PF2 cross >=2 orders (kept); PF3 is single-order
    (dropped by the gate); a multi-family + a no-Pfam row are dropped by single-family; A12 (taxid 99)
    is unresolvable -> resolve-dropped (feeds the bias table)."""
    rows = [
        # acc,  taxid, pfam,        ec
        ("A01", 10, "PF1", "3.4.24.1"), ("A02", 11, "PF1", "3.4.24.1"),
        ("A03", 12, "PF1", "3.4.24.1"), ("A04", 13, "PF1", "3.4.24.1"),
        ("A05", 10, "PF2", "2.7.11.1"), ("A06", 12, "PF2", "2.7.11.1"), ("A07", 14, "PF2", "2.7.11.1"),
        ("A08", 10, "PF3", ""), ("A09", 10, "PF3", ""),               # PF3: 2 members, 1 order -> fail
        ("A10", 11, "PF9;PF8", ""),                                    # multi-family -> dropped
        ("A11", 12, "", ""),                                           # no Pfam -> dropped
        ("A12", 99, "PF1", "3.4.24.1"),                                # taxid 99 unresolvable -> dropped
    ]
    rng = np.random.default_rng(2)
    aas = "ACDEFGHIKLMNPQRSTVWY"
    raw = pd.DataFrame({
        "Entry": [r[0] for r in rows],
        "Organism (ID)": [str(r[1]) for r in rows],
        "Organism": [f"org{r[1]}" for r in rows],
        "Sequence": ["".join(rng.choice(list(aas), size=40)) for _ in rows],
        "Length": ["40"] * len(rows),
        "Pfam": [r[2] for r in rows],
        "EC number": [r[3] for r in rows],
        "Protein names": ["x"] * len(rows),
        "Keywords": [""] * len(rows),
    })
    seq_lookup = dict(zip(raw["Entry"], raw["Sequence"]))
    return raw, seq_lookup


def _fake_freeze(accessions, sequences, out_tsv, manifest_path):
    pd.DataFrame({"accession": list(accessions), "cluster_id": list(accessions)}).to_csv(
        out_tsv, sep="\t", index=False)
    from pathlib import Path
    Path(manifest_path).write_text(json.dumps({"method": "fake"}))
    return "fake"


@pytest.fixture
def wired(tmp_path, monkeypatch):
    cfg = B.config
    raw, seq_lookup = _snapshot()
    # fake h5: every RESOLVED single-family accession (A01..A09) present, 1024-d float16
    h5p = tmp_path / "emb.h5"
    rng = np.random.default_rng(3)
    with h5py.File(h5p, "w") as h:
        for a in [f"A0{i}" for i in range(1, 10)]:
            h.create_dataset(a, data=rng.standard_normal(1024).astype(np.float16))
    # manifest (release read by main())
    man = tmp_path / "ann_manifest.json"; man.write_text(json.dumps({"release": "test_rel", "rows": 9}))
    # --- patch acquisition / load seams ---
    monkeypatch.setattr(B, "_ensure_inputs", lambda: None)
    monkeypatch.setattr(B, "_read_snapshot", lambda: (raw, seq_lookup))
    monkeypatch.setattr(B, "_load_emb", lambda target="metazoa": FakeEmb())
    monkeypatch.setattr(B, "_load_resolver", lambda: FakeResolver())
    monkeypatch.setattr(B, "_h5_path", lambda: str(h5p))
    monkeypatch.setattr(B, "freeze_cluster_ids", _fake_freeze)
    # --- patch config paths -> tmp ---
    for name, fn in [("SP_METAZOA_RESOLUTION", "resolution.tsv"), ("SP_GATE_SCAN", "gate_scan.tsv"),
                     ("SP_EC_GATE_SCAN", "ec_gate_scan.tsv"), ("SP_CLUSTER_IDS", "cluster_ids.tsv"),
                     ("SP_CLUSTER_IDS_MANIFEST", "cluster_ids_manifest.json"),
                     ("SP_BUILD_LOG", "build_log.md"), ("SP_METAZOA_PANEL", "panel.tsv"),
                     ("SP_METAZOA_PANEL_FULL", "panel_full.tsv"), ("SP_METAZOA_EC_PANEL", "ec_panel.tsv"),
                     ("SP_METAZOA_EC_PANEL_FULL", "ec_panel_full.tsv"),
                     ("SP_METAZOA_EXCLUSIONS", "excl.tsv"), ("SP_METAZOA_EC_EXCLUSIONS", "ec_excl.tsv"),
                     ("SP_METAZOA_MANIFEST", "ann_manifest.json"), ("DATA", "")]:
        monkeypatch.setattr(cfg, name, (tmp_path / fn) if fn else tmp_path)
    monkeypatch.setattr(cfg, "SP_METAZOA_MANIFEST", man)
    # --- patch thresholds DOWN so the tiny snapshot crosses the gate ---
    monkeypatch.setattr(cfg, "SP_PFAM_MEMBER_FLOOR", 2)
    monkeypatch.setattr(cfg, "SP_CROSS_MIN_ORDERS", 2)
    monkeypatch.setattr(cfg, "SP_CROSS_MIN_EFF_ORDERS", 1.5)
    monkeypatch.setattr(cfg, "SP_MIN_CROSSED_FAMILIES", 1)
    monkeypatch.setattr(cfg, "SP_NESTING_MAX", 0.999)
    monkeypatch.setattr(cfg, "SP_JOIN_COVERAGE_MIN", 0.0)
    # round-trip: clean_eval h5 + TE seams
    monkeypatch.setattr(clean_eval, "_load_sp_h5_path", lambda tb=None: str(h5p))
    monkeypatch.setattr(clean_eval, "_load_te", lambda target="metazoa": FakeTE())
    return cfg, tmp_path


def test_gate_scan_writes_deterministic_artifacts(wired):
    cfg, tmp = wired
    out = B.main(["--stage", "gate-scan"])
    assert out["pfam_kept"] == {"PF1", "PF2"} and out["ec_kept"] == {"3.4.24", "2.7.11"}
    assert out["pfam_acc"]["accepted"] and out["join_coverage"] == 1.0
    for p in ["resolution.tsv", "gate_scan.tsv", "ec_gate_scan.tsv", "cluster_ids.tsv", "build_log.md"]:
        assert (tmp / p).exists(), p
    scan = pd.read_csv(tmp / "gate_scan.tsv", sep="\t")
    assert "clade_signal_mi" in scan.columns
    assert dict(zip(scan.pfam_family, scan.keep)) == {"PF1": True, "PF2": True, "PF3": False}
    res = pd.read_csv(tmp / "resolution.tsv", sep="\t")
    assert "A12" not in set(res.accession)                      # unresolvable taxid dropped
    assert list(res.accession) == sorted(res.accession)         # sorted by accession (determinism)


def test_apply_freezes_panels_with_full_schema_and_twins(wired):
    cfg, tmp = wired
    B.main(["--stage", "all"])
    panel = pd.read_csv(tmp / "panel.tsv", sep="\t")
    aa_cols = [f"aa_{c}" for c in "ACDEFGHIKLMNPQRSTVWY"]
    assert all(c in panel.columns for c in aa_cols)             # all 20 composition cols frozen
    assert "cluster_id" in panel.columns and "ec" in panel.columns
    assert set(panel.pfam_family) == {"PF1", "PF2"}
    assert list(panel.accession) == sorted(panel.accession)
    # zero-exclusion twin == panel (no exclusions file present)
    full = pd.read_csv(tmp / "panel_full.tsv", sep="\t")
    assert list(full.accession) == list(panel.accession)
    # INDEPENDENT EC panel + its full twin
    ec = pd.read_csv(tmp / "ec_panel.tsv", sep="\t")
    assert set(ec.ec) == {"3.4.24", "2.7.11"} and all(c in ec.columns for c in aa_cols)
    assert (tmp / "ec_panel_full.tsv").exists()


def test_round_trip_through_load_sp_clean_pfam_and_ec(wired):
    cfg, tmp = wired
    B.main(["--stage", "all"])
    # Pfam testbed -> Pfam panel; EC testbed -> the INDEPENDENT EC panel (loader-routing fix)
    ann_p, reps_p, pos_p, _ = clean_eval.load_sp_clean("sp_metazoa_pfam")
    assert set(ann_p.func) == {"PF1", "PF2"} and reps_p.shape[0] == len(ann_p) == pos_p.shape[0]
    ann_e, reps_e, pos_e, _ = clean_eval.load_sp_clean("sp_metazoa_ec")
    assert set(ann_e.func) == {"3.4.24", "2.7.11"}              # the EC panel, NOT the Pfam ec column
    assert reps_e.shape[0] == len(ann_e)


def test_determinism_byte_identical_on_rerun(wired):
    cfg, tmp = wired
    B.main(["--stage", "all"])
    first = {p: (tmp / p).read_bytes() for p in
             ["panel.tsv", "panel_full.tsv", "ec_panel.tsv", "gate_scan.tsv", "resolution.tsv"]}
    B.main(["--stage", "all", "--force"])                       # re-run over existing -> --force
    for p, b in first.items():
        assert (tmp / p).read_bytes() == b, f"{p} not byte-identical on re-run"
