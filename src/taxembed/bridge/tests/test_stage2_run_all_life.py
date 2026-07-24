from taxembed.bridge import run_all_life


class _FakeEmb:
    MAP = {9606: 10, 562: 11, 2157: 12}

    def idx_of_taxid(self, t):
        return self.MAP.get(int(t))


class _FakeResolver:
    SK = {9606: "Eukaryota", 562: "Bacteria", 2157: "Archaea"}
    ORD = {9606: "Primates", 562: "Enterobacterales", 2157: "Methanobacteriales"}

    def canonical(self, t):
        return int(t)

    def species_parent(self, t):
        return None

    def lineage(self, t):
        t = int(t)
        return [("superkingdom", t, self.SK[t]), ("order", t, self.ORD[t])]


_RAW = [
    "P1\t9606\tPF1;\t100\t1.1.1.1\treviewed",      # single-Pfam PF1, Eukaryota
    "P2\t562\tPF1;\t110\t1.1.1.1\tunreviewed",      # PF1, Bacteria
    "P3\t2157\tPF1;\t120\t\treviewed",               # PF1, Archaea
    "P4\t9606\tPF2;PF9;\t90\t\treviewed",            # TWO Pfam -> dropped by the filter
    "P5\t562\tPF2;\t95\t\tunreviewed",               # PF2, Bacteria
    "P6\t2157\tPF2;\t130\t\treviewed",               # PF2, Archaea
]


def test_phase_a_chain_runs_and_auto_picks_breadth_rank():
    frame, scan, diag = run_all_life.phase_a(iter(_RAW), _FakeEmb(), _FakeResolver(),
                                             resolution_floor=0.0)
    assert diag["n_survivors"] == 5                  # P4 dropped (two Pfam)
    assert diag["breadth_rank"] == "tax_order"        # 100% non-null order in Bacteria + Archaea
    assert set(frame["tax_superkingdom"]) == {"Eukaryota", "Bacteria", "Archaea"}
    assert "source" in frame.columns and set(frame["source"]) == {"sp", "trembl"}
    assert "idx" in frame.columns and "resolved_taxid" in frame.columns
    assert diag["per_superkingdom_retention"]["Bacteria"] == 2


def test_phase_a_resolution_floor_nogo():
    import pytest

    class _NoneEmb:
        def idx_of_taxid(self, t):
            return None

    with pytest.raises(SystemExit):
        run_all_life.phase_a(iter(_RAW), _NoneEmb(), _FakeResolver(), resolution_floor=0.5)


def test_phase_b_caps_gate_passing(monkeypatch):
    # run_all_life.config is the shared bare `config` module that build_sp_panel also uses
    monkeypatch.setattr(run_all_life.config, "SP_PFAM_MEMBER_FLOOR", 2)
    monkeypatch.setattr(run_all_life.config, "SP_CROSS_MIN_ORDERS", 2)
    monkeypatch.setattr(run_all_life.config, "SP_CROSS_MIN_EFF_ORDERS", 1.5)
    frame, scan, diag = run_all_life.phase_a(iter(_RAW), _FakeEmb(), _FakeResolver(), resolution_floor=0.0)
    capped, panel = run_all_life.phase_b(frame, scan, seq_lookup={})   # freeze=False -> no panel
    assert "pfam_family" in capped.columns
    assert set(capped["pfam_family"]) <= {"PF1", "PF2"}
    assert panel is None
