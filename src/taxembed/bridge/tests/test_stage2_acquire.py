import gzip
import io

import pytest

from taxembed.bridge import acquire_all_life


# --- B1: shard-count probe + length-band bisection (disjoint + complete) ---

def test_plan_shards_bisects_until_under_cap_and_tiles():
    TOTAL = 12_000_000

    def probe(q):
        lo, hi = acquire_all_life._band_of(q)               # uniform length density
        return round(TOTAL * (hi - lo + 1) / 2000)

    shards = acquire_all_life.plan_shards("(taxonomy_id:2)", probe, cap=10_000_000)
    assert all(t < 10_000_000 for _, t in shards)
    bands = sorted(acquire_all_life._band_of(q) for q, _ in shards)
    assert bands[0][0] == 1 and bands[-1][1] == 2000        # tiles [1,2000]
    for (a, b), (c, d) in zip(bands, bands[1:]):
        assert c == b + 1                                   # disjoint, contiguous


def test_plan_shards_single_when_under_cap():
    shards = acquire_all_life.plan_shards("(taxonomy_id:2157)", lambda q: 500_000)
    assert len(shards) == 1 and shards[0][1] == 500_000


def test_plan_shards_completeness_uses_in_band_total_not_all_length():
    """M-6: the completeness assert compares the leaf sum to the IN-BAND (length-filtered) full-band
    probe, computed internally — NOT an externally-supplied all-length superkingdom total. A
    non-uniform density that still tiles [1,2000] disjoint+complete must NOT false-raise."""
    def probe(q):
        lo, hi = acquire_all_life._band_of(q)
        return (hi * hi) - ((lo - 1) * (lo - 1))            # additive over disjoint contiguous bands

    shards = acquire_all_life.plan_shards("(taxonomy_id:2)", probe, cap=1_000_000)   # no raise
    assert sum(t for _, t in shards) == probe("length:[1 TO 2000]")


def test_with_band_strips_only_length_clause_not_whole_query():
    """R1-6: a pre-existing length clause is regex-stripped; the superkingdom term survives."""
    q = acquire_all_life._with_band("(taxonomy_id:2) AND length:[1 TO 50]", 51, 100)
    assert "(taxonomy_id:2)" in q
    assert acquire_all_life._band_of(q) == (51, 100)
    assert q.count("length:[") == 1


# --- B2: streaming/resumable reader + single-release pin + cap-truncation fail-loud ---

class _Resp:
    def __init__(self, rows, release):
        body = ("\n".join(rows) + "\n").encode()
        self._gz = io.BytesIO(gzip.compress(body))
        self.headers = {"x-uniprot-release": release}

    def read(self, n=-1):
        return self._gz.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_stream_shard_counts_and_pins_release():
    rows = ["acc\torg", "P1\t9606", "P2\t562"]
    got = []
    n, rel = acquire_all_life.stream_shard("u", expected_total=2, release_pin="2026_01",
                                           open_url=lambda u, timeout=0: _Resp(rows, "2026_01"),
                                           sink=got.append)
    assert n == 2 and rel == "2026_01" and len(got) == 2


def test_stream_shard_raises_on_release_mismatch():
    rows = ["acc\torg", "P1\t9606"]
    with pytest.raises(SystemExit):
        acquire_all_life.stream_shard("u", 1, "2026_01",
                                      open_url=lambda u, timeout=0: _Resp(rows, "2025_04"),
                                      sink=lambda r: None)


def test_stream_shard_raises_on_count_mismatch():
    rows = ["acc\torg", "P1\t9606"]
    with pytest.raises(SystemExit):
        acquire_all_life.stream_shard("u", expected_total=99, release_pin="2026_01",
                                      open_url=lambda u, timeout=0: _Resp(rows, "2026_01"),
                                      sink=lambda r: None)


# --- B3: chunked single-Pfam filter, carrying the source/reviewed flag (B-2 dependency) ---

def test_filter_single_pfam_keeps_exactly_one_and_carries_source():
    rows = [
        "P1\t9606\tPF001;\t100\t1.1.1.1\treviewed",     # one Pfam -> keep, Swiss-Prot
        "P2\t562\tPF001;PF002;\t120\t\treviewed",        # two Pfam -> drop
        "P3\t2157\t\t90\t\tunreviewed",                  # no Pfam  -> drop
        "P4\t7227\tPF009;\t2000\t2.7.1.1\tunreviewed",   # one Pfam -> keep, TrEMBL
    ]
    out = list(acquire_all_life.filter_single_pfam(iter(rows)))
    assert [r["accession"] for r in out] == ["P1", "P4"]
    assert out[0]["pfam"] == "PF001" and out[0]["organism_id"] == "9606"
    assert out[0]["source"] == "sp" and out[1]["source"] == "trembl"
    assert out[1]["length"] == 2000 and out[1]["ec"] == "2.7.1.1"
