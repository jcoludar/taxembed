"""All-life acquisition (spec v3 §4.1): sharded, streamed, resumable, sized.

No single REST request may exceed the 10M-result cap. Shards are disjoint + complete by construction
(the leaf sum is asserted against the IN-BAND full-band probe — M-6: NOT an all-length total, which
the ≤2000-aa cutoff makes strictly larger). The single-Pfam filter streams row-by-row so Phase-A RAM
is O(survivors), never the ~250M-row snapshot, and carries the source/reviewed flag the §4.4 twin and
§7 batch-effect guard depend on.
"""
import gzip
import re
import urllib.request

CAP = 10_000_000

_LEN_RE = re.compile(r"\s*AND\s*length:\[\d+ TO \d+\]")
_BAND_RE = re.compile(r"length:\[(\d+) TO (\d+)\]")
_REVIEWED_TRUE = {"reviewed", "true", "True", "1"}


def _band_of(query):
    m = _BAND_RE.search(query)
    return (int(m.group(1)), int(m.group(2))) if m else (1, 2000)


def _with_band(base, lo, hi):
    """Attach a length band, regex-stripping ONLY a pre-existing length clause (R1-6 — the naive
    'drop the whole string if it contains length:[' nukes the superkingdom term)."""
    base = _LEN_RE.sub("", base)
    return f"{base} AND length:[{lo} TO {hi}]"


def plan_shards(superkingdom_query, probe, cap=CAP, lo=1, hi=2000, _total=None):
    """Recursively bisect the length band until every shard < cap. `probe(query)->int` reads the
    REST `size=0` `x-total-results`. Disjoint + complete by construction: the top-level call captures
    the IN-BAND full-band total and asserts the leaf sum equals it."""
    q = _with_band(superkingdom_query, lo, hi)
    n = probe(q)
    if _total is None:
        _total = n                                           # in-band (length-filtered) full-band total
    if n < cap or lo >= hi:
        return [(q, n)]
    mid = (lo + hi) // 2
    left = plan_shards(superkingdom_query, probe, cap, lo, mid, _total)
    right = plan_shards(superkingdom_query, probe, cap, mid + 1, hi, _total)
    shards = left + right
    if lo == 1 and hi == 2000:                               # top-level completeness assert
        s = sum(t for _, t in shards)
        if s != _total:
            raise SystemExit(f"[acquire] shard totals {s} != in-band total {_total} "
                             "(incomplete/overlap; spec v3 §4.1)")
    return shards


def stream_shard(url, expected_total, release_pin, open_url=urllib.request.urlopen, sink=None, cap=CAP):
    """Stream a gzipped TSV line-by-line; skip the header row; pin the single release; fail loud on a
    release mismatch, on exactly the cap (silent truncation), or on a count != probed-total."""
    n = 0
    with open_url(url, timeout=1800) as resp:
        release = resp.headers.get("x-uniprot-release", "unknown")
        if release_pin is not None and release != release_pin:
            raise SystemExit(f"[acquire] release {release} != pin {release_pin} (single-release invariant)")
        gz = gzip.GzipFile(fileobj=resp)
        for i, raw in enumerate(gz):
            if i == 0:
                continue                                     # header row
            if sink is not None:
                sink(raw.decode("utf-8").rstrip("\n"))
            n += 1
    if n == cap:
        raise SystemExit(f"[acquire] stream returned exactly the cap {cap} — silent truncation (incomplete)")
    if n != expected_total:
        raise SystemExit(f"[acquire] streamed {n} != probed {expected_total} for this shard")
    return n, release


def filter_single_pfam(rows_iter):
    """Stream the metadata snapshot row-by-row; yield survivors carrying exactly ONE Pfam signature
    (spec v3 §4.1/§4.4 — 'one Pfam signature', NOT 'single-domain'). RAM is O(survivors). Carries
    `reviewed` + a derived `source` ('sp'|'trembl') for the §4.4 twin / §7 batch-effect guard.
    TSV column order = SP2_METADATA_FIELDS = accession,organism_id,xref_pfam,length,ec,reviewed."""
    for line in rows_iter:
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        acc, org, pfam_raw, length = parts[0], parts[1], parts[2], parts[3]
        ec = parts[4] if len(parts) > 4 else ""
        reviewed_tok = parts[5].strip() if len(parts) > 5 else ""
        pfams = [p for p in pfam_raw.strip().rstrip(";").split(";") if p]
        if len(pfams) != 1:
            continue
        try:
            ln = int(length)
        except ValueError:
            continue
        source = "sp" if reviewed_tok in _REVIEWED_TRUE else "trembl"
        yield {"accession": acc, "organism_id": org, "pfam": pfams[0], "length": ln,
               "ec": ec, "reviewed": reviewed_tok, "source": source}
