"""All-life metadata + survivor-sequence download (spec v3 §4.1) — LOGIN-NODE only (internet).

Uses the UniProt REST /stream endpoint (one gzipped response per shard, up to the 10M cap → shard by
length band) for the narrow metadata, and the batched /accessions endpoint for survivor sequences.
Resumable: a completed shard writes a `.done` sentinel and is skipped. Stdlib only (urllib/gzip) so it
runs with the login node's python3 — no container.

Corrections from the LIVE API (mocked Mac tests could not catch these):
  - the count probe needs `size=1` (size=0 → HTTP 400);
  - "has a Pfam xref" is `database:pfam` (`xref_pfam` is a RETURN field, not a SEARCH field);
    single-Pfam ⊆ has-Pfam, so this server-side pre-filter is result-preserving and shrinks the download.
"""
import gzip
import json
import os
import time
import urllib.parse
import urllib.request

BASE = "https://rest.uniprot.org/uniprotkb"
FIELDS = "accession,organism_id,xref_pfam,length,ec,reviewed"
CAP = 10_000_000
# all-life = the three cellular superkingdom subtrees (taxonomy_id is lineage-inclusive on UniProt).
SUPERKINGDOMS = {"Bacteria": 2, "Archaea": 2157, "Eukaryota": 2759}


def _get(url, timeout=120, retries=4):
    last = None
    for k in range(retries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url), timeout=timeout)
        except Exception as e:                               # noqa: BLE001 — retry transient REST/network errors
            last = e
            time.sleep(5 * (k + 1))
    raise SystemExit(f"[download] GET failed after {retries} retries: {url}\n  {last}")


def probe(query):
    """x-total-results for a query (size=1; size=0 is rejected by the live API)."""
    url = f"{BASE}/search?query={urllib.parse.quote(query)}&fields=accession&size=1"
    with _get(url) as r:
        return int(r.headers.get("x-total-results", "0"))


def _band_query(taxid, lo, hi):
    return f"(database:pfam) AND taxonomy_id:{taxid} AND length:[{lo} TO {hi}]"


def plan_bands(taxid, lo=1, hi=2000, cap=CAP):
    """Recursively bisect [lo,hi] until each band's count < cap. Returns [(lo,hi,count)], disjoint+complete."""
    n = probe(_band_query(taxid, lo, hi))
    if n < cap or lo >= hi:
        return [(lo, hi, n)]
    mid = (lo + hi) // 2
    return plan_bands(taxid, lo, mid, cap) + plan_bands(taxid, mid + 1, hi, cap)


def stream_metadata(query, out_gz, expected, release_pin):
    """Stream a shard's gzipped TSV to `out_gz` (atomic .part→replace). Returns (release, n_rows)."""
    part = out_gz + ".part"
    url = f"{BASE}/stream?query={urllib.parse.quote(query)}&format=tsv&fields={FIELDS}&compressed=true"
    with _get(url, timeout=1800) as resp:
        release = resp.headers.get("x-uniprot-release", "unknown")
        if release_pin is not None and release != release_pin:
            raise SystemExit(f"[download] release {release} != pin {release_pin} (single-release invariant)")
        with open(part, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    n = 0                                                    # verify: decompress-count == expected (minus header)
    with gzip.open(part, "rt") as f:
        for _ in f:
            n += 1
    n_rows = max(0, n - 1)
    if expected and n_rows != expected:
        raise SystemExit(f"[download] {out_gz}: streamed {n_rows} != probed {expected}")
    os.replace(part, out_gz)
    return release, n_rows


def download_metadata(out_dir, superkingdoms=None, length_cap=2000):
    """Download all-life has-Pfam metadata, sharded by superkingdom × length band. Resumable."""
    os.makedirs(out_dir, exist_ok=True)
    sk = superkingdoms or SUPERKINGDOMS
    release_pin = None
    manifest = {"release": None, "shards": []}
    for name, taxid in sk.items():
        bands = plan_bands(taxid, 1, length_cap)
        print(f"[{name}] {len(bands)} bands, total {sum(c for *_, c in bands)}", flush=True)
        for lo, hi, expected in bands:
            tag = f"{name}.{lo}_{hi}"
            out_gz = os.path.join(out_dir, f"snapshot.{tag}.tsv.gz")
            done = out_gz + ".done"
            if os.path.exists(done):
                print(f"  resume: {tag} already done", flush=True)
                manifest["shards"].append({"tag": tag, "expected": expected, "resumed": True})
                continue
            rel, n = stream_metadata(_band_query(taxid, lo, hi), out_gz, expected, release_pin)
            release_pin = release_pin or rel
            open(done, "w").write(rel)
            manifest["shards"].append({"tag": tag, "expected": expected, "streamed": n, "release": rel})
            print(f"  {tag}: {n} rows (release {rel})", flush=True)
    manifest["release"] = release_pin
    json.dump(manifest, open(os.path.join(out_dir, "download_manifest.json"), "w"), indent=2)
    return manifest


def _bare_accession(header):
    """UniProt FASTA headers are `>db|ACC|ENTRY ...` (db = sp|tr). Emit the BARE primary accession so the
    self-embed h5 is keyed by it (spec §6 accession-key contract; otherwise subset_h5 silently collapses
    the rep set and load_sp_clean's set-containment guard aborts the battery)."""
    tok = header[1:].split()[0]
    parts = tok.split("|")
    return parts[1] if len(parts) >= 3 and parts[0] in ("sp", "tr") else tok


def fetch_sequences(accessions, out_fasta, batch=500):
    """Pull sequences for survivor accessions via the batched /accessions endpoint (≤500/request),
    rewriting each header to the BARE accession (spec §6). Returns the count written. Login-node only."""
    accs = list(dict.fromkeys(accessions))
    n = 0
    with open(out_fasta, "w") as out:
        for i in range(0, len(accs), batch):
            chunk = accs[i:i + batch]
            url = f"{BASE}/accessions?accessions={','.join(chunk)}&format=fasta"
            with _get(url, timeout=600) as r:
                for raw in r:
                    line = raw.decode("utf-8")
                    if line.startswith(">"):
                        n += 1
                        out.write(">" + _bare_accession(line.rstrip("\n")) + "\n")
                    else:
                        out.write(line)
    return n


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--archaea-only", action="store_true", help="slice: Archaea subtree only (validation)")
    ap.add_argument("--length-cap", type=int, default=2000, help="max length (slice validation uses a small cap)")
    args = ap.parse_args()
    sk = {"Archaea": 2157} if args.archaea_only else None
    m = download_metadata(args.out_dir, superkingdoms=sk, length_cap=args.length_cap)
    print(f"DOWNLOAD_DONE shards={len(m['shards'])} release={m['release']}", flush=True)
