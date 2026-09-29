#!/usr/bin/env python3
"""Spec §4.5 TimeTree — probe what is actually downloadable, and TIME it. Downloads nothing large.

WHY THIS FIRST. §4.5 is headed "External ground truth — SCHEDULED, not optional" and closes
"Schedule it, or delete the relatedness claim from the paper." No code and no data for it exist in
this repo (checked), so it is a BUILD, not a submit. Before any of it is planned, the cost gets
MEASURED rather than guessed: yesterday "days on LRZ" turned out to be twelve seconds of downloading
([[feedback_submit_to_lrz_early_queue_time_is_concurrent]]), and the standing correction is that I
overblow time ([[feedback_sequence_dont_cut_and_dont_overblow_time]]).

WHAT §4.5 NEEDS, minimally: divergence times for a few hundred well-sampled vertebrates and insects,
enough to report

    Spearman(embedded distance, divergence time)  BESIDE  Spearman(NCBI path length, divergence time)

🎯 The second term is not decoration. Finding 4 §4.7 is that outside-the-tree information must enter
the ANSWER, not merely the QUESTION -- a protocol whose label is unseen but predictable from a cheap
statistic of the training data measures that statistic. §4.5 is the one remaining protocol that
already names its cheap baseline by construction, which is exactly why it survives Finding 4.

This probe issues HEAD-style requests (streamed GET, aborted after the first chunk) so it learns
status + size + latency without pulling megabytes. It writes no data files.

Written 2026-09-29.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path("/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings")
UA = {"User-Agent": "Mozilla/5.0 (research; TaxEmbed spec 4.5 feasibility probe)"}
TIMEOUT = 25

CANDIDATES = [
    ("TTOL 2015 newick (classic full tree)",
     "http://www.timetree.org/public/data/TimetreeOfLife2015.nwk"),
    ("TTOL 2015 newick, https",
     "https://timetree.org/public/data/TimetreeOfLife2015.nwk"),
    ("TimeTree5 'Latest' data page",
     "https://timetree.org/home"),
    ("public data directory",
     "http://www.timetree.org/public/data/"),
    ("species-pair API shape (expect 4xx if absent)",
     "http://www.timetree.org/api/pairwise/9606/10090"),
]


def probe(label: str, url: str) -> dict:
    t0 = time.time()
    rec = {"label": label, "url": url}
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            first = r.read(4096)
            rec.update(
                status=r.status,
                content_type=r.headers.get("Content-Type", "?"),
                content_length=r.headers.get("Content-Length", "?"),
                final_url=r.url,
                first_bytes=first[:110].decode("utf-8", "replace").replace("\n", " "),
            )
    except urllib.error.HTTPError as e:
        rec.update(status=e.code, error=f"HTTPError {e.code} {e.reason}")
    except Exception as e:                                   # noqa: BLE001
        rec.update(status=None, error=f"{type(e).__name__}: {e}")
    rec["seconds"] = round(time.time() - t0, 3)
    return rec


def main() -> None:
    print("=" * 96)
    print("§4.5 TimeTree — download feasibility probe (measures; downloads nothing large)")
    print("=" * 96)
    out = []
    for label, url in CANDIDATES:
        rec = probe(label, url)
        out.append(rec)
        status = rec.get("status")
        print(f"\n  {label}")
        print(f"    {url}")
        if rec.get("error"):
            print(f"    ✗ {rec['error']}   ({rec['seconds']}s)")
            continue
        cl = rec["content_length"]
        mb = f"{int(cl)/1e6:.1f} MB" if str(cl).isdigit() else "unknown size"
        print(f"    ✓ HTTP {status}  {rec['content_type']}  {mb}  ({rec['seconds']}s)")
        if rec["final_url"] != url:
            print(f"      → redirected to {rec['final_url']}")
        print(f"      head: {rec['first_bytes'][:100]}")

    outp = ROOT / "results" / "timetree_fetch_probe_20260929.json"
    outp.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwritten: {outp}")


if __name__ == "__main__":
    main()
