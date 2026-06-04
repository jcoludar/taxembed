"""Strip ANSI + tqdm carriage-return spam from a train_small.py log and print the epoch table rows.

Usage: _parse_train_log.py <logfile>
Reusable diagnostic for reading background-training progress (Sep/kNN%/DepthCorr per epoch).
"""
import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
# An epoch summary row: optional leading junk, then "<epoch> | <loss> | ... | <status>"
ROW = re.compile(r"(\d+)\s*\|\s*([\d.]+)\s*\|.*\|\s*([+-]?[\d.]+)\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)\s*\|\s*([A-Za-z✓ ]+)\s*$")

path = sys.argv[1]
with open(path, "rb") as fh:
    raw = fh.read().decode("utf-8", errors="replace")
raw = ANSI.sub("", raw)
# tqdm uses \r; split on both so each table row (printed after a \r-cleared line) is isolated.
lines = re.split(r"[\r\n]", raw)

print(f"{'ep':>4} {'loss':>10} {'depthCorr':>10} {'Hier%':>7} {'kNN%':>7} {'Sep':>6}  status")
seen = set()
for ln in lines:
    m = ROW.search(ln)
    if not m:
        continue
    ep = int(m.group(1))
    if ep in seen:
        continue
    seen.add(ep)
    loss, depth, hier, knn, sep, status = m.group(2), m.group(3), m.group(4), m.group(5), m.group(6), m.group(7).strip()
    print(f"{ep:>4} {loss:>10} {depth:>10} {hier:>7} {knn:>7} {sep:>6}  {status}")
print(f"\n[{len(seen)} epochs parsed]")
