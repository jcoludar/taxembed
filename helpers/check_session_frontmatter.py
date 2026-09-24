"""Structural check that a session log's YAML frontmatter is well formed.

WHY: `followups:` is the field the next session greps. A frontmatter that fails to parse makes every
followup invisible while the file still LOOKS complete -- the same shape as a check that cannot fail.
PyYAML is not installed in this venv or in system python, so this validates structurally instead of
by parsing: delimiters, required keys, and that every followup is a single fully-quoted scalar.

Usage: <venv-python> helpers/check_session_frontmatter.py <path-to-session-log.md>
"""

from __future__ import annotations

import sys
from pathlib import Path

REQUIRED = ("date", "started_at", "slug", "status")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    path = Path(sys.argv[1])
    lines = path.read_text().splitlines()

    if not lines or lines[0].strip() != "---":
        print("FAIL: file does not open with '---'")
        return 1
    try:
        end = next(i for i, l in enumerate(lines[1:], start=1) if l.strip() == "---")
    except StopIteration:
        print("FAIL: frontmatter is never closed with '---'")
        return 1

    fm = lines[1:end]
    print(f"frontmatter: lines 2..{end} ({len(fm)} lines)")

    keys = [l.split(":", 1)[0] for l in fm if l and not l.startswith((" ", "-", "\t"))]
    missing = [k for k in REQUIRED if k not in keys]
    if missing:
        print(f"FAIL: missing required key(s): {missing}")
        return 1
    print(f"top-level keys: {keys}")

    if "followups" not in keys:
        print("WARN: no followups key")
        return 0

    start = next(i for i, l in enumerate(fm) if l.startswith("followups:"))
    items, bad = 0, []
    for l in fm[start + 1:]:
        if l and not l.startswith((" ", "\t")):
            break                      # next top-level key
        s = l.strip()
        if not s:
            continue
        if not s.startswith("- "):
            bad.append(("continuation line inside followups (breaks the list)", l[:70]))
            continue
        items += 1
        v = s[2:].strip()
        if not (v.startswith('"') and v.endswith('"') and len(v) >= 2):
            bad.append(("item is not a single fully double-quoted scalar", v[:70]))
        elif '"' in v[1:-1]:
            bad.append(("unescaped inner double quote", v[:70]))

    print(f"followups: {items} items")
    if bad:
        print(f"\nFAIL ({len(bad)}):")
        for why, frag in bad:
            print(f"  {why}: {frag}")
        return 1
    print("\nOK: frontmatter is structurally sound and every followup is one quoted scalar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
