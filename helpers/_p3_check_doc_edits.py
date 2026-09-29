#!/usr/bin/env python3
"""Verify today's edits to the two TaxEmbed docs actually landed as structure, not as body text.

Two things a markdown edit can silently get wrong, both with real history here:
  - a heading with no blank line before it is NOT a heading
    ([[feedback_a_heading_without_a_blank_line_is_not_a_heading]]);
  - a claimed cross-reference (§4.7, C10, a results path) can point at nothing.

Checks structure and every anchor this session introduced, then asserts that the files still
contain the passages that were meant to be PRESERVED rather than overwritten (Rule 5/Rule 7:
superseded text is kept, not deleted).

Written 2026-09-29.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path("/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings")
FINDING = ROOT / "docs" / "FINDING_protocols_that_are_vacuous_on_taxonomy_trees.md"
CORR = ROOT / "docs" / "MANUSCRIPT_CORRECTIONS_PENDING.md"

REQUIRED_HEADINGS = {
    FINDING: [
        "## Finding 4 — outside-the-tree information is NECESSARY BUT NOT SUFFICIENT: "
        "the placement test measured subtree size",
        "### 4.1 The protocol, and what it was built to escape",
        "### 4.2 The prescribed control could not have failed",
        "### 4.3 The moved taxon is inert — P3 was never a held-out-taxon test",
        "### 4.4 A one-line tree statistic beats the embedding outright",
        "### 4.5 Conditioned on subtree size, nothing remains",
        "### 4.6 The gate that reversed the verdict — a method lesson worth more than the result",
        "### 4.7 The general statement: §3.7 is necessary but not sufficient",
    ],
    CORR: [
        "## C10 — the P3 placement arm measured subtree size; `ANTICIPATES` is withdrawn",
    ],
}

# text that MUST still be present -- superseded material is preserved, never deleted
MUST_PRESERVE = {
    FINDING: [
        "needs information from outside the tree — later NCBI releases (spec §P3)",
        "Root edges do not exist, so the three cases are exhaustive over node types.",
    ],
    CORR: [
        "*(superseded)* **What the manuscript should say",
        "## C8 — P2's held-out protocol",
    ],
}

# numbers this session put into prose; each must appear verbatim
MUST_CONTAIN = {
    FINDING: ["1,241 / 1,241 pools (100.00 %)", "−0.0044", "0.3146", "+0.1298",
              "94 % / 103 % / 113 %", "+0.0122", "166,852"],
    CORR: ["0.3146", "+0.1298", "94 % / 103 % / 113 %", "p3_amendment_3_20260929",
           "p3_placement_result_v2_20260929.json"],
}


def main() -> int:
    bad = 0
    for path in (FINDING, CORR):
        text = path.read_text()
        lines = text.split("\n")
        print(f"\n=== {path.name} ({len(lines):,} lines) ===")

        # every ATX heading must be preceded by a blank line (or be line 1)
        for i, ln in enumerate(lines):
            if re.match(r"^#{1,6} ", ln) and i > 0 and lines[i - 1].strip() != "":
                print(f"  ✗ heading not preceded by a blank line, L{i+1}: {ln[:70]}")
                bad += 1

        for h in REQUIRED_HEADINGS[path]:
            ok = any(ln.rstrip() == h for ln in lines)
            print(f"  {'✓' if ok else '✗'} heading present: {h[:72]}")
            bad += 0 if ok else 1

        for s in MUST_PRESERVE[path]:
            ok = s in text
            print(f"  {'✓' if ok else '✗'} PRESERVED: {s[:72]}")
            bad += 0 if ok else 1

        for s in MUST_CONTAIN[path]:
            ok = s in text
            print(f"  {'✓' if ok else '✗'} number/anchor: {s}")
            bad += 0 if ok else 1

    # every results/ path named in either doc must exist on disk
    print("\n=== referenced results/ and helpers/ paths exist ===")
    both = FINDING.read_text() + CORR.read_text()
    for rel in sorted(set(re.findall(r"(?:results|helpers)/[A-Za-z0-9_./-]+\.(?:json|py)", both))):
        ok = (ROOT / rel).exists()
        print(f"  {'✓' if ok else '✗'} {rel}")
        bad += 0 if ok else 1

    print(f"\n{'ALL CHECKS PASS' if bad == 0 else f'{bad} PROBLEM(S)'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
