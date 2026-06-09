"""Leg-B release-diff core (spec §9B/§9C): canonicalize two taxdump releases via merged.dmp +
delnodes.dmp, then define 'reclassification' = direct-parent change after canonicalization,
EXCLUDING pure ID-merges and rank-only changes. No network here (the CLI does the fetch).
"""
from __future__ import annotations

from pathlib import Path


def _rows(dmp_path: Path):
    with open(dmp_path) as fh:
        for line in fh:
            yield [c.strip() for c in line.rstrip("\n").rstrip("|").split("|")]


def parse_merged(merged_path) -> dict:
    """merged.dmp: 'old_taxid | new_taxid |' -> {old: new}."""
    out = {}
    for parts in _rows(Path(merged_path)):
        if len(parts) >= 2 and parts[0] and parts[1]:
            out[int(parts[0])] = int(parts[1])
    return out


def parse_delnodes(delnodes_path) -> set:
    """delnodes.dmp: 'taxid |' -> {taxid, ...} of deleted ids."""
    out = set()
    for parts in _rows(Path(delnodes_path)):
        if parts and parts[0]:
            out.add(int(parts[0]))
    return out


def parse_parents(nodes_path) -> dict:
    """nodes.dmp tab-pipe rows -> {taxid: parent_taxid}. Root points to itself."""
    out = {}
    with open(nodes_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").rstrip("|").split("\t|\t")
            if len(parts) < 2:
                continue
            out[int(parts[0].strip())] = int(parts[1].strip())
    return out


def canonicalize_taxid(taxid: int, merged: dict, delnodes: set, max_hops: int = 64):
    """Follow the merge chain to the surviving id; return None if (eventually) deleted."""
    cur = int(taxid)
    if cur in delnodes:
        return None
    hops = 0
    while cur in merged and hops < max_hops:
        cur = merged[cur]
        hops += 1
    return None if cur in delnodes else cur


def reclassified_taxa(scored_taxids, old_parent: dict, new_parent: dict,
                      new_merged: dict, new_delnodes: set) -> set:
    """Set of scored taxids whose DIRECT PARENT changed old->new after canonicalization.

    Excludes: pure ID-merges (taxid itself merged away), deletions, and taxa absent from either
    release. Parent identities are compared in the NEW release's canonical id space.
    """
    out = set()
    for t in scored_taxids:
        if t not in old_parent:
            continue
        t_canon = canonicalize_taxid(t, new_merged, new_delnodes)
        if t_canon is None or t_canon != t:
            continue
        if t not in new_parent:
            continue
        op = canonicalize_taxid(old_parent[t], new_merged, new_delnodes)
        npar = new_parent[t]
        if op is None:
            continue
        if op != npar:
            out.add(t)
    return out
