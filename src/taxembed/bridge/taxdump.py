"""Hand-rolled NCBI taxdump resolver (no taxopy dependency; proven sufficient, spec §7).

Parses nodes.dmp (taxid->parent,rank), names.dmp (scientific-name<->taxid), merged.dmp
(old->new taxid). Fields are '\t|\t'-separated, rows end '\t|'.
"""
from __future__ import annotations
from pathlib import Path


class TaxonResolver:
    def __init__(self, dump_dir: Path | str):
        d = Path(dump_dir)
        self.parent: dict[int, int] = {}
        self.rank: dict[int, str] = {}
        self.name: dict[int, str] = {}          # taxid -> scientific name
        self._name2taxid: dict[str, int] = {}
        self.merged: dict[int, int] = {}
        for line in (d / "nodes.dmp").read_text().splitlines():
            f = [c.strip() for c in line.split("\t|")]
            tid, par, rk = int(f[0]), int(f[1]), f[2]
            self.parent[tid] = par
            self.rank[tid] = rk
        for line in (d / "names.dmp").read_text().splitlines():
            f = [c.strip() for c in line.split("\t|")]
            tid, nm, cls = int(f[0]), f[1], f[3]
            if cls == "scientific name":
                self.name[tid] = nm
                self._name2taxid[nm] = tid
        mp = d / "merged.dmp"
        if mp.exists():
            for line in mp.read_text().splitlines():
                f = [c.strip() for c in line.split("\t|")]
                self.merged[int(f[0])] = int(f[1])

    def name_to_taxid(self, scientific_name: str) -> int | None:
        return self._name2taxid.get(scientific_name)

    def canonical(self, taxid: int) -> int:
        return self.merged.get(taxid, taxid)

    def lineage(self, taxid: int) -> list[tuple[str, int, str]]:
        """Root-ward chain [(rank, taxid, name), ...] from `taxid` up to the root."""
        out, seen, t = [], set(), self.canonical(taxid)
        while t in self.parent and t not in seen:
            seen.add(t)
            out.append((self.rank.get(t, "no rank"), t, self.name.get(t, "")))
            if self.parent[t] == t:
                break
            t = self.parent[t]
        return out

    def species_parent(self, taxid: int) -> int | None:
        """Walk root-ward to the `species`-rank ancestor taxid; None if no species in lineage."""
        for rank, tid, _name in self.lineage(taxid):
            if rank == "species":
                return tid
        return None

    def resolve(self, species_name: str, aliases: dict[str, str] | None = None) -> int | None:
        nm = (aliases or {}).get(species_name, species_name)
        tid = self.name_to_taxid(nm)
        return self.canonical(tid) if tid is not None else None
