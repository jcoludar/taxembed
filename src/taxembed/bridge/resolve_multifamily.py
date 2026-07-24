"""Task 12 — freeze the multi-family (ToxFam) species resolution table (M1c).

For each ToxFam accession (84 bare UniProt accessions, headers in toxfam_v2.fasta / rows in
toxfam_v2_labels.csv):
  accession --[UniProt REST batch]--> organism taxid --[TaxonomyEmbedding.idx_of_taxid]--> metazoa idx.

Writes data/multifamily_species_resolution.tsv with columns
  identifier, family, taxid, idx, organism, via
where via = direct (taxid resolved into the 498k metazoa embedding),
            unresolved_taxid (UniProt returned no organism_id),
            not_in_embedding (taxid not present in the metazoa Poincaré embedding).

Unresolved rows are SURFACED (never silently dropped); the testbed downstream RESTRICTS to via==direct.
The metazoa embedding only covers Metazoa (33208), so non-metazoan venomous taxa (e.g. cone snails are
metazoan, but some bacterial/plant/fungal toxins or taxa absent from the 498k node set) will land as
not_in_embedding — reported with the n.

Run once; output is a committed artifact:
  python -m taxembed.bridge.resolve_multifamily
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent

from . import config  # noqa: E402
from .core import TaxonomyEmbedding  # noqa: E402

OUT = config.DATA / "multifamily_species_resolution.tsv"
UNIPROT_BATCH = "https://rest.uniprot.org/uniprotkb/accessions"
CHUNK = 100  # accessions per REST call (endpoint accepts up to a few hundred)


def _fetch_chunk(accessions: list[str]) -> dict[str, dict]:
    """Return {accession: {'organism_id': int|None, 'organism_name': str}} for one batch."""
    params = {
        "accessions": ",".join(accessions),
        "fields": "accession,organism_id,organism_name",
        "format": "tsv",
    }
    url = UNIPROT_BATCH + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "tax-disentangle/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        text = resp.read().decode("utf-8")
    out: dict[str, dict] = {}
    lines = text.strip().splitlines()
    if not lines:
        return out
    header = lines[0].split("\t")
    # tsv columns: Entry, Organism (ID), Organism
    try:
        i_acc = header.index("Entry")
        i_oid = header.index("Organism (ID)")
        i_onm = header.index("Organism")
    except ValueError:
        raise SystemExit(f"unexpected UniProt tsv header: {header}")
    for ln in lines[1:]:
        cells = ln.split("\t")
        acc = cells[i_acc]
        oid = cells[i_oid].strip() if i_oid < len(cells) else ""
        onm = cells[i_onm].strip() if i_onm < len(cells) else ""
        out[acc] = {"organism_id": int(oid) if oid.isdigit() else None, "organism_name": onm}
    return out


def main():
    labels = pd.read_csv(config.TOXFAM_LABELS)
    accs = labels["identifier"].tolist()
    print(f"[resolve_mf] {len(accs)} accessions, {labels['family'].nunique()} families")

    # --- batch UniProt: accession -> taxid ---
    acc2org: dict[str, dict] = {}
    for i in range(0, len(accs), CHUNK):
        chunk = accs[i:i + CHUNK]
        got = _fetch_chunk(chunk)
        acc2org.update(got)
        print(f"[resolve_mf] UniProt batch {i}-{i+len(chunk)}: {len(got)}/{len(chunk)} returned")
        time.sleep(0.5)

    missing_from_uniprot = [a for a in accs if a not in acc2org]
    if missing_from_uniprot:
        print(f"[resolve_mf] WARNING {len(missing_from_uniprot)} accessions absent from UniProt response: "
              f"{missing_from_uniprot}")

    # --- taxid -> metazoa embedding idx ---
    te = TaxonomyEmbedding(config.CKPT, config.TAXMAP, config.EDGELIST)

    rows, n_direct, n_unres_taxid, n_not_in_emb = [], 0, 0, 0
    for _, r in labels.iterrows():
        acc = r["identifier"]
        fam = r["family"]
        org = acc2org.get(acc, {})
        taxid = org.get("organism_id")
        organism = org.get("organism_name", "")
        if taxid is None:
            via, idx = "unresolved_taxid", None
            n_unres_taxid += 1
        else:
            idx = te.idx_of_taxid(taxid)
            if idx is None:
                via, n_not_in_emb = "not_in_embedding", n_not_in_emb + 1
            else:
                via, n_direct = "direct", n_direct + 1
        rows.append({
            "identifier": acc, "family": fam,
            "taxid": taxid if taxid is not None else "NA",
            "idx": idx if idx is not None else "NA",
            "organism": organism, "via": via,
        })

    df = pd.DataFrame(rows, columns=["identifier", "family", "taxid", "idx", "organism", "via"])
    df.to_csv(OUT, sep="\t", index=False)

    print(f"\n[resolve_mf] wrote {OUT}")
    print(f"[resolve_mf] direct (resolved into metazoa embedding): {n_direct}/{len(accs)}")
    print(f"[resolve_mf] unresolved_taxid (no UniProt organism_id): {n_unres_taxid}")
    print(f"[resolve_mf] not_in_embedding (taxid absent from metazoa): {n_not_in_emb}")
    print(f"[resolve_mf] families with >=1 resolved: "
          f"{df[df['via']=='direct']['family'].nunique()} of {labels['family'].nunique()}")
    # surface the unresolved explicitly
    for _, r in df[df["via"] != "direct"].iterrows():
        print(f"  UNRESOLVED [{r['via']}] {r['identifier']} ({r['family']}) "
              f"taxid={r['taxid']} organism={r['organism']!r}")


if __name__ == "__main__":
    main()
