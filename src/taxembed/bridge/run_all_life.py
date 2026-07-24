"""All-life Phase-A+B orchestrator (spec v3 §4–§6) — runs on the GPU cluster LOGIN NODE.

Rationale (discovered against the live GPU cluster): compute nodes have NO internet, and the resolve needs the
2.5 GB cellular bridge + taxdump, so the internet-facing download + the resolve + the light pandas
gate-scan/cap all run here on the login node. Only the GPU self-embed (J1) and the memory-heavy CLEAN
battery (J2) go to SLURM. This chains the already-unit-tested components:

  read raw metadata shards -> filter_single_pfam (stream) -> resolve_taxids (cellular) +
  resolution-rate floor + merged.dmp redirect/subset audits -> attach_ranks (superkingdom) +
  span-denominator floor -> AUTO breadth-rank (pre-stated rule) -> gate_scan (kingdom-span) ->
  apply_family_cap -> freeze Pfam+EC panels (+ source col) + bare-accession FASTA + self-embed manifest.

Everything is dependency-injected (emb, resolver, survivors, seq_lookup) so the chain is Mac-smoke-tested
on a tiny fixture; the real cellular emb/resolver + the 112M raw shards run on GPU cluster.
"""
import json
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent

from . import config  # noqa: E402
from . import acquire_all_life as A  # noqa: E402
from . import resolve_all_life as R  # noqa: E402
from . import twin_batch_eval as TB  # noqa: E402
from . import build_sp_panel as BP  # noqa: E402


def survivors_frame(raw_lines_iter):
    """Stream raw metadata rows through the single-Pfam filter into a survivor DataFrame."""
    recs = list(A.filter_single_pfam(raw_lines_iter))
    if not recs:
        return pd.DataFrame(columns=["accession", "taxid", "pfam_family", "ec", "length", "reviewed", "source"])
    df = pd.DataFrame(recs)
    df = df.rename(columns={"organism_id": "taxid", "pfam": "pfam_family"})
    df["taxid"] = df["taxid"].astype(int)
    return df[["accession", "taxid", "pfam_family", "ec", "length", "reviewed", "source"]]


def choose_breadth_rank(frame):
    """Decision #1 — apply the PRE-STATED rule (MEASURE_THEN_FREEZE.md) deterministically from the
    realized per-superkingdom non-null tax_order fraction: keep 'tax_order' iff >= 0.70 in BOTH
    Bacteria AND Archaea, else 'tax_phylum'. (p-hack-safe: rule frozen before the data was seen.)"""
    fr = {}
    for sk in ("Bacteria", "Archaea"):
        sub = frame[frame["tax_superkingdom"] == sk]
        fr[sk] = (sub["tax_order"].notna().mean() if len(sub) else 0.0)
    rank = "tax_order" if (fr.get("Bacteria", 0) >= 0.70 and fr.get("Archaea", 0) >= 0.70) else "tax_phylum"
    return rank, fr


def phase_a(raw_lines_iter, emb, resolver, *, merged_dmp=None, node_set=None,
            resolution_floor=None, redirect_bound=None):
    """Filter -> resolve (+ floor + audits) -> attach ranks (+ span floor) -> auto breadth-rank ->
    gate_scan. Returns (frame, scan, diagnostics). `frame` carries idx/resolved_taxid/ranks/source."""
    floor = config.SP_RESOLUTION_RATE_MIN if resolution_floor is None else resolution_floor
    surv = survivors_frame(raw_lines_iter)

    # taxid -> cellular idx (direct/merged/species-parent), then the GO/NO-GO resolution-rate floor.
    resolved, dropped = BP.resolve_taxids(surv, emb, resolver)
    uniq = surv["taxid"].nunique()
    rate = (resolved["resolved_taxid"].nunique() / uniq) if uniq else 0.0
    if rate < floor:
        raise SystemExit(f"[all_life] NO-GO: resolution rate {rate:.3f} < {floor} (spec §4.3) — "
                         "escalate (claim-scope change); the floor is NOT lowered to rescue the run")

    # merged.dmp redirect audit + mapping-subset audit (Mac-tested set arithmetic), when provided.
    if merged_dmp is not None and node_set is not None:
        R.redirect_audit(merged_dmp, node_set,
                         config.TAXDUMP_REDIRECT_ABSENT_MAX if redirect_bound is None else redirect_bound)
        R.subset_audit(set(resolved["resolved_taxid"]), node_set)

    frame = BP.attach_ranks(resolved, resolver)
    TB.assert_span_denominator(frame["tax_superkingdom"].tolist(), config.ALL_LIFE_SPAN_DENOM_MIN)

    order_col, frac = choose_breadth_rank(frame)
    scan = BP._gate_scan_for(frame, "pfam_family", order_col=order_col, span_superkingdoms=True)
    diagnostics = {"n_survivors": len(surv), "n_resolved": len(resolved), "resolution_rate": round(rate, 4),
                   "breadth_rank": order_col, "breadth_fraction": frac,
                   "per_superkingdom_retention": TB.per_superkingdom_retention(frame["tax_superkingdom"].tolist()),
                   "n_crossed_families": int(scan["keep"].sum()) if len(scan) else 0}
    return frame, scan, diagnostics


def phase_b(frame, scan, seq_lookup, *, target="cellular", emb=None, freeze=False, fasta_path=None):
    """Cap the gate-passing members; optionally freeze the Pfam panel TSV + bare-accession FASTA.
    `seq_lookup` maps accession->seq (needed for mmseqs90 clustering + aa-composition + FASTA when
    freeze=True). Returns (capped_frame, panel_or_None). The self-embed manifest is built POST-embed
    (it needs the merged h5), not here."""
    kept = set(scan.loc[scan["keep"], "pfam_family"]) if len(scan) else set()
    gp = frame[frame["pfam_family"].isin(kept)].copy()
    capped = BP.apply_family_cap(gp)
    if not freeze:
        return capped, None
    accs = capped["accession"].tolist()
    BP.freeze_cluster_ids(accs, [seq_lookup[a] for a in accs],          # mmseqs90 leak-guard cluster ids
                          str(config.SP_CLUSTER_IDS), str(config.SP_CLUSTER_IDS_MANIFEST))
    cdf = pd.read_csv(config.SP_CLUSTER_IDS, sep="\t")
    cluster_map = dict(zip(cdf["accession"], cdf["cluster_id"]))
    panel = BP._freeze_panel(capped, "pfam_family", {}, cluster_map, seq_lookup,
                             str(config.ALL_LIFE_PANEL), target=target, emb=emb)
    fasta = fasta_path or str(config.DATA / "all_life_survivors.fasta")
    BP.write_panel_fasta(panel, seq_lookup, fasta)
    return capped, panel


def _iter_raw_shards(raw_dir):
    import glob
    import gzip
    import os
    for path in sorted(glob.glob(os.path.join(raw_dir, "snapshot.*.tsv.gz"))):
        with gzip.open(path, "rt") as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue                                  # UniProt display-name header row
                yield line.rstrip("\n")


def _cmd_phase_a(args):
    """CONTAINER: bridge + filter+resolve+rank+gate+cap → write the capped frame (TSV, all columns) +
    the capped accession list + meta (n_nodes, breadth_rank, diagnostics). No internet needed."""
    emb = BP._load_emb("cellular")
    resolver = BP._load_resolver()
    print(f"[all_life] cellular bridge n_nodes={emb.n_nodes}", flush=True)
    frame, scan, diag = phase_a(_iter_raw_shards(args.raw_dir), emb, resolver, resolution_floor=args.floor)
    diag["n_gate_scanned_families"] = int(len(scan))
    capped, _ = phase_b(frame, scan, seq_lookup={}, freeze=False)       # cap (≤ TAXEMBED_EMBED_CEILING)
    diag["n_capped"] = int(len(capped))
    diag["n_nodes"] = int(emb.n_nodes)
    capped.to_csv(args.frame_out, sep="\t", index=False)
    with open(args.accs_out, "w") as f:
        f.write("\n".join(capped["accession"].tolist()))
    json.dump(diag, open(args.out_json, "w"), indent=2)
    print("PHASE_A_DIAGNOSTICS " + json.dumps(diag), flush=True)
    print(f"PHASE_A_DONE n_capped={len(capped)} n_crossed_families={diag['n_crossed_families']} "
          f"resolution_rate={diag['resolution_rate']} breadth_rank={diag['breadth_rank']}", flush=True)


def _cmd_phase_b(args):
    """CONTAINER: read the capped frame + the fetched FASTA → mmseqs90 cluster + freeze the Pfam panel
    + bare-accession FASTA + target stamp. Needs the bridge only for the n_nodes stamp."""
    capped = pd.read_csv(args.frame_in, sep="\t", dtype={"ec": str}, keep_default_na=False)
    capped = capped.replace({"": None})
    seq_lookup = {sid: seq for sid, seq in _iter_fasta(args.fasta)}
    capped = capped[capped["accession"].isin(seq_lookup)].reset_index(drop=True)   # only embeddable survivors
    emb = BP._load_emb("cellular")
    BP.freeze_cluster_ids(capped["accession"].tolist(), [seq_lookup[a] for a in capped["accession"]],
                          str(config.SP_CLUSTER_IDS), str(config.SP_CLUSTER_IDS_MANIFEST))
    cdf = pd.read_csv(config.SP_CLUSTER_IDS, sep="\t")
    cluster_map = dict(zip(cdf["accession"], cdf["cluster_id"]))
    panel = BP._freeze_panel(capped, "pfam_family", {}, cluster_map, seq_lookup,
                             str(config.ALL_LIFE_PANEL), target="cellular", emb=emb)
    BP.write_panel_fasta(panel, seq_lookup, args.fasta_out)
    print(f"PHASE_B_DONE n_panel={len(panel)} panel={config.ALL_LIFE_PANEL} fasta={args.fasta_out}", flush=True)


def main():
    """GPU cluster all-life Phase-A/B driver. Phases split for the container(bridge)/login(internet) boundary."""
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("phase-a")
    a.add_argument("--raw-dir", required=True)
    a.add_argument("--frame-out", required=True)
    a.add_argument("--accs-out", required=True)
    a.add_argument("--out-json", required=True)
    a.add_argument("--floor", type=float, default=0.0)
    b = sub.add_parser("phase-b")
    b.add_argument("--frame-in", required=True)
    b.add_argument("--fasta", required=True)        # the login-fetched survivor sequences
    b.add_argument("--fasta-out", required=True)    # the bare-accession panel FASTA J1 embeds
    args = ap.parse_args()
    (_cmd_phase_a if args.cmd == "phase-a" else _cmd_phase_b)(args)


def _iter_fasta(path):
    sid, seq = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if sid is not None:
                    yield sid, "".join(seq)
                sid = line[1:].split()[0]
                seq = []
            else:
                seq.append(line.strip())
    if sid is not None:
        yield sid, "".join(seq)


if __name__ == "__main__":
    main()
