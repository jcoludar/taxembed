"""Deterministic SP-Metazoa Pfam panel build (spec 2026-06-17 v3)."""
from __future__ import annotations
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent

from . import config  # noqa: E402

_COL = {"Entry":"accession","Organism (ID)":"taxid","Organism":"organism_name",
        "Length":"length","Pfam":"pfam","EC number":"ec",
        "Protein names":"protein_name","Keywords":"keyword"}

def _split_field(val: str) -> tuple:
    if not isinstance(val, str) or not val.strip():
        return ()
    return tuple(sorted({t for t in (s.strip() for s in val.split(";")) if t}))

def parse_annotations(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns=_COL).copy()
    df["taxid"] = df["taxid"].astype(int)
    df["length"] = df["length"].astype(int)
    df["pfam_set"] = df["pfam"].apply(_split_field)
    df["ec_set"] = df["ec"].apply(_split_field)
    return df[["accession","taxid","organism_name","length","pfam_set","ec_set","protein_name","keyword"]]

def filter_single_family(df):
    mask = df["pfam_set"].apply(lambda s: len(s) == 1)
    out = df.loc[mask].copy()
    out["pfam_family"] = out["pfam_set"].apply(lambda s: s[0])
    return out

def filter_coverage(df, min_cov):
    """Pfam domain coverage gate (spec §4.3). NaN domain_aa -> reported no-op (stage-1 default)."""
    d = df.copy()
    if d["domain_aa"].isna().all():
        d["coverage"] = float("nan")
        return d
    d["coverage"] = d["domain_aa"] / d["length"]
    return d.loc[d["coverage"].isna() | (d["coverage"] >= min_cov)].copy()

def resolve_taxids(df, emb, resolver):
    """taxid -> embedding idx (spec §4.4): direct -> merged -> species-parent; else drop."""
    rows, dropped = [], []
    for r in df.itertuples(index=False):
        t = int(r.taxid)
        idx = emb.idx_of_taxid(t)
        if idx is not None: rows.append((*r, t, idx, "direct", 0)); continue
        c = resolver.canonical(t); idx = emb.idx_of_taxid(c)
        if idx is not None: rows.append((*r, c, idx, "merged", 0)); continue
        sp = resolver.species_parent(t); idx = emb.idx_of_taxid(sp) if sp is not None else None
        if idx is not None: rows.append((*r, sp, idx, "subspecies_parent", 1)); continue
        dropped.append(r)
    cols = list(df.columns) + ["resolved_taxid","idx","via","rank_gap"]
    return pd.DataFrame(rows, columns=cols), pd.DataFrame(dropped, columns=df.columns)

def attach_ranks(df, resolver):
    """Attach tax_superkingdom/class/order/phylum names; None where the rank node is absent
    (spec §4.6 + v3 §4.2.1). superkingdom is REQUIRED for the all-life kingdom-span gate; null-
    superkingdom rows are excluded from the span numerator downstream (gate_scan)."""
    def _names(t):
        by = {rank: name for rank, _tid, name in resolver.lineage(int(t))}
        # NCBI renamed the top rank `superkingdom`→`domain` (2024); the real 2026 dump uses `domain`.
        # Prefer `domain`, fall back to `superkingdom` for older dumps / Stage-1 fixtures.
        sk = by.get("domain") or by.get("superkingdom")
        return sk, by.get("class"), by.get("order"), by.get("phylum")
    tr = df["resolved_taxid"].apply(_names)
    out = df.copy()
    out["tax_superkingdom"] = [a for a, _, _, _ in tr]
    out["tax_class"] = [b for _, b, _, _ in tr]; out["tax_order"] = [c for _, _, c, _ in tr]
    out["tax_phylum"] = [d for _, _, _, d in tr]
    return out

def _entropy(counts):
    """Shannon entropy (nats) of a count/weight vector; zeros ignored; empty -> 0."""
    c = np.asarray([x for x in counts if x > 0], dtype=float)
    if c.size == 0:
        return 0.0
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())

def effective_orders(orders):
    """exp(H(order)) for one family's member orders — balance-sensitive order count.
    Balanced k orders -> k; fully skewed -> ~1; empty -> 0."""
    s = pd.Series(list(orders)).dropna()
    if s.empty:
        return 0.0
    return float(np.exp(_entropy(s.value_counts().to_numpy())))

def _cond_reduction(cond_labels, target_labels):
    """1 - H(target|cond)/H(target): fraction of target uncertainty explained by cond.
    Perfectly nested (cond -> one target) -> 1; independent -> 0."""
    cond = pd.Series(list(cond_labels)).to_numpy()
    tgt = pd.Series(list(target_labels)).to_numpy()
    _, tcounts = np.unique(tgt, return_counts=True)
    H_t = _entropy(tcounts)
    if H_t <= 0:
        return 0.0
    n = len(tgt); H_cond = 0.0
    for cv in np.unique(cond):
        m = cond == cv
        _, c = np.unique(tgt[m], return_counts=True)
        H_cond += (m.sum() / n) * _entropy(c)
    return float(1.0 - H_cond / H_t)

def panel_nesting(func_labels, tax_labels):
    """Bidirectional nesting (spec §4.7): both 1-H(tax|func)/H(tax) and 1-H(func|tax)/H(func)."""
    return {"tax_given_func": _cond_reduction(func_labels, tax_labels),
            "func_given_tax": _cond_reduction(tax_labels, func_labels)}

def gate_scan(df, member_floor, min_orders, min_eff_orders,
              order_col="tax_order", family_col="pfam_family",
              span_superkingdoms=False, min_superkingdoms=2):
    """Per-family crossing stats + keep decision (spec §4.7/§6.1 + v3 §4.2.3 kingdom-span). Sorted by
    family (deterministic). Returns ONE row per family with a `keep` flag — it does NOT filter.
    When `span_superkingdoms`, a family must ALSO span >= `min_superkingdoms` of {Bacteria, Archaea,
    Eukaryota} (null-superkingdom rows excluded from the span count, v3 §4.2.1). Default off keeps the
    Stage-1 metazoa behaviour byte-identical."""
    cols = [family_col, "n_members", "raw_orders", "effective_orders", "n_superkingdoms", "keep"]
    if len(df) == 0:                                  # NIT-1: a zero-family run is a clean negative, not a crash
        return pd.DataFrame(columns=cols)
    rows = []
    for fam, g in df.groupby(family_col, sort=True):
        orders = g[order_col].dropna()
        n_members, raw_orders, eff = len(g), int(orders.nunique()), effective_orders(orders)
        n_kingdoms = int(g["tax_superkingdom"].dropna().nunique()) if "tax_superkingdom" in g else 0
        keep = (n_members >= member_floor) and (raw_orders >= min_orders) and (eff >= min_eff_orders)
        if span_superkingdoms:
            keep = keep and (n_kingdoms >= min_superkingdoms)
        rows.append({family_col: fam, "n_members": n_members, "raw_orders": raw_orders,
                     "effective_orders": eff, "n_superkingdoms": n_kingdoms, "keep": bool(keep)})
    return pd.DataFrame(rows).sort_values(family_col).reset_index(drop=True)

def within_family_clade_mi(reps, clades, n_clusters=None, seed=0):
    """MI(within-family embedding cluster, clade) in nats (spec §4.7a; REPORTED, non-gating).
    n_clusters defaults to #distinct clades, capped at n_members-1."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import mutual_info_score
    import numpy as np
    clades = list(clades); n = len(clades)
    uniq = sorted(set(clades))
    k = n_clusters or len(uniq)
    k = max(1, min(k, n - 1))
    if k < 2 or len(uniq) < 2:
        return 0.0
    lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(np.asarray(reps, float))
    return float(mutual_info_score(clades, lab))   # nats

def panel_acceptance(n_crossed_families, nesting, min_families, nesting_max):
    """Panel-level accept gate (spec §6.1): enough crossed families AND both-direction nesting low."""
    enough = n_crossed_families >= min_families
    both_low = (nesting["tax_given_func"] < nesting_max) and (nesting["func_given_tax"] < nesting_max)
    return {"accepted": bool(enough and both_low), "enough_families": bool(enough),
            "nesting_ok": bool(both_low), "n_crossed_families": int(n_crossed_families)}

def apply_exclusions(panel, exclusions):
    """Deterministic apply (spec §4.9): drop families with action=='drop'; stable sort for byte-identity."""
    drop = set(exclusions.loc[exclusions["action"] == "drop", "pfam_family"])
    out = panel.loc[~panel["pfam_family"].isin(drop)].copy()
    return out.sort_values(["pfam_family", "accession"]).reset_index(drop=True)


import os, gzip, hashlib, time, urllib.request, urllib.parse, numpy as np, h5py

def collapse_ec(ec, level):
    spec = []
    for p in ec.split("."):
        if p in ("-", ""): break
        spec.append(p)
    return ".".join(spec[:level]) if len(spec) >= level else None


def nmi_pfam_ec(pfam_labels, ec_labels):
    """Normalized mutual information between the Pfam-family and EC labels (spec §6.4).

    TESTED. The Pfam panel and EC panel are an *independent* function contrast only if the
    two label sets are not redundant. NMI=1 -> a bijection (the EC contrast is the Pfam
    contrast relabelled, i.e. redundant); NMI~0 -> the two label families vary independently.
    The caller flags the contrast redundant when NMI >= config.SP_NMI_MAX.

    Both arguments are aligned label sequences over the SAME rows (one Pfam label and one EC
    label per protein). Uses sklearn's arithmetic-mean normalization so the score lands in
    [0, 1] regardless of the (possibly very different) cardinalities of the two label sets.
    """
    from sklearn.metrics import normalized_mutual_info_score
    return float(normalized_mutual_info_score(
        list(pfam_labels), list(ec_labels), average_method="arithmetic"))


def build_ec_panel(resolved, cluster_map, seq_lookup, config, ec_exclusions=None,
                   target="metazoa", emb=None):
    """Build the INDEPENDENT EC contrast (spec §5; reconciled with `main()` 2026-06-18).

    The EC contrast labels enzymes by 3rd-level EC instead of Pfam family. It is built from the SAME
    resolved-and-ranked frame the Pfam pipeline produces, so the two panels share row identity /
    taxonomy resolution and the only thing that changes is the FUNCTION label. To guarantee the Pfam
    and EC panels CANNOT DRIFT, both freeze through the shared `_freeze_panel` helper (same column
    schema incl. `cluster_id` + the `aa_A..aa_Y` composition columns the C2c gate requires).

    Steps (spec §4/§5 order, EC variant):
      1. `_ec_relabel`: collapse each row's first EC to `config.SP_EC_LEVEL` (e.g. "3.4.24.1" ->
         "3.4.24") into a fresh `ec` column; DROP rows with no EC3. `clean_eval.load_sp_clean` reads
         the EC testbed off this `ec` column (renamed to `func`).
      2. Crossing gate-scan over the EC3 label at the SAME pre-committed thresholds, grouping by `ec`.
      3. Freeze BOTH the post-exclusion EC panel (`SP_METAZOA_EC_PANEL`) AND its zero-exclusion full
         twin (`SP_METAZOA_EC_PANEL_FULL`, B2 twin-run) via `_freeze_panel`.

    `resolved` MUST already carry `tax_order`/`idx`/`ec_set` (from the §4 pipeline). `cluster_map`
    (accession->cluster_id, frozen by `freeze_cluster_ids`) and `seq_lookup` (accession->sequence, for
    the aa composition) are produced once in `main()` and shared so the EC and Pfam panels are built
    from identical inputs.

    When `emb` is provided, a target stamp sidecar is written beside each frozen EC panel (spec v3 §2.N1).

    Returns (panel, panel_full, scan, acceptance, nesting).
    """
    ec_frame, _ec_report = _ec_relabel(resolved)
    scan = _gate_scan_for(ec_frame, "ec")
    kept = set(scan.loc[scan["keep"], "ec"])
    gp = ec_frame.loc[ec_frame["ec"].isin(kept)]
    nesting, acceptance, _gp = _panel_level_stats(ec_frame, "ec", kept)
    panel = _freeze_panel(gp, "ec", ec_exclusions, cluster_map, seq_lookup, config.SP_METAZOA_EC_PANEL,
                          target=target, emb=emb)
    panel_full = _freeze_panel(gp, "ec", None, cluster_map, seq_lookup, config.SP_METAZOA_EC_PANEL_FULL,
                               target=target, emb=emb)
    return panel, panel_full, scan, acceptance, nesting

def subset_h5(h5_path, accessions):
    present, vecs, missing = [], [], []
    with h5py.File(h5_path, "r") as h:
        for a in accessions:
            if a in h: present.append(a); vecs.append(np.asarray(h[a], dtype=np.float32))
            else: missing.append(a)
    reps = np.vstack(vecs) if vecs else np.empty((0, 1024), np.float32)   # ProtT5 width; (0,0) breaks PCA
    return reps, present, missing

def write_panel_fasta(panel, seq_lookup, out_fa):
    """Write the survivor FASTA with BARE primary-accession headers (spec v3 §6 accession-key contract):
    the self-embed h5 is keyed by the FASTA-header token, so '>'+accession guarantees subset_h5's
    exact-string join holds and never silently collapses the rep set (no 'sp|...|', no version/isoform
    suffix). Rows whose accession has no sequence in `seq_lookup` are skipped."""
    with open(out_fa, "w") as fh:
        for a in panel["accession"]:
            seq = seq_lookup.get(a)
            if seq:
                fh.write(f">{a}\n{seq}\n")

def _gunzip_all(data: bytes) -> bytes:
    """Peel EVERY gzip layer. UniProt's `compressed=true` returns a gzip body; if the request also
    advertises `Accept-Encoding: gzip` the transfer layer gzips it AGAIN (Content-Encoding: gzip),
    yielding 2 nested layers — a single `gzip.decompress` then leaves an inner gzip stream and the
    `.decode("utf-8")` fails on the 0x1f 0x8b magic. A decoded TSV never begins with that magic, so
    looping on it is safe for 1 OR 2 layers (and is a no-op on already-plain text)."""
    while len(data) >= 2 and data[0] == 0x1F and data[1] == 0x8B:
        data = gzip.decompress(data)
    return data


def fetch_annotations(query, fields, out_tsv, manifest_path):
    """REST stream compressed=true with bounded retry -> .part -> atomic rename (spec §3.3).
    NOTE: for connection-drop resilience a cursor-paginated /search fallback can replace this;
    stage-1 uses one-shot-with-retry, which is sufficient at ~110k rows.
    We do NOT send Accept-Encoding: gzip — `compressed=true` already gzips the body, and asking for
    transfer gzip on top double-wraps it; `_gunzip_all` is the belt-and-suspenders against either."""
    base = "https://rest.uniprot.org/uniprotkb/stream"
    url = f"{base}?query={urllib.parse.quote(query)}&format=tsv&fields={fields}&compressed=true"
    last = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "tax_disentangle/1.0 (stage-1)"})
            with urllib.request.urlopen(req, timeout=900) as resp:
                release = resp.headers.get("x-uniprot-release", "unknown")
                raw = resp.read()
            text = _gunzip_all(raw).decode("utf-8")
            break
        except Exception as e:                       # noqa
            last = e
            if attempt == 3: raise
            time.sleep(5 * (attempt + 1))
    part = out_tsv + ".part"
    with open(part, "w") as f: f.write(text)
    os.replace(part, out_tsv)
    sha = hashlib.sha256(text.encode()).hexdigest()
    n = text.count("\n") - 1
    if release == "unknown": print("WARNING: x-uniprot-release header absent")
    with open(manifest_path, "w") as f:
        json.dump({"query": query, "fields": fields, "release": release,
                   "rows": n, "sha256": sha}, f, indent=2)
    return release, n

def _h5_is_valid(path):
    """Spec §3.2 validation: at least one bare-accession dataset of shape (1024,)."""
    try:
        with h5py.File(path, "r") as h:
            for k in h:                                  # first key only — cheap
                return tuple(h[k].shape) == (1024,)
        return False
    except Exception:                                    # noqa — corrupt/truncated file
        return False

def download_embeddings(url, dest):
    """Resumable-safe one-shot download to .part -> verify Content-Length -> atomic rename.
    v3 fix: skip-guard validates h5 contents (not just size) so a >1GB-but-corrupt dest is re-fetched."""
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000_000 and _h5_is_valid(dest):
        return dest
    part = dest + ".part"
    with urllib.request.urlopen(url, timeout=1800) as resp:
        expected = int(resp.headers.get("Content-Length", "0"))
        with open(part, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk: break
                f.write(chunk)
    if expected and os.path.getsize(part) != expected:
        raise IOError(f"download size mismatch: {os.path.getsize(part)} != {expected}")
    os.replace(part, dest)
    return dest

def freeze_cluster_ids(accessions, sequences, out_tsv, manifest_path):
    """90%-id mmseqs clusters via tools/embeddings/cluster_ids.mmseqs_cluster (FASTA-based).
    Frozen so the battery never recomputes (spec §8). FAILS LOUDLY if mmseqs absent unless
    explicitly allowed -- the self-cluster fallback silently disables the leak guard."""
    import os, tempfile, subprocess
    from . import clusters as CI       # tools/ is on sys.path (see module top)
    if not CI.have_mmseqs():
        raise RuntimeError("mmseqs not on PATH; refusing silent self-cluster fallback (spec §8)")
    ver = subprocess.run(["mmseqs", "version"], capture_output=True, text=True).stdout.strip()
    with tempfile.NamedTemporaryFile("w", suffix=".fasta", delete=False) as fa:
        for a, s in zip(accessions, sequences): fa.write(f">{a}\n{s}\n")
        fa_path = fa.name
    try:
        rep = CI.mmseqs_cluster(fa_path, min_seq_id=0.90)  # {member_id: cluster_rep}; dict[str,str]
    finally:
        os.unlink(fa_path)                                 # don't leak the temp FASTA
    for a in accessions:                                  # v3 fix: mmseqs omits singletons -> they rep themselves
        rep.setdefault(a, a)                              # mirrors clean_eval.py:208-209 / read_eval.py:114-115
    pd.DataFrame({"accession": list(accessions),
                  "cluster_id": [rep[a] for a in accessions]}).to_csv(out_tsv, sep="\t", index=False)
    with open(manifest_path, "w") as f:
        json.dump({"method": "mmseqs90", "mmseqs_version": ver, "n": len(accessions)}, f, indent=2)
    return "mmseqs90"


# ---------------------------------------------------------------------------- per-family cap (B5, spec §5)
def _stratified_sample(g, n, seed, strat_col="tax_order"):
    """Sample exactly n rows from g, stratified by `strat_col` (largest-remainder allocation across
    strata, then seed-fixed within-stratum sample). Preserves the per-order distribution so the cap
    cannot silently destroy order-grain feasibility (R-4/spec §5)."""
    if n >= len(g):
        return g
    if strat_col not in g.columns:
        return g.sample(n=n, random_state=seed)
    groups = [(k, sub) for k, sub in g.groupby(strat_col, sort=True)]
    sizes = np.array([len(sub) for _, sub in groups])
    quota = n * sizes / sizes.sum()
    alloc = np.minimum(np.floor(quota).astype(int), sizes)
    remaining = int(n - alloc.sum())
    rema = quota - np.floor(quota)
    order = sorted(range(len(groups)), key=lambda i: (-rema[i], i))   # largest fractional remainder, deterministic
    j = 0
    while remaining > 0 and j < len(order) * (n + 1):
        i = order[j % len(order)]
        if alloc[i] < sizes[i]:
            alloc[i] += 1
            remaining -= 1
        j += 1
    parts = [sub.sample(n=int(alloc[i]), random_state=seed)
             for i, (_, sub) in enumerate(groups) if alloc[i] > 0]
    return pd.concat(parts) if parts else g.iloc[:0]


def _trim_surplus(out, ceiling, seed):
    """Deterministic seed-fixed surplus trim to bring an overshooting frame to <= ceiling WITHOUT
    wiping any admitted family (M-7). Each family keeps an anchor row (smallest accession); the
    surplus is dropped largest-family-first. If ceiling < #families the frame cannot honor both
    invariants — it returns one-per-family (> ceiling) and the §11 sizing gate NO-GOs honestly."""
    surplus = len(out) - ceiling
    if surplus <= 0:
        return out
    out2 = out.sort_values(["pfam_family", "accession"]).reset_index(drop=True)
    fam_pos = out2.groupby("pfam_family", sort=True).indices       # {fam: positions}
    droppable = []
    for fam, idxs in fam_pos.items():
        idxs = sorted(int(p) for p in idxs)
        droppable.extend((len(idxs), fam, p) for p in idxs[1:])    # idxs[0] = protected anchor
    droppable.sort(key=lambda x: (-x[0], x[1], x[2]))              # largest family first, deterministic
    keep = np.ones(len(out2), dtype=bool)
    for _, _, p in droppable[:surplus]:
        keep[p] = False
    return out2[keep].reset_index(drop=True)


def apply_family_cap(members):
    """Per-family cap + total-ceiling down-sample (spec §5). cap = min(SP_CAP_C, members) per family,
    stratified-by-order, seed-fixed (SP_CAP_SEED). Applied ONLY to gate-admitted families; never drops
    a whole family (cap-after-gate invariant). If Σcap > SP_TOTAL_EMBED_CEILING, a mechanical seed-fixed
    proportional-by-family down-sample + surplus trim hits the ceiling exactly (M-7)."""
    seed = config.SP_CAP_SEED
    capped = []
    for fam, g in members.groupby("pfam_family", sort=True):
        n = min(config.SP_CAP_C, len(g))
        capped.append(_stratified_sample(g, n, seed) if n < len(g) else g)
    out = pd.concat(capped, ignore_index=True) if capped else members.iloc[:0]
    ceiling = config.SP_TOTAL_EMBED_CEILING
    if len(out) > ceiling:
        frac = ceiling / len(out)
        downs = []
        for fam, g in out.groupby("pfam_family", sort=True):
            k = max(1, int(np.floor(len(g) * frac)))               # >=1: never wipe an admitted family
            downs.append(_stratified_sample(g, min(k, len(g)), seed))
        out = pd.concat(downs, ignore_index=True)
        if len(out) > ceiling:                                     # M-7: max(1) overshoot -> deterministic trim
            out = _trim_surplus(out, ceiling, seed)
    return out.sort_values(["pfam_family", "accession"]).reset_index(drop=True)


def outputs_valid(out_tsv, manifest_path, n_expected):
    """Validate-and-skip (spec §5): exists AND row-count == n_expected AND sha256 == manifest sha.
    Lets a requeued Phase B resume instead of HALTing — NOT a blind --force (Rule 2)."""
    import hashlib
    import os
    if not (os.path.exists(out_tsv) and os.path.exists(manifest_path)):
        return False
    text = open(out_tsv).read()
    rows = text.count("\n") - 1                                    # minus header
    sha = hashlib.sha256(text.encode()).hexdigest()
    man = json.load(open(manifest_path))
    return bool(rows == n_expected and man.get("rows") == n_expected and man.get("sha256") == sha)


def build_self_embed_manifest(panel_accessions, h5_path, model_meta, out_json):
    """Self-embed manifest (spec §6) — the largest NEW artifact, no upstream SHA to inherit.
    n_missing/miss_set computed by SET-CONTAINMENT (a count-neutral set drift is otherwise invisible)."""
    import h5py
    panel = list(dict.fromkeys(panel_accessions))
    with h5py.File(h5_path, "r") as f:
        keys = set(f.keys())
        dim = int(f[next(iter(keys))].shape[0]) if keys else 0
    present = [a for a in panel if a in keys]
    missing = [a for a in panel if a not in keys]
    man = {"n_panel_survivors": len(panel), "n_h5_present": len(present), "n_missing": len(missing),
           "dim": dim, "dtype": "float16",
           "miss_set_breakdown": {"missing": missing[:1000], "n_missing": len(missing)}}
    man.update(model_meta)
    json.dump(man, open(out_json, "w"), indent=2)
    return man


def write_gate_scan_sidecar(tsv_path):
    """M-2/§9: write <gate_scan_tsv>.provenance.json beside the gate-scan TSV (a TSV can't carry a JSON
    provenance block). Records the mac-captured config_git_sha + the TSV sha256."""
    import hashlib
    import os
    sidecar = tsv_path + ".provenance.json"
    git_sha = None
    prov = os.path.join(os.path.dirname(__file__), "config_provenance.json")
    if os.path.exists(prov):
        try:
            git_sha = json.load(open(prov)).get("config_git_sha")
        except Exception:
            git_sha = None
    sha = None
    if os.path.exists(tsv_path):
        h = hashlib.sha256()
        with open(tsv_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        sha = h.hexdigest()
    rec = {"config_git_sha": git_sha, "gate_scan_path": tsv_path, "gate_scan_sha256": sha}
    with open(sidecar, "w") as f:
        json.dump(rec, f, indent=2)
    return sidecar


# ============================================================================ panel freeze + orchestration (B11)
# Column order MATCHES clean_eval's aa_cols expectation exactly (clean_eval.py:1220/1514 use the 20
# canonical AAs in ALPHABETICAL order). All 20 columns must be present for the C2c composition gate.
STD_AA = "ACDEFGHIKLMNPQRSTVWY"
_AA_COLS = [f"aa_{c}" for c in STD_AA]
_PANEL_COLS = (["accession", "pfam_family", "ec", "taxid", "resolved_taxid", "idx",
                "tax_superkingdom", "tax_class", "tax_order", "tax_phylum", "cluster_id", "length",
                "source"]                                   # B-2: SP/TrEMBL flag → §4.4 twin / §7 batch guard
               + _AA_COLS)


def _aa_composition(seq):
    """20-D amino-acid fraction (dict `aa_<C>` -> frac) over the 20 standard residues; non-standard
    (X/U/B/Z/*/gaps) ignored and the fractions renormalized over standard residues only (spec B11).
    Empty / all-non-standard sequence -> all-zero vector (a benign zero row for the LEACE concept)."""
    counts = {c: 0 for c in STD_AA}
    total = 0
    for ch in (seq or "").upper():
        if ch in counts:
            counts[ch] += 1
            total += 1
    if total == 0:
        return {f"aa_{c}": 0.0 for c in STD_AA}
    return {f"aa_{c}": counts[c] / total for c in STD_AA}


def _apply_exclusions_on(panel, exclusions, func_col):
    """Generalized deterministic exclusions-apply keyed on `func_col` (`pfam_family` OR `ec`), sorted
    by `accession` (spec §4.9 / plan B12 — committed panels sorted by accession). Empty/None
    exclusions => keep every gate-passing family (the zero-exclusion twin)."""
    out = panel
    if exclusions is not None and len(exclusions) and "action" in getattr(exclusions, "columns", []):
        drop = set(exclusions.loc[exclusions["action"] == "drop", func_col])
        out = panel.loc[~panel[func_col].isin(drop)]
    return out.copy().sort_values("accession").reset_index(drop=True)


def _ec_relabel(frame):
    """Relabel the FUNCTION axis to 3rd-level EC, DROPPING multi-EC3 proteins (spec v3 §7). The previous
    first-alphabetical pick over-selected low EC numbers (a clade-skewed label x taxonomy confound), so
    a protein whose DISTINCT EC3 set has >1 element is dropped rather than arbitrarily labelled. Returns
    (frame_with_ec, drop_report) where drop_report maps superkingdom -> n_dropped_multi_ec3 (+ '_total')
    so the corroborative-panel filter's clade skew is auditable. NOTE: shared by the metazoa build too —
    the drop now applies to both the Stage-1 EC panel and the all-life EC corroborative panel."""
    df = frame.copy()
    def _ec3_set(s):
        if not isinstance(s, (tuple, list)):
            return ()
        ec3 = {collapse_ec(e, config.SP_EC_LEVEL) for e in s if isinstance(e, str) and e.strip()}
        return tuple(sorted(x for x in ec3 if x))
    sets = df["ec_set"].apply(_ec3_set)
    multi = sets.apply(lambda s: len(s) > 1)
    sk = df.get("tax_superkingdom", pd.Series([None] * len(df), index=df.index))
    report = {str(k): int(v) for k, v in sk[multi].fillna("(null)").value_counts().items()}
    report["_total"] = int(multi.sum())
    df = df.loc[~multi].copy()
    df["ec"] = [s[0] if s else None for s in sets[~multi]]
    if report["_total"]:
        print(f"[sp:ec] dropped {report['_total']} multi-EC3 proteins per-superkingdom={report} (v3 §7)")
    return df.loc[df["ec"].notna()].reset_index(drop=True), report


def _build_resolved_frame(snapshot, emb, resolver, h5_path):
    """Deterministic §4.1–§4.6 pipeline: parse -> single-family -> coverage(no-op) -> taxid-resolve ->
    h5-availability -> attach ranks. Returns (frame, reps, dropped, missing_h5, counts) where `frame`
    is row-aligned to `reps` (both in the h5-`present` order). `reps` (n,1024 float32) feeds clade-MI."""
    counts = {}
    df = parse_annotations(snapshot); counts["parsed"] = len(df)
    sf = filter_single_family(df); counts["single_family"] = len(sf)
    sf = sf.copy(); sf["domain_aa"] = float("nan")                 # §4.3 coverage is a reported no-op
    cov = filter_coverage(sf, config.SP_COVERAGE_MIN); counts["coverage_noop"] = len(cov)
    resolved, dropped = resolve_taxids(cov, emb, resolver); counts["resolved"] = len(resolved)
    assert resolved["accession"].is_unique, "snapshot accessions must be unique (.loc[present] mis-aligns)"
    reps, present, missing = subset_h5(h5_path, resolved["accession"].tolist())   # §4.5 availability
    counts["h5_available"] = len(present)
    avail = resolved.set_index("accession").loc[present].reset_index()            # reorder to reps order
    ranked = attach_ranks(avail, resolver)                                        # §4.6 class/order/phylum
    return ranked, reps, dropped, missing, counts


def _gate_scan_for(frame, func_col, order_col="tax_order", span_superkingdoms=False):
    """Crossing gate-scan (spec §4.7) over `func_col` at the pre-committed thresholds (§6.1). The all-life
    caller (Stage-2) passes a domain-appropriate `order_col` (§4.2.2) + span_superkingdoms=True (§4.2.3);
    the metazoa caller keeps the defaults (Stage-1 byte-identical)."""
    return gate_scan(frame, member_floor=config.SP_PFAM_MEMBER_FLOOR,
                     min_orders=config.SP_CROSS_MIN_ORDERS,
                     min_eff_orders=config.SP_CROSS_MIN_EFF_ORDERS,
                     order_col=order_col, family_col=func_col,
                     span_superkingdoms=span_superkingdoms, min_superkingdoms=2)


def _attach_clade_signal(scan, frame, reps, func_col):
    """Within-family clade-predictivity MI per SURVIVING family (spec §4.7a / B1; REPORTED). `reps` is
    row-aligned to `frame`. Non-kept families -> NaN; families with <2 non-null clades -> 0.0."""
    mi = {}
    for fam in scan.loc[scan["keep"], func_col]:
        m = (frame[func_col] == fam).to_numpy()
        clades = frame.loc[m, "tax_class"]
        ok = clades.notna().to_numpy()
        if ok.sum() >= 2 and clades[ok].nunique() >= 2:
            mi[fam] = within_family_clade_mi(reps[m][ok], clades[ok].tolist())
        else:
            mi[fam] = 0.0
    out = scan.copy()
    out["clade_signal_mi"] = [mi.get(f, float("nan")) for f in out[func_col]]
    return out


def _panel_level_stats(frame, func_col, kept_families):
    """Bidirectional func<->order nesting (§4.7) + panel acceptance (§6.1) over the gate-passing
    member set (members of `kept_families`), using rows that carry a non-null `tax_order`."""
    gp = frame.loc[frame[func_col].isin(kept_families)]
    ok = gp["tax_order"].notna()
    nesting = panel_nesting(gp.loc[ok, func_col], gp.loc[ok, "tax_order"])
    acc = panel_acceptance(len(kept_families), nesting,
                           min_families=config.SP_MIN_CROSSED_FAMILIES,
                           nesting_max=config.SP_NESTING_MAX)
    return nesting, acc, gp


def _compute_nmi(frame, ec_frame, pfam_kept, ec_kept):
    """NMI(Pfam family, EC3) on the proteins shared between the two gate-passing panels (spec §6.4).
    Returns (nmi, n_shared); NaN if too few shared proteins or a degenerate (single-label) overlap."""
    p = frame.loc[frame["pfam_family"].isin(pfam_kept), ["accession", "pfam_family"]]
    e = ec_frame.loc[ec_frame["ec"].isin(ec_kept), ["accession", "ec"]]
    merged = p.merge(e, on="accession", how="inner")
    if len(merged) < 2 or merged["pfam_family"].nunique() < 2 or merged["ec"].nunique() < 2:
        return float("nan"), len(merged)
    return nmi_pfam_ec(merged["pfam_family"].tolist(), merged["ec"].tolist()), len(merged)


def _freeze_panel(gate_passing, func_col, exclusions, cluster_map, seq_lookup, out_path,
                  target="metazoa", emb=None):
    """Shared panel-freeze (spec §4.9) used by BOTH the Pfam and EC paths so they cannot drift.
    Applies exclusions (keyed on `func_col`), attaches `cluster_id` (frozen mmseqs90) + the protein's
    EC3 (when absent) + the 20-D `aa_A..aa_Y` composition (from `seq_lookup`), selects the canonical
    panel columns, and writes sorted by `accession`. Returns the frozen panel DataFrame.
    When `emb` is provided, a target stamp sidecar is written alongside the panel (spec v3 §2.N1)."""
    panel = _apply_exclusions_on(gate_passing, exclusions, func_col)
    panel["cluster_id"] = panel["accession"].map(cluster_map)
    if "ec" not in panel.columns:                                  # Pfam path: derive protein-matched EC3
        firsts = panel["ec_set"].apply(lambda s: s[0] if isinstance(s, (tuple, list)) and len(s) else None)
        panel["ec"] = [collapse_ec(e, config.SP_EC_LEVEL) if isinstance(e, str) and e.strip() else None
                       for e in firsts]
    if "pfam_family" not in panel.columns:                         # EC path: carry pfam_family if present
        panel["pfam_family"] = None
    comp = [_aa_composition(seq_lookup.get(a, "")) for a in panel["accession"]]
    for col in _AA_COLS:
        panel[col] = [d[col] for d in comp]
    cols = [c for c in _PANEL_COLS if c in panel.columns]
    panel = panel[cols]
    panel.to_csv(out_path, sep="\t", index=False)
    if emb is not None:
        _write_target_stamp(out_path, target, emb)
    return panel


# --- monkeypatchable acquisition / load seams (the C-phase tests patch the clean_eval analogues; the
#     B11 integration test patches these so the real ckpt / h5 / network are never touched in CI) ---
def _h5_path():
    return str(config.SP_METAZOA_H5)


def _load_emb(target="metazoa"):
    from .core import TaxonomyEmbedding
    if target == "metazoa":
        return TaxonomyEmbedding(config.CKPT, config.TAXMAP, config.EDGELIST)
    if target == "cellular":
        return TaxonomyEmbedding(config.CELLULAR_CKPT, config.CELLULAR_TAXMAP, config.CELLULAR_EDGELIST)
    raise ValueError(f"unknown taxonomy target {target!r} (expected 'metazoa' or 'cellular')")


def _write_target_stamp(out_path, target, emb):
    """Stamp {target_name, n_nodes} beside the frozen panel (spec v3 §2.N1) so the battery can bind its
    embedding to the one the panel was resolved against. Does NOT hash the checkpoint: reading the
    ~700MB .pth on every freeze is prohibitively slow, and the runtime invariant binds via target_name
    + n_nodes + the idx->resolved_taxid round-trip (the load-bearing guard), never via a ckpt hash."""
    import json as _json, os
    stamp_path = os.path.splitext(str(out_path))[0] + ".target.json"
    _json.dump({"target_name": target, "n_nodes": int(emb.n_nodes)}, open(stamp_path, "w"))
    return stamp_path


def _load_resolver():
    from .taxdump import TaxonResolver
    taxdump = config.TAXDUMP_DIR_FULL or config.TAXDUMP_DIR    # GPU cluster run repoints via TAXEMBED_FULL_TAXDUMP_DIR
    return TaxonResolver(str(taxdump))


def _read_snapshot():
    """Read the frozen REST annotations snapshot (human-readable UniProt TSV headers) + a
    accession->sequence lookup. `keep_default_na=False` so empty Pfam/EC cells are '' (not NaN),
    which `parse_annotations._split_field` turns into ()."""
    raw = pd.read_csv(config.SP_METAZOA_ANNOTATIONS, sep="\t", dtype=str, keep_default_na=False)
    seq_lookup = dict(zip(raw["Entry"], raw["Sequence"]))
    return raw, seq_lookup


def _ensure_inputs():
    """Acquire the two gated inputs IF absent (idempotent). The annotations snapshot is fetched ONLY
    when missing so it stays byte-stable between a gate-scan and a later apply (the stream row order
    is unstable across re-pulls); the h5 has its own content-validity skip-guard."""
    if not (os.path.exists(config.SP_METAZOA_ANNOTATIONS) and os.path.exists(config.SP_METAZOA_MANIFEST)):
        print("[sp:build] fetching annotations snapshot (absent) …", flush=True)
        fetch_annotations(config.SP_TAXONOMY_QUERY, config.SP_REST_FIELDS,
                          str(config.SP_METAZOA_ANNOTATIONS), str(config.SP_METAZOA_MANIFEST))
    download_embeddings(config.SP_EMB_URL, str(config.SP_METAZOA_H5))


def _read_cluster_map():
    cm = pd.read_csv(config.SP_CLUSTER_IDS, sep="\t")
    return dict(zip(cm["accession"], cm["cluster_id"]))


def _read_exclusions(path, func_col):
    """Read a committed exclusions decision file (`<func_col>, action, reason`); None if absent."""
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, sep="\t")


def _write_resolution(frame):
    """Freeze the deterministic resolution record (spec §4.4), sorted by accession."""
    cols = ["accession", "taxid", "resolved_taxid", "idx", "via", "rank_gap", "organism_name"]
    out = frame[[c for c in cols if c in frame.columns]].sort_values("accession").reset_index(drop=True)
    out.to_csv(config.SP_METAZOA_RESOLUTION, sep="\t", index=False)


def _resolution_drop_bias(dropped, frame, resolver):
    """Order-distribution of resolve-DROPPED taxids vs KEPT (spec §4.4): is the drop enriched in
    non-model orders (removing exactly the breadth the gate needs)? Returned as a small text table."""
    def _order_of(tid):
        try:
            return {rk: nm for rk, _t, nm in resolver.lineage(int(tid))}.get("order")
        except Exception:                                          # noqa — unknown/obsolete taxid
            return None
    drop_orders = pd.Series([_order_of(t) for t in dropped["taxid"]]).fillna("(no order)") if len(dropped) else pd.Series([], dtype=str)
    kept_orders = frame["tax_order"].fillna("(no order)")
    dc = drop_orders.value_counts()
    kc = kept_orders.value_counts()
    allo = sorted(set(dc.index) | set(kc.index), key=lambda o: -(int(dc.get(o, 0)) + int(kc.get(o, 0))))
    lines = ["| order | dropped | kept |", "|---|---:|---:|"]
    for o in allo[:25]:
        lines.append(f"| {o} | {int(dc.get(o, 0))} | {int(kc.get(o, 0))} |")
    if len(allo) > 25:
        lines.append(f"| …(+{len(allo) - 25} more orders) | | |")
    return "\n".join(lines)


def _write_build_log(counts, frame, dropped, missing_h5, resolver,
                     pfam_scan, ec_scan, pfam_kept, ec_kept,
                     pfam_nest, ec_nest, pfam_acc, ec_acc, nmi, n_shared, release):
    """Write the §4/§6.5 build log (per-filter n, join-coverage, resolution-drop bias, crossing
    counts, both nesting numbers, clade-signal range, NMI, acceptance verdicts). Also prints the
    headline verdict to stdout."""
    join_cov = counts["h5_available"] / counts["resolved"] if counts["resolved"] else float("nan")
    mi_kept = pfam_scan.loc[pfam_scan["keep"], "clade_signal_mi"].dropna()
    mi_rng = (f"{mi_kept.min():.3f}–{mi_kept.max():.3f} (median {mi_kept.median():.3f})"
              if len(mi_kept) else "n/a")
    L = []
    L.append("# SP-Metazoa stage-1 panel build log\n")
    L.append(f"- UniProt release: **{release}**  ·  query: `{config.SP_TAXONOMY_QUERY}`")
    L.append(f"- PCA_DIM (reps): {config.PCA_DIM}  ·  EC level: {config.SP_EC_LEVEL}\n")
    L.append("## Per-filter counts (§4.1–4.6)")
    L.append(f"- parsed (reviewed Metazoa): {counts['parsed']}")
    L.append(f"- single-Pfam-family: {counts['single_family']}")
    L.append(f"- coverage filter: {counts['coverage_noop']}  "
             f"**(NOT ENFORCED — `xref_pfam` carries no domain coords; reported no-op, spec §4.3/B4)**")
    L.append(f"- taxid-resolved into embedding: {counts['resolved']}  (dropped {len(dropped)})")
    L.append(f"- present in per-protein.h5: {counts['h5_available']}  (missing {len(missing_h5)})")
    L.append(f"- **join-coverage = {join_cov:.4f}**  "
             f"(gate §6.5: ≥ {config.SP_JOIN_COVERAGE_MIN} → {'PASS' if join_cov >= config.SP_JOIN_COVERAGE_MIN else 'FAIL'})\n")
    L.append("## Resolution-drop order bias (§4.4)")
    L.append(_resolution_drop_bias(dropped, frame, resolver) + "\n")
    L.append("## Crossing gate-scan (§4.7 / §6.1)")
    L.append(f"- Pfam: {len(pfam_kept)} crossed families (of {pfam_scan['pfam_family'].nunique()})  "
             f"| floor M={config.SP_PFAM_MEMBER_FLOOR}, raw≥{config.SP_CROSS_MIN_ORDERS}, "
             f"eff≥{config.SP_CROSS_MIN_EFF_ORDERS}")
    L.append(f"- Pfam nesting: 1−H(tax|func)={pfam_nest['tax_given_func']:.3f}  "
             f"1−H(func|tax)={pfam_nest['func_given_tax']:.3f}  (both < {config.SP_NESTING_MAX}? "
             f"{pfam_nest['tax_given_func'] < config.SP_NESTING_MAX and pfam_nest['func_given_tax'] < config.SP_NESTING_MAX})")
    L.append(f"- Pfam within-family clade-signal MI range (§4.7a, reported): {mi_rng}")
    L.append(f"- **Pfam acceptance (§6.1): {'ACCEPTED' if pfam_acc['accepted'] else 'REJECTED'}** "
             f"(enough_families={pfam_acc['enough_families']} [≥{config.SP_MIN_CROSSED_FAMILIES}], "
             f"nesting_ok={pfam_acc['nesting_ok']})\n")
    L.append(f"- EC: {len(ec_kept)} crossed EC3 families (of {ec_scan['ec'].nunique()})")
    L.append(f"- EC nesting: 1−H(tax|func)={ec_nest['tax_given_func']:.3f}  "
             f"1−H(func|tax)={ec_nest['func_given_tax']:.3f}")
    L.append(f"- **EC acceptance: {'ACCEPTED' if ec_acc['accepted'] else 'REJECTED'}**")
    L.append(f"- **NMI(Pfam, EC3) on {n_shared} shared proteins = "
             f"{nmi:.3f}**  (independent iff < {config.SP_NMI_MAX} → "
             f"{'INDEPENDENT' if (nmi == nmi and nmi < config.SP_NMI_MAX) else 'REDUNDANT/NA'})\n")
    L.append("## Notes")
    L.append("- Thresholds are PRE-COMMITTED (§6.1, anti-p-hacking): a failing panel is a real negative, "
             "not a retune trigger.")
    L.append("- Exclusions (if any) are mechanical-category only (§4.8/B2); the headline must AGREE "
             "between the post-exclusion and zero-exclusion (`_full`) twin in E2.")
    config.SP_BUILD_LOG.write_text("\n".join(L) + "\n")
    print(f"\n[sp:build] join-coverage={join_cov:.4f}  Pfam crossed={len(pfam_kept)} "
          f"accepted={pfam_acc['accepted']}  EC crossed={len(ec_kept)} accepted={ec_acc['accepted']}  "
          f"NMI={nmi:.3f}  → build log: {config.SP_BUILD_LOG}", flush=True)


def _existing_outputs(stage):
    """Committed artifacts a stage would overwrite — for the file-safety guard (CLAUDE.md Rule 1/2)."""
    gate = [config.SP_GATE_SCAN, config.SP_EC_GATE_SCAN, config.SP_METAZOA_RESOLUTION, config.SP_CLUSTER_IDS]
    appl = [config.SP_METAZOA_PANEL, config.SP_METAZOA_PANEL_FULL,
            config.SP_METAZOA_EC_PANEL, config.SP_METAZOA_EC_PANEL_FULL]
    targets = (gate if stage == "gate-scan" else appl if stage == "apply" else gate + appl)
    return [str(p) for p in targets if os.path.exists(p)]


def main(argv=None):
    """SP-Metazoa stage-1 panel build (spec §4). `--stage gate-scan` runs §4.1–§4.7a + writes the
    gate-scans/resolution/cluster-ids + build log + acceptance verdict (no exclusions needed);
    `apply` reads the committed exclusions and freezes the Pfam + EC panels (+ their `_full` twins);
    `all` runs both. Deterministic from the frozen snapshot; committed artifacts sorted before write."""
    import argparse
    ap = argparse.ArgumentParser(description="SP-Metazoa stage-1 panel build (B11)")
    ap.add_argument("--stage", choices=["gate-scan", "apply", "all"], default="all")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing committed artifacts (Rule 2: explicit acknowledgement)")
    args = ap.parse_args(argv)

    existing = _existing_outputs(args.stage)
    if existing and not args.force:
        raise SystemExit("[sp:build] STOP (CLAUDE.md Rule 1/2) — these committed artifacts already "
                         "exist; re-run with --force to overwrite (they are deterministic regenerables):\n  "
                         + "\n  ".join(existing))

    config.SP_GATE_SCAN.parent.mkdir(parents=True, exist_ok=True)
    config.DATA.mkdir(parents=True, exist_ok=True)

    _ensure_inputs()
    snapshot, seq_lookup = _read_snapshot()
    release = json.loads(Path(config.SP_METAZOA_MANIFEST).read_text()).get("release", "unknown")
    target = "metazoa"                           # SP-Metazoa stage-1; all-life stage-2 will pass "cellular"
    emb, resolver = _load_emb(target), _load_resolver()
    frame, reps, dropped, missing_h5, counts = _build_resolved_frame(snapshot, emb, resolver, _h5_path())

    # gate-scans (kept-family sets feed both stages; clade MI only in the gate-scan write)
    pfam_scan = _gate_scan_for(frame, "pfam_family")
    ec_frame, _ec_report = _ec_relabel(frame)
    ec_scan = _gate_scan_for(ec_frame, "ec")
    pfam_kept = set(pfam_scan.loc[pfam_scan["keep"], "pfam_family"])
    ec_kept = set(ec_scan.loc[ec_scan["keep"], "ec"])

    if args.stage in ("gate-scan", "all"):
        pfam_scan_mi = _attach_clade_signal(pfam_scan, frame, reps, "pfam_family")
        # freeze cluster ids over the UNION of gate-passing members (superset of every panel) — sorted
        gp_acc = sorted(set(frame.loc[frame["pfam_family"].isin(pfam_kept), "accession"])
                        | set(ec_frame.loc[ec_frame["ec"].isin(ec_kept), "accession"]))
        freeze_cluster_ids(gp_acc, [seq_lookup[a] for a in gp_acc],
                           str(config.SP_CLUSTER_IDS), str(config.SP_CLUSTER_IDS_MANIFEST))
        pfam_nest, pfam_acc, _ = _panel_level_stats(frame, "pfam_family", pfam_kept)
        ec_nest, ec_acc, _ = _panel_level_stats(ec_frame, "ec", ec_kept)
        nmi, n_shared = _compute_nmi(frame, ec_frame, pfam_kept, ec_kept)
        _write_resolution(frame)
        pfam_scan_mi.sort_values("pfam_family").reset_index(drop=True).to_csv(
            config.SP_GATE_SCAN, sep="\t", index=False)
        ec_scan.sort_values("ec").reset_index(drop=True).to_csv(
            config.SP_EC_GATE_SCAN, sep="\t", index=False)
        write_gate_scan_sidecar(str(config.SP_GATE_SCAN))      # M-2/§9: provenance JSON beside each TSV
        write_gate_scan_sidecar(str(config.SP_EC_GATE_SCAN))
        _write_build_log(counts, frame, dropped, missing_h5, resolver, pfam_scan_mi, ec_scan,
                         pfam_kept, ec_kept, pfam_nest, ec_nest, pfam_acc, ec_acc, nmi, n_shared, release)
        if args.stage == "gate-scan":
            return {"pfam_kept": pfam_kept, "ec_kept": ec_kept, "pfam_acc": pfam_acc, "ec_acc": ec_acc,
                    "nmi": nmi, "join_coverage": counts["h5_available"] / counts["resolved"]}

    if args.stage in ("apply", "all"):
        if not os.path.exists(config.SP_CLUSTER_IDS):   # NIT-2: apply needs the gate-scan's frozen clusters
            raise SystemExit(f"[sp:build] STOP — {config.SP_CLUSTER_IDS} absent; run `--stage gate-scan` "
                             "first (apply reads the frozen mmseqs90 cluster ids).")
        cluster_map = _read_cluster_map()
        pfam_excl = _read_exclusions(config.SP_METAZOA_EXCLUSIONS, "pfam_family")
        ec_excl = _read_exclusions(config.SP_METAZOA_EC_EXCLUSIONS, "ec")
        gp_pfam = frame.loc[frame["pfam_family"].isin(pfam_kept)]
        panel = _freeze_panel(gp_pfam, "pfam_family", pfam_excl, cluster_map, seq_lookup,
                              config.SP_METAZOA_PANEL, target=target, emb=emb)
        _freeze_panel(gp_pfam, "pfam_family", None, cluster_map, seq_lookup, config.SP_METAZOA_PANEL_FULL,
                      target=target, emb=emb)
        ec_panel, ec_full, _es, _ea, _en = build_ec_panel(frame, cluster_map, seq_lookup, config, ec_excl,
                                                           target=target, emb=emb)
        # post-exclusion re-acceptance (plan B12 step 4): the headline panel must still pass §6.1
        post_kept = set(panel["pfam_family"].unique())
        _pn, post_acc, _ = _panel_level_stats(frame, "pfam_family", post_kept)
        print(f"[sp:build] apply: Pfam panel n={len(panel)} fams={len(post_kept)} "
              f"(full twin all-gate-passing); post-exclusion acceptance={post_acc['accepted']}; "
              f"EC panel n={len(ec_panel)} (full twin n={len(ec_full)})", flush=True)
        return {"panel_n": len(panel), "panel_families": len(post_kept),
                "post_exclusion_accepted": post_acc["accepted"], "ec_panel_n": len(ec_panel)}


if __name__ == "__main__":
    main()
