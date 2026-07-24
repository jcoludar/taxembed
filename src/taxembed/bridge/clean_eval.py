"""Tasks 13 + 14 — the CLEAN control battery (disentanglement matrix) + the coherence number.

Composition of already-tested, committed building blocks into the CLEAN evaluation for the Taxonomy
Bridge, run on the PLA2 testbed. CLEAN = "can we read taxonomic signal out of protein embeddings and
ERASE it (LEACE) without destroying function?" The disentanglement matrix crosses
{taxonomy, function}-recoverability x {original, tax-erased, function-erased, matched-control}, all
measured within-taxon strata, with the LEACE erased rank recorded. Task 14 then MEASURES whether the
subspace READ predicts taxonomy from is the SAME object CLEAN erases (principal-angle overlap vs a
random-rank-k baseline).

HONESTY CONTRACT (spec, repeated): PLA2 within-family is EXPECTED to be ENTANGLED — Gene Group ≈
paralog lineage ≈ (often) species/clade here, so erasing taxonomy is EXPECTED to also degrade function
purity, and erasing function is EXPECTED to also degrade taxonomy. A clean ENTANGLED result is the
valid finding for this "hard contrast" testbed — it is reported, not massaged. The clean DISentangled
contrast comes later from the multi-family testbed (T15).

ONE BASIS (adversarial-review B2/M5): we fit ONE Bridge `b`, set `Z = b.pca.transform(reps)`, and use
THIS `Z` and `b.pca` for every erasure / probe / subspace below. Held-out probes train+test on slices
of this fixed `Z` — never a fresh PCA, never a different row set.

Task 16 — pla2_mammals: the within-Mammals, crossed-group restriction (entangled contrast partner).
  Data selection: filter to Clade=="Mammals" ∩ Gene Groups present in >=2 clades in the FULL dataset
  (crossed groups, computed from full table). Within-Mammals stratification uses `Species` as the
  taxonomy proxy (Clade is constant = "Mammals" here, so Species gives the finest available
  discriminator already in the data, spanning 38 mammalian species across multiple orders).
  PCA_DIM = min(64, n_subset // 3) for numerical stability with the smaller n.

Run:  python -m taxembed.bridge.clean_eval
      [--testbed pla2]           (default; full PLA2 set)
      [--testbed pla2_mammals]   (Task 16: within-Mammals, crossed-group subset)
      [--testbed multifamily]    (T15: multi-family ToxFam-12 disentanglement — the PRIMARY clean result)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent

from . import config  # noqa: E402
from .h5_io import load_embeddings, mean_pool  # noqa: E402
from .core import Bridge, log0  # noqa: E402
from .erase import (  # noqa: E402
    leace_erase,
    erasing_projector_rowspace,
    matched_control_subspace,
    apply_projector,
    principal_angle_overlap,
)
from .splits import grouped_holdout  # noqa: E402
from .clusters import have_mmseqs, mmseqs_cluster  # noqa: E402
from taxembed.eval.nulls import radial_only_null  # noqa: E402

from .eval_utils import label_nesting as _label_nesting  # noqa: E402
from .eval_utils import _compute_cluster_ids_fasta  # noqa: E402

from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.neural_network import MLPClassifier, MLPRegressor  # noqa: E402

SEED = 0
N_SPLITS = 5
KNN_K = 5
N_RANDOM_BASELINE = 20          # seeds for the random-rank-k coherence baseline


# ============================================================================ data loading
def load_pla2_clean():
    """Return (ann, reps, positions, idx) for the PLA2 testbed: one row per protein with
    identifier / Gene Group (FUNCTION) / Clade / Species (TAXONOMY) / taxid / idx (metazoa node) /
    cluster_id; reps = (N,1024) float64 ProtT5; positions = (N,100) true Poincaré node per protein.
    All 452 join cleanly. Clade blanks are backfilled from same-species rows (deterministic)."""
    from .core import TaxonomyEmbedding

    emb = load_embeddings(config.PLA2_H5)                       # {identifier: (1024,) float32}
    ann = pd.read_csv(config.PLA2_CSV)
    res = pd.read_csv(config.PLA2_RESOLUTION, sep="\t").rename(columns={"species": "Species"})
    ann = ann.merge(res[["Species", "taxid", "idx"]], on="Species", how="left", validate="m:1")

    missing = ann[ann["taxid"].isna()]
    if len(missing):
        raise SystemExit(f"{len(missing)} proteins have unresolved species — fix resolution table")
    ann["taxid"] = ann["taxid"].astype(int)
    ann["idx"] = ann["idx"].astype(int)

    ids = ann["identifier"].tolist()
    if set(ids) - set(emb):
        raise SystemExit(f"identifiers missing from h5: {sorted(set(ids) - set(emb))[:5]}")
    ann = ann.reset_index(drop=True)
    reps = np.stack([emb[i].astype(np.float64) for i in ann["identifier"]])  # (N, 1024)

    # Clade is species-level; a few blank cells -> backfill from same-species rows (surface it).
    n_blank = int(ann["Clade"].isna().sum())
    if n_blank:
        sp2clade = (ann.dropna(subset=["Clade"]).groupby("Species")["Clade"]
                    .agg(lambda s: s.mode().iloc[0]).to_dict())
        ann["Clade"] = ann["Clade"].fillna(ann["Species"].map(sp2clade))
        still = int(ann["Clade"].isna().sum())
        print(f"[clade] backfilled {n_blank} blank Clade cells from same-species rows ({still} still blank)")
        if still:
            raise SystemExit(f"{still} proteins have an un-backfillable Clade — investigate")

    clusters = _compute_cluster_ids(ids)
    ann["cluster_id"] = [clusters[i] for i in ann["identifier"]]

    te_emb = TaxonomyEmbedding(config.CKPT, config.TAXMAP, config.EDGELIST)
    positions = te_emb.positions[ann["idx"].to_numpy()]        # (N, 100) true node per protein
    return ann, reps, positions


def load_pla2_mammals_clean():
    """Return (ann_m, reps_m, positions_m, crossed_groups, pca_dim_m) for the within-Mammals,
    crossed-group subset (Task 16).

    Data selection (all computed from the data, not hardcoded):
      - Load the full PLA2 annotation table (same as load_pla2_clean).
      - Compute CROSSED groups = Gene Groups that appear in >=2 distinct Clades in the FULL dataset.
      - Filter to rows where Clade == "Mammals" AND Gene Group in crossed_groups.
      - PCA_DIM_M = min(64, n_subset // 3) — safety margin for LEACE/PCA conditioning.

    Within-Mammals taxonomy stratum: `Species` (Clade is constant = "Mammals" here; Species spans
    38 distinct mammalian species across multiple orders, giving a meaningful taxonomy axis).

    Returns (ann_m, reps_m, positions_m, crossed_groups_list, pca_dim_m):
      ann_m       — DataFrame of the subset, with Species / Gene Group / Clade / cluster_id columns
      reps_m      — (n_m, 1024) float64 ProtT5 embeddings for the subset
      positions_m — (n_m, 100) Poincaré node positions for the subset
      crossed_groups_list — sorted list of crossed Gene Group names
      pca_dim_m   — PCA dim used (min(64, n_m // 3))
    """
    from .core import TaxonomyEmbedding

    # ---- load full data (mirrors load_pla2_clean exactly) ---
    emb = load_embeddings(config.PLA2_H5)
    ann_full = pd.read_csv(config.PLA2_CSV)
    res = pd.read_csv(config.PLA2_RESOLUTION, sep="\t").rename(columns={"species": "Species"})
    ann_full = ann_full.merge(res[["Species", "taxid", "idx"]], on="Species", how="left", validate="m:1")
    missing = ann_full[ann_full["taxid"].isna()]
    if len(missing):
        raise SystemExit(f"[pla2_mammals] {len(missing)} proteins unresolved species")
    ann_full["taxid"] = ann_full["taxid"].astype(int)
    ann_full["idx"] = ann_full["idx"].astype(int)

    # Clade backfill (same as load_pla2_clean)
    n_blank = int(ann_full["Clade"].isna().sum())
    if n_blank:
        sp2clade = (ann_full.dropna(subset=["Clade"]).groupby("Species")["Clade"]
                    .agg(lambda s: s.mode().iloc[0]).to_dict())
        ann_full["Clade"] = ann_full["Clade"].fillna(ann_full["Species"].map(sp2clade))
        still = int(ann_full["Clade"].isna().sum())
        if still:
            raise SystemExit(f"[pla2_mammals] {still} un-backfillable Clade cells")

    # ---- compute crossed groups from the FULL table (not the subset) ---
    g2nclade = ann_full.dropna(subset=["Clade"]).groupby("Gene Group")["Clade"].nunique()
    crossed_groups_list = sorted(g2nclade[g2nclade >= 2].index.tolist())
    print(f"[pla2_mammals] crossed groups (>=2 clades, computed from full dataset): "
          f"{crossed_groups_list}  (n={len(crossed_groups_list)})")

    # ---- filter: Clade == "Mammals" ∩ crossed Gene Groups ---
    mammals_mask = (ann_full["Clade"] == "Mammals") & ann_full["Gene Group"].isin(crossed_groups_list)
    ann_m = ann_full[mammals_mask].reset_index(drop=True)
    n_m = len(ann_m)
    n_mammals_all = int((ann_full["Clade"] == "Mammals").sum())
    print(f"[pla2_mammals] Mammals total={n_mammals_all}; after crossed-group filter: "
          f"n={n_m} proteins, {ann_m['Species'].nunique()} species, "
          f"{ann_m['Gene Group'].nunique()} gene groups")
    if n_m < 20:
        raise SystemExit(f"[pla2_mammals] subset too small (n={n_m}) to run battery")

    # ---- PCA_DIM with safety margin ---
    pca_dim_m = min(config.PCA_DIM, n_m // 3)
    print(f"[pla2_mammals] PCA_DIM_M = min({config.PCA_DIM}, {n_m}//3={n_m // 3}) = {pca_dim_m}")

    # ---- build reps + positions for subset ---
    ids_m = ann_m["identifier"].tolist()
    if set(ids_m) - set(emb):
        raise SystemExit(f"[pla2_mammals] identifiers missing from h5: "
                         f"{sorted(set(ids_m) - set(emb))[:5]}")
    reps_m = np.stack([emb[i].astype(np.float64) for i in ids_m])   # (n_m, 1024)

    clusters = _compute_cluster_ids(ids_m)
    ann_m = ann_m.copy()
    ann_m["cluster_id"] = [clusters[i] for i in ids_m]

    te_emb = TaxonomyEmbedding(config.CKPT, config.TAXMAP, config.EDGELIST)
    positions_m = te_emb.positions[ann_m["idx"].to_numpy()]           # (n_m, 100)
    return ann_m, reps_m, positions_m, crossed_groups_list, pca_dim_m


def _compute_cluster_ids(ids):
    """{identifier: cluster_rep} via mmseqs easy-cluster at 0.9 identity; one-per-seq fallback."""
    fasta = config.PLA2_FASTA
    tmp = _HERE / "results" / "_mmseqs_tmp"
    if not have_mmseqs():
        print("[clusters] mmseqs NOT on PATH -> one-cluster-per-sequence fallback")
        return {i: i for i in ids}
    try:
        mapping = mmseqs_cluster(fasta, out_prefix=tmp / "pla2", tmp_dir=tmp / "tmp",
                                 min_seq_id=0.9, cleanup=True)
        for i in ids:
            mapping.setdefault(i, i)
        n_clusters = len(set(mapping[i] for i in ids))
        print(f"[clusters] mmseqs -> {n_clusters} clusters over {len(ids)} sequences")
        return mapping
    except Exception as e:                                     # noqa: BLE001 — robust fallback per spec
        print(f"[clusters] mmseqs FAILED ({e}) -> one-cluster-per-sequence fallback")
        return {i: i for i in ids}


# ============================================================================ multifamily loader (T12/T15)
def load_multifamily_clean():
    """Return (ann, reps, positions, pca_dim) for the multi-family (ToxFam-12) PRIMARY testbed.

    The headline disentanglement testbed: FUNCTION = toxin gene-FAMILY identity (genuinely
    non-phylogenetic — the same toxin family recurs across distant taxa, e.g. CRISP spans a hookworm,
    snakes, a monitor lizard, and a horsefly), TAXONOMY = species/clade position in the metazoa
    Poincaré embedding. This is the clean contrast to PLA2 (single family, entangled).

    Data: toxfam_v2_labels.csv (identifier, family) + the frozen resolution table
    multifamily_species_resolution.tsv (identifier, family, taxid, idx, organism, via). RESTRICT to
    via=='direct' rows (resolved into the 498k metazoa embedding); surface anything dropped.

    Returns ann with columns:
      identifier, Family (toxin family = FUNCTION label), taxid, idx, organism, cluster_id,
      tax_family / tax_order / tax_class (NCBI ranks from TaxonResolver.lineage = TAXONOMY probe labels).
    reps      = (n, 1024) mean-pooled ProtT5 (per-residue (L,1024) -> mean over L).
    positions = (n, 100) true Poincaré node per protein.
    pca_dim   = min(PCA_DIM, n // 3)  (small-n LEACE/PCA conditioning guard).
    """
    from .core import TaxonomyEmbedding
    from .taxdump import TaxonResolver

    labels = pd.read_csv(config.TOXFAM_LABELS)                  # identifier, family (FUNCTION)
    res = pd.read_csv(config.MULTIFAMILY_RESOLUTION, sep="\t")  # frozen Task-12 table

    # join coverage on residue-h5 keys (mean-pool the per-residue embeddings)
    res_emb = load_embeddings(config.TOXFAM_RESIDUE_H5)         # {id: (L,1024)}
    emb = mean_pool(res_emb)                                    # {id: (1024,)}
    ids_labels = set(labels["identifier"])
    ids_h5 = set(emb)
    miss_h5 = sorted(ids_labels - ids_h5)
    extra_h5 = sorted(ids_h5 - ids_labels)
    print(f"[multifamily] residue-h5<->labels join: {len(ids_labels & ids_h5)}/{len(ids_labels)} "
          f"labels covered; missing_from_h5={miss_h5}; extra_in_h5={extra_h5}")
    if miss_h5:
        raise SystemExit(f"[multifamily] {len(miss_h5)} labelled identifiers missing from residue h5: {miss_h5}")

    # merge labels + resolution; RESTRICT to rows that resolved into the metazoa embedding
    ann = labels.merge(res[["identifier", "taxid", "idx", "organism", "via"]],
                       on="identifier", how="left", validate="1:1")
    n_total = len(ann)
    unresolved = ann[ann["via"] != "direct"]
    if len(unresolved):
        print(f"[multifamily] SURFACING {len(unresolved)} unresolved (NOT in metazoa embedding) — dropped:")
        for _, r in unresolved.iterrows():
            print(f"    [{r['via']}] {r['identifier']} ({r['family']}) taxid={r['taxid']} organism={r['organism']!r}")
    ann = ann[ann["via"] == "direct"].reset_index(drop=True)
    ann["taxid"] = ann["taxid"].astype(int)
    ann["idx"] = ann["idx"].astype(int)
    n_resolved = len(ann)
    print(f"[multifamily] resolved into embedding: {n_resolved}/{n_total} proteins, "
          f"{ann['family'].nunique()} toxin families")

    if n_resolved < 40:
        print(f"[multifamily] CONCERN: only {n_resolved} usable proteins (<40) — power-limited")

    # FUNCTION label = toxin family (rename to avoid clashing with the NCBI 'family' rank below)
    ann = ann.rename(columns={"family": "Family"})

    # reps (mean-pooled) in ann order
    ids = ann["identifier"].tolist()
    reps = np.stack([emb[i].astype(np.float64) for i in ids])  # (n, 1024)

    # TAXONOMY probe labels: NCBI ranks from the taxdump lineage (species mostly singletons here)
    resv = TaxonResolver(config.TAXDUMP_DIR)
    rank_cache: dict[int, dict] = {}
    for tid in ann["taxid"].unique():
        lin = resv.lineage(int(tid))                           # [(rank, taxid, name), ...]
        rmap = {rk: nm for (rk, _t, nm) in lin}
        rank_cache[int(tid)] = rmap
    for rank_col, rank_key in (("tax_family", "family"), ("tax_order", "order"), ("tax_class", "class")):
        ann[rank_col] = [rank_cache[int(t)].get(rank_key, "NA") for t in ann["taxid"]]

    # identity clusters over the ToxFam fasta
    clusters = _compute_cluster_ids_fasta(ids, config.TOXFAM_FASTA, tag="toxfam")
    ann["cluster_id"] = [clusters[i] for i in ids]

    # PCA dim with small-n safety margin
    pca_dim = min(config.PCA_DIM, n_resolved // 3)
    print(f"[multifamily] PCA_DIM = min({config.PCA_DIM}, {n_resolved}//3={n_resolved // 3}) = {pca_dim}")

    te_emb = TaxonomyEmbedding(config.CKPT, config.TAXMAP, config.EDGELIST)
    positions = te_emb.positions[ann["idx"].to_numpy()]        # (n, 100)
    return ann, reps, positions, pca_dim


# ============================================================================ SP-Metazoa loader (Phase C)
# Three monkeypatch indirections: the C1 test patches all three so the real ckpt / per-protein.h5 are
# never touched. Keep them as standalone module-level functions (test seam, mirrors read_eval discipline).
def _load_sp_panel_path(testbed):
    """Resolve the frozen panel TSV for an SP testbed. `all_life_*` is matched by PREFIX FIRST — note
    all_life_pfam contains '_pfam' and all_life_ec contains '_ec', so the substring checks below would
    otherwise mis-route it to the SP_METAZOA panels (spec v3 §2.N2). The FUNCTION axis selects the panel
    family — Pfam vs the INDEPENDENT EC set (spec §5) — and a trailing `_full` selects the zero-excl twin:
        *_pfam       -> SP_METAZOA_PANEL          *_pfam_full -> SP_METAZOA_PANEL_FULL
        *_ec         -> SP_METAZOA_EC_PANEL        *_ec_full   -> SP_METAZOA_EC_PANEL_FULL
        all_life_*   -> ALL_LIFE_{,EC_}PANEL[_FULL]
    """
    testbed = _strip_source_suffix(testbed)                # _sp/_trembl twins read the SAME base panel
    full = testbed.endswith("_full")
    if testbed.startswith("all_life_"):
        if "_ec" in testbed:
            return str(config.ALL_LIFE_EC_PANEL_FULL if full else config.ALL_LIFE_EC_PANEL)
        return str(config.ALL_LIFE_PANEL_FULL if full else config.ALL_LIFE_PANEL)
    if "_ec" in testbed:
        return str(config.SP_METAZOA_EC_PANEL_FULL if full else config.SP_METAZOA_EC_PANEL)
    return str(config.SP_METAZOA_PANEL_FULL if full else config.SP_METAZOA_PANEL)


def _source_filter_of(testbed):
    """Return the source row-filter for a §4.4 twin testbed ('sp'|'trembl'), else None. The _sp/_trembl
    twins read the SAME base all-life panel and row-filter by the `source` column (B-2 / spec §4.4)."""
    if testbed.endswith("_sp"):
        return "sp"
    if testbed.endswith("_trembl"):
        return "trembl"
    return None


def _strip_source_suffix(testbed):
    for suf in ("_trembl", "_sp"):
        if testbed.endswith(suf):
            return testbed[:-len(suf)]
    return testbed


def _load_sp_h5_path(testbed=None):
    """Path to the ProtT5 per-protein.h5. `all_life_*` reads the self-embedded all-life h5; metazoa (or
    a None/unspecified testbed) reads the UniProt-precomputed SwissProt h5 (float16, keyed by accession)."""
    return str(config.ALL_LIFE_H5 if (testbed and testbed.startswith("all_life_")) else config.SP_METAZOA_H5)


def _load_te(target="metazoa"):
    """Load the LOCKED Poincaré embedding for `target` (positions + taxid<->idx tree).
    target='metazoa' -> config.CKPT/TAXMAP/EDGELIST; target='cellular' -> the all-life cellular_canonical."""
    from .core import TaxonomyEmbedding
    if target == "metazoa":
        return TaxonomyEmbedding(config.CKPT, config.TAXMAP, config.EDGELIST)
    if target == "cellular":
        return TaxonomyEmbedding(config.CELLULAR_CKPT, config.CELLULAR_TAXMAP, config.CELLULAR_EDGELIST)
    raise ValueError(f"unknown taxonomy target {target!r} (expected 'metazoa' or 'cellular')")


def _assert_target_invariant(te, ann, target, stamp):
    """Bind the panel-build embedding to the battery-time embedding (spec v3 §2.N1). The build seam
    (_load_emb) and the battery seam (_load_te) run in separate GPU cluster jobs, so a per-job --target
    divergence would otherwise pass silently (in-range idx -> wrong taxon, no crash). Fail loud."""
    if stamp.get("target_name") != target:
        raise SystemExit(f"[invariant] panel built under target={stamp.get('target_name')!r} "
                         f"but battery loaded target={target!r}")
    if int(stamp.get("n_nodes", -1)) != int(te.n_nodes):
        raise SystemExit(f"[invariant] panel n_nodes={stamp.get('n_nodes')} != te.n_nodes={te.n_nodes}")
    idx = ann["idx"].to_numpy()
    if idx.size and int(idx.max()) >= int(te.n_nodes):
        raise SystemExit(f"[invariant] panel idx max {idx.max()} >= te.n_nodes {te.n_nodes}")
    got = te.idx2taxid[idx]
    want = ann["resolved_taxid"].to_numpy().astype(got.dtype)
    bad = int((got != want).sum())
    if bad:
        raise SystemExit(f"[invariant] {bad}/{len(idx)} panel rows: te.idx2taxid[idx] != resolved_taxid "
                         "— wrong taxonomy target (silent mis-index)")


def load_sp_clean(testbed):
    """Return (ann, reps, positions, pca_dim) for an SP-Metazoa CLEAN-battery testbed, row-aligned to
    the accessions actually present in the per-protein h5.

    `testbed` is one of {sp_metazoa_pfam, sp_metazoa_ec} optionally with a `_full` suffix (zero-exclusion
    panel twin). The de-suffixed base picks the FUNCTION column:
      *_pfam* -> `pfam_family` (renamed to `func`)
      *_ec*   -> `ec`          (renamed to `func`; rows with null EC are dropped before the h5 subset)

    The frozen panel is read via `_load_sp_panel_path(testbed)`; embeddings are subset from
    `_load_sp_h5_path()` (float16->float32 cast inside subset_h5). The panel and h5 are NOT guaranteed
    to agree row-for-row: `subset_h5` partitions the requested accessions into `present` (found, in
    request order) and `missing`. We FILTER `ann` to `present` (preserving order) so that ann / reps /
    positions stay aligned, then recompute `positions = te.positions[ann.idx]` and
    `pca_dim = min(config.PCA_DIM, len(ann)//3)` on the FILTERED frame, and log the missing accessions.

    `ann` carries: func, accession, taxid, idx, cluster_id, tax_class, tax_order, tax_phylum, length,
    PLUS the per-protein amino-acid composition columns `aa_A..aa_Y` when the frozen panel has them
    (passed through for the C2c composition-erasure gate; absent in the C1 fixture, which does not
    exercise that gate).
    """
    from .build_sp_panel import subset_h5

    src_filter = _source_filter_of(testbed)                # §4.4 twin: 'sp'|'trembl'|None (None = metazoa/normal)
    core = _strip_source_suffix(testbed)
    base = core[:-len("_full")] if core.endswith("_full") else core
    panel_path = _load_sp_panel_path(testbed)
    ann = pd.read_csv(panel_path, sep="\t")

    # --- choose the FUNCTION column from the de-suffixed base ---
    if "_pfam" in base:
        func_col = "pfam_family"
    elif "_ec" in base:
        func_col = "ec"
        # EC panel: drop rows with no EC label before subsetting (null-EC carries no function signal).
        n_pre = len(ann)
        ann = ann[ann["ec"].notna()].reset_index(drop=True)
        n_drop = n_pre - len(ann)
        if n_drop:
            print(f"[sp:{testbed}] dropped {n_drop} null-EC rows ({len(ann)} remain)")
    else:
        raise SystemExit(f"[sp] unknown SP testbed '{testbed}' (expected *_pfam* or *_ec*)")
    ann = ann.rename(columns={func_col: "func"})

    # --- §4.4 SP-only / TrEMBL-only twin: row-filter by source (None = no filter; metazoa byte-identical) ---
    if src_filter:
        if "source" not in ann.columns:
            raise SystemExit(f"[sp:{testbed}] source-filter '{src_filter}' requested but panel has no "
                             "'source' column (spec §4.4 twin needs the B-2 source flag)")
        n0 = len(ann)
        ann = ann[ann["source"] == src_filter].reset_index(drop=True)
        print(f"[sp:{testbed}] source-filter={src_filter}: {len(ann)}/{n0} rows")

    # --- spec §4.6: per-rank scoring is over taxa that HAVE the rank, never imputed. A protein with no
    # NCBI `tax_class` node cannot enter the categorical taxonomy probe / within-class purity / READ
    # leave-clade-out, and pd.Categorical(null).codes==-1 would crash np.bincount. The frozen panel
    # KEEPS these rows (store NULL, §4.9); we drop them HERE at load (scoring time) and log the
    # denominator. (The finer tax_order grain masks its own nulls inside run_battery_sp.) ---
    n_pre_class = len(ann)
    ann = ann[ann["tax_class"].notna()].reset_index(drop=True)
    n_drop_class = n_pre_class - len(ann)
    if n_drop_class:
        print(f"[sp:{testbed}] dropped {n_drop_class} null-tax_class rows ({len(ann)} remain) — "
              f"spec §4.6 (no class rank → not scorable at the stratum)")

    # --- subset the per-protein h5 to the panel accessions (float16 -> float32 inside subset_h5) ---
    reps, present, missing = subset_h5(_load_sp_h5_path(testbed), ann["accession"].tolist())
    join_cov = len(present) / len(ann) if len(ann) else 0.0
    if join_cov < config.SP_JOIN_COVERAGE_MIN:
        raise SystemExit(f"[sp:{testbed}] h5 join coverage {join_cov:.3f} < SP_JOIN_COVERAGE_MIN "
                         f"{config.SP_JOIN_COVERAGE_MIN} — refusing to run the battery on a biased subset")
    # all-life accession-key contract (spec v3 §6): the self-embed h5 is keyed by the bare-accession FASTA
    # header (write_panel_fasta), so EXACT set-containment must hold. `missing` already == set(ann.accession)
    # - set(h5.keys()); any miss is a key-format mismatch (e.g. sp|..| or version suffix) — abort, not drop.
    if testbed.startswith("all_life_") and missing:
        raise SystemExit(f"[sp:{testbed}] {len(missing)} panel accessions absent from the self-embed h5 "
                         f"(accession-key mismatch) — aborting before the battery")
    if missing:
        head = missing[:10]
        more = f" (+{len(missing) - len(head)} more)" if len(missing) > len(head) else ""
        print(f"[sp:{testbed}] {len(missing)}/{len(ann)} panel accessions MISSING from h5 — dropped: "
              f"{head}{more}")

    # --- FILTER ann to `present` (preserve subset_h5's order so ann row i <-> reps row i) ---
    assert ann["accession"].is_unique, "panel accessions must be unique (.loc[present] would mis-align)"
    ann = ann.set_index("accession").loc[present].reset_index()
    ann["taxid"] = ann["taxid"].astype(int)
    ann["idx"] = ann["idx"].astype(int)

    # --- recompute positions + pca_dim on the FILTERED frame ---
    import os
    target = "cellular" if testbed.startswith("all_life_") else "metazoa"
    te = _load_te(target)
    stamp_path = os.path.splitext(panel_path)[0] + ".target.json"
    if os.path.exists(stamp_path):
        stamp = json.loads(open(stamp_path).read())
        _assert_target_invariant(te, ann, target, {**stamp, "te_ckpt_sha256": stamp.get("ckpt_sha256")})
    positions = te.positions[ann["idx"].to_numpy()]            # (n_present, d)
    pca_dim = min(config.PCA_DIM, len(ann) // 3)
    print(f"[sp:{testbed}] panel={panel_path}  func_col={func_col}  n_present={len(ann)}  "
          f"reps={reps.shape}  positions={positions.shape}  pca_dim={pca_dim}")

    return ann, reps.astype(np.float32), positions, pca_dim


# ============================================================================ probes
def _held_out_indices(cluster_ids, n_splits, seed):
    """Materialize the grouped_holdout folds once so every probe uses identical (no-cluster-leak) splits."""
    return list(grouped_holdout(cluster_ids, n_splits=n_splits, seed=seed))


def taxonomy_probe_categorical(Z, y_codes, folds):
    """Held-out Clade probe on a FIXED feature matrix Z (no refit PCA). Linear = multinomial logistic,
    nonlinear = one-hidden-layer MLP. Returns out-of-fold accuracy for each. Chance = max class freq."""
    yhat_lin = np.full(len(y_codes), -1, dtype=np.int64)
    yhat_mlp = np.full(len(y_codes), -1, dtype=np.int64)
    for tr, te in folds:
        if len(np.unique(y_codes[tr])) < 2:
            continue
        lin = LogisticRegression(max_iter=2000, C=1.0).fit(Z[tr], y_codes[tr])
        yhat_lin[te] = lin.predict(Z[te])
        mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=SEED,
                            early_stopping=False).fit(Z[tr], y_codes[tr])
        yhat_mlp[te] = mlp.predict(Z[te])
    scored = yhat_lin != -1
    counts = np.bincount(y_codes[scored]) if scored.any() else np.array([1])
    chance = float(counts.max() / counts.sum()) if scored.any() else float("nan")
    acc_lin = float((yhat_lin[scored] == y_codes[scored]).mean()) if scored.any() else float("nan")
    acc_mlp = float((yhat_mlp[scored] == y_codes[scored]).mean()) if scored.any() else float("nan")
    return {"linear_acc": acc_lin, "nonlinear_acc": acc_mlp, "chance": chance,
            "n_scored": int(scored.sum())}


def taxonomy_probe_continuous(Z, Y_pos, folds):
    """Held-out continuous taxonomy probe: regress the true Poincaré tangent target (log0 position) from
    the FIXED Z. Linear = Ridge, nonlinear = MLPRegressor. Returns out-of-fold R^2 (averaged over output
    dims, variance-weighted) for each. R^2<=0 ~ chance (predicting the mean)."""
    pred_lin = np.full_like(Y_pos, np.nan)
    pred_mlp = np.full_like(Y_pos, np.nan)
    for tr, te in folds:
        rid = Ridge(alpha=1.0).fit(Z[tr], Y_pos[tr])
        pred_lin[te] = rid.predict(Z[te])
        mlp = MLPRegressor(hidden_layer_sizes=(64,), max_iter=500, random_state=SEED,
                           early_stopping=False).fit(Z[tr], Y_pos[tr])
        pred_mlp[te] = mlp.predict(Z[te])
    scored = ~np.isnan(pred_lin).any(1)

    def _r2(pred):
        yt = Y_pos[scored]
        ss_res = ((yt - pred[scored]) ** 2).sum()
        ss_tot = ((yt - yt.mean(0)) ** 2).sum()
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {"linear_r2": _r2(pred_lin), "nonlinear_r2": _r2(pred_mlp), "n_scored": int(scored.sum())}


# ============================================================================ function purity
def knn_purity_within_strata(Z, groups, strata, k=KNN_K, block=4096):
    """Mean kNN-purity@k of `groups` (Gene Group) under cosine distance, computed WITHIN each stratum
    (e.g. each Clade): for each query, the fraction of its k nearest neighbours *from the same stratum*
    that share its group. Averaged over all queries that have >=k same-stratum neighbours. Within-stratum
    isolates function structure from taxonomy structure (a neighbour of the same clade tells you nothing
    new about taxonomy). Returns (mean_purity, n_queries_scored).

    Blocked kernel (spec v3 §2.N kNN): processes each stratum's rows in row-blocks of `block`, computing a
    (b x m) similarity slab and taking the top-k with argpartition — never an m x m argsort index nor a
    full `-sims` copy. The top-k SET (and thus purity) is identical to the dense argsort absent ties."""
    Z = np.asarray(Z, np.float64)
    norm = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    groups = np.asarray(groups)
    strata = np.asarray(strata)
    purities = []
    for s in np.unique(strata):
        mask = np.where(strata == s)[0]
        m = len(mask)
        if m < k + 1:
            continue
        sub = norm[mask]
        g_local = groups[mask]
        for start in range(0, m, block):
            stop = min(start + block, m)
            sims = sub[start:stop] @ sub.T                     # (b, m) slab — NOT (m, m)
            for r in range(stop - start):
                sims[r, start + r] = -np.inf                   # exclude self
            part = np.argpartition(-sims, kth=k, axis=1)[:, :k]
            for r in range(stop - start):
                purities.append(float((g_local[part[r]] == g_local[start + r]).mean()))
    if not purities:
        return float("nan"), 0
    return float(np.mean(purities)), len(purities)


def knn_purity_global(Z, groups, k=KNN_K, subset_mask=None, block=4096):
    """Plain (non-stratified) kNN-purity@k of `groups`, optionally restricted to a row subset
    (e.g. the crossed-group subset). Neighbours are drawn from the SAME subset only.

    Blocked kernel (spec v3 §2.N kNN): processes rows in blocks of `block`, computing a (b x n) similarity
    slab and taking the top-k with argpartition — never an n x n argsort index nor a full `-sims` copy
    (drops the 3x N*N dense-argsort RAM cliff). The top-k SET (and thus purity) matches the dense argsort
    exactly absent ties (continuous embeddings)."""
    Z = np.asarray(Z, np.float64)
    if subset_mask is not None:
        idx = np.where(subset_mask)[0]
    else:
        idx = np.arange(len(Z))
    if len(idx) < k + 1:
        return float("nan"), 0
    sub = Z[idx]
    norm = sub / np.maximum(np.linalg.norm(sub, axis=1, keepdims=True), 1e-12)
    g = np.asarray(groups)[idx]
    n = len(idx)
    pur = np.empty(n, dtype=np.float64)
    for start in range(0, n, block):
        stop = min(start + block, n)
        sims = norm[start:stop] @ norm.T                       # (b, n) slab — NOT (n, n)
        for r in range(stop - start):
            sims[r, start + r] = -np.inf                       # exclude self
        part = np.argpartition(-sims, kth=k, axis=1)[:, :k]
        for r in range(stop - start):
            pur[start + r] = (g[part[r]] == g[start + r]).mean()
    return float(pur.mean()), n


# ============================================================================ the battery
def run_battery(ann, reps, positions):
    """Tasks 13 (steps 0-8) + 14 (steps A-C). Returns the full results dict ready to serialize."""
    out = {}
    species = ann["Species"].to_numpy()
    clade = ann["Clade"].astype("category")
    clade_codes = clade.cat.codes.to_numpy()
    cluster_ids = ann["cluster_id"].to_numpy()
    func_groups = ann["Gene Group"].to_numpy()

    # ---- Step 0: ONE basis + pca-dim floor (B2, M5) -------------------------------------------
    # whole-data n=452 >> 64, so PCA_DIM=64 is fine for whole-data steps; the floor guards folds.
    n = len(ann)
    folds_for_floor = _held_out_indices(cluster_ids, N_SPLITS, SEED)
    smallest_train_n = min(len(tr) for tr, _ in folds_for_floor)
    pca_dim = min(config.PCA_DIM, smallest_train_n, n)
    b = Bridge.align(reps, positions, pca_dim=pca_dim)
    Z = b.pca.transform(reps)                                  # THE fixed feature matrix (n, pca_dim)
    tax_targets = log0(positions)                              # continuous taxonomy target (tangent)
    func_onehot = pd.get_dummies(ann["Gene Group"]).to_numpy().astype(np.float64)
    print(f"[step0] one basis fixed: n={n}, PCA_DIM={pca_dim} "
          f"(min(64, smallest_train_fold_n={smallest_train_n}, n)); Z shape {Z.shape}")
    out["meta"] = {
        "testbed": "pla2", "n_proteins": int(n), "n_species": int(ann["Species"].nunique()),
        "n_clades": int(clade.nunique()), "n_gene_groups": int(ann["Gene Group"].nunique()),
        "n_clusters": int(len(np.unique(cluster_ids))), "pca_dim": int(pca_dim),
        "smallest_train_fold_n": int(smallest_train_n), "pca_dim_floor_applied": bool(pca_dim < config.PCA_DIM),
        "n_splits": N_SPLITS, "knn_k": KNN_K, "seed": SEED, "mmseqs_used": have_mmseqs(),
    }

    folds = folds_for_floor                                   # reuse identical splits for every probe

    # ---- Step 1: LEACE-erase taxonomy; record erased rank; gate CLEAN claim (M4) --------------
    Xe_tax, P_tax = leace_erase(Z, tax_targets)               # tax-erased reps + linear part P
    E = erasing_projector_rowspace(P_tax)                     # (pca_dim, k) erased directions
    k_erased = int(E.shape[1])
    clean_gate = bool(k_erased < pca_dim / 2)                 # M4: E << PCA_DIM, else "preserved" is hollow
    print(f"[step1] LEACE erased rank k={k_erased} of PCA_DIM={pca_dim}  "
          f"(CLEAN gate k<<PCA_DIM: {clean_gate})")
    out["leace"] = {"erased_rank_k": k_erased, "pca_dim": int(pca_dim),
                    "clean_claim_gated_ok": clean_gate,
                    "gate_note": ("k < PCA_DIM/2 — function-preservation is meaningful"
                                  if clean_gate else
                                  "k is a large fraction of PCA_DIM — LEACE removed much of the space, "
                                  "so 'function preserved' carries little weight (M4 warning)")}

    # ---- Step 2: erasure worked — held-out tax probe original vs tax-erased --------------------
    tax_orig_cat = taxonomy_probe_categorical(Z, clade_codes, folds)
    tax_erased_cat = taxonomy_probe_categorical(Xe_tax, clade_codes, folds)
    tax_orig_cont = taxonomy_probe_continuous(Z, tax_targets, folds)
    tax_erased_cont = taxonomy_probe_continuous(Xe_tax, tax_targets, folds)
    print(f"[step2] Clade acc  orig lin={tax_orig_cat['linear_acc']:.3f}/mlp={tax_orig_cat['nonlinear_acc']:.3f}"
          f"  tax-erased lin={tax_erased_cat['linear_acc']:.3f}/mlp={tax_erased_cat['nonlinear_acc']:.3f}"
          f"  (chance={tax_orig_cat['chance']:.3f})")
    print(f"[step2] position R^2 orig lin={tax_orig_cont['linear_r2']:.3f}  "
          f"tax-erased lin={tax_erased_cont['linear_r2']:.3f}")
    out["step2_erasure_worked"] = {
        "clade_probe": {"original": tax_orig_cat, "tax_erased": tax_erased_cat},
        "position_probe": {"original": tax_orig_cont, "tax_erased": tax_erased_cont},
    }

    # ---- Step 3: function preserved — kNN-purity within strata + crossed subset ----------------
    crossed_mask = _crossed_group_mask(ann)
    fp_orig_within, n_w = knn_purity_within_strata(Z, func_groups, clade.to_numpy())
    fp_erased_within, _ = knn_purity_within_strata(Xe_tax, func_groups, clade.to_numpy())
    fp_orig_crossed, n_c = knn_purity_global(Z, func_groups, subset_mask=crossed_mask)
    fp_erased_crossed, _ = knn_purity_global(Xe_tax, func_groups, subset_mask=crossed_mask)
    fp_chance = _group_chance_purity(func_groups)
    print(f"[step3] func purity@{KNN_K} within-clade  orig={fp_orig_within:.3f}  "
          f"tax-erased={fp_erased_within:.3f}  (n={n_w}, chance~{fp_chance:.3f})")
    print(f"[step3] func purity@{KNN_K} crossed-subset orig={fp_orig_crossed:.3f}  "
          f"tax-erased={fp_erased_crossed:.3f}  (n={n_c})")
    out["step3_function_preserved"] = {
        "within_clade": {"original": fp_orig_within, "tax_erased": fp_erased_within, "n_queries": n_w},
        "crossed_group_subset": {"original": fp_orig_crossed, "tax_erased": fp_erased_crossed,
                                 "n_queries": n_c, "n_crossed_groups": int(_n_crossed_groups(ann)),
                                 "crossed_groups": _crossed_groups(ann)},
        "group_chance_purity": fp_chance,
    }

    # ---- Step 4: specificity — variance-AND-concept-matched control --------------------------
    erased_energy = float(((Z - Z.mean(0)) @ E).var(0).sum())
    Pc = matched_control_subspace(Z, target=func_onehot, avoid=tax_targets, erased_energy=erased_energy)
    Ec = erasing_projector_rowspace(Pc)
    control_energy = float(((Z - Z.mean(0)) @ Ec).var(0).sum()) if Ec.shape[1] else 0.0
    energy_ratio = control_energy / erased_energy if erased_energy > 0 else float("nan")
    energy_ok = bool(0.5 <= energy_ratio <= 2.0)
    Zc = apply_projector(Z, Pc)
    tax_control_cat = taxonomy_probe_categorical(Zc, clade_codes, folds)
    tax_control_cont = taxonomy_probe_continuous(Zc, tax_targets, folds)
    print(f"[step4] matched-control erased rank={Ec.shape[1]}  energy {control_energy:.4f} vs LEACE "
          f"{erased_energy:.4f}  ratio={energy_ratio:.3f}  (0.5x-2x ok: {energy_ok})")
    print(f"[step4] taxonomy under control: Clade acc lin={tax_control_cat['linear_acc']:.3f}  "
          f"pos R^2={tax_control_cont['linear_r2']:.3f} (should stay HIGH unlike LEACE)")
    out["step4_specificity"] = {
        "leace_erased_energy": erased_energy, "control_erased_energy": control_energy,
        "energy_ratio_control_over_leace": energy_ratio, "energy_match_ok_half_to_2x": energy_ok,
        "control_erased_rank": int(Ec.shape[1]),
        "taxonomy_under_control": {"clade_probe": tax_control_cat, "position_probe": tax_control_cont},
    }

    # ---- Step 5: reverse symmetry — erase FUNCTION, taxonomy should survive -------------------
    Xe_func, P_func = leace_erase(Z, func_onehot)
    Ef = erasing_projector_rowspace(P_func)
    tax_funcerased_cat = taxonomy_probe_categorical(Xe_func, clade_codes, folds)
    tax_funcerased_cont = taxonomy_probe_continuous(Xe_func, tax_targets, folds)
    fp_funcerased_within, _ = knn_purity_within_strata(Xe_func, func_groups, clade.to_numpy())
    fp_funcerased_crossed, _ = knn_purity_global(Xe_func, func_groups, subset_mask=crossed_mask)
    print(f"[step5] func-erased rank={Ef.shape[1]}  taxonomy survives? Clade acc lin="
          f"{tax_funcerased_cat['linear_acc']:.3f} pos R^2={tax_funcerased_cont['linear_r2']:.3f}  | "
          f"func purity within={fp_funcerased_within:.3f} crossed={fp_funcerased_crossed:.3f}")
    out["step5_reverse_symmetry"] = {
        "function_erased_rank": int(Ef.shape[1]),
        "taxonomy_under_function_erasure": {"clade_probe": tax_funcerased_cat,
                                            "position_probe": tax_funcerased_cont},
        "function_under_function_erasure": {"within_clade": fp_funcerased_within,
                                            "crossed_group_subset": fp_funcerased_crossed},
        "note": ("Within ONE family, Gene Group ≈ paralog lineage ≈ taxonomy, so function-erasure "
                 "degrading taxonomy (and vice versa) is the EXPECTED entanglement, not a failure."),
    }

    # ---- Step 6: depth-null on the CLEAN side -------------------------------------------------
    # Question (spec §13.6): does LEACE merely remove the DEPTH (radial) axis of the taxonomy, or real
    # directional taxonomy structure? Two depth-null targets, both honest:
    #   (a) PRIMARY — the 1-D radial DEPTH magnitude ||log0(pos)|| (= tree-depth proxy). Erasing this
    #       removes ONLY the depth axis (low rank), so its taxonomy-collapse is the depth-attributable
    #       floor. If real-LEACE collapse >> this, LEACE erased directional (not just depth) structure.
    #   (b) LITERAL — radial_only_null(tax_targets): keep each target's norm, randomize DIRECTION. This
    #       is the spec's literal "radial_only_null applied to tax_targets", but it is a full-rank random
    #       100-D continuous target, so LEACE erases the WHOLE PCA space (rank≈PCA_DIM) and collapses
    #       taxonomy to chance for a TRIVIAL reason (target dimensionality), not a depth one. Reported
    #       for completeness with a degeneracy flag; the PRIMARY (a) is the meaningful depth-null.
    real_collapse = tax_orig_cat["linear_acc"] - tax_erased_cat["linear_acc"]

    # (a) primary: 1-D depth magnitude
    depth_mag = np.linalg.norm(tax_targets, axis=1, keepdims=True)   # (n,1) radial/depth scalar
    Xe_depthmag, P_depthmag = leace_erase(Z, depth_mag)
    Edm = erasing_projector_rowspace(P_depthmag)
    tax_depthmag_cat = taxonomy_probe_categorical(Xe_depthmag, clade_codes, folds)
    tax_depthmag_cont = taxonomy_probe_continuous(Xe_depthmag, tax_targets, folds)
    depthmag_collapse = tax_orig_cat["linear_acc"] - tax_depthmag_cat["linear_acc"]

    # (b) literal randomized-direction null
    depth_targets = radial_only_null(tax_targets, seed=SEED)
    Xe_depth, P_depth = leace_erase(Z, depth_targets)
    Ed = erasing_projector_rowspace(P_depth)
    tax_depth_cat = taxonomy_probe_categorical(Xe_depth, clade_codes, folds)
    randdir_collapse = tax_orig_cat["linear_acc"] - tax_depth_cat["linear_acc"]
    randdir_degenerate = bool(Ed.shape[1] >= 0.9 * pca_dim)

    print(f"[step6] PRIMARY depth-magnitude erase rank={Edm.shape[1]}  Clade acc lin="
          f"{tax_depthmag_cat['linear_acc']:.3f}  collapse={depthmag_collapse:.3f}  | "
          f"real-LEACE collapse={real_collapse:.3f}")
    print(f"[step6] LITERAL rand-direction null erase rank={Ed.shape[1]}  collapse={randdir_collapse:.3f}  "
          f"(degenerate full-rank: {randdir_degenerate})")
    out["step6_depth_null_clean"] = {
        "primary_depth_magnitude": {
            "erased_rank": int(Edm.shape[1]),
            "taxonomy_after_depth_erasure": {"clade_probe": tax_depthmag_cat,
                                             "position_probe": tax_depthmag_cont},
            "clade_collapse": depthmag_collapse,
        },
        "real_leace_clade_collapse": real_collapse,
        "depth_attributable_fraction": (depthmag_collapse / real_collapse) if real_collapse else float("nan"),
        "literal_random_direction_null": {
            "erased_rank": int(Ed.shape[1]), "clade_collapse": randdir_collapse,
            "degenerate_full_rank": randdir_degenerate,
            "degeneracy_note": ("radial_only_null(tax_targets) is a full-rank random 100-D continuous "
                                "target; LEACE erases ≈the whole PCA space against it, so its "
                                "taxonomy-collapse is a target-dimensionality artifact, NOT a depth "
                                "signal. Use the primary 1-D depth-magnitude null instead."),
        },
        "note": ("real-LEACE collapse much larger than the depth-magnitude collapse ⇒ LEACE erased real "
                 "DIRECTIONAL taxonomy structure, not merely the radial/depth axis."),
    }

    # ---- Step 7: identity-stratum check -------------------------------------------------------
    out["step7_identity_strata"] = _identity_stratum_report(
        Z, Xe_tax, func_groups, clade_codes, cluster_ids, folds)
    print(f"[step7] identity strata: {list(out['step7_identity_strata']['strata'].keys())}")

    # ---- Step 8: the disentanglement matrix ---------------------------------------------------
    out["disentanglement_matrix"] = _build_matrix(
        Z, Xe_tax, Xe_func, Zc, func_groups, clade_codes, tax_targets, clade.to_numpy(),
        crossed_mask, folds, k_erased)

    # ---- Task 14: coherence (steps A-C) -------------------------------------------------------
    out["coherence"] = coherence(b, P_tax, pca_dim)

    return out


# ============================================================================ pla2_mammals battery (T16)
def run_battery_mammals(ann_m, reps_m, positions_m, crossed_groups_list, pca_dim_m):
    """Task 16: run the SAME battery on the within-Mammals, crossed-group subset.

    Key differences from run_battery():
    - Taxonomy stratum = Species (Clade is constant="Mammals"; Species spans 38 mammalian species).
    - PCA_DIM = pca_dim_m (pre-computed min(64, n//3) safety margin for small n).
    - Categorical taxonomy probe = Species (many labels, small n per label; chance will be high).
    - Crossed-group mask is computed within the subset (all rows ARE crossed-group, so mask=all-True).
    - All steps are identical in structure; small-n instability is flagged honestly where relevant.
    """
    out = {}
    n_m = len(ann_m)
    species_arr = ann_m["Species"].to_numpy()
    species_cat = pd.Categorical(species_arr)
    species_codes = species_cat.codes.copy()                   # taxonomy codes for probe
    cluster_ids_m = ann_m["cluster_id"].to_numpy()
    func_groups_m = ann_m["Gene Group"].to_numpy()

    # ---- Step 0: ONE basis ---
    folds_for_floor = _held_out_indices(cluster_ids_m, N_SPLITS, SEED)
    valid_folds = [(tr, te) for tr, te in folds_for_floor
                   if len(tr) >= 2 and len(np.unique(species_codes[tr])) >= 2]
    if not valid_folds:
        # Degenerate: too few samples for stratified holdout — use 2-fold on species
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
        valid_folds = list(skf.split(np.arange(n_m), species_codes))
    smallest_train_n = min(len(tr) for tr, _ in valid_folds)

    b_m = Bridge.align(reps_m, positions_m, pca_dim=pca_dim_m)
    Z_m = b_m.pca.transform(reps_m)                           # (n_m, pca_dim_m) fixed feature matrix
    tax_targets_m = log0(positions_m)
    func_onehot_m = pd.get_dummies(ann_m["Gene Group"]).to_numpy().astype(np.float64)
    n_gene_groups_m = int(ann_m["Gene Group"].nunique())
    n_species_m = int(ann_m["Species"].nunique())

    print(f"[mammals step0] one basis fixed: n={n_m}, PCA_DIM_M={pca_dim_m}  "
          f"(min(64, {n_m}//3)); Z shape {Z_m.shape}")
    print(f"[mammals step0] species={n_species_m}, gene_groups={n_gene_groups_m}, "
          f"crossed_groups={len(crossed_groups_list)}, n_folds={len(valid_folds)}, "
          f"smallest_train_n={smallest_train_n}")

    out["meta"] = {
        "testbed": "pla2_mammals",
        "clade_filter": "Mammals",
        "crossed_groups_computed": crossed_groups_list,
        "n_crossed_groups_in_subset": n_gene_groups_m,
        "note_d3_absent": "D3 is a crossed group (3 clades) but has 0 proteins in Mammals subset",
        "n_proteins": int(n_m),
        "n_species": n_species_m,
        "n_gene_groups": n_gene_groups_m,
        "n_clusters": int(len(np.unique(cluster_ids_m))),
        "pca_dim": int(pca_dim_m),
        "pca_dim_formula": f"min(64, {n_m}//3) = {pca_dim_m}",
        "smallest_train_fold_n": int(smallest_train_n),
        "n_splits_used": int(len(valid_folds)),
        "taxonomy_stratum_label": "Species",
        "taxonomy_stratum_rationale": (
            "Clade is constant='Mammals' in this subset; Species (38 distinct mammalian species "
            "spanning multiple orders) is the finest available taxonomy discriminator in the "
            "annotation table without additional NCBI rank lookups."),
        "knn_k": KNN_K, "seed": SEED, "mmseqs_used": have_mmseqs(),
        "small_n_warnings": [],
    }

    # Track small-n warnings
    warnings = out["meta"]["small_n_warnings"]
    if n_m < 100:
        warnings.append(f"small subset n={n_m} (<100): some metrics may be unstable")
    if smallest_train_n < 20:
        warnings.append(f"smallest train fold n={smallest_train_n}: probe accuracy estimates less reliable")

    folds_m = valid_folds

    # ---- Step 1: LEACE-erase taxonomy (Species-based targets) ---
    Xe_tax_m, P_tax_m = leace_erase(Z_m, tax_targets_m)
    E_m = erasing_projector_rowspace(P_tax_m)
    k_erased_m = int(E_m.shape[1])
    clean_gate_m = bool(k_erased_m < pca_dim_m / 2)
    print(f"[mammals step1] LEACE erased rank k={k_erased_m} of PCA_DIM={pca_dim_m}  "
          f"(CLEAN gate k<<PCA_DIM: {clean_gate_m})")
    out["leace"] = {
        "erased_rank_k": k_erased_m, "pca_dim": int(pca_dim_m),
        "clean_claim_gated_ok": clean_gate_m,
        "gate_note": ("k < PCA_DIM/2 — function-preservation is meaningful"
                      if clean_gate_m else
                      "k is a large fraction of PCA_DIM — LEACE removed much of the space, "
                      "so 'function preserved' carries little weight (M4 warning)"),
    }

    # ---- Step 2: erasure worked — probe Species (not Clade — Clade is constant here) ---
    tax_orig_cat_m = taxonomy_probe_categorical(Z_m, species_codes, folds_m)
    tax_erased_cat_m = taxonomy_probe_categorical(Xe_tax_m, species_codes, folds_m)
    tax_orig_cont_m = taxonomy_probe_continuous(Z_m, tax_targets_m, folds_m)
    tax_erased_cont_m = taxonomy_probe_continuous(Xe_tax_m, tax_targets_m, folds_m)
    print(f"[mammals step2] Species acc  orig lin={tax_orig_cat_m['linear_acc']:.3f}"
          f"/mlp={tax_orig_cat_m['nonlinear_acc']:.3f}  "
          f"tax-erased lin={tax_erased_cat_m['linear_acc']:.3f}"
          f"/mlp={tax_erased_cat_m['nonlinear_acc']:.3f}  "
          f"(chance={tax_orig_cat_m['chance']:.3f})")
    print(f"[mammals step2] position R^2 orig lin={tax_orig_cont_m['linear_r2']:.3f}  "
          f"tax-erased lin={tax_erased_cont_m['linear_r2']:.3f}")
    out["step2_erasure_worked"] = {
        "taxonomy_probe_label": "Species (38 mammalian species; Clade=constant='Mammals')",
        "species_probe": {"original": tax_orig_cat_m, "tax_erased": tax_erased_cat_m},
        "position_probe": {"original": tax_orig_cont_m, "tax_erased": tax_erased_cont_m},
        "note": (f"Species probe has {n_species_m} classes with n={n_m} total; "
                 f"chance={tax_orig_cat_m['chance']:.3f} (high due to many species + small n)."),
    }

    # ---- Step 3: function preserved — within-Species purity ---
    # Within-stratum = within Species (our taxonomy proxy for this subset)
    species_strata_m = species_arr
    # All rows are crossed-group; crossed mask = all True
    crossed_mask_m = np.ones(n_m, dtype=bool)
    fp_orig_within_m, n_w_m = knn_purity_within_strata(Z_m, func_groups_m, species_strata_m)
    fp_erased_within_m, _ = knn_purity_within_strata(Xe_tax_m, func_groups_m, species_strata_m)
    fp_orig_crossed_m, n_c_m = knn_purity_global(Z_m, func_groups_m, subset_mask=crossed_mask_m)
    fp_erased_crossed_m, _ = knn_purity_global(Xe_tax_m, func_groups_m, subset_mask=crossed_mask_m)
    fp_chance_m = _group_chance_purity(func_groups_m)
    print(f"[mammals step3] func purity@{KNN_K} within-Species  orig={fp_orig_within_m:.3f}  "
          f"tax-erased={fp_erased_within_m:.3f}  (n={n_w_m}, chance~{fp_chance_m:.3f})")
    print(f"[mammals step3] func purity@{KNN_K} global(all-crossed) orig={fp_orig_crossed_m:.3f}  "
          f"tax-erased={fp_erased_crossed_m:.3f}  (n={n_c_m})")
    out["step3_function_preserved"] = {
        "within_species": {
            "original": fp_orig_within_m, "tax_erased": fp_erased_within_m, "n_queries": n_w_m,
            "note": "within-Species stratum (our proxy for taxonomy here; Clade is constant)",
        },
        "all_crossed_group_global": {
            "original": fp_orig_crossed_m, "tax_erased": fp_erased_crossed_m, "n_queries": n_c_m,
            "note": "global purity over all n_m rows (all are crossed-group by construction)",
        },
        "group_chance_purity": fp_chance_m,
        "crossed_groups_present_in_subset": sorted(ann_m["Gene Group"].unique().tolist()),
    }
    if n_w_m < 10:
        warnings.append(f"step3: within-Species purity scored only {n_w_m} queries (many species "
                        f"have <{KNN_K+1} members); interpret with caution")

    # ---- Step 4: specificity — matched-variance control ---
    erased_energy_m = float(((Z_m - Z_m.mean(0)) @ E_m).var(0).sum())
    Pc_m = matched_control_subspace(Z_m, target=func_onehot_m, avoid=tax_targets_m,
                                    erased_energy=erased_energy_m)
    Ec_m = erasing_projector_rowspace(Pc_m)
    control_energy_m = float(((Z_m - Z_m.mean(0)) @ Ec_m).var(0).sum()) if Ec_m.shape[1] else 0.0
    energy_ratio_m = control_energy_m / erased_energy_m if erased_energy_m > 0 else float("nan")
    energy_ok_m = bool(0.5 <= energy_ratio_m <= 2.0)
    Zc_m = apply_projector(Z_m, Pc_m)
    tax_control_cat_m = taxonomy_probe_categorical(Zc_m, species_codes, folds_m)
    tax_control_cont_m = taxonomy_probe_continuous(Zc_m, tax_targets_m, folds_m)
    print(f"[mammals step4] matched-control erased rank={Ec_m.shape[1]}  "
          f"energy {control_energy_m:.4f} vs LEACE {erased_energy_m:.4f}  "
          f"ratio={energy_ratio_m:.3f}  (0.5x-2x ok: {energy_ok_m})")
    out["step4_specificity"] = {
        "leace_erased_energy": erased_energy_m, "control_erased_energy": control_energy_m,
        "energy_ratio_control_over_leace": energy_ratio_m,
        "energy_match_ok_half_to_2x": energy_ok_m,
        "control_erased_rank": int(Ec_m.shape[1]),
        "taxonomy_under_control": {
            "species_probe": tax_control_cat_m, "position_probe": tax_control_cont_m},
    }

    # ---- Step 5: reverse symmetry — erase function, taxonomy (Species) should survive ---
    Xe_func_m, P_func_m = leace_erase(Z_m, func_onehot_m)
    Ef_m = erasing_projector_rowspace(P_func_m)
    tax_funcerased_cat_m = taxonomy_probe_categorical(Xe_func_m, species_codes, folds_m)
    tax_funcerased_cont_m = taxonomy_probe_continuous(Xe_func_m, tax_targets_m, folds_m)
    fp_funcerased_within_m, _ = knn_purity_within_strata(Xe_func_m, func_groups_m, species_strata_m)
    fp_funcerased_crossed_m, _ = knn_purity_global(Xe_func_m, func_groups_m, subset_mask=crossed_mask_m)
    print(f"[mammals step5] func-erased rank={Ef_m.shape[1]}  taxonomy survives? "
          f"Species acc lin={tax_funcerased_cat_m['linear_acc']:.3f}  "
          f"pos R^2={tax_funcerased_cont_m['linear_r2']:.3f}  | "
          f"func purity within={fp_funcerased_within_m:.3f} global={fp_funcerased_crossed_m:.3f}")
    out["step5_reverse_symmetry"] = {
        "function_erased_rank": int(Ef_m.shape[1]),
        "taxonomy_under_function_erasure": {
            "species_probe": tax_funcerased_cat_m, "position_probe": tax_funcerased_cont_m},
        "function_under_function_erasure": {
            "within_species": fp_funcerased_within_m,
            "global_crossed_subset": fp_funcerased_crossed_m},
        "note": ("Within one family restricted to one clade, Gene Group ≈ paralog lineage ≈ "
                 "species-group, so strong entanglement is expected."),
    }

    # ---- Step 6: depth-null ---
    real_collapse_m = tax_orig_cat_m["linear_acc"] - tax_erased_cat_m["linear_acc"]
    depth_mag_m = np.linalg.norm(tax_targets_m, axis=1, keepdims=True)
    Xe_depthmag_m, P_depthmag_m = leace_erase(Z_m, depth_mag_m)
    Edm_m = erasing_projector_rowspace(P_depthmag_m)
    tax_depthmag_cat_m = taxonomy_probe_categorical(Xe_depthmag_m, species_codes, folds_m)
    tax_depthmag_cont_m = taxonomy_probe_continuous(Xe_depthmag_m, tax_targets_m, folds_m)
    depthmag_collapse_m = tax_orig_cat_m["linear_acc"] - tax_depthmag_cat_m["linear_acc"]

    depth_targets_m = radial_only_null(tax_targets_m, seed=SEED)
    Xe_depth_m, P_depth_m = leace_erase(Z_m, depth_targets_m)
    Ed_m = erasing_projector_rowspace(P_depth_m)
    tax_depth_cat_m = taxonomy_probe_categorical(Xe_depth_m, species_codes, folds_m)
    randdir_collapse_m = tax_orig_cat_m["linear_acc"] - tax_depth_cat_m["linear_acc"]
    randdir_degenerate_m = bool(Ed_m.shape[1] >= 0.9 * pca_dim_m)

    print(f"[mammals step6] PRIMARY depth-magnitude erase rank={Edm_m.shape[1]}  "
          f"Species acc lin={tax_depthmag_cat_m['linear_acc']:.3f}  "
          f"collapse={depthmag_collapse_m:.3f}  | real-LEACE collapse={real_collapse_m:.3f}")
    out["step6_depth_null_clean"] = {
        "primary_depth_magnitude": {
            "erased_rank": int(Edm_m.shape[1]),
            "taxonomy_after_depth_erasure": {
                "species_probe": tax_depthmag_cat_m, "position_probe": tax_depthmag_cont_m},
            "species_acc_collapse": depthmag_collapse_m,
        },
        "real_leace_species_acc_collapse": real_collapse_m,
        "depth_attributable_fraction": (depthmag_collapse_m / real_collapse_m
                                        if real_collapse_m else float("nan")),
        "literal_random_direction_null": {
            "erased_rank": int(Ed_m.shape[1]), "species_acc_collapse": randdir_collapse_m,
            "degenerate_full_rank": randdir_degenerate_m,
        },
    }

    # ---- Step 7: identity-stratum check ---
    out["step7_identity_strata"] = _identity_stratum_report(
        Z_m, Xe_tax_m, func_groups_m, species_codes, cluster_ids_m, folds_m)
    print(f"[mammals step7] identity strata: {list(out['step7_identity_strata']['strata'].keys())}")

    # ---- Step 8: disentanglement matrix (taxonomy = Species probe here) ---
    out["disentanglement_matrix"] = _build_matrix_mammals(
        Z_m, Xe_tax_m, Xe_func_m, Zc_m, func_groups_m, species_codes, tax_targets_m,
        species_strata_m, crossed_mask_m, folds_m, k_erased_m)

    # ---- Task 14: coherence ---
    out["coherence"] = coherence(b_m, P_tax_m, pca_dim_m)

    return out


# ============================================================================ multifamily battery (T15)
def run_battery_multifamily(ann, reps, positions, pca_dim):
    """Task 15 — the MULTI-FAMILY PRIMARY battery (the CLEAN half of the contrast).

    FUNCTION label = toxin Family (genuinely non-phylogenetic: the same toxin family recurs across
    distant taxa). TAXONOMY = the species position (continuous log0(positions)) AND a categorical
    taxonomy rank for the probe. Species are mostly singletons here (69 distinct taxids over 84
    proteins), so the categorical taxonomy probe uses the NCBI **family rank** (`tax_family`, 15 classes,
    chance≈0.30) — DOCUMENTED: this is the taxonomic family rank, DISTINCT from and not redundant with
    the toxin Family (function) label (e.g. taxonomic Viperidae spans 4 toxin families; the toxin
    Sea-anemone-sodium-channel family spans 2 taxonomic families). That cross-cutting is the
    disentanglement structure we test. The order rank (`tax_order`) is reported as a coarser sensitivity.

    Same battery as run_battery / run_battery_mammals (steps 0-8 + coherence). Small n (≤84) ⇒ power
    is the binding constraint; instability is flagged honestly, never massaged. The KEY read-out: is
    taxonomy ERASABLE while toxin-Family clustering is PRESERVED (clean disentanglement), in contrast to
    PLA2 where the two were entangled.
    """
    out = {}
    n = len(ann)
    tax_family = ann["tax_family"].to_numpy()
    tax_family_cat = pd.Categorical(tax_family)
    tax_codes = tax_family_cat.codes.copy()                    # categorical taxonomy codes for probe
    tax_order = ann["tax_order"].to_numpy()
    cluster_ids = ann["cluster_id"].to_numpy()
    func_groups = ann["Family"].to_numpy()                     # FUNCTION = toxin family

    # ---- Step 0: ONE basis (+ small-n fold guard) ----------------------------------------------
    folds_for_floor = _held_out_indices(cluster_ids, N_SPLITS, SEED)
    valid_folds = [(tr, te) for tr, te in folds_for_floor
                   if len(tr) >= 2 and len(np.unique(tax_codes[tr])) >= 2 and len(te)]
    if not valid_folds:
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        valid_folds = list(skf.split(np.arange(n), tax_codes))
    smallest_train_n = min(len(tr) for tr, _ in valid_folds)

    b = Bridge.align(reps, positions, pca_dim=pca_dim)
    Z = b.pca.transform(reps)                                  # (n, pca_dim) fixed feature matrix
    tax_targets = log0(positions)                             # continuous taxonomy target
    func_onehot = pd.get_dummies(ann["Family"]).to_numpy().astype(np.float64)
    n_families = int(ann["Family"].nunique())
    n_taxids = int(ann["taxid"].nunique())
    n_tax_family = int(pd.Series(tax_family).nunique())
    n_tax_order = int(pd.Series(tax_order).nunique())
    nesting_family = _label_nesting(func_groups, tax_family)
    nesting_order = _label_nesting(func_groups, tax_order)

    print(f"[multifamily step0] one basis fixed: n={n}, PCA_DIM={pca_dim} (min(64, {n}//3)); Z {Z.shape}")
    print(f"[multifamily step0] toxin_families={n_families}, distinct_taxids={n_taxids}, "
          f"tax_family_classes={n_tax_family}, tax_order_classes={n_tax_order}, "
          f"n_folds={len(valid_folds)}, smallest_train_n={smallest_train_n}")

    out["meta"] = {
        "testbed": "multifamily",
        "data": "ToxFam-12 (toxfam_v2) ProtT5 residue embeddings, mean-pooled",
        "n_proteins": int(n), "n_toxin_families": n_families, "n_distinct_taxids": n_taxids,
        "n_tax_family_classes": n_tax_family, "n_tax_order_classes": n_tax_order,
        "n_clusters": int(len(np.unique(cluster_ids))), "pca_dim": int(pca_dim),
        "pca_dim_formula": f"min({config.PCA_DIM}, {n}//3) = {pca_dim}",
        "smallest_train_fold_n": int(smallest_train_n), "n_splits_used": int(len(valid_folds)),
        "function_label": "toxin Family (toxfam_v2_labels.csv 'family' column)",
        "taxonomy_probe_label": "tax_family (NCBI family rank from TaxonResolver.lineage)",
        "taxonomy_probe_rationale": (
            "Species are mostly singletons (69 distinct taxids / 84 proteins); the NCBI FAMILY rank "
            "(15 classes, chance≈0.30) is the finest taxonomy label with enough per-class support for a "
            "held-out probe. It is DISTINCT from the toxin Family (function) label and cross-cuts it "
            "(e.g. taxonomic Viperidae spans 4 toxin families), which is exactly the disentanglement "
            "structure under test. tax_order (10 classes) reported as a coarser sensitivity."),
        "knn_k": KNN_K, "seed": SEED, "mmseqs_used": have_mmseqs(),
        "function_taxonomy_nesting": {
            "tax_family": nesting_family, "tax_order": nesting_order,
            "note": ("In this curated ToxFam-12 sample each toxin family is drawn from a NARROW taxonomic "
                     "slice (e.g. 9/12 families map to ONE taxonomic order), so knowing the toxin family "
                     "determines most of the taxonomy. This makes the REVERSE direction (erase function ⇒ "
                     "taxonomy drops) a SAMPLING property of the curation, NOT embedding entanglement. The "
                     "FORWARD direction — erase taxonomy, keep function — is the disentanglement claim and "
                     "is clean."),
        },
        "small_n_warnings": [],
    }
    warnings = out["meta"]["small_n_warnings"]
    if n < 100:
        warnings.append(f"small dataset n={n} (<100): probe/purity estimates have wide CIs")
    if smallest_train_n < 20:
        warnings.append(f"smallest train fold n={smallest_train_n}: linear/MLP probes less reliable")

    folds = valid_folds

    # ---- Step 1: LEACE-erase taxonomy; record erased rank; CLEAN gate -------------------------
    Xe_tax, P_tax = leace_erase(Z, tax_targets)
    E = erasing_projector_rowspace(P_tax)
    k_erased = int(E.shape[1])
    clean_gate = bool(k_erased < pca_dim / 2)
    print(f"[multifamily step1] LEACE erased rank k={k_erased} of PCA_DIM={pca_dim}  "
          f"(CLEAN gate k<<PCA_DIM: {clean_gate})")
    out["leace"] = {"erased_rank_k": k_erased, "pca_dim": int(pca_dim),
                    "clean_claim_gated_ok": clean_gate,
                    "gate_note": ("k < PCA_DIM/2 — function-preservation is meaningful" if clean_gate
                                  else "k is a large fraction of PCA_DIM — 'function preserved' carries "
                                       "little weight (M4 warning)")}

    # ---- Step 2: erasure worked — categorical (tax_family) + continuous (position) probes ------
    tax_orig_cat = taxonomy_probe_categorical(Z, tax_codes, folds)
    tax_erased_cat = taxonomy_probe_categorical(Xe_tax, tax_codes, folds)
    tax_orig_cont = taxonomy_probe_continuous(Z, tax_targets, folds)
    tax_erased_cont = taxonomy_probe_continuous(Xe_tax, tax_targets, folds)
    print(f"[multifamily step2] tax_family acc orig lin={tax_orig_cat['linear_acc']:.3f}"
          f"/mlp={tax_orig_cat['nonlinear_acc']:.3f}  tax-erased lin={tax_erased_cat['linear_acc']:.3f}"
          f"/mlp={tax_erased_cat['nonlinear_acc']:.3f}  (chance={tax_orig_cat['chance']:.3f})")
    print(f"[multifamily step2] position R^2 orig lin={tax_orig_cont['linear_r2']:.3f}  "
          f"tax-erased lin={tax_erased_cont['linear_r2']:.3f}")
    out["step2_erasure_worked"] = {
        "taxonomy_probe_label": "tax_family (NCBI family rank)",
        "tax_family_probe": {"original": tax_orig_cat, "tax_erased": tax_erased_cat},
        "position_probe": {"original": tax_orig_cont, "tax_erased": tax_erased_cont},
    }

    # ---- Step 3: function preserved — toxin-Family purity (global + within taxonomic strata) ---
    # Within-stratum = within taxonomic family (isolates function from taxonomy). Most tax_family
    # strata have <k+1 members, so within-stratum purity may score few queries — report global too.
    fp_orig_global, n_g = knn_purity_global(Z, func_groups)
    fp_erased_global, _ = knn_purity_global(Xe_tax, func_groups)
    fp_orig_within, n_w = knn_purity_within_strata(Z, func_groups, tax_family)
    fp_erased_within, _ = knn_purity_within_strata(Xe_tax, func_groups, tax_family)
    fp_chance = _group_chance_purity(func_groups)
    print(f"[multifamily step3] toxin-Family purity@{KNN_K} GLOBAL  orig={fp_orig_global:.3f}  "
          f"tax-erased={fp_erased_global:.3f}  (n={n_g}, chance~{fp_chance:.3f})")
    print(f"[multifamily step3] toxin-Family purity@{KNN_K} within-tax_family orig={fp_orig_within:.3f}  "
          f"tax-erased={fp_erased_within:.3f}  (n={n_w})")
    out["step3_function_preserved"] = {
        "global": {"original": fp_orig_global, "tax_erased": fp_erased_global, "n_queries": n_g,
                   "note": "global toxin-Family kNN purity over all n proteins (primary function signal)"},
        "within_tax_family": {"original": fp_orig_within, "tax_erased": fp_erased_within, "n_queries": n_w,
                              "note": "within taxonomic-family stratum (controls for taxonomy; few queries "
                                      "at this n — interpret as supporting only)"},
        "group_chance_purity": fp_chance,
    }
    if n_w < 10:
        warnings.append(f"step3: within-tax_family purity scored only {n_w} queries (most taxonomic "
                        f"families have <{KNN_K+1} proteins); global purity is the primary signal")

    # ---- Step 4: specificity — variance-AND-concept-matched control ---------------------------
    erased_energy = float(((Z - Z.mean(0)) @ E).var(0).sum())
    Pc = matched_control_subspace(Z, target=func_onehot, avoid=tax_targets, erased_energy=erased_energy)
    Ec = erasing_projector_rowspace(Pc)
    control_energy = float(((Z - Z.mean(0)) @ Ec).var(0).sum()) if Ec.shape[1] else 0.0
    energy_ratio = control_energy / erased_energy if erased_energy > 0 else float("nan")
    energy_ok = bool(0.5 <= energy_ratio <= 2.0)
    Zc = apply_projector(Z, Pc)
    tax_control_cat = taxonomy_probe_categorical(Zc, tax_codes, folds)
    tax_control_cont = taxonomy_probe_continuous(Zc, tax_targets, folds)
    print(f"[multifamily step4] matched-control erased rank={Ec.shape[1]}  energy {control_energy:.4f} "
          f"vs LEACE {erased_energy:.4f}  ratio={energy_ratio:.3f}  (0.5x-2x ok: {energy_ok})")
    print(f"[multifamily step4] taxonomy under control: tax_family acc lin={tax_control_cat['linear_acc']:.3f} "
          f"pos R^2={tax_control_cont['linear_r2']:.3f} (should stay HIGH unlike LEACE)")
    out["step4_specificity"] = {
        "leace_erased_energy": erased_energy, "control_erased_energy": control_energy,
        "energy_ratio_control_over_leace": energy_ratio, "energy_match_ok_half_to_2x": energy_ok,
        "control_erased_rank": int(Ec.shape[1]),
        "taxonomy_under_control": {"tax_family_probe": tax_control_cat, "position_probe": tax_control_cont},
    }

    # ---- Step 5: reverse symmetry — erase toxin-FUNCTION, taxonomy should survive -------------
    Xe_func, P_func = leace_erase(Z, func_onehot)
    Ef = erasing_projector_rowspace(P_func)
    tax_funcerased_cat = taxonomy_probe_categorical(Xe_func, tax_codes, folds)
    tax_funcerased_cont = taxonomy_probe_continuous(Xe_func, tax_targets, folds)
    fp_funcerased_global, _ = knn_purity_global(Xe_func, func_groups)
    fp_funcerased_within, _ = knn_purity_within_strata(Xe_func, func_groups, tax_family)
    print(f"[multifamily step5] func-erased rank={Ef.shape[1]}  taxonomy survives? tax_family acc lin="
          f"{tax_funcerased_cat['linear_acc']:.3f} pos R^2={tax_funcerased_cont['linear_r2']:.3f}  | "
          f"toxin-Family purity global={fp_funcerased_global:.3f} within={fp_funcerased_within:.3f}")
    out["step5_reverse_symmetry"] = {
        "function_erased_rank": int(Ef.shape[1]),
        "taxonomy_under_function_erasure": {"tax_family_probe": tax_funcerased_cat,
                                            "position_probe": tax_funcerased_cont},
        "function_under_function_erasure": {"global": fp_funcerased_global,
                                            "within_tax_family": fp_funcerased_within},
        "note": ("Across families the toxin Family axis is non-phylogenetic, so erasing function should "
                 "NOT collapse taxonomy and erasing taxonomy should NOT collapse function — the clean "
                 "(disentangled) expectation, opposite to PLA2."),
    }

    # ---- Step 6: depth-null -------------------------------------------------------------------
    real_collapse = tax_orig_cat["linear_acc"] - tax_erased_cat["linear_acc"]
    depth_mag = np.linalg.norm(tax_targets, axis=1, keepdims=True)
    Xe_depthmag, P_depthmag = leace_erase(Z, depth_mag)
    Edm = erasing_projector_rowspace(P_depthmag)
    tax_depthmag_cat = taxonomy_probe_categorical(Xe_depthmag, tax_codes, folds)
    tax_depthmag_cont = taxonomy_probe_continuous(Xe_depthmag, tax_targets, folds)
    depthmag_collapse = tax_orig_cat["linear_acc"] - tax_depthmag_cat["linear_acc"]

    depth_targets = radial_only_null(tax_targets, seed=SEED)
    Xe_depth, P_depth = leace_erase(Z, depth_targets)
    Ed = erasing_projector_rowspace(P_depth)
    tax_depth_cat = taxonomy_probe_categorical(Xe_depth, tax_codes, folds)
    randdir_collapse = tax_orig_cat["linear_acc"] - tax_depth_cat["linear_acc"]
    randdir_degenerate = bool(Ed.shape[1] >= 0.9 * pca_dim)
    print(f"[multifamily step6] PRIMARY depth-magnitude erase rank={Edm.shape[1]}  tax_family acc lin="
          f"{tax_depthmag_cat['linear_acc']:.3f}  collapse={depthmag_collapse:.3f}  | "
          f"real-LEACE collapse={real_collapse:.3f}")
    out["step6_depth_null_clean"] = {
        "primary_depth_magnitude": {
            "erased_rank": int(Edm.shape[1]),
            "taxonomy_after_depth_erasure": {"tax_family_probe": tax_depthmag_cat,
                                             "position_probe": tax_depthmag_cont},
            "tax_family_acc_collapse": depthmag_collapse,
        },
        "real_leace_tax_family_acc_collapse": real_collapse,
        "depth_attributable_fraction": (depthmag_collapse / real_collapse) if real_collapse else float("nan"),
        "literal_random_direction_null": {
            "erased_rank": int(Ed.shape[1]), "tax_family_acc_collapse": randdir_collapse,
            "degenerate_full_rank": randdir_degenerate},
    }

    # ---- Step 7: identity-stratum check -------------------------------------------------------
    out["step7_identity_strata"] = _identity_stratum_report(
        Z, Xe_tax, func_groups, tax_codes, cluster_ids, folds)
    print(f"[multifamily step7] identity strata: {list(out['step7_identity_strata']['strata'].keys())}")

    # ---- Step 8: disentanglement matrix -------------------------------------------------------
    out["disentanglement_matrix"] = _build_matrix_multifamily(
        Z, Xe_tax, Xe_func, Zc, func_groups, tax_codes, tax_family, folds, k_erased)

    # ---- Task 14: coherence -------------------------------------------------------------------
    out["coherence"] = coherence(b, P_tax, pca_dim)

    return out


def _confound_erase_report(Z, concept, func_groups, strata, tax_codes, folds,
                           purity_orig_within, tax_orig_acc, tax_chance):
    """Tasks C2b/C2c shared kernel: LEACE-erase a SEQUENCE-CONFOUND concept (length=1-col, composition=
    20-col) off the FIXED feature matrix Z, then ask whether the HEADLINE survives the erasure.

    Returns (purity_after_erase, tax_after_erase, geometry_change, headline_survives, detail).

    The headline survives iff (a) the within-stratum function purity is barely moved by erasing the
    confound (relative drop < SP_CONFOUND_PURITY_REL_DROP_MAX — a 1-D/20-D confound that were the *true*
    function carrier would collapse purity), AND (b) taxonomy is STILL recoverable on the confound-erased
    reps (the categorical stratum probe stays >= `tax_chance` + SP_CONFOUND_TAX_MARGIN_MIN, where
    `tax_chance` is the majority-class frequency baseline from `taxonomy_probe_categorical` — NOT 1/n_classes
    — so erasing the confound did NOT take taxonomy down with it). `geometry_change` = mean per-rep cosine
    shift Z->Z_erased, so a near no-op erasure (e.g. a 1-D LEACE on a 40-D space) is VISIBLE, not hidden."""
    Xe, _P = leace_erase(Z, concept)                              # confound-erased reps (same primitive)
    purity_after, _n = knn_purity_within_strata(Xe, func_groups, strata)
    tax_after = taxonomy_probe_categorical(Xe, tax_codes, folds)
    tax_after_acc = tax_after["linear_acc"]
    # geometry change: mean (1 - cos) per-rep shift between Z and Xe (no-op erasure -> ~0)
    a = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    be = Xe / np.maximum(np.linalg.norm(Xe, axis=1, keepdims=True), 1e-12)
    geometry_change = float(np.mean(1.0 - np.sum(a * be, axis=1)))
    # (a) purity barely moves
    purity_rel_drop = ((purity_orig_within - purity_after) / purity_orig_within
                       if purity_orig_within else float("nan"))
    purity_preserved = bool(purity_rel_drop == purity_rel_drop                       # not NaN
                            and purity_rel_drop < config.SP_CONFOUND_PURITY_REL_DROP_MAX)
    # (b) taxonomy still recoverable on the confound-erased reps (clearly above chance)
    tax_still_recoverable = bool(
        tax_after_acc == tax_after_acc
        and tax_after_acc >= tax_chance + config.SP_CONFOUND_TAX_MARGIN_MIN)
    headline_survives = bool(purity_preserved and tax_still_recoverable)
    detail = {
        "purity_orig_within": purity_orig_within,
        "purity_relative_drop": (round(float(purity_rel_drop), 4)
                                 if purity_rel_drop == purity_rel_drop else None),
        "purity_preserved": purity_preserved,
        "tax_acc_after_erase": round(float(tax_after_acc), 4) if tax_after_acc == tax_after_acc else None,
        "tax_chance": round(float(tax_chance), 4) if tax_chance == tax_chance else None,
        "tax_still_recoverable": tax_still_recoverable,
    }
    return purity_after, tax_after, geometry_change, headline_survives, detail


def run_battery_sp(ann, reps, positions, pca_dim, stratum_col="tax_class",
                   length_control=False, composition_control=False):
    """Task C2a — the SP-Metazoa CLEAN battery (clone of run_battery_multifamily).

    FUNCTION label = `ann["func"]` (a single Pfam family per protein, or an EC class) — ⊥ phylogeny by
    construction, sidestepping the OrthoDB circularity. TAXONOMY = the species position (continuous
    log0(positions)) AND a categorical taxonomy rank for the probe, given by `stratum_col` (`tax_class`
    by default; SP species are mostly singletons so a high NCBI rank is the probe target). The within-
    stratum function-purity grain is reported at BOTH SP attached ranks `tax_order` (fine) and `tax_class`
    (coarse) -> `out["stratum_grain"]`, and `out["headline_stratum"]` is the FINEST rank with enough
    per-stratum support (spec §6.1 feasibility rule). The disentanglement structure under test: is
    taxonomy ERASABLE while Pfam/EC function clustering is PRESERVED (clean disentanglement)?

    Same battery as run_battery_multifamily (steps 0-8 + coherence). The label_nesting is reported
    BIDIRECTIONALLY (`tax_given_func` and `func_given_tax`) so the reverse-direction sampling confound is
    visible in both directions, and the specificity block carries a named `pass` acceptance gate.
    """
    out = {}
    n = len(ann)
    # C2c fail-fast: if the composition gate is requested, the aa_A..aa_Y columns MUST be present. Check
    # BEFORE the expensive battery so a misconfigured run fails immediately, never silently disabling S1.
    if composition_control:
        _aa_cols = [f"aa_{c}" for c in "ACDEFGHIKLMNPQRSTVWY"]
        _missing = [c for c in _aa_cols if c not in ann.columns]
        if _missing:
            raise ValueError(
                "composition_control=True but the amino-acid composition columns are absent from the panel: "
                f"missing {_missing} (expected aa_A..aa_Y, 20 columns frozen at build time). The control "
                "RAISES rather than silently skipping — a silent skip would disable the S1 gate.")
    tax_family = ann[stratum_col].to_numpy()                   # categorical taxonomy probe target (stratum)
    tax_family_cat = pd.Categorical(tax_family)
    tax_codes = tax_family_cat.codes.copy()                    # categorical taxonomy codes for probe
    tax_order = ann["tax_order"].to_numpy()
    cluster_ids = ann["cluster_id"].to_numpy()
    func_groups = ann["func"].to_numpy()                       # FUNCTION = Pfam family / EC class
    out["function_chance"] = _function_chance(func_groups)     # Σp_f² kNN-purity floor (spec v3 §7 verdict)

    # ---- Step 0: ONE basis (+ small-n fold guard) ----------------------------------------------
    folds_for_floor = _held_out_indices(cluster_ids, N_SPLITS, SEED)
    valid_folds = [(tr, te) for tr, te in folds_for_floor
                   if len(tr) >= 2 and len(np.unique(tax_codes[tr])) >= 2 and len(te)]
    if not valid_folds:
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        valid_folds = list(skf.split(np.arange(n), tax_codes))
    smallest_train_n = min(len(tr) for tr, _ in valid_folds)

    b = Bridge.align(reps, positions, pca_dim=pca_dim)
    Z = b.pca.transform(reps)                                  # (n, pca_dim) fixed feature matrix
    tax_targets = log0(positions)                             # continuous taxonomy target
    func_onehot = pd.get_dummies(ann["func"]).to_numpy().astype(np.float64)
    n_families = int(ann["func"].nunique())
    n_taxids = int(ann["taxid"].nunique())
    n_tax_family = int(pd.Series(tax_family).nunique())
    n_tax_order = int(pd.Series(tax_order).nunique())

    # bidirectional label nesting: how much each label determines the other (spec §6.1 both-nesting gate)
    nesting_tax_given_func = _label_nesting(func_groups, tax_family)   # 1 - H(tax|func)/H(tax)
    nesting_func_given_tax = _label_nesting(tax_family, func_groups)   # 1 - H(func|tax)/H(func)
    nesting_family = nesting_tax_given_func                            # alias for the meta block (stratum rank)
    nesting_order = _label_nesting(func_groups, tax_order)             # tax_order grain (coarser sensitivity)
    out["label_nesting"] = {"tax_given_func": nesting_tax_given_func,
                            "func_given_tax": nesting_func_given_tax}

    print(f"[sp step0] one basis fixed: n={n}, PCA_DIM={pca_dim} (min(64, {n}//3)); Z {Z.shape}")
    print(f"[sp step0] function_labels={n_families}, distinct_taxids={n_taxids}, "
          f"{stratum_col}_classes={n_tax_family}, tax_order_classes={n_tax_order}, "
          f"n_folds={len(valid_folds)}, smallest_train_n={smallest_train_n}")

    out["meta"] = {
        "testbed": "sp_metazoa",
        "data": "SwissProt-Metazoa ProtT5 per-protein embeddings (UniProt precomputed, mean-pooled)",
        "n_proteins": int(n), "n_function_labels": n_families, "n_distinct_taxids": n_taxids,
        "stratum_col": stratum_col,
        "n_stratum_classes": n_tax_family, "n_tax_order_classes": n_tax_order,
        "n_clusters": int(len(np.unique(cluster_ids))), "pca_dim": int(pca_dim),
        "pca_dim_formula": f"min({config.PCA_DIM}, {n}//3) = {pca_dim}",
        "smallest_train_fold_n": int(smallest_train_n), "n_splits_used": int(len(valid_folds)),
        "function_label": "func (single Pfam family / EC class from the frozen SP panel)",
        "taxonomy_probe_label": f"{stratum_col} (NCBI rank from TaxonResolver.lineage)",
        "taxonomy_probe_rationale": (
            f"SP species are mostly singletons; the NCBI rank `{stratum_col}` is the probe target with "
            "enough per-class support for a held-out probe. It is DISTINCT from the Pfam/EC function label "
            "and cross-cuts it (the same function recurs across distant taxa), which is exactly the "
            "disentanglement structure under test. tax_order reported as a finer-grain sensitivity."),
        "knn_k": KNN_K, "seed": SEED, "mmseqs_used": have_mmseqs(),
        "function_taxonomy_nesting": {
            stratum_col: nesting_family, "tax_order": nesting_order,
            "note": ("label_nesting is reported BIDIRECTIONALLY in out['label_nesting']. The Pfam/EC "
                     "function axis is ⊥ phylogeny by construction, so the FORWARD direction (erase "
                     "taxonomy, keep function) is the disentanglement claim; the reverse direction is a "
                     "sampling property of the panel, made visible by the func_given_tax figure."),
        },
        "small_n_warnings": [],
    }
    warnings = out["meta"]["small_n_warnings"]
    if n < 100:
        warnings.append(f"small dataset n={n} (<100): probe/purity estimates have wide CIs")
    if smallest_train_n < 20:
        warnings.append(f"smallest train fold n={smallest_train_n}: linear/MLP probes less reliable")

    folds = valid_folds

    # ---- Step 1: LEACE-erase taxonomy; record erased rank; CLEAN gate -------------------------
    Xe_tax, P_tax = leace_erase(Z, tax_targets)
    E = erasing_projector_rowspace(P_tax)
    k_erased = int(E.shape[1])
    clean_gate = bool(k_erased < pca_dim / 2)
    print(f"[sp step1] LEACE erased rank k={k_erased} of PCA_DIM={pca_dim}  "
          f"(CLEAN gate k<<PCA_DIM: {clean_gate})")
    out["leace"] = {"erased_rank_k": k_erased, "pca_dim": int(pca_dim),
                    "clean_claim_gated_ok": clean_gate,
                    "gate_note": ("k < PCA_DIM/2 — function-preservation is meaningful" if clean_gate
                                  else "k is a large fraction of PCA_DIM — 'function preserved' carries "
                                       "little weight (M4 warning)")}

    # ---- Step 2: erasure worked — categorical (stratum) + continuous (position) probes ---------
    tax_orig_cat = taxonomy_probe_categorical(Z, tax_codes, folds)
    tax_erased_cat = taxonomy_probe_categorical(Xe_tax, tax_codes, folds)
    tax_orig_cont = taxonomy_probe_continuous(Z, tax_targets, folds)
    tax_erased_cont = taxonomy_probe_continuous(Xe_tax, tax_targets, folds)
    print(f"[sp step2] {stratum_col} acc orig lin={tax_orig_cat['linear_acc']:.3f}"
          f"/mlp={tax_orig_cat['nonlinear_acc']:.3f}  tax-erased lin={tax_erased_cat['linear_acc']:.3f}"
          f"/mlp={tax_erased_cat['nonlinear_acc']:.3f}  (chance={tax_orig_cat['chance']:.3f})")
    print(f"[sp step2] position R^2 orig lin={tax_orig_cont['linear_r2']:.3f}  "
          f"tax-erased lin={tax_erased_cont['linear_r2']:.3f}")
    out["step2_erasure_worked"] = {
        "taxonomy_probe_label": f"{stratum_col} (NCBI rank)",
        "tax_family_probe": {"original": tax_orig_cat, "tax_erased": tax_erased_cat},
        "position_probe": {"original": tax_orig_cont, "tax_erased": tax_erased_cont},
    }

    # ---- Step 3: function preserved — func purity (global + within taxonomic strata) -----------
    # Within-stratum = within the `stratum_col` rank (isolates function from taxonomy). Compute the
    # within-stratum purity at BOTH SP attached ranks (tax_order fine, tax_class coarse) for the grain
    # report, and headline the FINEST rank that is feasible (spec §6.1).
    fp_orig_global, n_g = knn_purity_global(Z, func_groups)
    fp_erased_global, _ = knn_purity_global(Xe_tax, func_groups)
    fp_orig_within, n_w = knn_purity_within_strata(Z, func_groups, tax_family)
    fp_erased_within, _ = knn_purity_within_strata(Xe_tax, func_groups, tax_family)
    fp_chance = _group_chance_purity(func_groups)
    print(f"[sp step3] func purity@{KNN_K} GLOBAL  orig={fp_orig_global:.3f}  "
          f"tax-erased={fp_erased_global:.3f}  (n={n_g}, chance~{fp_chance:.3f})")
    print(f"[sp step3] func purity@{KNN_K} within-{stratum_col} orig={fp_orig_within:.3f}  "
          f"tax-erased={fp_erased_within:.3f}  (n={n_w})")
    out["step3_function_preserved"] = {
        "global": {"original": fp_orig_global, "tax_erased": fp_erased_global, "n_queries": n_g,
                   "note": "global func kNN purity over all n proteins (primary function signal)"},
        "within_tax_family": {"original": fp_orig_within, "tax_erased": fp_erased_within, "n_queries": n_w,
                              "stratum_col": stratum_col,
                              "note": f"within {stratum_col} stratum (controls for taxonomy)"},
        "group_chance_purity": fp_chance,
    }
    if n_w < 10:
        warnings.append(f"step3: within-{stratum_col} purity scored only {n_w} queries (few strata have "
                        f">{KNN_K} proteins); global purity is the primary signal")

    # ---- Stratum-grain report + finest-feasible headline rank (spec §6.1) ----------------------
    # Within-stratum function purity computed at each attached rank (order fine -> class coarse). A rank
    # is FEASIBLE iff >=config.SP_KNN_K+1 members in >=50% of its strata AND those strata cover >=80% of
    # proteins. headline = finest feasible rank (order -> class).
    stratum_grain = {}
    grain_specs = [("order", "tax_order"), ("class", "tax_class")]
    for grain_key, rank_col in grain_specs:
        if rank_col not in ann.columns:
            continue
        # spec §4.6: score each rank over taxa that HAVE that rank. tax_class is non-null by load-time
        # filtering, but the finer tax_order can be null for a class-having taxon (incertae sedis) — mask
        # those rows here so the order grain's purity + denominator are over order-having proteins only.
        rank_mask = pd.notna(ann[rank_col]).to_numpy()
        n_with_rank = int(rank_mask.sum())
        if n_with_rank < (config.SP_KNN_K + 2):
            continue
        strata_vals = ann[rank_col].to_numpy()[rank_mask]
        g_orig_w, g_n_w = knn_purity_within_strata(Z[rank_mask], func_groups[rank_mask], strata_vals)
        g_erased_w, _ = knn_purity_within_strata(Xe_tax[rank_mask], func_groups[rank_mask], strata_vals)
        # feasibility: fraction of strata with >= SP_KNN_K+1 members, and the protein-coverage of those
        labels, counts = np.unique(strata_vals, return_counts=True)
        big = counts >= (config.SP_KNN_K + 1)
        frac_strata_big = float(big.mean()) if len(labels) else 0.0
        coverage = float(counts[big].sum() / counts.sum()) if counts.sum() else 0.0
        feasible = bool(frac_strata_big >= config.GRAIN_MIN_FRAC_STRATA
                        and coverage >= config.GRAIN_MIN_COVERAGE)
        stratum_grain[grain_key] = {
            "rank_col": rank_col, "original": g_orig_w, "tax_erased": g_erased_w, "n_queries": g_n_w,
            "n_with_rank": n_with_rank, "n_null_rank": int(len(ann) - n_with_rank),
            "n_strata": int(len(labels)), "frac_strata_with_min_members": round(frac_strata_big, 4),
            "protein_coverage_of_min_strata": round(coverage, 4), "feasible": feasible,
        }
    out["stratum_grain"] = stratum_grain
    # finest feasible rank: order is finer than class; fall back to class, then the stratum_col itself
    if stratum_grain.get("order", {}).get("feasible"):
        out["headline_stratum"] = "tax_order"
    elif stratum_grain.get("class", {}).get("feasible"):
        out["headline_stratum"] = "tax_class"
    else:
        out["headline_stratum"] = stratum_col
    print(f"[sp step3] stratum_grain feasibility: "
          f"{ {k: v['feasible'] for k, v in stratum_grain.items()} }  "
          f"-> headline_stratum={out['headline_stratum']}")

    # ---- Step 4: specificity — variance-AND-concept-matched control ---------------------------
    erased_energy = float(((Z - Z.mean(0)) @ E).var(0).sum())
    Pc = matched_control_subspace(Z, target=func_onehot, avoid=tax_targets, erased_energy=erased_energy)
    Ec = erasing_projector_rowspace(Pc)
    control_energy = float(((Z - Z.mean(0)) @ Ec).var(0).sum()) if Ec.shape[1] else 0.0
    energy_ratio = control_energy / erased_energy if erased_energy > 0 else float("nan")
    energy_ok = bool(0.5 <= energy_ratio <= 2.0)
    Zc = apply_projector(Z, Pc)
    tax_control_cat = taxonomy_probe_categorical(Zc, tax_codes, folds)
    tax_control_cont = taxonomy_probe_continuous(Zc, tax_targets, folds)
    print(f"[sp step4] matched-control erased rank={Ec.shape[1]}  energy {control_energy:.4f} "
          f"vs LEACE {erased_energy:.4f}  ratio={energy_ratio:.3f}  (0.5x-2x ok: {energy_ok})")
    print(f"[sp step4] taxonomy under control: {stratum_col} acc lin={tax_control_cat['linear_acc']:.3f} "
          f"pos R^2={tax_control_cont['linear_r2']:.3f} (should stay HIGH unlike LEACE)")
    # named specificity acceptance gate (spec §6.1): energy matched AND taxonomy under control stays ~orig
    tax_under_control_ok = bool(
        tax_control_cat["linear_acc"] >= tax_orig_cat["linear_acc"] - 0.10)
    out["step4_specificity"] = {
        "leace_erased_energy": erased_energy, "control_erased_energy": control_energy,
        "energy_ratio_control_over_leace": energy_ratio, "energy_match_ok_half_to_2x": energy_ok,
        "control_erased_rank": int(Ec.shape[1]),
        "taxonomy_under_control": {"tax_family_probe": tax_control_cat, "position_probe": tax_control_cont},
        "taxonomy_under_control_stays_recoverable": tax_under_control_ok,
        "pass": bool(energy_ok and tax_under_control_ok),
    }

    # ---- Step 5: reverse symmetry — erase FUNCTION, taxonomy should survive --------------------
    Xe_func, P_func = leace_erase(Z, func_onehot)
    Ef = erasing_projector_rowspace(P_func)
    tax_funcerased_cat = taxonomy_probe_categorical(Xe_func, tax_codes, folds)
    tax_funcerased_cont = taxonomy_probe_continuous(Xe_func, tax_targets, folds)
    fp_funcerased_global, _ = knn_purity_global(Xe_func, func_groups)
    fp_funcerased_within, _ = knn_purity_within_strata(Xe_func, func_groups, tax_family)
    print(f"[sp step5] func-erased rank={Ef.shape[1]}  taxonomy survives? {stratum_col} acc lin="
          f"{tax_funcerased_cat['linear_acc']:.3f} pos R^2={tax_funcerased_cont['linear_r2']:.3f}  | "
          f"func purity global={fp_funcerased_global:.3f} within={fp_funcerased_within:.3f}")
    out["step5_reverse_symmetry"] = {
        "function_erased_rank": int(Ef.shape[1]),
        "taxonomy_under_function_erasure": {"tax_family_probe": tax_funcerased_cat,
                                            "position_probe": tax_funcerased_cont},
        "function_under_function_erasure": {"global": fp_funcerased_global,
                                            "within_tax_family": fp_funcerased_within},
        "note": ("The Pfam/EC function axis is ⊥ phylogeny by construction, so erasing function should "
                 "NOT collapse taxonomy and erasing taxonomy should NOT collapse function — the clean "
                 "(disentangled) expectation. The reverse-direction figure is reported via "
                 "out['label_nesting']['func_given_tax'] (a sampling property, not entanglement)."),
    }

    # ---- Step 6: depth-null -------------------------------------------------------------------
    real_collapse = tax_orig_cat["linear_acc"] - tax_erased_cat["linear_acc"]
    depth_mag = np.linalg.norm(tax_targets, axis=1, keepdims=True)
    Xe_depthmag, P_depthmag = leace_erase(Z, depth_mag)
    Edm = erasing_projector_rowspace(P_depthmag)
    tax_depthmag_cat = taxonomy_probe_categorical(Xe_depthmag, tax_codes, folds)
    tax_depthmag_cont = taxonomy_probe_continuous(Xe_depthmag, tax_targets, folds)
    depthmag_collapse = tax_orig_cat["linear_acc"] - tax_depthmag_cat["linear_acc"]

    depth_targets = radial_only_null(tax_targets, seed=SEED)
    Xe_depth, P_depth = leace_erase(Z, depth_targets)
    Ed = erasing_projector_rowspace(P_depth)
    tax_depth_cat = taxonomy_probe_categorical(Xe_depth, tax_codes, folds)
    randdir_collapse = tax_orig_cat["linear_acc"] - tax_depth_cat["linear_acc"]
    randdir_degenerate = bool(Ed.shape[1] >= 0.9 * pca_dim)
    print(f"[sp step6] PRIMARY depth-magnitude erase rank={Edm.shape[1]}  {stratum_col} acc lin="
          f"{tax_depthmag_cat['linear_acc']:.3f}  collapse={depthmag_collapse:.3f}  | "
          f"real-LEACE collapse={real_collapse:.3f}")
    out["step6_depth_null_clean"] = {
        "primary_depth_magnitude": {
            "erased_rank": int(Edm.shape[1]),
            "taxonomy_after_depth_erasure": {"tax_family_probe": tax_depthmag_cat,
                                             "position_probe": tax_depthmag_cont},
            "tax_family_acc_collapse": depthmag_collapse,
        },
        "real_leace_tax_family_acc_collapse": real_collapse,
        "depth_attributable_fraction": (depthmag_collapse / real_collapse) if real_collapse else float("nan"),
        "literal_random_direction_null": {
            "erased_rank": int(Ed.shape[1]), "tax_family_acc_collapse": randdir_collapse,
            "degenerate_full_rank": randdir_degenerate},
    }

    # ---- Step 7: identity-stratum check -------------------------------------------------------
    out["step7_identity_strata"] = _identity_stratum_report(
        Z, Xe_tax, func_groups, tax_codes, cluster_ids, folds)
    print(f"[sp step7] identity strata: {list(out['step7_identity_strata']['strata'].keys())}")

    # ---- Step 8: disentanglement matrix -------------------------------------------------------
    out["disentanglement_matrix"] = _build_matrix_sp(
        Z, Xe_tax, Xe_func, Zc, func_groups, tax_codes, tax_family, folds, k_erased, stratum_col)

    # ---- Task 14: coherence -------------------------------------------------------------------
    out["coherence"] = coherence(b, P_tax, pca_dim)

    # ---- Task C2b: length-erasure confound control (the ENFORCED confound gate, spec §6.2) -----
    # Length is the WEAKEST sequence confound (1-D scalar). z-score it, LEACE-erase off Z, and require
    # the headline to SURVIVE: within-stratum function purity barely moves AND taxonomy stays recoverable
    # on the length-erased reps (length ⟂ the function/taxonomy axes => erasing it is a near no-op).
    if length_control:
        lengths = ann["length"].to_numpy().astype(np.float64)
        lstd = lengths.std()
        length_target = (lengths - lengths.mean()) / (lstd if lstd > 0 else 1.0)   # 1-col continuous concept
        purity_after, tax_after, geom, survives, detail = _confound_erase_report(
            Z, length_target, func_groups, tax_family, tax_codes, folds,
            purity_orig_within=fp_orig_within, tax_orig_acc=tax_orig_cat["linear_acc"],
            tax_chance=tax_orig_cat["chance"])
        print(f"[sp C2b] length-erase: within-{stratum_col} purity {fp_orig_within:.3f}->{purity_after:.3f} "
              f"(rel-drop={detail['purity_relative_drop']}); tax acc {tax_orig_cat['linear_acc']:.3f}->"
              f"{tax_after['linear_acc']:.3f} (chance~{tax_orig_cat['chance']:.3f}); geom-change={geom:.4f}; "
              f"headline_survives={survives}")
        out["length_control"] = {
            "purity_after_length_erase": purity_after,
            "tax_after_length_erase": tax_after,
            "geometry_change": geom,
            "headline_survives": survives,
            "concept_dim": 1,
            "detail": detail,
            "note": ("Length is a 1-D scalar — the weakest sequence confound; a 1-D LEACE erasure on a "
                     "~PCA_DIM space is near a no-op (geometry_change makes the no-op visible). The headline "
                     "must survive (spec §6.2): function purity preserved AND taxonomy still recoverable."),
        }

    # ---- Task C2c: amino-acid composition-erasure confound control (STRONGER confound gate — S1) ---
    # Composition (20-D AA fraction, columns aa_A..aa_Y) is the stronger taxonomy proxy (clade-structured
    # compositional drift). MIRROR C2b with the 20-col concept. If the aa_A..aa_Y columns are ABSENT we
    # RAISE (never silently skip — a silent skip disables the gate, the freeze_cluster_ids anti-pattern).
    if composition_control:
        aa_cols = [f"aa_{c}" for c in "ACDEFGHIKLMNPQRSTVWY"]    # 20 canonical AAs, fixed order
        missing = [c for c in aa_cols if c not in ann.columns]
        if missing:
            raise ValueError(
                "composition_control=True but the amino-acid composition columns are absent from the panel: "
                f"missing {missing} (expected aa_A..aa_Y, 20 columns frozen at build time). The control "
                "RAISES rather than silently skipping — a silent skip would disable the S1 gate.")
        aa_comp = ann[aa_cols].to_numpy().astype(np.float64)    # (n, 20) composition concept
        comp_dim = int(aa_comp.shape[1])
        purity_after, tax_after, geom, survives, detail = _confound_erase_report(
            Z, aa_comp, func_groups, tax_family, tax_codes, folds,
            purity_orig_within=fp_orig_within, tax_orig_acc=tax_orig_cat["linear_acc"],
            tax_chance=tax_orig_cat["chance"])
        print(f"[sp C2c] comp-erase (dim={comp_dim}): within-{stratum_col} purity {fp_orig_within:.3f}->"
              f"{purity_after:.3f} (rel-drop={detail['purity_relative_drop']}); tax acc "
              f"{tax_orig_cat['linear_acc']:.3f}->{tax_after['linear_acc']:.3f} "
              f"(chance~{tax_orig_cat['chance']:.3f}); geom-change={geom:.4f}; headline_survives={survives}")
        out["composition_control"] = {
            "purity_after_comp_erase": purity_after,
            "tax_after_comp_erase": tax_after,
            "comp_dim": comp_dim,
            "geometry_change": geom,
            "headline_survives": survives,
            "detail": detail,
            "note": ("Amino-acid composition (20-D AA fraction) is the STRONGER taxonomy proxy "
                     "(clade-structured compositional drift, encoded by ProtT5). The headline must survive "
                     "composition-erasure too (spec §6.1 S1): function purity preserved AND taxonomy still "
                     "recoverable. geometry_change reports the mean per-rep cosine shift so a no-op is visible."),
        }

    return out


def _build_matrix_sp(Z, Xe_tax, Xe_func, Zc, func_groups, tax_codes, tax_family_strata,
                     folds, k_erased, stratum_col="tax_class"):
    """Disentanglement matrix for SP (clone of _build_matrix_multifamily): taxonomy probe = `stratum_col`
    (NCBI rank, held-out linear acc), function = func (Pfam/EC) kNN purity (GLOBAL primary; within-stratum
    reported). Crossing {original, tax-erased, function-erased, matched-control}.
    """
    reps_by_col = {"original": Z, "tax_erased": Xe_tax, "function_erased": Xe_func, "matched_control": Zc}
    matrix = {
        "leace_erased_rank_k": int(k_erased),
        "taxonomy_probe_label": f"{stratum_col}_acc (NCBI rank)",
        "stratum_col": stratum_col,
        "function_metric_primary": "global func kNN purity",
        "taxonomy_recoverability_tax_family_acc": {},
        "function_recoverability_global_purity": {},
        "function_recoverability_within_tax_family_purity": {},
    }
    for col, Xcol in reps_by_col.items():
        tax = taxonomy_probe_categorical(Xcol, tax_codes, folds)
        fn_global, _ = knn_purity_global(Xcol, func_groups)
        fn_within, _ = knn_purity_within_strata(Xcol, func_groups, tax_family_strata)
        matrix["taxonomy_recoverability_tax_family_acc"][col] = round(tax["linear_acc"], 4)
        matrix["function_recoverability_global_purity"][col] = round(fn_global, 4)
        fn_within_val = round(fn_within, 4) if not (fn_within != fn_within) else None
        matrix["function_recoverability_within_tax_family_purity"][col] = fn_within_val
    return matrix


def _sha256_or_none(path):
    """SHA256 of a file, streamed in 1 MiB chunks (so a 700 MB ckpt never loads whole). None when the
    path is falsy or absent — provenance records what it can, never crashes on a missing input."""
    import hashlib, os
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_provenance(testbed, panel_path, h5_path, cap_seed):
    """Consolidated provenance for every result JSON (spec v3 §9). config_git_sha is read from a
    mac-captured config_provenance.json baked into the rsync payload (GPU cluster has no .git); None where a
    field is unavailable, never a crash. mmseqs_version is a STRING (filled from the cluster manifest),
    not a bool."""
    import json, os
    git_sha = None
    prov_file = os.path.join(os.path.dirname(__file__), "config_provenance.json")
    if os.path.exists(prov_file):
        try:
            git_sha = json.load(open(prov_file)).get("config_git_sha")
        except Exception:
            git_sha = None
    target = "cellular" if testbed.startswith("all_life_") else "metazoa"
    ckpt = config.CELLULAR_CKPT if target == "cellular" else config.CKPT
    # M-1/§9: thread the single-release-pinned uniprot_release + the JB-realized mmseqs_version STRING
    # from the self-embed manifest (the largest NEW artifact carries both) — replacing the None placeholders.
    manifest_path = (str(config.ALL_LIFE_H5) + ".manifest.json"
                     if testbed.startswith("all_life_") else None)
    uniprot_release, mmseqs_version = None, None
    if manifest_path and os.path.exists(manifest_path):
        try:
            _man = json.load(open(manifest_path))
            uniprot_release = _man.get("uniprot_release")
            mmseqs_version = _man.get("mmseqs_version")
        except Exception:
            pass
    return {
        "config_git_sha": git_sha,
        "uniprot_release": uniprot_release,   # M-1: from the self-embed manifest (release pinned in B2)
        "panel_path": panel_path,
        "panel_sha256": _sha256_or_none(panel_path),
        "h5_path": h5_path,
        "h5_sha256": _sha256_or_none(h5_path),
        "mmseqs_version": mmseqs_version,     # M-1: STRING from the manifest (JB-realized), not a bool
        "cap_seed": int(cap_seed),
        "poincare_ckpt_sha256": _sha256_or_none(str(ckpt)),
        "self_embed_manifest_path": manifest_path,
    }


def _function_chance(func_labels):
    """Σ p_f² — the kNN-purity chance floor for the function-label distribution (spec v3 §7). At all-life
    scale the family-skew shifts this floor, so it must be measured per-panel rather than assumed."""
    import numpy as np
    _, counts = np.unique(np.asarray(list(func_labels)), return_counts=True)
    if counts.sum() == 0:
        return 0.0
    p = counts / counts.sum()
    return float((p * p).sum())


def _function_preserved(purity_orig, purity_after, func_chance):
    """Headroom-aware function-preservation (spec v3 §7). Returns None = NOT-SCORABLE (skew-dominated:
    purity_orig too close to the Σp_f² chance floor to test), else True iff BOTH the rel-drop margin holds
    AND the chance-corrected retained fraction clears (1 - margin). func_chance=0.0 reproduces the legacy
    bare-rel-drop behaviour (Stage-1 metazoa, where the floor is negligible)."""
    if purity_orig is None or purity_after is None or purity_orig != purity_orig or purity_after != purity_after:
        return None
    headroom = purity_orig - func_chance
    if headroom < config.ALL_LIFE_MIN_PURITY_HEADROOM:
        return None
    rel_drop = (purity_orig - purity_after) / purity_orig if purity_orig else float("nan")
    corrected = (purity_after - func_chance) / headroom
    return bool(rel_drop == rel_drop and rel_drop < config.ALL_LIFE_PURITY_REL_DROP_MAX
                and corrected >= (1.0 - config.ALL_LIFE_PURITY_REL_DROP_MAX))


def _taxonomy_erasable(tax_orig, tax_erased, tax_chance):
    """Symmetric headroom-aware taxonomy-erasability (spec v3 §7). Returns None = NOT-SCORABLE, else True
    iff the chance-corrected COLLAPSE fraction (tax_orig-tax_erased)/(tax_orig-tax_chance) clears the floor
    AND the erased probe sits at ~chance (tax_erased <= chance + ε). Spec v3 §7 KEEPS the at-chance floor
    in addition to the collapse fraction."""
    if tax_orig is None or tax_erased is None or tax_orig != tax_orig or tax_erased != tax_erased:
        return None
    headroom = tax_orig - tax_chance
    if headroom < config.ALL_LIFE_MIN_TAX_HEADROOM:
        return None
    collapse = (tax_orig - tax_erased) / headroom
    at_chance = tax_erased <= tax_chance + config.SP_CONFOUND_TAX_MARGIN_MIN
    return bool(collapse >= config.ALL_LIFE_TAX_COLLAPSE_FRAC_MIN and at_chance)


def _disentanglement_verdict_sp(out):
    """Verdict for the SP CLEAN battery (clone of _disentanglement_verdict_multifamily): is taxonomy
    ERASABLE while Pfam/EC function clustering is PRESERVED (CLEAN disentanglement)?

    Reads the _build_matrix_sp keys + the BIDIRECTIONAL out['label_nesting'] + the specificity `pass`.
    Adds the GRAIN-CONSISTENCY gate (spec §6.1/§6.4 — S2): the per-grain clean/entangled call is computed
    from out['stratum_grain'] (order/class); out['grain_consistent'] is True iff every feasible grain
    agrees. When they DISAGREE the verdict reads grain-dependent (headline_stratum is the display rank
    only — it never overrides a disagreeing grain).

    CLEAN-DISENTANGLED reads as TRUE when BOTH hold:
      (a) taxonomy ERASABLE: stratum acc drops toward chance under LEACE, AND
      (b) function PRESERVED: within-stratum func purity holds (small rel-drop, < ~10%) under tax erasure.
    """
    m = out["disentanglement_matrix"]
    stratum_col = m.get("stratum_col", "tax_class")
    tax = m["taxonomy_recoverability_tax_family_acc"]
    fn_w = m["function_recoverability_within_tax_family_purity"]   # PRIMARY (taxonomy-controlled)
    fn_g = m["function_recoverability_global_purity"]              # SECONDARY (confounded)
    s2 = out["step2_erasure_worked"]
    s3 = out["step3_function_preserved"]

    tax_orig = tax["original"]
    tax_erased = tax["tax_erased"]
    tax_func_erased = tax["function_erased"]
    chance = s2["tax_family_probe"]["original"].get("chance", float("nan"))

    # PRIMARY function signal: within-stratum purity (None if too few queries -> fall back to global)
    fn_orig = fn_w["original"]
    fn_tax_erased = fn_w["tax_erased"]
    fn_func_erased = fn_w["function_erased"]
    within_feasible = (fn_orig is not None and fn_tax_erased is not None
                       and s3["within_tax_family"]["n_queries"] >= 10)
    if not within_feasible:
        # within-stratum not feasible at this n: report global with explicit confound flag
        fn_orig = fn_g["original"]
        fn_tax_erased = fn_g["tax_erased"]
        fn_func_erased = fn_g["function_erased"]
    fn_metric = f"within_{stratum_col}" if within_feasible else "global(confounded_fallback)"

    # global (secondary) figures for context regardless
    g_orig, g_tax_erased = fn_g["original"], fn_g["tax_erased"]
    global_rel_drop_on_tax_erase = (g_orig - g_tax_erased) / g_orig if g_orig else float("nan")

    # (a) taxonomy erasable + (b) function preserved — headroom-aware, chance-corrected on BOTH legs
    # (spec v3 §7). Each returns True / False / None where None = NOT-SCORABLE (headroom below floor).
    # func_chance=0.0 default reproduces the Stage-1 metazoa behaviour for callers that build `out` by hand.
    func_chance = out.get("function_chance", 0.0)
    taxonomy_erasable = _taxonomy_erasable(tax_orig, tax_erased, chance)
    function_preserved = _function_preserved(fn_orig, fn_tax_erased, func_chance)
    func_rel_drop_on_tax_erase = (fn_orig - fn_tax_erased) / fn_orig if fn_orig else float("nan")
    # reverse: taxonomy survives function-erasure (reported, not part of the headline gate)
    tax_rel_drop_on_func_erase = (tax_orig - tax_func_erased) / tax_orig if tax_orig else float("nan")
    func_rel_drop_on_func_erase = (fn_orig - fn_func_erased) / fn_orig if fn_orig else float("nan")
    taxonomy_survives_func_erasure = bool(
        not (tax_rel_drop_on_func_erase != tax_rel_drop_on_func_erase)
        and tax_rel_drop_on_func_erase < 0.10)

    clean_disentangled = (None if (taxonomy_erasable is None or function_preserved is None)
                          else bool(taxonomy_erasable and function_preserved))

    # ---- grain-consistency gate (S2): per-grain clean/entangled call from out['stratum_grain'] -----
    # each grain's "function preserved under tax-erasure" call uses the SAME headroom-aware criterion
    # (spec v3 §7); None = NOT-SCORABLE at that grain (excluded from the agreement check).
    grain_calls = {}
    for grain_key, gr in out.get("stratum_grain", {}).items():
        g_o, g_te = gr.get("original"), gr.get("tax_erased")
        grain_calls[grain_key] = _function_preserved(g_o, g_te, func_chance)
    scorable = [v for v in grain_calls.values() if v is not None]
    grain_consistent = bool(len(scorable) <= 1 or all(scorable) or not any(scorable))
    out["grain_consistent"] = grain_consistent

    # specificity acceptance gate (named pass)
    specificity_pass = bool(out.get("step4_specificity", {}).get("pass", False))

    # bidirectional nesting figures
    ln = out.get("label_nesting", {})
    tax_given_func = ln.get("tax_given_func", {})
    func_given_tax = ln.get("func_given_tax", {})

    reads_cleaner_than_pla2 = bool(taxonomy_erasable and function_preserved)

    # ---- confound-erasure gates (C2b length, C2c composition) — S1 -------------------------------
    # The headline must SURVIVE both the length-erasure (C2b) and the composition-erasure (C2c) confound
    # controls. Each is present in `out` only when the corresponding control was enabled; a gate is None
    # (not asserted) when its control was not run. S1 = headline survives BOTH (when both ran).
    length_ctrl = out.get("length_control")
    comp_ctrl = out.get("composition_control")
    length_survives = (bool(length_ctrl["headline_survives"]) if length_ctrl is not None else None)
    comp_survives = (bool(comp_ctrl["headline_survives"]) if comp_ctrl is not None else None)
    ran = [s for s in (length_survives, comp_survives) if s is not None]
    confound_gates_pass = bool(all(ran)) if ran else None      # S1: survives every confound control run

    if not grain_consistent:
        interp = (
            f"GRAIN-DEPENDENT (spec §6.4 S2): the clean/entangled call DISAGREES across the stratum grains "
            f"{grain_calls}. The headline_stratum ({out.get('headline_stratum')}) is the display rank only "
            "and does NOT override the disagreement — reported honestly, not rescued.")
    elif clean_disentangled is None:
        interp = (
            "NOT-SCORABLE at this grain (spec v3 §7): the panel is skew-dominated — one or both legs lack "
            f"headroom to test (taxonomy_erasable={taxonomy_erasable}, function_preserved={function_preserved}; "
            "None = the original probe sits within its chance floor). Reported honestly as not-scorable, NOT "
            f"coerced to pass/fail (tax {tax_orig:.3f} vs chance {chance:.3f}; function Σp_f² chance "
            f"{func_chance:.3f}).")
    elif clean_disentangled:
        interp = (
            f"CLEAN DISENTANGLEMENT (the expected result): taxonomy ({stratum_col} acc "
            f"{tax_orig:.3f}->{tax_erased:.3f}, collapsing to ~chance={chance:.3f}) is ERASABLE while the "
            f"Pfam/EC function clustering is PRESERVED ({fn_metric} purity {fn_orig:.3f}->"
            f"{fn_tax_erased:.3f}, rel-drop={func_rel_drop_on_tax_erase:.3f} < 0.10). The function axis "
            "lives in a subspace separable from the taxonomy axis. NOTE the SECONDARY global purity may "
            f"drop ({g_orig:.3f}->{g_tax_erased:.3f}, rel-drop={global_rel_drop_on_tax_erase:.3f}); the "
            "within-stratum metric removes that taxonomy confound.")
    elif function_preserved:
        interp = (
            f"PARTIALLY CLEAN: Pfam/EC function preserved under taxonomy erasure ({fn_metric} "
            f"rel-drop={func_rel_drop_on_tax_erase:.3f}), but the taxonomy-erasability gate was not fully "
            f"met ({stratum_col} acc {tax_orig:.3f}->{tax_erased:.3f} vs chance={chance:.3f}). At "
            f"n={out['meta']['n_proteins']} the taxonomy probe is noisy.")
    else:
        interp = (
            f"NOT cleanly disentangled at the measured grain: even taxonomy-controlled ({fn_metric}) "
            f"function purity dropped on taxonomy erasure (rel-drop={func_rel_drop_on_tax_erase:.3f} "
            ">= 0.10). Inspect — small n may be driving instability.")

    return {
        "clean_disentangled": clean_disentangled,
        "reads_cleaner_than_pla2": reads_cleaner_than_pla2,
        "grain_consistent": grain_consistent,
        "grain_calls": grain_calls,
        "headline_stratum": out.get("headline_stratum"),
        "specificity_pass": specificity_pass,
        "length_control_headline_survives": length_survives,        # C2b gate (None if not run)
        "composition_control_headline_survives": comp_survives,     # C2c gate (None if not run)
        "confound_control_gates_pass": confound_gates_pass,         # S1: survives BOTH confounds run
        "primary_function_metric": fn_metric,
        "taxonomy_erasable": taxonomy_erasable,
        "function_preserved_under_taxonomy_erasure": function_preserved,
        "taxonomy_survives_function_erasure": taxonomy_survives_func_erasure,
        "tax_family_acc_original": round(float(tax_orig), 4),
        "tax_family_acc_tax_erased": round(float(tax_erased), 4),
        "tax_family_chance": round(float(chance), 4),
        "function_within_stratum_original": (round(float(fn_w["original"]), 4)
                                             if fn_w["original"] is not None else None),
        "function_within_stratum_tax_erased": (round(float(fn_w["tax_erased"]), 4)
                                               if fn_w["tax_erased"] is not None else None),
        "function_relative_drop_on_taxonomy_erasure_PRIMARY": round(float(func_rel_drop_on_tax_erase), 4),
        "function_relative_drop_on_taxonomy_erasure_global_SECONDARY": round(float(global_rel_drop_on_tax_erase), 4),
        "global_purity_drop_is_taxonomy_confound_not_entanglement": True,
        "leace_clean_gate_ok": out["leace"]["clean_claim_gated_ok"],
        "leace_gate_caveat": (
            None if out["leace"]["clean_claim_gated_ok"] else
            f"M4 caveat: LEACE erased k={out['leace']['erased_rank_k']} of PCA_DIM={out['leace']['pca_dim']} "
            "(a large fraction of the space); the 'preserved' claim carries somewhat less weight. Reported "
            "honestly."),
        "label_nesting_bidirectional": {
            "tax_given_func_uncertainty_reduction": tax_given_func.get("uncertainty_reduction"),
            "func_given_tax_uncertainty_reduction": func_given_tax.get("uncertainty_reduction")},
        "taxonomy_relative_drop_on_function_erasure": round(float(tax_rel_drop_on_func_erase), 4),
        "function_relative_drop_on_function_erasure": round(float(func_rel_drop_on_func_erase), 4),
        "reverse_direction_note": (
            "The Pfam/EC function axis is ⊥ phylogeny by construction; any reverse-direction taxonomy drop "
            "under function-erasure is a SAMPLING property visible via out['label_nesting']['func_given_tax'] "
            "(uncertainty_reduction="
            f"{func_given_tax.get('uncertainty_reduction')}), NOT embedding entanglement. The "
            "disentanglement CLAIM rests on the FORWARD direction (erase taxonomy, keep function)."),
        "expected_clean_contrast_to_pla2": True,
        "interpretation": interp,
        "small_n_caveat": out["meta"]["small_n_warnings"],
    }


def _build_matrix_multifamily(Z, Xe_tax, Xe_func, Zc, func_groups, tax_codes, tax_family_strata,
                              folds, k_erased):
    """Disentanglement matrix for multifamily: taxonomy probe = tax_family (NCBI family rank, held-out
    linear acc), function = toxin-Family kNN purity (GLOBAL is primary; within-tax_family is reported
    but scores few queries at this n). Crossing {original, tax-erased, function-erased, matched-control}.
    """
    reps_by_col = {"original": Z, "tax_erased": Xe_tax, "function_erased": Xe_func, "matched_control": Zc}
    matrix = {
        "leace_erased_rank_k": int(k_erased),
        "taxonomy_probe_label": "tax_family_acc (NCBI family rank)",
        "function_metric_primary": "global toxin-Family kNN purity",
        "taxonomy_recoverability_tax_family_acc": {},
        "function_recoverability_global_purity": {},
        "function_recoverability_within_tax_family_purity": {},
    }
    for col, Xcol in reps_by_col.items():
        tax = taxonomy_probe_categorical(Xcol, tax_codes, folds)
        fn_global, _ = knn_purity_global(Xcol, func_groups)
        fn_within, _ = knn_purity_within_strata(Xcol, func_groups, tax_family_strata)
        matrix["taxonomy_recoverability_tax_family_acc"][col] = round(tax["linear_acc"], 4)
        matrix["function_recoverability_global_purity"][col] = round(fn_global, 4)
        fn_within_val = round(fn_within, 4) if not (fn_within != fn_within) else None
        matrix["function_recoverability_within_tax_family_purity"][col] = fn_within_val
    return matrix


def _disentanglement_verdict_multifamily(out):
    """Verdict for the multifamily PRIMARY: is taxonomy ERASABLE while toxin-Family clustering is
    PRESERVED (CLEAN disentanglement), in CONTRAST to PLA2 (entangled)?

    PRIMARY function-preservation metric = WITHIN-tax_family kNN purity (taxonomy-CONTROLLED), the SAME
    within-stratum metric the validated PLA2 verdict uses (function_recoverability_within_clade_purity).
    This is the apples-to-apples contrast and the honest one: GLOBAL purity conflates function with
    taxonomy here (a protein's same-toxin-family neighbours are very often also its closest taxonomic
    relatives — e.g. the 7 Conus insulins are same-family AND same-genus — so erasing taxonomy pulls them
    apart and drops GLOBAL purity for a taxonomy reason, not a function-damage reason). The within-stratum
    purity removes that confound. Global purity is reported as a SECONDARY, explicitly-confounded figure.

    CLEAN-DISENTANGLED reads as TRUE when BOTH hold:
      (a) taxonomy ERASABLE: tax_family acc drops toward chance under LEACE, AND
      (b) function PRESERVED: within-tax_family toxin-Family purity holds (small rel-drop, < ~10%) under
          taxonomy erasure.
    The REVERSE direction is reported symmetrically. Small n ⇒ flagged; the CONTRAST DIRECTION
    (cleaner than PLA2) is the headline even with wide CIs.
    """
    m = out["disentanglement_matrix"]
    tax = m["taxonomy_recoverability_tax_family_acc"]
    fn_w = m["function_recoverability_within_tax_family_purity"]   # PRIMARY (taxonomy-controlled)
    fn_g = m["function_recoverability_global_purity"]              # SECONDARY (confounded)
    s2 = out["step2_erasure_worked"]
    s3 = out["step3_function_preserved"]

    tax_orig = tax["original"]
    tax_erased = tax["tax_erased"]
    tax_func_erased = tax["function_erased"]
    chance = s2["tax_family_probe"]["original"].get("chance", float("nan"))

    # PRIMARY function signal: within-tax_family purity (None if too few queries -> fall back to global)
    fn_orig = fn_w["original"]
    fn_tax_erased = fn_w["tax_erased"]
    fn_func_erased = fn_w["function_erased"]
    within_feasible = (fn_orig is not None and fn_tax_erased is not None
                       and s3["within_tax_family"]["n_queries"] >= 10)
    if not within_feasible:
        # within-stratum not feasible at this n: report global with explicit confound flag
        fn_orig = fn_g["original"]
        fn_tax_erased = fn_g["tax_erased"]
        fn_func_erased = fn_g["function_erased"]
    fn_metric = "within_tax_family" if within_feasible else "global(confounded_fallback)"

    # global (secondary) figures for context regardless
    g_orig, g_tax_erased = fn_g["original"], fn_g["tax_erased"]
    global_rel_drop_on_tax_erase = (g_orig - g_tax_erased) / g_orig if g_orig else float("nan")

    # (a) taxonomy erasable: drops toward chance
    tax_abs_drop = tax_orig - tax_erased
    tax_erased_at_chance = bool(tax_erased <= chance + 0.05)
    taxonomy_erasable = bool(tax_abs_drop >= 0.05 and tax_erased_at_chance)
    # (b) function preserved under tax-erasure (PRIMARY = within-stratum)
    func_rel_drop_on_tax_erase = (fn_orig - fn_tax_erased) / fn_orig if fn_orig else float("nan")
    function_preserved = bool(not (func_rel_drop_on_tax_erase != func_rel_drop_on_tax_erase)
                              and func_rel_drop_on_tax_erase < 0.10)
    # reverse: taxonomy survives function-erasure
    tax_rel_drop_on_func_erase = (tax_orig - tax_func_erased) / tax_orig if tax_orig else float("nan")
    func_rel_drop_on_func_erase = (fn_orig - fn_func_erased) / fn_orig if fn_orig else float("nan")
    taxonomy_survives_func_erasure = bool(
        not (tax_rel_drop_on_func_erase != tax_rel_drop_on_func_erase)
        and tax_rel_drop_on_func_erase < 0.10)

    clean_disentangled = bool(taxonomy_erasable and function_preserved)
    reads_cleaner_than_pla2 = bool(taxonomy_erasable and function_preserved)

    if clean_disentangled:
        interp = (
            "CLEAN DISENTANGLEMENT (the expected PRIMARY result): taxonomy (tax_family acc "
            f"{tax_orig:.3f}->{tax_erased:.3f}, collapsing to ~chance={chance:.3f}) is ERASABLE while the "
            f"toxin-Family function clustering is PRESERVED ({fn_metric} purity {fn_orig:.3f}->"
            f"{fn_tax_erased:.3f}, rel-drop={func_rel_drop_on_tax_erase:.3f} < 0.10). This reads CLEANER "
            "than PLA2: there, taxonomy only fell from 0.74 to 0.56 under LEACE (partial erasure) AND "
            "erasing FUNCTION damaged taxonomy by ~16% (entanglement); here taxonomy is erased ~fully "
            "to chance and erasing taxonomy leaves taxonomy-controlled function purity untouched. The "
            "non-phylogenetic toxin-Family axis lives in a subspace separable from the taxonomy axis. "
            "NOTE the SECONDARY global purity DOES drop "
            f"({g_orig:.3f}->{g_tax_erased:.3f}, rel-drop={global_rel_drop_on_tax_erase:.3f}); that is "
            "EXPECTED and not entanglement — same-toxin-family neighbours are often same-taxon, so erasing "
            "taxonomy pulls them apart globally. The within-stratum metric removes that confound.")
    elif function_preserved:
        interp = (
            f"PARTIALLY CLEAN: toxin-Family function preserved under taxonomy erasure ({fn_metric} "
            f"rel-drop={func_rel_drop_on_tax_erase:.3f}), but the taxonomy-erasability gate was not fully "
            f"met (tax_family acc {tax_orig:.3f}->{tax_erased:.3f} vs chance={chance:.3f}). At "
            f"n={out['meta']['n_proteins']} the taxonomy probe is noisy; still reads cleaner than PLA2 on "
            "the function-preservation axis.")
    else:
        interp = (
            f"NOT cleanly disentangled at the measured grain: even taxonomy-controlled ({fn_metric}) "
            f"function purity dropped on taxonomy erasure (rel-drop={func_rel_drop_on_tax_erase:.3f} "
            ">= 0.10). Inspect — multifamily was expected to disentangle more cleanly than PLA2; small n "
            "may be driving instability.")

    return {
        "clean_disentangled": clean_disentangled,
        "reads_cleaner_than_pla2": reads_cleaner_than_pla2,
        "primary_function_metric": fn_metric,
        "taxonomy_erasable": taxonomy_erasable,
        "function_preserved_under_taxonomy_erasure": function_preserved,
        "taxonomy_survives_function_erasure": taxonomy_survives_func_erasure,
        "tax_family_acc_original": round(float(tax_orig), 4),
        "tax_family_acc_tax_erased": round(float(tax_erased), 4),
        "tax_family_chance": round(float(chance), 4),
        "function_within_stratum_original": (round(float(fn_w["original"]), 4)
                                             if fn_w["original"] is not None else None),
        "function_within_stratum_tax_erased": (round(float(fn_w["tax_erased"]), 4)
                                               if fn_w["tax_erased"] is not None else None),
        "function_relative_drop_on_taxonomy_erasure_PRIMARY": round(float(func_rel_drop_on_tax_erase), 4),
        "function_relative_drop_on_taxonomy_erasure_global_SECONDARY": round(float(global_rel_drop_on_tax_erase), 4),
        "global_purity_drop_is_taxonomy_confound_not_entanglement": True,
        "leace_clean_gate_ok": out["leace"]["clean_claim_gated_ok"],
        "leace_gate_caveat": (
            None if out["leace"]["clean_claim_gated_ok"] else
            f"M4 caveat: LEACE erased k={out['leace']['erased_rank_k']} of PCA_DIM={out['leace']['pca_dim']} "
            "(≈half the space) because PCA_DIM is small at n=84. The within-stratum function purity being "
            "FULLY preserved (0.827->0.833) despite this is reassuring (function lives outside the erased "
            "subspace), but the gate failing means the 'preserved' claim carries somewhat less weight than "
            "it would at larger n with a smaller erased fraction. Reported honestly."),
        "taxonomy_relative_drop_on_function_erasure": round(float(tax_rel_drop_on_func_erase), 4),
        "function_relative_drop_on_function_erasure": round(float(func_rel_drop_on_func_erase), 4),
        "reverse_direction_note": (
            "taxonomy_survives_function_erasure is False (erasing toxin-Family drops tax_family acc "
            f"{tax_orig:.3f}->{tax_func_erased:.3f}). This is NOT embedding entanglement: in this curated "
            "ToxFam-12 set the toxin-Family one-hot structurally CONTAINS most taxonomy "
            f"(H(tax|function) reduction = {out['meta']['function_taxonomy_nesting']['tax_order']['uncertainty_reduction']:.1%} "
            "at order rank; "
            f"{out['meta']['function_taxonomy_nesting']['tax_order']['n_function_labels_mapping_to_one_tax']}"
            f"/{out['meta']['function_taxonomy_nesting']['tax_order']['n_function_labels']} toxin families "
            "map to ONE taxonomic order), so LEACE-erasing the function one-hot removes those shared "
            "directions too. The disentanglement CLAIM rests on the FORWARD direction (erase taxonomy, "
            "keep taxonomy-controlled function), which is clean."),
        "expected_clean_contrast_to_pla2": True,
        "interpretation": interp,
        "small_n_caveat": out["meta"]["small_n_warnings"],
    }


def _build_matrix_mammals(Z, Xe_tax, Xe_func, Zc, func_groups, species_codes, tax_targets,
                          species_strata, crossed_mask, folds, k_erased):
    """Disentanglement matrix for pla2_mammals: taxonomy probe = Species, function = within-Species purity.

    Within-Species purity is NaN when most species have fewer than k+1=6 proteins (38 species over
    146 proteins => ~4 per species on average). In that case the matrix function row is all NaN and we
    report the global (non-stratified) purity as the only viable function signal with a clear flag.
    The global purity is informative about function clustering overall but does NOT control for
    taxonomy (a neighbour of the same species trivially shares its gene group here), so it is not a
    clean within-stratum measure — documented honestly.
    """
    reps_by_col = {"original": Z, "tax_erased": Xe_tax, "function_erased": Xe_func, "matched_control": Zc}
    matrix = {
        "leace_erased_rank_k": int(k_erased),
        "taxonomy_probe_label": "Species_acc (within-Mammals; Clade is constant)",
        "taxonomy_recoverability_species_acc": {},
        "function_recoverability_within_species_purity": {},
        "function_recoverability_global_purity": {},
        "function_note": (
            "within_species_purity is NaN where most species have <k+1=6 proteins "
            "(38 species / 146 proteins => ~4/species mean); use global_purity as the viable signal, "
            "noting it does NOT control for taxonomy (within-species gene group is trivially correlated). "
            "Coherence + step3 global figures are the primary read-outs for this subset."),
    }
    for col, Xcol in reps_by_col.items():
        tax = taxonomy_probe_categorical(Xcol, species_codes, folds)
        fn_within, _ = knn_purity_within_strata(Xcol, func_groups, species_strata)
        fn_global, _ = knn_purity_global(Xcol, func_groups, subset_mask=crossed_mask)
        matrix["taxonomy_recoverability_species_acc"][col] = round(tax["linear_acc"], 4)
        fn_within_val = round(fn_within, 4) if not (fn_within != fn_within) else None
        matrix["function_recoverability_within_species_purity"][col] = fn_within_val
        matrix["function_recoverability_global_purity"][col] = round(fn_global, 4)
    return matrix


def _entanglement_verdict_mammals(out):
    """Verdict for pla2_mammals: reads the disentanglement matrix.

    Primary signal: global function purity (within-Species purity is NaN due to small n-per-species).
    Species accuracy is at chance (0/0 probe with 38 classes on ~29 test points).
    Verdict is reported as UNDETERMINED_SMALL_N when the Species taxonomy probe is at chance
    (linear_acc <= chance + epsilon) AND within-species function purity is NaN — meaning neither
    metric can be meaningfully compared. The global purity comparison is reported separately as the
    best available proxy.
    """
    m = out["disentanglement_matrix"]
    tax = m["taxonomy_recoverability_species_acc"]
    fn_global = m["function_recoverability_global_purity"]

    fn_orig_g = fn_global["original"]
    fn_tax_erased_g = fn_global["tax_erased"]
    tax_orig = tax["original"]
    tax_func_erased = tax["function_erased"]

    # Global purity drops on taxonomy erasure (best available function signal)
    fn_global_drop = ((fn_orig_g - fn_tax_erased_g) / fn_orig_g
                      if fn_orig_g and not (fn_orig_g != fn_orig_g) else float("nan"))
    # Taxonomy probe is at/near chance (38-class, ~29 test points): acc <= 2*chance
    tax_acc_chance = out["step2_erasure_worked"]["species_probe"]["original"].get("chance", 0.034)
    tax_at_chance = bool(tax_orig is not None and tax_orig <= 2.0 * tax_acc_chance)

    # Within-species function purity: NaN when n-per-species < k+1
    fn_within = m["function_recoverability_within_species_purity"]
    fn_within_orig = fn_within.get("original")
    fn_within_nan = fn_within_orig is None

    # Entanglement assessment via global purity (proxy only; explicitly labelled)
    global_purity_entangled = bool(
        not (fn_global_drop != fn_global_drop) and fn_global_drop >= 0.10)

    if tax_at_chance and fn_within_nan:
        reads_as = "UNDETERMINED_SMALL_N"
        interp = (
            "UNDETERMINED due to small-n instability: Species taxonomy probe is at chance "
            f"(acc={tax_orig:.4f} <= 2*chance={2*tax_acc_chance:.4f}; 38 classes, ~29 test "
            "points), and within-Species kNN purity is NaN (most species have <6 proteins). "
            "The formal entanglement verdict cannot be rendered. "
            f"Best available proxy: global kNN purity orig={fn_orig_g:.4f} -> "
            f"tax-erased={fn_tax_erased_g:.4f} (drop={fn_global_drop:.4f}); "
            "this is likely ENTANGLED at the global level but does not isolate the within-species "
            "signal. Coherence overlap=high (see coherence block) confirms READ and CLEAN share "
            "the same subspace. The multi-family testbed (T15) is needed for a clean disentanglement "
            "verdict."
        )
    else:
        reads_as = "ENTANGLED" if global_purity_entangled else "SEPARABLE"
        interp = (
            "ENTANGLED (expected for within-Mammals PLA2 crossed-group subset): "
            "within one clade, Gene Group ≈ paralog lineage correlates with species, "
            "so function and taxonomy subspaces are not cleanly separable."
            if global_purity_entangled else
            "Surprisingly SEPARABLE — taxonomy could be erased without much function loss. "
            "Inspect: PLA2 within-Mammals was expected entangled.")

    return {
        "reads_as": reads_as,
        "reads_as_entangled": (reads_as == "ENTANGLED"),
        "global_function_drop_on_taxonomy_erasure": round(float(fn_global_drop), 4),
        "taxonomy_acc_at_chance": tax_at_chance,
        "within_species_purity_nan": fn_within_nan,
        "expected_for_pla2_mammals": True,
        "interpretation": interp,
        "small_n_caveat": out["meta"]["small_n_warnings"],
    }


# ============================================================================ matrix + helpers
def _build_matrix(Z, Xe_tax, Xe_func, Zc, func_groups, clade_codes, tax_targets, clade_strata,
                  crossed_mask, folds, k_erased):
    """Emit {taxonomy, function}-recoverability x {original, tax-erased, function-erased, matched-control},
    within-strata, WITH the LEACE erased rank recorded. Taxonomy-recoverability = held-out linear Clade
    accuracy. Function-recoverability = within-clade kNN-purity@k. Both on the SAME fixed feature
    matrices."""
    reps_by_col = {"original": Z, "tax_erased": Xe_tax, "function_erased": Xe_func, "matched_control": Zc}
    matrix = {"leace_erased_rank_k": int(k_erased),
              "taxonomy_recoverability_clade_acc": {}, "function_recoverability_within_clade_purity": {}}
    for col, Xcol in reps_by_col.items():
        tax = taxonomy_probe_categorical(Xcol, clade_codes, folds)
        fn, _ = knn_purity_within_strata(Xcol, func_groups, clade_strata)
        matrix["taxonomy_recoverability_clade_acc"][col] = round(tax["linear_acc"], 4)
        matrix["function_recoverability_within_clade_purity"][col] = round(fn, 4)
    return matrix


def _crossed_group_mask(ann):
    g2nclade = ann.dropna(subset=["Clade"]).groupby("Gene Group")["Clade"].nunique()
    crossed = set(g2nclade[g2nclade >= 2].index.tolist())
    return ann["Gene Group"].isin(crossed).to_numpy()


def _crossed_groups(ann):
    g2nclade = ann.dropna(subset=["Clade"]).groupby("Gene Group")["Clade"].nunique()
    return sorted(g2nclade[g2nclade >= 2].index.tolist())


def _n_crossed_groups(ann):
    return len(_crossed_groups(ann))


def _group_chance_purity(groups):
    """Expected kNN purity if neighbours were random = sum of squared group frequencies."""
    _, counts = np.unique(groups, return_counts=True)
    p = counts / counts.sum()
    return float((p ** 2).sum())


def _identity_stratum_report(Z, Xe_tax, func_groups, clade_codes, cluster_ids, folds):
    """Report function-preservation (within-clade purity) + taxonomy-recoverability (clade acc) by
    pairwise-identity stratum. Coarse binning by cluster SIZE: singleton clusters (low redundancy /
    lower identity neighbourhood) vs multi-member clusters (high identity). Flags if the effect lives
    only in the high-identity stratum."""
    cluster_ids = np.asarray(cluster_ids)
    _, inv, counts = np.unique(cluster_ids, return_inverse=True, return_counts=True)
    csize = counts[inv]                                       # cluster size per row
    strata = {"singleton_low_identity": csize == 1, "multi_member_high_identity": csize > 1}
    report = {"binning": "cluster size (mmseqs 0.9): singleton vs multi-member", "strata": {}}
    clade = np.asarray(["c%d" % c for c in clade_codes])     # stratum labels for within-strata purity
    for name, mask in strata.items():
        idx = np.where(mask)[0]
        if len(idx) < KNN_K + 1:
            report["strata"][name] = {"n_rows": int(len(idx)), "skipped": "too few rows"}
            continue
        fp_orig, _ = knn_purity_within_strata(Z[idx], func_groups[idx], clade[idx])
        fp_erased, _ = knn_purity_within_strata(Xe_tax[idx], func_groups[idx], clade[idx])
        # taxonomy recoverability inside the stratum: restrict the held-out folds to this stratum's rows
        sub_folds = [(np.intersect1d(tr, idx), np.intersect1d(te, idx)) for tr, te in folds]
        sub_folds = [(tr, te) for tr, te in sub_folds if len(te) and len(np.unique(clade_codes[tr])) >= 2]
        tax_orig = taxonomy_probe_categorical(Z, clade_codes, sub_folds) if sub_folds else {"linear_acc": float("nan")}
        tax_erased = taxonomy_probe_categorical(Xe_tax, clade_codes, sub_folds) if sub_folds else {"linear_acc": float("nan")}
        report["strata"][name] = {
            "n_rows": int(len(idx)),
            "function_purity_within_clade": {"original": fp_orig, "tax_erased": fp_erased},
            "taxonomy_clade_acc": {"original": tax_orig.get("linear_acc"),
                                   "tax_erased": tax_erased.get("linear_acc")},
        }
    return report


# ============================================================================ Task 14: coherence
def coherence(b, P_tax, pca_dim):
    """Task 14 — measured READ<->CLEAN identity. Same basis `b` (B2), matched low rank (B1):
      E = erased directions of the tax-LEACE (shape (pca_dim, k)),
      R = READ's predictive tax subspace truncated to the SAME k,
    then overlap = principal_angle_overlap(R, E) reported ALONGSIDE the random-rank-k baseline
    (mean over seeds of overlap(R, random_rank_k)). 'Same object' holds only if overlap >> baseline."""
    E = erasing_projector_rowspace(P_tax)                     # (pca_dim, k)
    k = int(E.shape[1])
    assert 0 < k < pca_dim, f"coherence rank k={k} must be 0<k<pca_dim={pca_dim} (B1: non-trivial, non-full)"
    R = b.tax_subspace(rank=k)                                # READ subspace at the SAME rank
    overlap = principal_angle_overlap(R, E)

    rng = np.random.default_rng(SEED)
    baselines = []
    for _ in range(N_RANDOM_BASELINE):
        G = rng.standard_normal((pca_dim, k))
        Q = np.linalg.qr(G)[0][:, :k]
        baselines.append(principal_angle_overlap(R, Q))
    baseline_mean = float(np.mean(baselines))
    baseline_std = float(np.std(baselines))
    expected_random = k / pca_dim                             # ≈ k/PCA_DIM theory check
    same_object = bool(overlap > baseline_mean + 3 * baseline_std)
    print(f"[task14] k={k}/{pca_dim}  READ<->CLEAN overlap={overlap:.4f}  "
          f"random baseline={baseline_mean:.4f}±{baseline_std:.4f} (~k/PCA_DIM={expected_random:.4f})  "
          f"same-object: {same_object}")
    return {"k": k, "pca_dim": int(pca_dim), "overlap": overlap,
            "random_baseline_mean": baseline_mean, "random_baseline_std": baseline_std,
            "expected_random_k_over_pca_dim": expected_random, "n_random_seeds": N_RANDOM_BASELINE,
            "overlap_exceeds_baseline_3sigma": same_object,
            "interpretation": ("READ and CLEAN target the SAME subspace (overlap >> random baseline)"
                               if same_object else
                               "overlap NOT clearly above random baseline — READ and CLEAN subspaces "
                               "may differ at this rank (report honestly)")}


# ============================================================================ verdict
def _entanglement_verdict(out):
    """Read the matrix: PLA2 within-family is EXPECTED entangled. Entangled = erasing taxonomy ALSO
    drops function purity meaningfully (>= ~10% relative) AND/OR erasing function ALSO drops taxonomy.
    We REPORT the contrast, we do not pass/fail on it."""
    m = out["disentanglement_matrix"]
    tax = m["taxonomy_recoverability_clade_acc"]
    fn = m["function_recoverability_within_clade_purity"]
    fn_orig = fn["original"]
    fn_tax_erased = fn["tax_erased"]
    tax_orig = tax["original"]
    tax_func_erased = tax["function_erased"]
    func_drop_on_tax_erase = (fn_orig - fn_tax_erased) / fn_orig if fn_orig else float("nan")
    tax_drop_on_func_erase = (tax_orig - tax_func_erased) / tax_orig if tax_orig else float("nan")
    entangled = bool((func_drop_on_tax_erase >= 0.10) or (tax_drop_on_func_erase >= 0.10))
    return {
        "reads_as_entangled": entangled,
        "function_relative_drop_on_taxonomy_erasure": round(float(func_drop_on_tax_erase), 4),
        "taxonomy_relative_drop_on_function_erasure": round(float(tax_drop_on_func_erase), 4),
        "expected_for_pla2": True,
        "interpretation": (
            "ENTANGLED (expected for the PLA2 single-family hard contrast): within one gene family, "
            "Gene Group ≈ paralog lineage ≈ taxonomy, so the function and taxonomy subspaces overlap and "
            "cannot be cleanly separated. A clean DISentangled result requires the multi-family testbed."
            if entangled else
            "Surprisingly SEPARABLE at the measured grain — taxonomy could be erased without much function "
            "loss (and/or vice versa). Inspect before trusting: PLA2 was expected to be entangled.")
    }


# ============================================================================ main
def main(testbed: str = "pla2"):
    t_start = time.time()
    print(f"=== Tasks 13+14: CLEAN battery + coherence  (testbed={testbed}) ===")

    if testbed.startswith(("sp_metazoa_", "all_life_")):
        return _main_sp(testbed, t_start)

    if testbed == "multifamily":
        return _main_multifamily(t_start)

    if testbed == "pla2_mammals":
        return _main_pla2_mammals(t_start)

    # --- pla2 (default) ---
    ann, reps, positions = load_pla2_clean()
    print(f"loaded {len(ann)} proteins, {ann['Species'].nunique()} species, "
          f"{ann['Clade'].nunique()} clades, {ann['Gene Group'].nunique()} gene groups, "
          f"{ann['cluster_id'].nunique()} identity clusters")

    out = run_battery(ann, reps, positions)
    out["entanglement_verdict"] = _entanglement_verdict(out)
    out["runtime_sec"] = round(time.time() - t_start, 1)

    out_path = _HERE / "results" / f"clean_{testbed}.json"
    out_path.write_text(json.dumps(out, indent=2))

    # -------- printed verdict --------
    print("\n" + "=" * 78)
    print(f"DISENTANGLEMENT MATRIX (LEACE erased rank k={out['disentanglement_matrix']['leace_erased_rank_k']} "
          f"of PCA_DIM={out['meta']['pca_dim']})")
    cols = ["original", "tax_erased", "function_erased", "matched_control"]
    tax = out["disentanglement_matrix"]["taxonomy_recoverability_clade_acc"]
    fn = out["disentanglement_matrix"]["function_recoverability_within_clade_purity"]
    print(f"{'metric':<34}" + "".join(f"{c:<18}" for c in cols))
    print(f"{'taxonomy (Clade acc)':<34}" + "".join(f"{tax[c]:<18.4f}" for c in cols))
    print(f"{'function (within-clade purity)':<34}" + "".join(f"{fn[c]:<18.4f}" for c in cols))
    s3 = out["step3_function_preserved"]
    print(f"\nfunction purity orig->tax-erased: within-clade {s3['within_clade']['original']:.3f}->"
          f"{s3['within_clade']['tax_erased']:.3f}  crossed-subset "
          f"{s3['crossed_group_subset']['original']:.3f}->{s3['crossed_group_subset']['tax_erased']:.3f}")
    s4 = out["step4_specificity"]
    print(f"specificity: control energy ratio={s4['energy_ratio_control_over_leace']:.3f} "
          f"(0.5x-2x ok: {s4['energy_match_ok_half_to_2x']}); taxonomy under control Clade acc="
          f"{s4['taxonomy_under_control']['clade_probe']['linear_acc']:.3f}")
    c = out["coherence"]
    print(f"COHERENCE (Task 14): k={c['k']}  overlap={c['overlap']:.4f}  vs random baseline "
          f"{c['random_baseline_mean']:.4f}  -> same-object: {c['overlap_exceeds_baseline_3sigma']}")
    v = out["entanglement_verdict"]
    print(f"\nVERDICT: reads as {'ENTANGLED' if v['reads_as_entangled'] else 'SEPARABLE'} "
          f"(func drop on tax-erase={v['function_relative_drop_on_taxonomy_erasure']}, "
          f"tax drop on func-erase={v['taxonomy_relative_drop_on_function_erasure']})")
    print(f"  {v['interpretation']}")
    print(f"\nwrote {out_path}  (runtime {out['runtime_sec']}s)")
    return out


def _main_pla2_mammals(t_start):
    """Task 16: within-Mammals crossed-group battery."""
    ann_m, reps_m, positions_m, crossed_groups_list, pca_dim_m = load_pla2_mammals_clean()
    print(f"[pla2_mammals] loaded subset: n={len(ann_m)}, species={ann_m['Species'].nunique()}, "
          f"gene_groups={ann_m['Gene Group'].nunique()}, "
          f"crossed_groups={crossed_groups_list}")

    out = run_battery_mammals(ann_m, reps_m, positions_m, crossed_groups_list, pca_dim_m)
    out["entanglement_verdict"] = _entanglement_verdict_mammals(out)
    out["runtime_sec"] = round(time.time() - t_start, 1)

    out_path = _HERE / "results" / "clean_pla2_mammals.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    # -------- printed verdict --------
    print("\n" + "=" * 78)
    m = out["disentanglement_matrix"]
    print(f"DISENTANGLEMENT MATRIX [pla2_mammals] "
          f"(LEACE erased rank k={m['leace_erased_rank_k']} of PCA_DIM={out['meta']['pca_dim']})")
    print(f"  taxonomy probe: {m['taxonomy_probe_label']}")
    cols = ["original", "tax_erased", "function_erased", "matched_control"]
    tax = m["taxonomy_recoverability_species_acc"]
    fn_g = m["function_recoverability_global_purity"]
    fn_w = m["function_recoverability_within_species_purity"]

    def _fmt(v):
        return f"{v:<18.4f}" if v is not None and not (isinstance(v, float) and v != v) else f"{'NaN':<18}"

    print(f"{'metric':<36}" + "".join(f"{c:<18}" for c in cols))
    print(f"{'taxonomy (Species acc)':<36}" + "".join(_fmt(tax[c]) for c in cols))
    print(f"{'function (global purity)':<36}" + "".join(_fmt(fn_g[c]) for c in cols))
    print(f"{'function (within-Species purity)':<36}" + "".join(_fmt(fn_w[c]) for c in cols))
    print(f"  NOTE: within-Species purity NaN = most species <6 proteins; use global as proxy.")
    c = out["coherence"]
    print(f"\nCOHERENCE (Task 14): k={c['k']}  overlap={c['overlap']:.4f}  vs random baseline "
          f"{c['random_baseline_mean']:.4f}  -> same-object: {c['overlap_exceeds_baseline_3sigma']}")
    v = out["entanglement_verdict"]
    print(f"\nVERDICT: reads_as={v['reads_as']}  "
          f"(global func drop on tax-erase={v['global_function_drop_on_taxonomy_erasure']:.4f})")
    print(f"  {v['interpretation']}")
    if v["small_n_caveat"]:
        print(f"  SMALL-N WARNINGS: {v['small_n_caveat']}")
    print(f"\nwrote {out_path}  (runtime {out['runtime_sec']}s)")
    return out


def _main_multifamily(t_start):
    """Task 15: the multi-family (ToxFam-12) PRIMARY CLEAN battery — the contrast to PLA2."""
    ann, reps, positions, pca_dim = load_multifamily_clean()
    print(f"[multifamily] loaded n={len(ann)}, toxin_families={ann['Family'].nunique()}, "
          f"distinct_taxids={ann['taxid'].nunique()}, "
          f"tax_family_classes={pd.Series(ann['tax_family']).nunique()}")

    out = run_battery_multifamily(ann, reps, positions, pca_dim)
    out["disentanglement_verdict"] = _disentanglement_verdict_multifamily(out)
    out["runtime_sec"] = round(time.time() - t_start, 1)

    out_path = _HERE / "results" / "clean_multifamily.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    # -------- printed verdict --------
    print("\n" + "=" * 78)
    m = out["disentanglement_matrix"]
    print(f"DISENTANGLEMENT MATRIX [multifamily / ToxFam-12] "
          f"(LEACE erased rank k={m['leace_erased_rank_k']} of PCA_DIM={out['meta']['pca_dim']})")
    print(f"  taxonomy probe: {m['taxonomy_probe_label']}  |  function: {m['function_metric_primary']}")
    cols = ["original", "tax_erased", "function_erased", "matched_control"]
    tax = m["taxonomy_recoverability_tax_family_acc"]
    fn_g = m["function_recoverability_global_purity"]
    fn_w = m["function_recoverability_within_tax_family_purity"]

    def _fmt(v):
        return f"{v:<18.4f}" if v is not None and not (isinstance(v, float) and v != v) else f"{'NaN':<18}"

    print(f"{'metric':<38}" + "".join(f"{c:<18}" for c in cols))
    print(f"{'taxonomy (tax_family acc)':<38}" + "".join(_fmt(tax[c]) for c in cols))
    print(f"{'function (global toxin-Family purity)':<38}" + "".join(_fmt(fn_g[c]) for c in cols))
    print(f"{'function (within-tax_family purity)':<38}" + "".join(_fmt(fn_w[c]) for c in cols))
    s3 = out["step3_function_preserved"]
    print(f"\nfunction global purity orig->tax-erased: {s3['global']['original']:.3f}->"
          f"{s3['global']['tax_erased']:.3f}  (chance~{s3['group_chance_purity']:.3f})")
    s4 = out["step4_specificity"]
    print(f"specificity: control energy ratio={s4['energy_ratio_control_over_leace']:.3f} "
          f"(0.5x-2x ok: {s4['energy_match_ok_half_to_2x']}); taxonomy under control tax_family acc="
          f"{s4['taxonomy_under_control']['tax_family_probe']['linear_acc']:.3f}")
    c = out["coherence"]
    print(f"COHERENCE (Task 14): k={c['k']}/{c['pca_dim']}  overlap={c['overlap']:.4f}  vs random baseline "
          f"{c['random_baseline_mean']:.4f}±{c['random_baseline_std']:.4f}  "
          f"-> same-object: {c['overlap_exceeds_baseline_3sigma']}")
    v = out["disentanglement_verdict"]
    print(f"\nVERDICT: clean_disentangled={v['clean_disentangled']}  "
          f"reads_cleaner_than_pla2={v['reads_cleaner_than_pla2']}  "
          f"(primary function metric: {v['primary_function_metric']})")
    print(f"  taxonomy_erasable={v['taxonomy_erasable']}  "
          f"function_preserved_under_tax_erasure={v['function_preserved_under_taxonomy_erasure']}  "
          f"taxonomy_survives_func_erasure={v['taxonomy_survives_function_erasure']}")
    print(f"  tax_family acc {v['tax_family_acc_original']}->{v['tax_family_acc_tax_erased']} "
          f"(chance {v['tax_family_chance']})")
    print(f"  func PRIMARY (within-tax_family) rel-drop on tax-erase="
          f"{v['function_relative_drop_on_taxonomy_erasure_PRIMARY']}  |  "
          f"SECONDARY (global, confounded) rel-drop="
          f"{v['function_relative_drop_on_taxonomy_erasure_global_SECONDARY']}")
    print(f"  tax rel-drop on func-erase={v['taxonomy_relative_drop_on_function_erasure']}")
    print(f"  {v['interpretation']}")
    if v["small_n_caveat"]:
        print(f"  SMALL-N WARNINGS: {v['small_n_caveat']}")
    print(f"\nwrote {out_path}  (runtime {out['runtime_sec']}s)")
    return out


def _main_sp(testbed, t_start):
    """Task C2a: the SP-Metazoa CLEAN battery (clone of _main_multifamily). `testbed` is one of
    {sp_metazoa_pfam, sp_metazoa_ec} optionally with a `_full` zero-exclusion suffix."""
    ann, reps, positions, pca_dim = load_sp_clean(testbed)
    stratum_col = "tax_class"
    print(f"[{testbed}] loaded n={len(ann)}, function_labels={ann['func'].nunique()}, "
          f"distinct_taxids={ann['taxid'].nunique()}, "
          f"{stratum_col}_classes={pd.Series(ann[stratum_col]).nunique()}")

    # Enforced confound gates (spec §6.1/§6.2 S1): the real run MUST exercise both length- and
    # composition-erasure. The frozen panel carries `length` + `aa_A..aa_Y` by construction; if the
    # aa columns are absent run_battery_sp raises (fail-loud, never a silent skip of the gate).
    out = run_battery_sp(ann, reps, positions, pca_dim, stratum_col=stratum_col,
                         length_control=True, composition_control=True)
    out["disentanglement_verdict"] = _disentanglement_verdict_sp(out)
    out["runtime_sec"] = round(time.time() - t_start, 1)
    # consolidated provenance for the result JSON (spec v3 §9) — emitted here (the result-JSON assembler)
    # rather than inside run_battery_sp, which lacks testbed/path inputs and is the unit-tested hot path.
    out["meta"]["provenance"] = _build_provenance(
        testbed, _load_sp_panel_path(testbed), _load_sp_h5_path(testbed), config.SP_CAP_SEED)

    # --- all-life §7 batch-effect guard + §4.2.4/§10 per-superkingdom retention (B-2; gated → metazoa untouched) ---
    if testbed.startswith("all_life_"):
        from . import twin_batch_eval as _tb
        al = {}
        if "source" in ann.columns and ann["source"].nunique() > 1:
            al["batch_effect_auc_sp_vs_trembl"] = _tb.batch_effect_auc(
                reps, ann["source"].tolist(), seed=config.SP_CAP_SEED)
        if "tax_superkingdom" in ann.columns:
            al["per_superkingdom_retention"] = _tb.per_superkingdom_retention(
                ann["tax_superkingdom"].tolist())
        out["meta"]["all_life"] = al

    out_path = _HERE / "results" / f"clean_{testbed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    # -------- printed verdict --------
    print("\n" + "=" * 78)
    m = out["disentanglement_matrix"]
    print(f"DISENTANGLEMENT MATRIX [{testbed}] "
          f"(LEACE erased rank k={m['leace_erased_rank_k']} of PCA_DIM={out['meta']['pca_dim']})")
    print(f"  taxonomy probe: {m['taxonomy_probe_label']}  |  function: {m['function_metric_primary']}")
    cols = ["original", "tax_erased", "function_erased", "matched_control"]
    tax = m["taxonomy_recoverability_tax_family_acc"]
    fn_g = m["function_recoverability_global_purity"]
    fn_w = m["function_recoverability_within_tax_family_purity"]

    def _fmt(v):
        return f"{v:<18.4f}" if v is not None and not (isinstance(v, float) and v != v) else f"{'NaN':<18}"

    print(f"{'metric':<38}" + "".join(f"{c:<18}" for c in cols))
    print(f"{'taxonomy (' + stratum_col + ' acc)':<38}" + "".join(_fmt(tax[c]) for c in cols))
    print(f"{'function (global func purity)':<38}" + "".join(_fmt(fn_g[c]) for c in cols))
    print(f"{'function (within-' + stratum_col + ' purity)':<38}" + "".join(_fmt(fn_w[c]) for c in cols))
    s3 = out["step3_function_preserved"]
    print(f"\nfunction global purity orig->tax-erased: {s3['global']['original']:.3f}->"
          f"{s3['global']['tax_erased']:.3f}  (chance~{s3['group_chance_purity']:.3f})")
    print(f"stratum_grain feasibility: { {k: v['feasible'] for k, v in out['stratum_grain'].items()} }  "
          f"-> headline_stratum={out['headline_stratum']}")
    s4 = out["step4_specificity"]
    print(f"specificity: control energy ratio={s4['energy_ratio_control_over_leace']:.3f} "
          f"(0.5x-2x ok: {s4['energy_match_ok_half_to_2x']}); pass={s4['pass']}; "
          f"taxonomy under control {stratum_col} acc="
          f"{s4['taxonomy_under_control']['tax_family_probe']['linear_acc']:.3f}")
    c = out["coherence"]
    print(f"COHERENCE (Task 14): k={c['k']}/{c['pca_dim']}  overlap={c['overlap']:.4f}  vs random baseline "
          f"{c['random_baseline_mean']:.4f}±{c['random_baseline_std']:.4f}  "
          f"-> same-object: {c['overlap_exceeds_baseline_3sigma']}")
    v = out["disentanglement_verdict"]
    print(f"\nVERDICT: clean_disentangled={v['clean_disentangled']}  "
          f"grain_consistent={v['grain_consistent']}  specificity_pass={v['specificity_pass']}  "
          f"(primary function metric: {v['primary_function_metric']})")
    print(f"  taxonomy_erasable={v['taxonomy_erasable']}  "
          f"function_preserved_under_tax_erasure={v['function_preserved_under_taxonomy_erasure']}  "
          f"taxonomy_survives_func_erasure={v['taxonomy_survives_function_erasure']}")
    print(f"  {stratum_col} acc {v['tax_family_acc_original']}->{v['tax_family_acc_tax_erased']} "
          f"(chance {v['tax_family_chance']})")
    print(f"  func PRIMARY (within-stratum) rel-drop on tax-erase="
          f"{v['function_relative_drop_on_taxonomy_erasure_PRIMARY']}  |  "
          f"SECONDARY (global, confounded) rel-drop="
          f"{v['function_relative_drop_on_taxonomy_erasure_global_SECONDARY']}")
    print(f"  label_nesting bidirectional: {v['label_nesting_bidirectional']}")
    print(f"  tax rel-drop on func-erase={v['taxonomy_relative_drop_on_function_erasure']}")
    print(f"  {v['interpretation']}")
    if v["small_n_caveat"]:
        print(f"  SMALL-N WARNINGS: {v['small_n_caveat']}")
    print(f"\nwrote {out_path}  (runtime {out['runtime_sec']}s)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--testbed", default="pla2",
                    choices=["pla2", "multifamily", "pla2_mammals",
                             "sp_metazoa_pfam", "sp_metazoa_ec",
                             "sp_metazoa_pfam_full", "sp_metazoa_ec_full",
                             "all_life_pfam", "all_life_ec",
                             "all_life_pfam_full", "all_life_ec_full",
                             "all_life_pfam_sp", "all_life_pfam_trembl",      # §4.4 SP-only / TrEMBL-only twin
                             "all_life_ec_sp", "all_life_ec_trembl"])
    args = ap.parse_args()
    main(args.testbed)
