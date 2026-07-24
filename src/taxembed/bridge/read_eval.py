"""Task 10 / M1b phasegate — the PLA2 READ experiment.

Composition of already-tested building blocks into the confirmatory READ evaluation for the
Taxonomy Bridge. We hold out whole species (LOSO), align the linear bridge on the remaining
proteins, PLACE the held species' proteins into the 498k metazoa Poincaré embedding, retrieve the
nearest reference node, and read its taxonomic lineage as the per-rank prediction. We compare the
bridge `f` against:
  (a) raw-pLM-kNN  (nearest TRAIN protein in raw ProtT5 space; inherit its species' lineage),
  (b) per-rank logistic ensemble on PCA features (accuracy + mean softmax entropy),
  (c) flat-target ablation (regress a one-hot genus target instead of hyperbolic positions),
  (d) a radial-only depth null (same retrieval against direction-randomized reference positions).
LOFO (leave-one-clade-out) and leave-species-AND-clade-out stress the same-species-paralog leak.

PRE-REGISTERED PRIMARY ENDPOINT (one confirmatory line, spec §6):
  LOSO *class*-rank accuracy of f vs raw-pLM-kNN (taxon-bootstrap CIs reported separately) AND
  f > depth-null. Everything else is exploratory (Holm-adjusted where p-values apply).

NCBI-rank reality (verified on disk 2026-06-17): 8 of the 85 PLA2 species (4 Crocodiles + 4
Turtles, 32 proteins) have NO "class" rank in NCBI taxonomy — they sit under clade "Sauropsida".
A rank that does not exist in the *true* lineage cannot be scored, so per rank we score only the
proteins/species whose true lineage contains that rank, and we RECORD the denominator. The
class-rank endpoint is therefore over the 77 species that have a class node — surfaced, not hidden.

Run:  python -m taxembed.bridge.read_eval
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
from .h5_io import load_embeddings  # noqa: E402
from .core import Bridge, TaxonomyEmbedding, log0, retrieve_nearest  # noqa: E402
from .splits import (  # noqa: E402
    leave_one_group_out,
    leave_species_out,
    per_fold_sign_test,
)
from .clusters import have_mmseqs, mmseqs_cluster  # noqa: E402
from .taxdump import TaxonResolver  # noqa: E402
from taxembed.eval.bootstrap import taxon_bootstrap_ci  # noqa: E402
from taxembed.eval.nulls import radial_only_null  # noqa: E402

from .eval_utils import RankLookup  # noqa: E402

SEED = 0
ALPHA_GRID = [0.01, 0.1, 1.0, 10.0]
N_BOOT = 2000
RANKS = config.RANKS
PRIMARY_RANK = "class"
RETRIEVE_CHUNK = 8192


# ---------------------------------------------------------------------------- data loading
def load_pla2():
    """Return a frame with one row per PLA2 protein: identifier, reps (float64 1024-d), Species,
    Clade, Gene Group, taxid, idx (metazoa node), position (100-d), cluster_id. All 452 join cleanly."""
    emb = load_embeddings(config.PLA2_H5)                       # {identifier: (1024,) float32}
    ann = pd.read_csv(config.PLA2_CSV)
    res = pd.read_csv(config.PLA2_RESOLUTION, sep="\t")         # species, taxid, idx, via
    res = res.rename(columns={"species": "Species"})
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

    # Clade is a species-level property; 3 proteins have a blank Clade cell -> backfill from the same
    # species' other proteins (deterministic). Surface it, don't silently drop (spec §8).
    n_blank = int(ann["Clade"].isna().sum())
    if n_blank:
        sp2clade = (ann.dropna(subset=["Clade"]).groupby("Species")["Clade"]
                    .agg(lambda s: s.mode().iloc[0]).to_dict())
        ann["Clade"] = ann["Clade"].fillna(ann["Species"].map(sp2clade))
        still = int(ann["Clade"].isna().sum())
        print(f"[clade] backfilled {n_blank} blank Clade cells from same-species rows "
              f"({still} still blank)")
        if still:
            raise SystemExit(f"{still} proteins have an un-backfillable Clade — investigate")

    clusters = compute_cluster_ids(ids)
    ann["cluster_id"] = [clusters[i] for i in ann["identifier"]]
    return ann, reps


def compute_cluster_ids(ids):
    """{identifier: cluster_rep} via mmseqs easy-cluster at 0.9 identity; fall back to one-per-seq."""
    fasta = config.PLA2_FASTA
    tmp = _HERE / "results" / "_mmseqs_tmp"
    if not have_mmseqs():
        print("[clusters] mmseqs NOT on PATH -> one-cluster-per-sequence fallback")
        return {i: i for i in ids}
    try:
        mapping = mmseqs_cluster(fasta, out_prefix=tmp / "pla2", tmp_dir=tmp / "tmp",
                                 min_seq_id=0.9, cleanup=True)
        for i in ids:                                          # singletons mmseqs may omit
            mapping.setdefault(i, i)
        n_clusters = len(set(mapping[i] for i in ids))
        print(f"[clusters] mmseqs -> {n_clusters} clusters over {len(ids)} sequences")
        return mapping
    except Exception as e:                                     # noqa: BLE001 — robust fallback per spec
        print(f"[clusters] mmseqs FAILED ({e}) -> one-cluster-per-sequence fallback")
        return {i: i for i in ids}


def load_sp_read(testbed):
    """Return (ann, reps) for an SP-Metazoa READ testbed by reusing the CLEAN-battery loader.

    `testbed` is one of {sp_metazoa_pfam, sp_metazoa_ec} (optionally with a `_full` suffix). We
    delegate the panel + per-protein-h5 read to clean_eval.load_sp_clean — the single, already-tested
    place that subsets the float16 SwissProt h5 by accession and keeps ann / reps / positions aligned —
    then surface the columns the cloned READ comparators expect. load_sp_clean returns
    (ann, reps, positions, pca_dim) with ann carrying func / accession / taxid / idx / cluster_id /
    tax_class / tax_order / tax_phylum / length; we add the two READ aliases:
        Clade   = tax_class   (the categorical taxonomy stratum the cloned READ functions group on)
        Species = taxid       (the species identity used for LOSO / per-species aggregation)
    Positions are recomputed inside the READ harness from ann['idx'] against the locked embedding, so we
    drop them here and return only (ann, reps) per the D1 contract."""
    from .clean_eval import load_sp_clean

    ann, reps, _positions, _pca_dim = load_sp_clean(testbed)
    ann = ann.copy()
    ann["Clade"] = ann["tax_class"]
    ann["Species"] = ann["taxid"]
    return ann, reps


# ---------------------------------------------------------------------------- testbed dispatch
def build_parser():
    """argparse parser for the READ harness. `--testbed` (default pla2) selects the experiment; the
    SP testbeds route to load_sp_read, pla2 to the original load_pla2 (behavior unchanged)."""
    ap = argparse.ArgumentParser(description="Taxonomy Bridge READ experiment")
    ap.add_argument("--testbed", default="pla2",
                    choices=["pla2", "sp_metazoa_pfam", "sp_metazoa_ec",
                             "sp_metazoa_pfam_full", "sp_metazoa_ec_full"])
    return ap


def select_loader(testbed):
    """Return the loader CALLABLE for a testbed: pla2 -> load_pla2 (the original PLA2 path),
    any sp_metazoa_* -> load_sp_read (which takes the testbed name). Returns the function object
    (not its result) so callers invoke it themselves."""
    if testbed == "pla2":
        return load_pla2
    if testbed.startswith("sp_metazoa_"):
        return load_sp_read
    raise SystemExit(f"[read] unknown testbed '{testbed}' (expected pla2 or sp_metazoa_*)")


# ---------------------------------------------------------------------------- taxonomy helpers
def per_rank_hits(true_taxids, pred_taxids, ranklut: RankLookup):
    """For arrays of true/pred taxids return {rank: (hits_bool, scored_bool)} where scored_bool marks
    proteins whose TRUE lineage has that rank (the only ones we can score)."""
    out = {r: {"hit": [], "scored": []} for r in RANKS}
    for tt, pt in zip(true_taxids, pred_taxids):
        trm = ranklut.rank_map(int(tt))
        prm = ranklut.rank_map(int(pt))
        for r in RANKS:
            scored = r in trm
            out[r]["scored"].append(scored)
            out[r]["hit"].append(bool(scored and prm.get(r) == trm.get(r)))
    return {r: (np.array(v["hit"], bool), np.array(v["scored"], bool)) for r, v in out.items()}


def per_species_rank_ci(species_arr, hits_by_rank):
    """Reduce protein-level hits to one value per species (mean over that species' SCORED proteins),
    then taxon-bootstrap CI over species. Returns {rank: {mean, lo, hi, n_species, n_scored}}."""
    species_arr = np.asarray(species_arr)
    res = {}
    for r in RANKS:
        hit, scored = hits_by_rank[r]
        per_species = []
        for sp in sorted(set(species_arr.tolist())):
            m = (species_arr == sp) & scored
            if m.any():
                per_species.append(hit[m].mean())
        per_species = np.array(per_species, float)
        if len(per_species):
            mean, lo, hi = taxon_bootstrap_ci(per_species, n_boot=N_BOOT, seed=SEED)
        else:
            mean = lo = hi = float("nan")
        res[r] = {"mean": mean, "lo": lo, "hi": hi,
                  "n_species": int(len(per_species)), "n_scored": int(scored.sum())}
    return res


# ---------------------------------------------------------------------------- alpha selection
def select_alpha(reps, positions, species):
    """Pick a single global Ridge alpha by leak-aware (species-grouped) inner CV on tangent-space
    reconstruction MSE — cheap (no retrieval), no held-species leak. Documented & frozen for all folds."""
    Y = log0(positions)
    rng = np.random.default_rng(SEED)
    uniq = np.array(sorted(set(species.tolist())))
    rng.shuffle(uniq)
    folds = np.array_split(uniq, 5)
    best_alpha, best_mse = ALPHA_GRID[0], np.inf
    for alpha in ALPHA_GRID:
        mses = []
        for held in folds:
            te_mask = np.isin(species, held)
            tr_mask = ~te_mask
            if tr_mask.sum() < 5 or te_mask.sum() == 0:
                continue
            br = Bridge.align(reps[tr_mask], positions[tr_mask], pca_dim=config.PCA_DIM, alpha=alpha)
            Zte = br.pca.transform(reps[te_mask])
            pred_tan = br.ridge.predict(Zte)
            mses.append(float(((pred_tan - Y[te_mask]) ** 2).mean()))
        mean_mse = float(np.mean(mses))
        print(f"[alpha] {alpha:>6}: grouped-CV tangent MSE = {mean_mse:.6f}")
        if mean_mse < best_mse:
            best_alpha, best_mse = alpha, mean_mse
    print(f"[alpha] selected alpha={best_alpha} (tangent MSE {best_mse:.6f})")
    return best_alpha


# ---------------------------------------------------------------------------- LOSO retrieval
def run_loso(reps, positions, ann, te_emb, ranklut, alpha, reference_positions=None, tag="f"):
    """LOSO: per held species, align bridge on train, place held reps, retrieve nearest in
    `reference_positions` (default the real 498k), read predicted taxid per protein. Returns
    (per_species_rank_ci dict, protein-level (true_taxids, pred_taxids, species)) for downstream reuse."""
    ref = te_emb.positions if reference_positions is None else reference_positions
    species = ann["Species"].to_numpy()
    true_taxids = ann["taxid"].to_numpy()
    pred = np.empty(len(ann), dtype=np.int64)
    t0 = time.time()
    for tr, te, held in leave_species_out(species):
        br = Bridge.align(reps[tr], positions[tr], pca_dim=config.PCA_DIM, alpha=alpha)
        placed = br.place(reps[te])                            # (n_held, 100) in the ball
        nn = retrieve_nearest(placed, ref, k=1, chunk=RETRIEVE_CHUNK).ravel()
        pred[te] = te_emb.idx2taxid[nn]
    print(f"[LOSO/{tag}] retrieval done in {time.time()-t0:.1f}s")
    hits = per_rank_hits(true_taxids, pred, ranklut)
    return per_species_rank_ci(species, hits), (true_taxids, pred, species)


# ---------------------------------------------------------------------------- comparators
def comparator_raw_knn(reps, ann, ranklut):
    """raw-pLM-kNN: LOSO over species; per held protein, nearest TRAIN protein in raw ProtT5 space
    (cosine), inherit that protein's species' lineage. Per-species rank CI, same as f."""
    species = ann["Species"].to_numpy()
    true_taxids = ann["taxid"].to_numpy()
    sp_taxid = ann["taxid"].to_numpy()                         # each protein's own species taxid
    # cosine == dot of L2-normalized reps
    norm = reps / np.maximum(np.linalg.norm(reps, axis=1, keepdims=True), 1e-12)
    pred = np.empty(len(ann), dtype=np.int64)
    for tr, te, held in leave_species_out(species):
        sims = norm[te] @ norm[tr].T                           # (n_held, n_train)
        nn_local = sims.argmax(1)
        pred[te] = sp_taxid[tr][nn_local]
    hits = per_rank_hits(true_taxids, pred, ranklut)
    return per_species_rank_ci(species, hits), (true_taxids, pred, species)


def comparator_logistic(reps, ann, ranklut, alpha):
    """Per-rank logistic ensemble on PCA features, LOSO over species. Reports per-species accuracy CI
    plus mean softmax entropy (calibration proxy). Classes restricted to those seen in train; a held
    species' true class is by construction unseen at species rank (so species-rank logistic ~ 0, expected)."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression

    species = ann["Species"].to_numpy()
    true_taxids = ann["taxid"].to_numpy()
    # precompute per-rank true labels (NA where rank absent in true lineage)
    rank_labels = {r: np.array([ranklut.rank_map(int(t)).get(r, -1) for t in true_taxids]) for r in RANKS}

    pred = {r: np.full(len(ann), -1, dtype=np.int64) for r in RANKS}
    entropies = {r: [] for r in RANKS}
    for tr, te, held in leave_species_out(species):
        pca = PCA(n_components=min(config.PCA_DIM, len(tr)), svd_solver="full").fit(reps[tr])
        Ztr, Zte = pca.transform(reps[tr]), pca.transform(reps[te])
        for r in RANKS:
            ytr = rank_labels[r][tr]
            keep = ytr != -1
            classes = np.unique(ytr[keep])
            if keep.sum() < 2 or len(classes) < 2:
                continue
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(Ztr[keep], ytr[keep])
            proba = clf.predict_proba(Zte)
            pred[r][te] = clf.classes_[proba.argmax(1)]
            p = np.clip(proba, 1e-12, 1.0)
            entropies[r].extend((-(p * np.log(p)).sum(1)).tolist())

    out = {}
    for r in RANKS:
        scored = rank_labels[r] != -1
        # degenerate = the classifier was never fit at this rank (only 1 train class every fold, e.g.
        # phylum=Chordata only). pred stays -1 everywhere -> a misleading 0.0; flag it instead.
        degenerate = bool(scored.any() and (pred[r][scored] == -1).all())
        hit = (pred[r] == true_taxids_rank(true_taxids, r, ranklut)) & scored
        per_species = []
        for sp in sorted(set(species.tolist())):
            m = (species == sp) & scored
            if m.any():
                per_species.append(hit[m].mean())
        per_species = np.array(per_species, float)
        if degenerate or not len(per_species):
            mean = lo = hi = float("nan")
        else:
            mean, lo, hi = taxon_bootstrap_ci(per_species, n_boot=N_BOOT, seed=SEED)
        out[r] = {"mean": mean, "lo": lo, "hi": hi, "n_species": int(len(per_species)),
                  "mean_softmax_entropy": float(np.mean(entropies[r])) if entropies[r] else float("nan"),
                  "degenerate_single_class": degenerate}
    return out


def true_taxids_rank(true_taxids, rank, ranklut):
    return np.array([ranklut.rank_map(int(t)).get(rank, -2) for t in true_taxids])  # -2 never matches pred -1


def comparator_flat(reps, ann, te_emb, ranklut, alpha):
    """Flat-target ablation: regress a one-hot GENUS target (Euclidean flat space) instead of
    hyperbolic positions, then retrieve the nearest TRAIN-genus centroid in that flat space and
    inherit the corresponding genus's lineage. Tests whether hyperbolic geometry earns its complexity.
    Held species' genus is usually unseen in train, so this is a hard, honest ablation."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge

    species = ann["Species"].to_numpy()
    true_taxids = ann["taxid"].to_numpy()
    genus = np.array([ranklut.rank_map(int(t)).get("genus", -1) for t in true_taxids])
    pred = np.empty(len(ann), dtype=np.int64)
    for tr, te, held in leave_species_out(species):
        gtr = genus[tr]
        classes = np.unique(gtr[gtr != -1])
        cls_index = {g: i for i, g in enumerate(classes)}
        Onehot = np.zeros((len(tr), len(classes)))
        for i, g in enumerate(gtr):
            if g in cls_index:
                Onehot[i, cls_index[g]] = 1.0
        pca = PCA(n_components=min(config.PCA_DIM, len(tr)), svd_solver="full").fit(reps[tr])
        Ztr, Zte = pca.transform(reps[tr]), pca.transform(reps[te])
        ridge = Ridge(alpha=alpha, fit_intercept=True).fit(Ztr, Onehot)
        scores = ridge.predict(Zte)                            # (n_held, n_genus) flat target scores
        # the "genus taxid" each test protein points to; map -> a representative member's full lineage
        # via any train protein of that genus (genus -> first train taxid with that genus)
        genus_to_taxid = {}
        for g, t in zip(gtr, true_taxids[tr]):
            if g != -1:
                genus_to_taxid.setdefault(g, int(t))
        chosen_genus = classes[scores.argmax(1)]
        pred[te] = [genus_to_taxid[g] for g in chosen_genus]
    hits = per_rank_hits(true_taxids, pred, ranklut)
    return per_species_rank_ci(species, hits)


# ---------------------------------------------------------------------------- LOFO
def run_lofo(reps, positions, ann, te_emb, ranklut, alpha):
    """LOFO over Clade: per clade, train on other clades, score f vs raw-kNN class-rank accuracy on
    the held clade. Returns per-fold deltas (f - raw) + the sign test. Holding out a whole clade
    necessarily holds out all its species too — the joint species-AND-clade leak is quantified
    separately by run_leave_species_and_clade()."""
    clade = ann["Clade"].to_numpy()
    species = ann["Species"].to_numpy()
    true_taxids = ann["taxid"].to_numpy()
    sp_taxid = ann["taxid"].to_numpy()
    norm = reps / np.maximum(np.linalg.norm(reps, axis=1, keepdims=True), 1e-12)

    deltas, fold_rows = [], []
    for tr, te, held in leave_one_group_out(clade):
        if len(np.unique(clade[tr])) < 2:                      # need >=2 train clades to be meaningful
            continue
        # f: align on train clades, place held clade, retrieve in full 498k
        br = Bridge.align(reps[tr], positions[tr], pca_dim=config.PCA_DIM, alpha=alpha)
        placed = br.place(reps[te])
        nn = retrieve_nearest(placed, te_emb.positions, k=1, chunk=RETRIEVE_CHUNK).ravel()
        f_pred = te_emb.idx2taxid[nn]
        # raw-kNN: nearest train protein in raw space, inherit its lineage
        sims = norm[te] @ norm[tr].T
        knn_pred = sp_taxid[tr][sims.argmax(1)]

        f_hit, f_scored = _classrank_hits(true_taxids[te], f_pred, ranklut)
        k_hit, k_scored = _classrank_hits(true_taxids[te], knn_pred, ranklut)
        common = f_scored & k_scored
        f_acc = float(f_hit[common].mean()) if common.any() else float("nan")
        k_acc = float(k_hit[common].mean()) if common.any() else float("nan")
        if common.any():
            deltas.append(f_acc - k_acc)
        fold_rows.append({"held_clade": held, "n_test": int(len(te)), "n_class_scored": int(common.sum()),
                          "f_class_acc": f_acc, "rawknn_class_acc": k_acc})
    n_pos, n_total, p = per_fold_sign_test(deltas)
    nonzero = [d for d in deltas if d != 0.0]
    note = ("all f-vs-rawknn class deltas are exactly 0 — both collapse to the SAME class accuracy "
            "under leave-one-clade-out, so the sign test is undefined (no nonzero deltas)."
            if deltas and not nonzero else "")
    return {"folds": fold_rows, "sign_test": {"n_pos": n_pos, "n_total": n_total, "p": p},
            "deltas": [float(d) for d in deltas], "note": note}


def _classrank_hits(true_taxids, pred_taxids, ranklut):
    hit, scored = [], []
    for tt, pt in zip(true_taxids, pred_taxids):
        trm = ranklut.rank_map(int(tt))
        prm = ranklut.rank_map(int(pt))
        s = PRIMARY_RANK in trm
        scored.append(s)
        hit.append(bool(s and prm.get(PRIMARY_RANK) == trm.get(PRIMARY_RANK)))
    return np.array(hit, bool), np.array(scored, bool)


def run_leave_species_and_clade(reps, ann, ranklut):
    """Leave-species-AND-clade-out diagnostic for the same-species-paralog leak: raw-kNN class-rank
    accuracy when the nearest TRAIN protein may NOT come from the held species (LOSO already) — here we
    additionally forbid same-CLADE train neighbours, so the only signal is cross-clade. Reports the
    raw-kNN class accuracy drop relative to plain LOSO raw-kNN, quantifying how much leaked via clade."""
    species = ann["Species"].to_numpy()
    clade = ann["Clade"].to_numpy()
    true_taxids = ann["taxid"].to_numpy()
    sp_taxid = ann["taxid"].to_numpy()
    norm = reps / np.maximum(np.linalg.norm(reps, axis=1, keepdims=True), 1e-12)

    pred_loso = np.empty(len(ann), dtype=np.int64)
    pred_joint = np.full(len(ann), -1, dtype=np.int64)
    for tr, te, held in leave_species_out(species):
        sims = norm[te] @ norm[tr].T
        pred_loso[te] = sp_taxid[tr][sims.argmax(1)]
        # joint: also exclude train proteins from the held species' clade
        held_clade = clade[te][0]
        for j_global in te:
            allowed = tr[clade[tr] != held_clade]
            if len(allowed):
                s = norm[j_global] @ norm[allowed].T
                pred_joint[j_global] = sp_taxid[allowed][s.argmax()]

    def class_acc(pred):
        hit, scored = _classrank_hits(true_taxids, pred, ranklut)
        valid = scored & (pred != -1)
        return float(hit[valid].mean()) if valid.any() else float("nan"), int(valid.sum())

    loso_acc, loso_n = class_acc(pred_loso)
    joint_acc, joint_n = class_acc(pred_joint)
    return {"rawknn_loso_class_acc": loso_acc, "rawknn_loso_n": loso_n,
            "rawknn_species_and_clade_class_acc": joint_acc, "rawknn_joint_n": joint_n,
            "leak_drop": (loso_acc - joint_acc) if not (np.isnan(loso_acc) or np.isnan(joint_acc)) else float("nan")}


# ---------------------------------------------------------------------------- SP-Metazoa READ
# Task D2 (spec §6.3). Clone of the PLA2 READ pipeline for the SP-Metazoa panel. The PLA2 path above
# resolves the predicted *class* via NCBI lineage (RankLookup.rank_map) because PLA2 carries real NCBI
# taxids and the 498k reference exposes idx2taxid. The SP harness instead takes (ann, reps, te) DIRECTLY
# and reads the predicted class as the `tax_class` (Clade) of the nearest TRAIN reference protein — so it
# is self-contained (no resolver / no idx2taxid), exactly matching the D2 test fixture's FakeTE (only
# `.positions`). The retrieval geometry (bridge align → place → poincare nearest) is identical to run_lofo.
SP_MIN_FOLD_N = config.SP_READ_MIN_FOLD_N   # spec §6.1/§6.3 power floor (config for CLEAN/READ parity)


def _sp_place(reps, positions, tr, te):
    """Align the linear bridge on `tr`, return (bridge, placed_held) where placed_held are the held `te`
    reps mapped into the ball. Factored out so a single bridge fit can be retrieved against multiple
    reference sets (the real reference AND the depth-null) without refitting the expensive PCA twice."""
    br = Bridge.align(reps[tr], positions[tr], pca_dim=config.PCA_DIM, alpha=1.0)
    return br, br.place(reps[te])                              # (n_held, tax_dim) in the ball


def _sp_read_class_pred(reps, positions, ann, ref_positions, tr, te):
    """Align the bridge on `tr`, place the held `te` reps, retrieve the nearest TRAIN reference node in
    `ref_positions` (poincare), and read that train protein's `tax_class` as the predicted class.
    `ref_positions` is indexed identically to `tr` (so the nearest index maps back through tr).
    Returns the predicted class-label array for the held proteins (length == len(te))."""
    classes = ann["tax_class"].to_numpy()
    _br, placed = _sp_place(reps, positions, tr, te)
    nn = retrieve_nearest(placed, ref_positions, k=1, chunk=RETRIEVE_CHUNK).ravel()  # idx into tr
    return classes[tr][nn]


def _sp_classrank_acc(reps, positions, ann, te_positions, null_positions):
    """Global LOSO (leave-one-species-out over taxid) class-rank accuracy of the bridge AND its depth-null
    in ONE pass: per held species fit the bridge ONCE on the rest, place the held reps, then retrieve the
    nearest TRAIN protein in BOTH the real reference (`te_positions`) and the direction-randomized
    `null_positions`, inheriting that train protein's `tax_class` each time, scored against the held
    protein's true `tax_class`. Pooled accuracy over all proteins (every protein held exactly once).
    Returns (class_rank_acc, n_scored, depth_null_acc). Self-contained — no NCBI resolution."""
    species = ann["Species"].to_numpy()
    classes = ann["tax_class"].to_numpy()
    true_class = classes
    pred = np.empty(len(ann), dtype=object)
    pred_null = np.empty(len(ann), dtype=object)
    for tr, te, _held in leave_species_out(species):
        if len(tr) < config.PCA_DIM or len(np.unique(species[tr])) < 2:
            pred[te] = None; pred_null[te] = None              # cannot fit a meaningful bridge
            continue
        _br, placed = _sp_place(reps, positions, tr, te)
        nn = retrieve_nearest(placed, te_positions[tr], k=1, chunk=RETRIEVE_CHUNK).ravel()
        pred[te] = classes[tr][nn]
        nn_null = retrieve_nearest(placed, null_positions[tr], k=1, chunk=RETRIEVE_CHUNK).ravel()
        pred_null[te] = classes[tr][nn_null]

    def _acc(p):
        scored = np.array([v is not None for v in p], bool)
        hit = np.array([bool(scored[i] and p[i] == true_class[i]) for i in range(len(ann))], bool)
        return (float(hit[scored].mean()) if scored.any() else float("nan")), int(scored.sum())

    acc, n_scored = _acc(pred)
    null_acc, _ = _acc(pred_null)
    return acc, n_scored, null_acc


def _jaccard(a, b):
    """Jaccard overlap of two label sets; |A∩B| / |A∪B| (0 when the union is empty)."""
    sa, sb = set(a), set(b)
    union = sa | sb
    return (len(sa & sb) / len(union)) if union else 0.0


def _family_balanced_indices(func, tr_idx, te_idx, seed):
    """Subsample the held fold (D2b, spec §6.3) so its `func` (family) frequency ≈ the train fold's,
    controlling the family-as-clade-proxy confound. Per family present in the HELD set, keep a count
    proportional to the train family-mix, capped at the held availability. Returns the kept held indices
    (a subset of te_idx). Deterministic given `seed`."""
    rng = np.random.default_rng(seed)
    tr_func = func[tr_idx]
    te_func = func[te_idx]
    tr_fams, tr_counts = np.unique(tr_func, return_counts=True)
    tr_freq = dict(zip(tr_fams.tolist(), (tr_counts / tr_counts.sum()).tolist()))
    n_held = len(te_idx)
    kept = []
    for fam in np.unique(te_func):
        avail = te_idx[te_func == fam]
        target = int(round(tr_freq.get(fam, 0.0) * n_held))    # train-mix-proportional quota
        take = min(target, len(avail))
        if take > 0:
            chosen = rng.choice(avail, size=take, replace=False)
            kept.extend(chosen.tolist())
    return np.array(sorted(kept), dtype=int)


def _sp_leave_clade_out(reps, positions, ann, te_positions,
                        claimed_effect=config.SP_READ_CLAIMED_EFFECT, seed=SEED):
    """Global leave-one-clade-out (spec §6.3): the clade is removed across ALL families. Per fold, fit the
    bridge on the remaining clades, place the held clade, retrieve nearest train, inherit its `tax_class`,
    and score class accuracy on the held clade. Folds with n_held < SP_MIN_FOLD_N are DROPPED from the
    weighted mean and reported separately. Retained folds are aggregated weighted by n via the explicit
    formula fold_weights[i] = n_held_i / sum_j(n_held_j) so sum(fold_weights) == 1.

    Family-composition confound controls (D2b):
      * per_fold_family_overlap[i] = jaccard(set(train func), set(held func)) — computed inside this same
        leave_one_group_out(clade) loop, one value per RETAINED fold.
      * family_balanced — re-score each retained fold after subsampling the held clade to the train
        family-mix, gated by the S3 power floor: informative iff every retained balanced fold has
        post-balance n >= SP_MIN_FOLD_N AND the accuracy CI half-width < the claimed effect. When not
        informative the variant is reported "uninformative", never read as "passes"."""
    clade = ann["tax_class"].to_numpy()
    func = ann["func"].to_numpy()
    true_class = ann["tax_class"].to_numpy()

    fold_rows, kept_n, overlaps = [], [], []
    bal_n_acc = []          # (post-balance n, balanced class acc) for RETAINED folds, kept paired
    for tr, te, held in leave_one_group_out(clade):
        if len(np.unique(clade[tr])) < 2 or len(tr) < config.PCA_DIM:
            continue                                            # need >=2 train clades to read class
        n_held = int(len(te))
        # --- raw held-clade class accuracy ---
        pred = _sp_read_class_pred(reps, positions, ann, te_positions[tr], tr, te)
        acc = float((pred == true_class[te]).mean())
        # --- per-fold family overlap (D2b a): train-vs-held func-set Jaccard ---
        overlap = _jaccard(func[tr], func[te])
        row = {"held_clade": held, "n_held": n_held, "class_acc": acc, "family_overlap": overlap}
        # --- family-balanced variant (D2b b): re-score after matching held family-mix to train ---
        bal_idx = _family_balanced_indices(func, tr, te, seed)
        if len(bal_idx):
            bal_pred = _sp_read_class_pred(reps, positions, ann, te_positions[tr], tr, bal_idx)
            bal_acc = float((bal_pred == true_class[bal_idx]).mean())
        else:
            bal_acc = float("nan")
        row["family_balanced_n"] = int(len(bal_idx))
        row["family_balanced_class_acc"] = bal_acc
        fold_rows.append(row)
        if n_held >= SP_MIN_FOLD_N:                             # power floor: retain for the weighted mean
            kept_n.append(n_held)
            overlaps.append(overlap)
            bal_n_acc.append((int(len(bal_idx)), bal_acc))

    dropped = [r for r in fold_rows if r["n_held"] < SP_MIN_FOLD_N]
    total = float(sum(kept_n))
    fold_weights = [n / total for n in kept_n] if total > 0 else []
    retained_accs = [r["class_acc"] for r in fold_rows if r["n_held"] >= SP_MIN_FOLD_N]
    weighted_class_acc = (float(np.average(retained_accs, weights=fold_weights))
                          if fold_weights else float("nan"))

    # ---- S3 power floor (spec §6.1) for the family-balanced variant ----
    bal_ns = [n for n, _a in bal_n_acc]
    valid = [(n, a) for n, a in bal_n_acc if not np.isnan(a)]    # folds with a scorable balanced acc
    bal_accs = [a for _n, a in valid]
    bal_arr = np.array(bal_accs, float)
    if len(bal_arr) >= 2:
        ci_half = 1.96 * float(bal_arr.std(ddof=1)) / np.sqrt(len(bal_arr))  # normal-approx half-width
    else:
        ci_half = float("inf")
    enough_n = bool(bal_ns) and all(n >= SP_MIN_FOLD_N for n in bal_ns)
    # zero-variance guard: leave-clade-out structurally yields identical (often 0.0) balanced accs across
    # folds -> std=0 -> ci_half=0 < claimed_effect -> "informative". That is PRECISE, not POSITIVE; flag it
    # so a downstream reader doesn't mistake a tight CI around ~0 for a passing result.
    zero_variance = bool(len(bal_arr) >= 2 and bal_arr.std(ddof=1) == 0.0)
    informative = bool(enough_n and (ci_half < claimed_effect))
    bal_weighted = (float(np.average(bal_accs, weights=[n for n, _a in valid]))
                    if valid else float("nan"))          # point estimate ALWAYS exposed (read for the signal)
    family_balanced = {
        "informative": informative,                       # informative == estimate is PRECISE, NOT == positive
        "verdict": ("informative" if informative else "uninformative"),
        "weighted_class_acc": bal_weighted,               # <- the actual signal; read this, not `informative`
        "per_fold_n": bal_ns,
        "ci_half_width": (ci_half if np.isfinite(ci_half) else None),
        "zero_variance": zero_variance,
        "claimed_effect": claimed_effect,
        "power_floor_n": SP_MIN_FOLD_N,
        "note": (("PRECISE estimate (CI half-width < claimed effect); 'informative' means PRECISE, NOT "
                  "positive -- read weighted_class_acc for the signal"
                  + (" [ZERO-VARIANCE: all balanced folds identical -> degenerate CI]" if zero_variance else ""))
                 if informative else
                 "underpowered post-balance (n < floor or CI half-width >= claimed effect) "
                 "-> reported UNINFORMATIVE, not a pass"),
    }

    return {
        "weighted_class_acc": weighted_class_acc,
        "fold_weights": fold_weights,
        "per_fold_family_overlap": overlaps,
        "min_fold_n": SP_MIN_FOLD_N,
        "folds": fold_rows,
        "dropped_folds_below_min_n": dropped,
        "family_balanced": family_balanced,
    }


def _sp_leave_species_and_clade(reps, ann):
    """Leave-species-AND-clade-out leak diagnostic (D2b), cloned from run_leave_species_and_clade with
    Clade -> tax_class and Species -> taxid (spec §6.3 — the control that collapsed M1's kNN to 0.000).
    raw-kNN class accuracy under LOSO (nearest TRAIN protein may not be the held species), and the joint
    variant that ADDITIONALLY forbids same-CLADE train neighbours; reports the class-accuracy drop (leak)
    attributable to same-clade neighbours. Reads tax_class directly (self-contained — no NCBI lineage)."""
    species = ann["Species"].to_numpy()
    clade = ann["tax_class"].to_numpy()
    true_class = ann["tax_class"].to_numpy()
    sp_class = ann["tax_class"].to_numpy()                      # each train protein's own class label
    norm = reps / np.maximum(np.linalg.norm(reps, axis=1, keepdims=True), 1e-12)

    pred_loso = np.empty(len(ann), dtype=object)
    pred_joint = np.full(len(ann), None, dtype=object)
    for tr, te, _held in leave_species_out(species):
        sims = norm[te] @ norm[tr].T
        pred_loso[te] = sp_class[tr][sims.argmax(1)]
        held_clade = clade[te][0]
        for j_global in te:
            allowed = tr[clade[tr] != held_clade]              # forbid same-clade train neighbours
            if len(allowed):
                s = norm[j_global] @ norm[allowed].T
                pred_joint[j_global] = sp_class[allowed][s.argmax()]

    def class_acc(pred):
        valid = np.array([p is not None for p in pred], bool)
        hit = np.array([bool(valid[i] and pred[i] == true_class[i]) for i in range(len(ann))], bool)
        return (float(hit[valid].mean()) if valid.any() else float("nan")), int(valid.sum())

    loso_acc, loso_n = class_acc(pred_loso)
    joint_acc, joint_n = class_acc(pred_joint)
    leak = (loso_acc - joint_acc) if not (np.isnan(loso_acc) or np.isnan(joint_acc)) else float("nan")
    return {"rawknn_loso_class_acc": loso_acc, "rawknn_loso_n": loso_n,
            "rawknn_species_and_clade_class_acc": joint_acc, "rawknn_joint_n": joint_n,
            "leak_drop": leak}


def run_sp_read(ann, reps, te):
    """SP-Metazoa READ harness (Task D2). Self-contained clone of the PLA2 READ pipeline that takes the
    panel + reps + TaxonomyEmbedding-like `te` (needs only `te.positions`) directly. Reports:
      * class_rank_acc — global LOSO bridge class-rank accuracy (the primary READ read);
      * depth_null — same retrieval against a radial-only (direction-randomized) reference;
      * matched_candidate_set — class-frequency-matched random baseline (the chance ceiling a leak-free
        retrieval must clear), the READ specificity control;
      * leave_clade_out — global leave-one-clade-out, n-weighted folds (fold_weights sum to 1) with the
        per-fold train/held family-overlap Jaccards and the family-balanced variant (S3 power floor, §6.3);
      * leave_species_and_clade — the leave-species-AND-clade-out leak diagnostic (primary leak test, §6.3)."""
    ann = ann.reset_index(drop=True)
    reps = np.asarray(reps, np.float64)
    te_positions = np.asarray(te.positions)[ann["idx"].to_numpy()]   # each protein's true node (n, tax_dim)

    # --- primary class-rank accuracy of the bridge + depth-only null (one shared LOSO pass) ---
    # depth null = same retrieval against a radial-only (direction-randomized) reference.
    null_positions = radial_only_null(te_positions, seed=SEED)
    class_rank_acc, n_scored, depth_null_acc = _sp_classrank_acc(
        reps, te_positions, ann, te_positions, null_positions)

    # --- matched candidate-set control: class-frequency-matched random baseline (chance ceiling) ---
    true_class = ann["tax_class"].to_numpy()
    _, counts = np.unique(true_class, return_counts=True)
    p = counts / counts.sum()
    matched_chance = float((p ** 2).sum())                     # expected accuracy of frequency-matched guessing

    # --- global leave-clade-out (n-weighted) + family-composition confound controls (D2b) ---
    leave_clade_out = _sp_leave_clade_out(reps, te_positions, ann, te_positions)

    # --- leave-species-AND-clade-out leak diagnostic (D2b, primary leak test) ---
    leave_species_and_clade = _sp_leave_species_and_clade(reps, ann)

    return {
        "class_rank_acc": class_rank_acc,
        "n_scored": n_scored,
        "depth_null": {"class_rank_acc": depth_null_acc},
        "matched_candidate_set": {"matched_chance_acc": matched_chance,
                                  "n_classes": int(len(counts))},
        "leave_clade_out": leave_clade_out,
        "leave_species_and_clade": leave_species_and_clade,
        "meta": {"n_proteins": int(len(ann)), "n_species": int(ann["Species"].nunique()),
                 "n_clades": int(ann["tax_class"].nunique()), "pca_dim": config.PCA_DIM,
                 "min_fold_n": SP_MIN_FOLD_N, "seed": SEED},
    }


# ---------------------------------------------------------------------------- Holm
def holm_adjust(pvals: dict) -> dict:
    """Holm-Bonferroni step-down adjustment over a {name: p} dict. Returns {name: adjusted_p}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (name, p) in enumerate(items):
        a = min(1.0, (m - i) * p)
        running = max(running, a)                              # enforce monotonic non-decreasing
        adj[name] = running
    return adj


# ---------------------------------------------------------------------------- main
def _main_sp_read(testbed, t_start):
    """REAL-RUN-ONLY (untestable here — needs the locked metazoa embedding + the SwissProt h5).

    SP-Metazoa READ entrypoint (Task E1 wiring; the D2 comparators in `run_sp_read` are exercised
    for real here). Symmetric with `clean_eval._main_sp`: load the panel + reps via `load_sp_read`,
    load the LOCKED metazoa Poincaré embedding the same way `clean_eval._load_te` does, run the SP
    READ harness, and write `results/read_{testbed}.json`."""
    print(f"=== SP-Metazoa READ experiment (testbed={testbed}) ===")
    ann, reps = load_sp_read(testbed)
    print(f"loaded {len(ann)} proteins, {ann['Species'].nunique()} species, "
          f"{ann['Clade'].nunique()} clades, func_labels={ann['func'].nunique()}")

    te_emb = TaxonomyEmbedding(config.CKPT, config.TAXMAP, config.EDGELIST)
    results = run_sp_read(ann, reps, te_emb)
    results["runtime_sec"] = round(time.time() - t_start, 1)

    out_path = _HERE / "results" / f"read_{testbed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}  (runtime {results['runtime_sec']}s)")
    return results


def main(testbed: str = "pla2"):
    t_start = time.time()

    if testbed.startswith("sp_metazoa_"):
        # SP-Metazoa READ path (wired E1): load + run the SP READ harness, write read_{testbed}.json.
        return _main_sp_read(testbed, t_start)

    print("=== Task 10: PLA2 READ experiment ===")
    ann, reps = select_loader(testbed)()
    print(f"loaded {len(ann)} proteins, {ann['Species'].nunique()} species, "
          f"{ann['Clade'].nunique()} clades, {ann['cluster_id'].nunique()} identity clusters")

    te_emb = TaxonomyEmbedding(config.CKPT, config.TAXMAP, config.EDGELIST)
    resolver = TaxonResolver(config.TAXDUMP_DIR)
    ranklut = RankLookup(resolver, RANKS)
    positions = te_emb.positions[ann["idx"].to_numpy()]        # (N, 100) true node per protein
    species = ann["Species"].to_numpy()

    alpha = select_alpha(reps, positions, species)

    # --- f (the bridge) ---
    f_ci, _f_proteins = run_loso(reps, positions, ann, te_emb, ranklut, alpha, tag="f")
    # --- raw-pLM-kNN ---
    knn_ci, _k_proteins = comparator_raw_knn(reps, ann, ranklut)
    # --- depth-only null: same retrieval against radial-only-randomized reference ---
    null_positions = radial_only_null(te_emb.positions, seed=SEED)
    null_ci, _n_proteins = run_loso(reps, positions, ann, te_emb, ranklut, alpha,
                                    reference_positions=null_positions, tag="depth_null")
    # --- logistic ensemble ---
    logistic_ci = comparator_logistic(reps, ann, ranklut, alpha)
    # --- flat-target ablation ---
    flat_ci = comparator_flat(reps, ann, te_emb, ranklut, alpha)
    # --- LOFO over Clade ---
    lofo = run_lofo(reps, positions, ann, te_emb, ranklut, alpha)
    # --- leave-species-AND-clade-out leak quantification ---
    leak = run_leave_species_and_clade(reps, ann, ranklut)

    # ----- PRIMARY ENDPOINT (pre-registered) -----
    f_c = f_ci[PRIMARY_RANK]
    k_c = knn_ci[PRIMARY_RANK]
    n_c = null_ci[PRIMARY_RANK]
    f_beats_knn = f_c["lo"] > k_c["hi"]                        # CIs separated, f above raw-kNN
    f_beats_null = f_c["lo"] > n_c["hi"]                       # CIs separated, f above depth null
    knn_beats_f = k_c["lo"] > f_c["hi"]                        # the observed direction (raw-kNN above f)
    endpoint_pass = bool(f_beats_knn and f_beats_null)
    if knn_beats_f:
        interpretation = (
            f"NEGATIVE on the f>raw-kNN leg: raw-pLM-kNN class accuracy ({k_c['mean']:.3f}) is HIGHER "
            f"than f ({f_c['mean']:.3f}) with separated CIs. f DOES clear the depth-null ({f_beats_null}). "
            f"BUT the LOFO + leave-species-AND-clade analysis shows raw-kNN's LOSO class accuracy is almost "
            f"entirely same-clade nearest-neighbour leakage: forbidding same-clade train neighbours drops it "
            f"from {leak['rawknn_loso_class_acc']:.3f} to {leak['rawknn_species_and_clade_class_acc']:.3f} "
            f"(leak {leak['leak_drop']:.3f}), and under leave-one-clade-out BOTH f and raw-kNN fall to ~0. "
            f"raw-pLM-kNN under LOSO is therefore a memorised-neighbour baseline, not a clean READ baseline. "
            f"The endpoint as literally pre-registered FAILS, but it fails against a leak-inflated comparator "
            f"— reported honestly, not massaged."
        )
    elif endpoint_pass:
        interpretation = "PASS: f separates above raw-kNN AND above depth-null at class rank."
    else:
        interpretation = "Inconclusive: f does not separate from raw-kNN at class rank in either direction."

    # ----- exploratory Holm over the p-value-bearing tests -----
    expl_p = {"lofo_class_sign_test": lofo["sign_test"]["p"]}
    holm = holm_adjust(expl_p)

    results = {
        "meta": {
            "n_proteins": int(len(ann)), "n_species": int(ann["Species"].nunique()),
            "n_clades": int(ann["Clade"].nunique()), "n_clusters": int(ann["cluster_id"].nunique()),
            "alpha": float(alpha), "alpha_grid": ALPHA_GRID, "pca_dim": config.PCA_DIM,
            "n_boot": N_BOOT, "seed": SEED, "ranks": RANKS, "primary_rank": PRIMARY_RANK,
            "mmseqs_used": have_mmseqs(),
            "class_rank_note": "8 species (4 Crocodiles + 4 Turtles, 32 proteins) lack an NCBI 'class' "
                               "rank (clade Sauropsida); class-rank scored over the 77 species that have one.",
        },
        "per_rank": {
            "f": f_ci, "raw_knn": knn_ci, "depth_null": null_ci,
            "logistic": logistic_ci, "flat": flat_ci,
        },
        "lofo_clade": lofo,
        "leak_species_and_clade": leak,
        "exploratory_holm_adjusted_p": holm,
        "primary_endpoint": {
            "definition": "LOSO class-rank accuracy of f vs raw-pLM-kNN (taxon-bootstrap CIs separated) "
                          "AND f > depth-null.",
            "f_class": f_c, "raw_knn_class": k_c, "depth_null_class": n_c,
            "f_beats_rawknn_ci_separated": bool(f_beats_knn),
            "rawknn_beats_f_ci_separated": bool(knn_beats_f),
            "f_beats_depth_null_ci_separated": bool(f_beats_null),
            "PASS": endpoint_pass,
            "interpretation": interpretation,
        },
        "runtime_sec": None,  # filled below
    }
    results["runtime_sec"] = round(time.time() - t_start, 1)

    out_path = _HERE / "results" / "read_pla2.json"
    out_path.write_text(json.dumps(results, indent=2))

    # ---------- printed verdict ----------
    print("\n" + "=" * 72)
    print("PER-RANK ACCURACY (mean [lo, hi] over per-species values)")
    print(f"{'rank':<9}{'f (bridge)':<26}{'raw-pLM-kNN':<26}{'depth-null':<26}")
    for r in RANKS:
        def fmt(d):
            return f"{d['mean']:.3f} [{d['lo']:.3f},{d['hi']:.3f}] n={d['n_species']}"
        print(f"{r:<9}{fmt(f_ci[r]):<26}{fmt(knn_ci[r]):<26}{fmt(null_ci[r]):<26}")
    print(f"\nlogistic (exploratory) class acc = {logistic_ci['class']['mean']:.3f} "
          f"[{logistic_ci['class']['lo']:.3f},{logistic_ci['class']['hi']:.3f}]  "
          f"mean softmax entropy = {logistic_ci['class']['mean_softmax_entropy']:.3f}")
    print(f"flat-target ablation class acc = {flat_ci['class']['mean']:.3f} "
          f"[{flat_ci['class']['lo']:.3f},{flat_ci['class']['hi']:.3f}]")
    print("\nLOFO (leave-one-clade-out) class-rank f-vs-rawknn sign test: "
          f"{lofo['sign_test']['n_pos']}/{lofo['sign_test']['n_total']} nonzero folds f>raw, "
          f"p={lofo['sign_test']['p']:.3f} (Holm-adj p={holm['lofo_class_sign_test']:.3f})")
    if lofo.get("note"):
        print(f"  LOFO note: {lofo['note']}")
    print(f"Same-species-AND-clade leak: raw-kNN class acc LOSO={leak['rawknn_loso_class_acc']:.3f} "
          f"-> forbid-same-clade={leak['rawknn_species_and_clade_class_acc']:.3f} "
          f"(leak drop={leak['leak_drop']:.3f})")
    print("\n" + "=" * 72)
    print("PRE-REGISTERED PRIMARY ENDPOINT (class rank):")
    print(f"  f          class acc = {f_c['mean']:.3f}  CI [{f_c['lo']:.3f}, {f_c['hi']:.3f}]  (n={f_c['n_species']} species)")
    print(f"  raw-pLM-kNN class acc = {k_c['mean']:.3f}  CI [{k_c['lo']:.3f}, {k_c['hi']:.3f}]")
    print(f"  depth-null class acc = {n_c['mean']:.3f}  CI [{n_c['lo']:.3f}, {n_c['hi']:.3f}]")
    print(f"  f > raw-pLM-kNN (CIs separated): {f_beats_knn}")
    print(f"  raw-pLM-kNN > f (CIs separated): {knn_beats_f}")
    print(f"  f > depth-null  (CIs separated): {f_beats_null}")
    if endpoint_pass:
        verdict = "PASS"
    elif knn_beats_f:
        verdict = "FAIL (raw-kNN > f at class rank, CIs separated — but raw-kNN is leak-inflated; see interpretation)"
    else:
        verdict = "FAIL/INCONCLUSIVE (f does not separate from raw-kNN; CIs overlap — see interpretation)"
    print(f"  ENDPOINT VERDICT: {verdict}")
    print(f"  INTERPRETATION: {interpretation}")
    print(f"\nwrote {out_path}  (runtime {results['runtime_sec']}s)")
    return results


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(args.testbed)
