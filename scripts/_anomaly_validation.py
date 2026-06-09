"""Application #2 validation legs (spec §5#2, §9B):
  roc          — leg A: synthetic ROC stratified by phylogenetic displacement (AUC curve, not a number)
  releasediff  — leg B: NCBI release-diff odds ratio with matched background (the headline)
  enrichment   — leg C: incertae-sedis/environmental enrichment

Local analysis only: explicit --checkpoint (final/ep200) + --mapping; never rely on run.json.
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT / "src"))

from analyze_hierarchy_hyperbolic import (load_embeddings, load_mapping,
                                          load_taxonomy_with_depth, get_ancestor_at_rank)
from knn_purity_hyperbolic import _prep_sqnorms, _batch_distances, chance_purity, build_pool
from taxembed.eval.treedist import TreeDistance
from taxembed.eval.anomaly import (matched_null_z, trivial_baselines, baseline_aucs,
                                   relocate_nodes, enrichment_odds_ratio, match_background)
from taxembed.eval.release_diff import parse_parents, parse_merged, parse_delnodes, reclassified_taxa
from audit_taxonomy_noise import classify_name_noise, is_container, parse_names_dmp


def index_tree(idx2tax, taxonomy=None, parent_col=None):
    n = len(idx2tax)
    tax2idx = {t: i for i, t in idx2tax.items()}
    parent = np.arange(n, dtype=np.int64); depth = np.zeros(n, dtype=np.int64)
    for i in range(n):
        t = idx2tax[i]
        p = (parent_col[t] if parent_col is not None else taxonomy.get(t, {}).get("parent", t))
        parent[i] = tax2idx.get(p, i)
        if parent_col is not None:
            d, cur, seen = 0, t, set()
            while cur in parent_col and parent_col[cur] != cur and cur not in seen:
                seen.add(cur); cur = parent_col[cur]; d += 1
            depth[i] = d
        else:
            depth[i] = taxonomy.get(t, {}).get("depth", 0)
    degree = np.zeros(n, dtype=np.int64); clade_size = np.ones(n, dtype=np.int64)
    for i in range(n):
        if parent[i] != i:
            degree[parent[i]] += 1
    for i in np.argsort(-depth):
        if parent[i] != i:
            clade_size[parent[i]] += clade_size[i]
    return parent, depth, clade_size, degree


def observed_purity(emb, raw, clip, pool_idx, pool_lab, k):
    pe, pr, pc = emb[pool_idx], raw[pool_idx], clip[pool_idx]
    keff = min(k, len(pool_idx) - 1)
    out = np.empty(len(pool_idx), dtype=np.float64); B = 256
    for s in range(0, len(pool_idx), B):
        bq = np.arange(s, min(s + B, len(pool_idx)))
        d = _batch_distances(pe[bq], pr[bq], pc[bq], pe, pr, pc)
        d[np.arange(len(bq)), bq] = np.inf
        nn = np.argpartition(d, keff, axis=1)[:, :keff]
        out[s:s + len(bq)] = (pool_lab[nn] == pool_lab[bq][:, None]).mean(axis=1)
    return out


def matched_null(observed, pool_idx, depth, clade_size, n_null, n_bins, seed):
    rng = np.random.default_rng(seed)
    pd_, ps_ = depth[pool_idx], clade_size[pool_idx]
    def qb(x):
        qs = np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1]) if n_bins > 1 else np.array([])
        return np.digitize(x, qs)
    bins = qb(pd_) * (n_bins + 1) + qb(ps_)
    null = np.empty((len(pool_idx), n_null), dtype=np.float64)
    for b in np.unique(bins):
        members = np.flatnonzero(bins == b)
        for qi in members:
            null[qi] = observed[rng.choice(members, size=n_null, replace=True)]
    return null


def load_all(args):
    emb = load_embeddings(args.checkpoint).astype(np.float32)
    idx2tax = load_mapping(args.mapping)
    if args.parent_from_mapping:
        df = pd.read_csv(args.mapping, sep="\t")
        parent_col = {int(r.taxid): int(r.parent) for r in df.itertuples()}
        parent, depth, clade_size, degree = index_tree(idx2tax, parent_col=parent_col)
        taxonomy = None
    else:
        taxonomy = load_taxonomy_with_depth(set(idx2tax.values()), args.data_dir)
        parent, depth, clade_size, degree = index_tree(idx2tax, taxonomy=taxonomy)
    return emb, idx2tax, taxonomy, parent, depth, clade_size, degree


def build_labels(emb, idx2tax, taxonomy, args):
    if args.rank_from_mapping:
        df = pd.read_csv(args.mapping, sep="\t")
        lab_of = {int(r.taxid): int(getattr(r, args.rank_from_mapping)) for r in df.itertuples()}
        tax2idx = {t: i for i, t in idx2tax.items()}
        pi, pl = [], []
        for t, lab in lab_of.items():
            if lab >= 0 and t in tax2idx:
                pi.append(tax2idx[t]); pl.append(lab)
        return np.asarray(pi, np.int64), np.asarray(pl, np.int64)
    return build_pool(emb, idx2tax, taxonomy, args.rank)


def cmd_roc(args):
    emb, idx2tax, taxonomy, parent, depth, clade_size, degree = load_all(args)
    raw, clip = _prep_sqnorms(emb.astype(np.float64))
    td = TreeDistance(parent, depth)
    new_parent, moved = relocate_nodes(parent, depth, n=args.n_relocate, seed=args.seed)
    disp = td.path_length(parent[moved], new_parent[moved])
    edges = [0, 2, 4, 8, 10_000]
    disp_class = np.digitize(disp, edges[1:-1])

    pool_idx, pool_lab = build_labels(emb, idx2tax, taxonomy, args)
    obs = observed_purity(emb, raw, clip, pool_idx, pool_lab, args.k)
    null = matched_null(obs, pool_idx, depth, clade_size, args.n_null, args.n_bins, args.seed)
    score_z = matched_null_z(obs, null)
    base = trivial_baselines(emb, parent, depth, clade_size, degree)
    base_pool = {n: v[pool_idx] for n, v in base.items()}

    moved_set = set(moved.tolist())
    pool_is_moved = np.array([1 if int(i) in moved_set else 0 for i in pool_idx])
    moved_to_disp = {int(m): int(disp_class[j]) for j, m in enumerate(moved)}

    auc_by, base_by = {}, {}
    for dc in sorted(set(disp_class.tolist())):
        pos_mask = np.array([pool_is_moved[k] == 1 and moved_to_disp.get(int(pool_idx[k]), -1) == dc
                             for k in range(len(pool_idx))])
        labels = np.where(pos_mask, 1, 0)
        keep = (labels == 1) | (pool_is_moved == 0)
        if labels[keep].sum() < 1 or (labels[keep] == 0).sum() < 1:
            continue
        scores = {"score_z": score_z[keep], **{n: base_pool[n][keep] for n in base_pool}}
        all_aucs = baseline_aucs(labels[keep], scores)
        auc_by[str(dc)] = {"score_z": all_aucs["score_z"], "n_pos": int(labels[keep].sum())}
        base_by[str(dc)] = {n: all_aucs[n] for n in base_pool}

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    res = {"auc_by_displacement": auc_by, "baseline_auc_by_displacement": base_by,
           "displacement_edges": edges, "n_relocate": int(len(moved)), "k": args.k}
    (out / "roc_by_displacement.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def cmd_enrichment(args):
    data = np.load(args.pool_npz)
    pool_idx = data["pool_idx"]; score_z = data["score_z"]
    idx2tax = load_mapping(args.mapping)
    taxids = np.array([idx2tax[int(i)] for i in pool_idx], dtype=np.int64)
    names = parse_names_dmp(Path(args.names_dmp), set(taxids.tolist()))
    is_uncertain = np.array([
        bool(classify_name_noise(names.get(int(t), ""))) or
        (is_container(names.get(int(t), "")) is not None)
        for t in taxids], dtype=bool)
    res = enrichment_odds_ratio(score_z, is_uncertain, top_frac=args.top_frac)
    res["n_uncertain"] = int(is_uncertain.sum())
    res["pool_size"] = int(len(pool_idx))
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "enrichment.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def _parse_date(s):
    y, m, d = (int(x) for x in s.split("-"))
    return date(y, m, d)


def cmd_releasediff(args):
    old_d, new_d = _parse_date(args.old_date), _parse_date(args.new_date)
    dates_ok = old_d < new_d
    msg = []
    if not dates_ok:
        msg.append(f"scored old-date {old_d} must be strictly before new-date {new_d}")
    if args.training_date:
        tr = _parse_date(args.training_date)
        if tr > old_d:
            dates_ok = False
            msg.append(f"LEAKAGE: training-date {tr} is after scored old-date {old_d}")
    if not dates_ok:
        sys.stderr.write("releasediff REFUSED: " + "; ".join(msg) + "\n")
        sys.exit(2)

    data = np.load(args.pool_npz)
    pool_idx = data["pool_idx"]; score_z = data["score_z"]
    depth = data["depth"]; clade_size = data["clade_size"]; degree = data["degree"]
    idx2tax = load_mapping(args.mapping)
    scored_taxids = [int(idx2tax[int(i)]) for i in pool_idx]

    old_parent = parse_parents(args.old_nodes)
    new_parent = parse_parents(args.new_nodes)
    new_merged = parse_merged(args.new_merged) if args.new_merged else {}
    new_deln = parse_delnodes(args.new_delnodes) if args.new_delnodes else set()

    present = np.array([t in old_parent for t in scored_taxids])
    reclass = reclassified_taxa([t for t, p in zip(scored_taxids, present) if p],
                                old_parent, new_parent, new_merged, new_deln)
    is_reclass = np.array([t in reclass for t in scored_taxids]) & present

    res = enrichment_odds_ratio(score_z[present], is_reclass[present], top_frac=args.top_frac)
    flagged = np.flatnonzero((score_z >= np.quantile(score_z[present], 1 - args.top_frac)) & present)
    controls = match_background(flagged, depth, clade_size, degree, n_bins=args.n_bins, seed=0)
    res["flagged_reclass_rate"] = float(is_reclass[flagged].mean()) if len(flagged) else 0.0
    res["matched_control_reclass_rate"] = float(is_reclass[controls].mean()) if len(controls) else 0.0

    res.update({"n_reclassified": int(is_reclass[present].sum()),
                "n_scored_present_in_old": int(present.sum()),
                "training_date": args.training_date, "old_date": args.old_date,
                "new_date": args.new_date, "dates_ok": True})
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "releasediff.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def add_common(sp):
    sp.add_argument("--checkpoint", required=True)
    sp.add_argument("--mapping", required=True)
    sp.add_argument("--data-dir", default=str(ROOT / "data"))
    sp.add_argument("--parent-from-mapping", action="store_true")
    sp.add_argument("--rank", default="family")
    sp.add_argument("--rank-from-mapping", default=None)
    sp.add_argument("--k", type=int, default=10)
    sp.add_argument("--n-null", type=int, default=200)
    sp.add_argument("--n-bins", type=int, default=5)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("-o", "--output-dir", required=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    roc = sub.add_parser("roc"); add_common(roc); roc.add_argument("--n-relocate", type=int, default=2000)
    enr = sub.add_parser("enrichment")
    enr.add_argument("--pool-npz", required=True)
    enr.add_argument("--mapping", required=True)
    enr.add_argument("--names-dmp", default=str(ROOT / "data" / "names.dmp"))
    enr.add_argument("--top-frac", type=float, default=0.1)
    enr.add_argument("-o", "--output-dir", required=True)
    rd = sub.add_parser("releasediff")
    rd.add_argument("--pool-npz", required=True)
    rd.add_argument("--mapping", required=True)
    rd.add_argument("--old-nodes", required=True)
    rd.add_argument("--old-merged", default=None)
    rd.add_argument("--old-delnodes", default=None)
    rd.add_argument("--new-nodes", required=True)
    rd.add_argument("--new-merged", default=None)
    rd.add_argument("--new-delnodes", default=None)
    rd.add_argument("--training-date", default=None)
    rd.add_argument("--old-date", required=True)
    rd.add_argument("--new-date", required=True)
    rd.add_argument("--top-frac", type=float, default=0.1)
    rd.add_argument("--n-bins", type=int, default=5)
    rd.add_argument("-o", "--output-dir", required=True)
    args = ap.parse_args()
    if args.cmd == "roc":
        cmd_roc(args)
    elif args.cmd == "enrichment":
        cmd_enrichment(args)
    elif args.cmd == "releasediff":
        cmd_releasediff(args)


if __name__ == "__main__":
    main()
