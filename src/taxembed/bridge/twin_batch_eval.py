"""§4.4 SP-only twin + §7 SP-vs-TrEMBL batch-effect guard + §4.2.1 span floor + §4.2.4/§9 retention
+ #9 TrEMBL relabel bound (spec v3). Self-contained, Mac-TDD'd; the all-life battery calls these. They
depend on the `source` column the panel now carries (build_sp_panel._PANEL_COLS) — the data the §4.4
twin and §7 guard were previously unbuildable without (R-SC-2 blocker)."""
import pandas as pd


def source_subset(ann, which):
    """Row-filter the panel frame to one source ('sp'|'trembl') for the §4.4 fallback-verdict twin."""
    return ann[ann["source"] == which].reset_index(drop=True)


def batch_effect_auc(X, source_labels, seed=0, cv=3, max_n=10000):
    """§7 batch-effect guard: cross-validated AUC of an SP-vs-TrEMBL classifier on RAW embeddings.
    Near 0.5 = no batch effect (the desired outcome). NaN if only one source present. Subsamples to
    max_n (seed-fixed) so the guard stays fast at the 250k ceiling."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    y = np.asarray([1 if s == "trembl" else 0 for s in source_labels])
    X = np.asarray(X, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan")
    if len(y) > max_n:
        idx = np.random.default_rng(seed).choice(len(y), size=max_n, replace=False)
        X, y = X[idx], y[idx]
        if len(np.unique(y)) < 2:
            return float("nan")
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    return float(scores.mean())


def assert_span_denominator(superkingdoms, min_frac):
    """M-5/§4.2.1: the non-null-superkingdom span denominator must be >= min_frac of admitted proteins,
    so a kingdom-span verdict cannot be computed over a tiny populated denominator. Fail loud."""
    s = pd.Series(list(superkingdoms))
    n = len(s)
    nn = int(s.notna().sum())
    frac = (nn / n) if n else 0.0
    report = {"n": n, "n_non_null": nn, "frac": frac, "min_frac": min_frac, "ok": frac >= min_frac}
    if frac < min_frac:
        raise SystemExit(f"[span] non-null-superkingdom denominator {frac:.3f} < ALL_LIFE_SPAN_DENOM_MIN "
                         f"{min_frac} — kingdom-span verdict over a tiny denominator (spec §4.2.1)")
    return report


def per_superkingdom_retention(superkingdoms):
    """R-SC-8/§4.2.4/§10: per-superkingdom retention dict for every result-JSON denominator block."""
    s = pd.Series(list(superkingdoms))
    counts = s.value_counts(dropna=False).to_dict()
    return {(str(k) if pd.notna(k) else "null"): int(v) for k, v in counts.items()}


def annotation_completeness(counts):
    """§4.4: per-group (family or clade) single-Pfam annotation completeness. `counts` maps a group to
    {'total': n, 'single_pfam': m}; returns the fraction with exactly one Pfam signature (the TrEMBL
    '>20% carry no Pfam' caveat is read off these fractions)."""
    out = {}
    for g, c in counts.items():
        total = c.get("total", 0)
        out[g] = (c.get("single_pfam", 0) / total) if total else 0.0
    return out


def trembl_relabel_check(per_sk_null_frac, bound):
    """#9/§4.2.4: if a superkingdom's null-rank drop exceeds `bound`, the deliverable is RELABELLED to
    the retained domains (a reporting action, pre-stated NOW — not a post-hoc judgement)."""
    excluded = sorted(sk for sk, f in per_sk_null_frac.items() if f > bound)
    retained = sorted(sk for sk in per_sk_null_frac if sk not in excluded)
    return {"relabel": bool(excluded), "excluded_domains": excluded,
            "retained_domains": retained, "bound": bound}
