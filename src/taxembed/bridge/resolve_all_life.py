"""All-life taxid->node resolution + pre-flight audits (spec v3 §4.3).

Operate on the UNIQUE taxid set (millions), never the ~250M raw rows. Emits a GO/NO-GO before the GPU
DAG. The resolution floor (config.SP_RESOLUTION_RATE_MIN) is a pre-committed ONE-SIDED gate: a
sub-floor realized rate is a NO-GO that escalates to the user (a claim-scope change, like the Step-0
HALT), NEVER lowered to rescue the run (decision #2). The redirect/subset audits are pure set
arithmetic over the resolver node set — Mac-testable, the heavy 2.5 GB resolver instantiation is the
only part that lives in JA.
"""
import json


def memoized_resolve(taxids, resolver):
    out = {}
    for t in taxids:
        t = int(t)
        if t in out:
            continue
        out[t] = resolver.idx_of_taxid(t)
    return {t: i for t, i in out.items() if i is not None}


def resolution_preflight(survivor_taxids, resolver, floor, out_json=None):
    uniq = list(dict.fromkeys(int(t) for t in survivor_taxids))
    resolved = memoized_resolve(uniq, resolver)
    rate = len(resolved) / len(uniq) if uniq else 0.0
    report = {"n_unique": len(uniq), "n_resolved": len(resolved), "rate": round(rate, 4),
              "floor": floor, "go": bool(rate >= floor)}
    if out_json:
        json.dump(report, open(out_json, "w"), indent=2)
    if rate < floor:
        raise SystemExit(f"[resolve] NO-GO: resolution rate {rate:.3f} < SP_RESOLUTION_RATE_MIN {floor} "
                         "(spec v3 §4.3) — abort + escalate; the floor is NOT lowered to rescue the run")
    return report


def redirect_audit(merged_dmp_map, node_set, bound):
    """Count merged.dmp redirects whose TARGET id is ABSENT from the embedding node set. An
    absent-target redirect silently inflates the drop rate, so it fails loud above a pre-committed
    bound (config.TAXDUMP_REDIRECT_ABSENT_MAX)."""
    node_set = set(node_set)
    absent = sorted({new for new in merged_dmp_map.values() if new not in node_set})
    report = {"n_redirects": len(merged_dmp_map), "n_absent_target": len(absent),
              "absent_targets": absent[:1000], "bound": bound, "ok": len(absent) <= bound}
    if len(absent) > bound:
        raise SystemExit(f"[resolve] merged.dmp: {len(absent)} redirects target ids absent from the node "
                         f"set > bound {bound} (spec v3 §4.3) — these silently inflate the drop rate")
    return report


def subset_audit(mapping_taxids, node_set):
    """Assert the cellular mapping.tsv taxid set is a SUBSET of the resolver node set (spec §4.3); a
    mapping taxid absent from the node set cannot be indexed → fail loud."""
    node_set = set(node_set)
    missing = sorted({int(t) for t in mapping_taxids if int(t) not in node_set})
    report = {"n_mapping": len({int(t) for t in mapping_taxids}), "n_not_in_node_set": len(missing),
              "missing": missing[:1000], "ok": not missing}
    if missing:
        raise SystemExit(f"[resolve] mapping.tsv has {len(missing)} taxids absent from the node set "
                         "(not a subset; spec v3 §4.3)")
    return report
