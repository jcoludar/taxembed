"""Step-0 taxonomy-target gate (spec v3 §3) — a fail-loud HARD gate before the all-life DAG.

Five parts: (1) assert cellular_canonical.pth exists or HALT-and-ask the user; (2) assert the GPU cluster dump
is the FULL new_taxdump — per-superkingdom node counts above committed floors for Bacteria AND Archaea
AND Eukaryota (a eukaryote/Metazoa-subset dump that still resolves ONE bacterial taxid would otherwise
pass the idx_of_taxid lookup yet silently mis-scope 'all cellular life' — the exact failure §3 exists
to prevent; canary #6 alone cannot catch it); (3) verify resolved cellular taxids carry a superkingdom
rank label; (4) record dump release+SHA256; (5) repoint TAXDUMP_DIR → the GPU cluster dump.

The Mac-testable core below is fixture-mockable (counts/labels passed in); the real dump/ckpt reads
run in J0a/J0 on GPU cluster, where this module is invoked with the live node-count + rank-label scan.
"""
import hashlib
import os

_REQUIRED_SK = ("Bacteria", "Archaea", "Eukaryota")


def assert_ckpt_present(ckpt_path):
    if not os.path.exists(ckpt_path):
        raise SystemExit(
            f"[step0] cellular_canonical checkpoint absent: {ckpt_path}. HALT — ask the user. "
            "Swapping to eukaryota_canonical changes the deliverable from all cellular life to "
            "eukaryotes-only (a claim change), so it is a user decision, never a silent fallback (spec §3).")
    return ckpt_path


def check_superkingdom_node_counts(node_counts, floors):
    """Assert the dump spans all three superkingdoms above committed floors (spec §3 part 2)."""
    below = {sk: node_counts.get(sk, 0) for sk in _REQUIRED_SK if node_counts.get(sk, 0) < floors.get(sk, 0)}
    report = {"node_counts": dict(node_counts), "floors": dict(floors),
              "below_floor": below, "ok": not below}
    if below:
        raise SystemExit(f"[step0] dump is not the full new_taxdump — superkingdoms below floor: {below} "
                         "(a eukaryote/Metazoa-subset dump silently mis-scopes 'all cellular life'; spec §3)")
    return report


def assert_superkingdom_rank_labels(taxid_to_sk):
    """Assert every sampled resolved cellular taxid carries a superkingdom rank label (spec §3 part 2;
    couples to §4.2.1 kingdom-span). `taxid_to_sk` maps taxid -> superkingdom name (falsy = missing)."""
    missing = sorted(t for t, sk in taxid_to_sk.items() if not sk)
    report = {"n": len(taxid_to_sk), "n_missing_label": len(missing),
              "missing": missing[:100], "ok": not missing}
    if missing:
        raise SystemExit(f"[step0] {len(missing)} resolved cellular taxids lack a superkingdom rank label "
                         "(spec §3)")
    return report


def record_dump_provenance(taxdump_dir, release, nodes_file="nodes.dmp"):
    """Record the GPU cluster dump release + SHA256(nodes.dmp) for the result-JSON provenance (spec §3/§4.3)."""
    path = os.path.join(taxdump_dir, nodes_file)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {"release": release, "sha256": h.hexdigest(), "taxdump_dir": taxdump_dir, "nodes_file": nodes_file}
