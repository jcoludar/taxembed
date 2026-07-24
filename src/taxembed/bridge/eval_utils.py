"""Shared eval helpers for the tax_disentangle READ/CLEAN/calibration batteries.

Behavior-preserving extraction (Phase A) of three helpers that previously lived inside the M1 eval
scripts, so the SP-Metazoa code can reuse them without copy-paste drift:

  * label_nesting (alias _label_nesting) — conditional-entropy nesting of taxonomy under function
    (moved verbatim from clean_eval.py); the return-dict keys are part of the contract read by
    clean_eval._disentanglement_verdict_multifamily — DO NOT rename them.
  * _compute_cluster_ids_fasta — mmseqs 0.9-identity clustering over an arbitrary fasta
    (moved verbatim from clean_eval.py). The PLA2-specific _compute_cluster_ids stays in clean_eval.py.
  * RankLookup — {taxid -> {rank: taxid}} cache from TaxonResolver.lineage (moved verbatim from
    read_eval.py; radius_calib.py had a duplicate — both now import this one).

Imports are relative within the ``taxembed.bridge`` package.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent

from . import config  # noqa: E402,F401  (kept for parity with the scripts' bare-import style)
from .clusters import have_mmseqs, mmseqs_cluster  # noqa: E402
from .taxdump import TaxonResolver  # noqa: E402


def _compute_cluster_ids_fasta(ids, fasta, tag):
    """{identifier: cluster_rep} via mmseqs easy-cluster at 0.9 identity over an ARBITRARY fasta
    (multifamily uses the ToxFam fasta); one-per-seq fallback. For diverse cross-family sequences
    most clusters will be singletons at 0.9 identity — that is the correct 'low redundancy' picture."""
    tmp = _HERE / "results" / "_mmseqs_tmp"
    if not have_mmseqs():
        print("[clusters] mmseqs NOT on PATH -> one-cluster-per-sequence fallback")
        return {i: i for i in ids}
    try:
        mapping = mmseqs_cluster(fasta, out_prefix=tmp / tag, tmp_dir=tmp / "tmp",
                                 min_seq_id=0.9, cleanup=True)
        for i in ids:
            mapping.setdefault(i, i)
        n_clusters = len(set(mapping[i] for i in ids))
        print(f"[clusters] mmseqs -> {n_clusters} clusters over {len(ids)} sequences")
        return mapping
    except Exception as e:                                     # noqa: BLE001 — robust fallback per spec
        print(f"[clusters] mmseqs FAILED ({e}) -> one-cluster-per-sequence fallback")
        return {i: i for i in ids}


def label_nesting(func_labels, tax_labels):
    """How much knowing FUNCTION (toxin family) determines TAXONOMY at a rank, via the conditional-entropy
    reduction 1 - H(tax|func)/H(tax). High reduction ⇒ in THIS sample each toxin family is drawn from a
    narrow taxonomic slice, so the function one-hot structurally contains taxonomy — which makes the
    REVERSE direction (erase function ⇒ taxonomy drops) a SAMPLING property of the curated set, not
    embedding entanglement. Also returns how many function labels map to a single tax value."""
    func_labels = list(func_labels)
    tax_labels = list(tax_labels)
    tot = len(tax_labels)
    _, c = np.unique(tax_labels, return_counts=True)
    p = c / c.sum()
    h_tax = float(-(p * np.log2(p)).sum())
    h_cond = 0.0
    for fv in set(func_labels):
        sub = [tax_labels[i] for i in range(tot) if func_labels[i] == fv]
        _, cc = np.unique(sub, return_counts=True)
        pp = cc / cc.sum()
        h_cond += (len(sub) / tot) * float(-(pp * np.log2(pp)).sum())
    reduction = (1 - h_cond / h_tax) if h_tax > 0 else float("nan")
    pure = sum(1 for fv in set(func_labels)
               if len(set(tax_labels[i] for i in range(tot) if func_labels[i] == fv)) == 1)
    return {"H_tax_bits": round(h_tax, 4), "H_tax_given_function_bits": round(h_cond, 4),
            "uncertainty_reduction": round(float(reduction), 4),
            "n_function_labels_mapping_to_one_tax": int(pure),
            "n_function_labels": int(len(set(func_labels)))}


# preserve the historical underscore name so internal references by `_label_nesting` still resolve
_label_nesting = label_nesting


class RankLookup:
    """Caches {taxid -> {rank: taxid}} from TaxonResolver.lineage for both true and predicted taxids."""

    def __init__(self, resolver: TaxonResolver, ranks):
        self.resolver = resolver
        self.ranks = ranks
        self._cache: dict[int, dict] = {}

    def rank_map(self, taxid: int) -> dict:
        taxid = int(taxid)
        if taxid not in self._cache:
            lin = self.resolver.lineage(taxid)
            rm = {}
            for rank, tid, _name in lin:
                if rank in self.ranks and rank not in rm:      # first (lowest) occurrence wins
                    rm[rank] = int(tid)
            self._cache[taxid] = rm
        return self._cache[taxid]
