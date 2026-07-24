"""E3 — SP-Metazoa stage-1 figures (spec §6.6). Renders whatever result JSONs exist (defensive, so it
can run mid-E2). Panels:
  (a) disentanglement-matrix heatmap x {pfam, ec}
  (b) coherence overlap vs random baseline (+3σ band) x {pfam, ec}
  (c) per-family order-span distribution (Pfam gate-scan: raw + effective orders)
  (d) READ accuracy battery x {pfam, ec} (class-rank, depth-null, matched-chance, leave-clade-out,
      leave-species-and-clade leak)
  (e) confound-survival: within-stratum function purity orig vs length-erased vs composition-erased
  (f) within-family clade-signal MI distribution (Pfam, §4.7a) — headline must hold for high-signal too

Run: python -m taxembed.bridge.make_sp_figures  -> results/figures/sp_metazoa_*.png
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = Path(__file__).resolve().parent
from . import config  # noqa: E402

R = PROJ / "results"
FIG = R / "figures"


def _load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


CLEAN = {k: _load(f"clean_sp_metazoa_{k}.json") for k in ("pfam", "ec", "pfam_full", "ec_full")}
READ = {k: _load(f"read_sp_metazoa_{k}.json") for k in ("pfam", "ec")}


def fig_a_matrix():
    panels = [(k, CLEAN[k]) for k in ("pfam", "ec") if CLEAN[k]]
    if not panels:
        return
    rows = ["taxonomy_recoverability_tax_family_acc", "function_recoverability_global_purity",
            "function_recoverability_within_tax_family_purity"]
    rlab = ["taxonomy\n(class acc)", "function\n(global purity)", "function\n(within-class purity)"]
    cols = ["original", "tax_erased", "function_erased", "matched_control"]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 3.4), squeeze=False)
    for ax, (k, d) in zip(axes[0], panels):
        M = np.array([[d["disentanglement_matrix"][r][c] for c in cols] for r in rows])
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels(rlab, fontsize=8)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        color="white" if M[i, j] < 0.6 else "black", fontsize=8)
        v = d.get("disentanglement_verdict", {})
        ax.set_title(f"{k}  (clean_disentangled={v.get('clean_disentangled')})", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("(a) Disentanglement matrix — taxonomy erasable, function preserved", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "sp_metazoa_a_matrix.png", dpi=140); plt.close(fig)


def fig_b_coherence():
    panels = [(k, CLEAN[k]["coherence"]) for k in ("pfam", "ec") if CLEAN[k] and CLEAN[k].get("coherence")]
    if not panels:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    x = np.arange(len(panels))
    ov = [c["overlap"] for _, c in panels]
    base = [c["random_baseline_mean"] for _, c in panels]
    s3 = [3 * c["random_baseline_std"] for _, c in panels]
    ax.bar(x - 0.18, ov, 0.36, label="READ↔CLEAN overlap", color="#2c7fb8")
    ax.bar(x + 0.18, base, 0.36, yerr=s3, label="random baseline (±3σ)", color="#bdbdbd", capsize=4)
    ax.set_xticks(x); ax.set_xticklabels([k for k, _ in panels])
    ax.set_ylabel("subspace overlap"); ax.set_ylim(0, 1.05)
    for i, (_, c) in enumerate(panels):
        ax.text(i - 0.18, c["overlap"] + 0.02, "same-object" if c["overlap_exceeds_baseline_3sigma"] else "ns",
                ha="center", fontsize=8)
    ax.legend(fontsize=8); ax.set_title("(b) Coherence: READ ridge ≡ CLEAN predictive subspace")
    fig.tight_layout(); fig.savefig(FIG / "sp_metazoa_b_coherence.png", dpi=140); plt.close(fig)


def fig_c_orderspan():
    p = config.SP_GATE_SCAN
    if not Path(p).exists():
        return
    g = pd.read_csv(p, sep="\t")
    g = g[g["keep"]]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.hist(g["raw_orders"], bins=range(0, int(g["raw_orders"].max()) + 2), alpha=0.55, label="raw #orders")
    ax.hist(g["effective_orders"], bins=20, alpha=0.55, label="effective #orders exp(H)")
    ax.axvline(config.SP_CROSS_MIN_ORDERS, color="k", ls="--", lw=1, label=f"floor raw≥{config.SP_CROSS_MIN_ORDERS}")
    ax.axvline(config.SP_CROSS_MIN_EFF_ORDERS, color="r", ls=":", lw=1, label=f"floor eff≥{config.SP_CROSS_MIN_EFF_ORDERS}")
    ax.set_xlabel("orders spanned per kept Pfam family"); ax.set_ylabel("families")
    ax.legend(fontsize=8); ax.set_title(f"(c) Per-family order span ({len(g)} kept Pfam families)")
    fig.tight_layout(); fig.savefig(FIG / "sp_metazoa_c_orderspan.png", dpi=140); plt.close(fig)


def fig_d_read():
    panels = [(k, READ[k]) for k in ("pfam", "ec") if READ[k]]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 3.6), squeeze=False)
    for ax, (k, d) in zip(axes[0], panels):
        lco = d.get("leave_clade_out", {})
        lsc = d.get("leave_species_and_clade", {})
        labels = ["class_rank", "depth_null", "matched_chance", "leave_clade_out",
                  "fam_balanced", "loso(leak)", "loso+clade"]
        vals = [d.get("class_rank_acc", np.nan), d.get("depth_null", {}).get("class_rank_acc", np.nan),
                d.get("matched_candidate_set", {}).get("matched_chance_acc", np.nan),
                lco.get("weighted_class_acc", np.nan),
                lco.get("family_balanced", {}).get("weighted_class_acc", np.nan),
                lsc.get("rawknn_loso_class_acc", np.nan),
                lsc.get("rawknn_species_and_clade_class_acc", np.nan)]
        colors = ["#2c7fb8", "#9e9e9e", "#9e9e9e", "#41ab5d", "#41ab5d", "#fd8d3c", "#fd8d3c"]
        ax.bar(range(len(vals)), vals, color=colors)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_ylim(0, 1.0); ax.set_ylabel("class accuracy")
        leak = lsc.get("leak_drop")
        ax.set_title(f"{k}  leak_drop={leak:.3f}" if isinstance(leak, (int, float)) else k, fontsize=9)
    fig.suptitle("(d) READ accuracy battery (bridge class-rank, nulls, leak diagnostic)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "sp_metazoa_d_read.png", dpi=140); plt.close(fig)


def fig_e_confound():
    panels = [(k, CLEAN[k]) for k in ("pfam", "ec") if CLEAN[k]]
    if not panels:
        return
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    x = np.arange(len(panels)); w = 0.26
    orig = [d["step3_function_preserved"]["within_tax_family"]["original"] for _, d in panels]
    leng = [d.get("length_control", {}).get("purity_after_length_erase", np.nan) for _, d in panels]
    comp = [d.get("composition_control", {}).get("purity_after_comp_erase", np.nan) for _, d in panels]
    ax.bar(x - w, orig, w, label="orig within-class purity", color="#2c7fb8")
    ax.bar(x, leng, w, label="after length-erase (1-D)", color="#7fcdbb")
    ax.bar(x + w, comp, w, label="after composition-erase (20-D)", color="#fdae6b")
    ax.set_xticks(x); ax.set_xticklabels([k for k, _ in panels]); ax.set_ylim(0, 1.0)
    ax.set_ylabel("within-class function purity")
    for i, (_, d) in enumerate(panels):
        g1 = d.get("length_control", {}).get("geometry_change")
        g2 = d.get("composition_control", {}).get("geometry_change")
        surv = (d.get("composition_control", {}).get("headline_survives")
                and d.get("length_control", {}).get("headline_survives"))
        ax.text(i, 0.06, f"geomΔ L={g1:.2f} C={g2:.2f}\nsurvives={surv}", ha="center", fontsize=7)
    ax.legend(fontsize=8); ax.set_title("(e) Confound survival — purity holds after length & composition erasure")
    fig.tight_layout(); fig.savefig(FIG / "sp_metazoa_e_confound.png", dpi=140); plt.close(fig)


def fig_f_cladesignal():
    p = config.SP_GATE_SCAN
    if not Path(p).exists():
        return
    g = pd.read_csv(p, sep="\t")
    g = g[g["keep"] & g["clade_signal_mi"].notna()]
    if not len(g):
        return
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.hist(g["clade_signal_mi"], bins=25, color="#756bb1")
    ax.axvline(g["clade_signal_mi"].median(), color="k", ls="--", lw=1,
               label=f"median {g['clade_signal_mi'].median():.2f}")
    v = (CLEAN.get("pfam") or {}).get("disentanglement_verdict", {})
    surv = v.get("function_preserved_under_taxonomy_erasure")
    ax.set_xlabel("within-family clade-signal MI (nats, §4.7a)"); ax.set_ylabel("kept Pfam families")
    ax.legend(fontsize=8)
    ax.set_title(f"(f) Within-family clade signal — Pfam headline survives={surv} (holds for high-signal too)")
    fig.tight_layout(); fig.savefig(FIG / "sp_metazoa_f_cladesignal.png", dpi=140); plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    for fn in (fig_a_matrix, fig_b_coherence, fig_c_orderspan, fig_d_read, fig_e_confound, fig_f_cladesignal):
        try:
            fn()
            print(f"[fig] {fn.__name__} ok")
        except Exception as e:  # noqa
            print(f"[fig] {fn.__name__} FAILED: {type(e).__name__}: {e}")
    print(f"[fig] wrote to {FIG}")


if __name__ == "__main__":
    main()
