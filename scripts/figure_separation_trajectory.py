#!/usr/bin/env python3
"""Key result figure: the separation-ratio trajectory through the curriculum.

Tells the "radial-flat / lateral-climb" story — depth↔norm stays pinned near +0.98 (the radial frame
is in place from the start) while per-rank separation climbs from ~1× to 2–10× as the curriculum
widens. Reads a `<tag>_summary.tsv` produced by `_sweep_diagnostic_analyses.py` (columns: run,
checkpoint, depth_pearson, depth_spearman, phylum_sep, class_sep, order_sep, family_sep) and draws the
milestone_epNNN rows ordered by epoch.

Usage:
    .venv/bin/python scripts/figure_separation_trajectory.py \
        --summary artifacts/tags/metazoa_lower_lr_bigger_batch_summary.tsv \
        --title "Metazoa (498k nodes) — canonical recipe" \
        --transitions 40 80 120 \
        --out paper/figures/metazoa_separation_trajectory
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RANKS = [("phylum_sep", "phylum", "#1b9e77"),
         ("class_sep", "class", "#d95f02"),
         ("order_sep", "order", "#7570b3"),
         ("family_sep", "family", "#e7298a")]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", required=True, help="<tag>_summary.tsv from the sweep")
    ap.add_argument("--out", required=True, help="Output path stem (writes .png and .pdf)")
    ap.add_argument("--title", default="Separation trajectory")
    ap.add_argument("--transitions", type=int, nargs="*", default=[],
                    help="Epochs of curriculum-phase transitions to annotate (e.g. 40 80 120)")
    args = ap.parse_args()

    df = pd.read_csv(args.summary, sep="\t")
    # Keep only milestone rows; parse epoch number.
    df = df[df["checkpoint"].astype(str).str.startswith("milestone_ep")].copy()
    df["epoch"] = df["checkpoint"].str.extract(r"milestone_ep0*(\d+)").astype(int)
    df = df.sort_values("epoch")
    if df.empty:
        raise SystemExit("No milestone_epNNN rows in summary — nothing to plot.")

    fig, ax = plt.subplots(figsize=(9, 5.5))

    sep_cols = [c for c, _, _ in RANKS if c in df]
    ymax = float(df[sep_cols].to_numpy().max()) * 1.18
    x0, x1 = df["epoch"].min(), df["epoch"].max()

    # Quality bands (scaled to the data range, not blown up to 100).
    ax.axhspan(2.0, ymax, color="#2ca02c", alpha=0.06, zorder=0)
    ax.axhline(2.0, color="#2ca02c", ls=":", lw=1, alpha=0.7)
    ax.axhline(1.5, color="#888888", ls=":", lw=1, alpha=0.6)
    ax.text(x0, 2.05, "EXCELLENT (≥2×)", color="#2ca02c", fontsize=8, ha="left", va="bottom")
    ax.text(x0, 1.55, "GOOD (≥1.5×)", color="#666666", fontsize=8, ha="left", va="bottom")

    # Per-rank separation lines.
    for col, label, color in RANKS:
        if col in df:
            ax.plot(df["epoch"], df[col], marker="o", ms=4, lw=1.8, color=color,
                    label=f"{label} separation")
            ax.annotate(f"{df[col].iloc[-1]:.1f}×",
                        (df["epoch"].iloc[-1], df[col].iloc[-1]),
                        textcoords="offset points", xytext=(7, 0),
                        fontsize=8, fontweight="bold", color=color, va="center")

    ax.set_ylim(0, ymax)

    # Curriculum transition markers (vertical labels riding each line).
    for t in args.transitions:
        ax.axvline(t, color="#444444", ls="--", lw=0.9, alpha=0.55)
        ax.text(t, ymax * 0.62, f"dd-window widens (ep {t})", rotation=90,
                fontsize=7, ha="right", va="center", color="#444444",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Separation ratio  (mean inter / mean intra, Poincaré)")
    ax.set_title(args.title, fontweight="bold")
    ax.grid(True, alpha=0.25)

    # Twin axis: depth↔norm correlation (the "radial-flat" line).
    ax2 = ax.twinx()
    ax2.plot(df["epoch"], df["depth_pearson"], color="#333333", lw=1.3, ls="-",
             alpha=0.5, label="depth↔norm (Pearson)")
    ax2.set_ylabel("depth↔norm correlation", color="#333333")
    ax2.set_ylim(0.0, 1.02)
    ax2.tick_params(axis="y", labelcolor="#333333")

    # Merge legends.
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, framealpha=0.9)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.with_suffix('.png')}\n      {out.with_suffix('.pdf')}")

    # Console summary of the story.
    first, last = df.iloc[0], df.iloc[-1]
    print(f"\nradial-flat: depth↔norm {first['depth_pearson']:.3f} (ep{first['epoch']}) → "
          f"{last['depth_pearson']:.3f} (ep{last['epoch']})")
    for col, label, _ in RANKS:
        if col in df:
            print(f"lateral-climb {label:7s}: {first[col]:.2f}× → {last[col]:.2f}×")


if __name__ == "__main__":
    main()
