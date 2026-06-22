"""Run hyperbolic hierarchy analysis across all diagnostic checkpoints (A/B/C).

For A (metazoa_softmax_nocurric) and B (metazoa_v2_nocurric): analyze best.pth + final.pth.
For C (metazoa_softmax_milestones): analyze best.pth + final.pth + all 20 milestone_epoch*.pth
to build the per-epoch trajectory through the curriculum transitions.

Outputs:
- Per-checkpoint analysis dir at artifacts/tags/<tag>/analysis_<which>/ (PNG plots + console)
- One combined TSV at artifacts/tags/metazoa_diagnostics_summary.tsv aggregating
  (run, checkpoint, depth_norm_pearson, phylum_sep, class_sep, order_sep, family_sep).
- A markdown table written to stdout for paste into PROJECT_STATE.md.

Designed to be one-shot: assumes all .pth files have been pulled to local
TaxPointCare/poincare-embeddings/artifacts/tags/<tag>/. Skip any checkpoint whose .pth
isn't found locally (logged + continued).
"""

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path("/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings")
VENV_PY = REPO_ROOT / ".venv" / "bin" / "python"
ANALYZER = REPO_ROOT / "scripts" / "analyze_hierarchy_hyperbolic.py"
MAPPING = REPO_ROOT / "data" / "taxopy" / "metazoa_33208_clean" / "taxonomy_edges_metazoa_33208_clean.mapping.tsv"
SUMMARY_TSV = REPO_ROOT / "artifacts" / "tags" / "metazoa_diagnostics_summary.tsv"

# (tag, [checkpoint-label, checkpoint-file-relative-to-tag-dir])
TAG_TO_CHECKPOINTS = {
    "metazoa_softmax_nocurric": [
        ("best", "metazoa_softmax_nocurric_best.pth"),
        ("final", "metazoa_softmax_nocurric.pth"),
    ],
    "metazoa_v2_nocurric": [
        ("best", "metazoa_v2_nocurric_best.pth"),
        ("final", "metazoa_v2_nocurric.pth"),
    ],
    "metazoa_softmax_milestones": [
        ("best", "metazoa_softmax_milestones_best.pth"),
        ("final", "metazoa_softmax_milestones.pth"),
        # Milestones auto-discovered below (ep 10, 20, ..., 200).
    ],
    "metazoa_lower_lr_bigger_batch": [   # Experiment 1 (job 5664609) — added 2026-06-03
        ("best", "metazoa_lower_lr_bigger_batch_best.pth"),
        ("final", "metazoa_lower_lr_bigger_batch.pth"),
        # Milestones auto-discovered below.
    ],
}

# Tags that save per-epoch milestone checkpoints (auto-discover *_milestone_epoch*.pth).
MILESTONE_TAGS = ("metazoa_softmax_milestones", "metazoa_lower_lr_bigger_batch")


def discover_milestones(tag_dir: Path) -> list[tuple[str, str]]:
    """Find milestone_epoch{N}.pth files in tag_dir, return sorted (label, filename)."""
    milestones = []
    for path in tag_dir.glob("*_milestone_epoch*.pth"):
        match = re.search(r"_milestone_epoch(\d+)\.pth$", path.name)
        if match:
            ep = int(match.group(1))
            milestones.append((f"milestone_ep{ep:03d}", path.name, ep))
    milestones.sort(key=lambda x: x[2])
    return [(label, fname) for label, fname, _ in milestones]


# Result schema: dict per (tag, checkpoint_label).
def parse_analyzer_stdout(stdout: str) -> dict:
    """Extract metrics from analyze_hierarchy_hyperbolic.py's printed summary."""
    out = {
        "depth_pearson": None,
        "depth_spearman": None,
        "phylum_sep": None,
        "class_sep": None,
        "order_sep": None,
        "family_sep": None,
    }
    # Pearson/Spearman lines look like:
    #   Pearson:  r = +0.8782 (p = 0.00e+00)
    #   Spearman: rho = +0.8734 (p = 0.00e+00)
    m = re.search(r"Pearson:\s+r\s*=\s*([+-]?[\d.]+)", stdout)
    if m:
        out["depth_pearson"] = float(m.group(1))
    m = re.search(r"Spearman:\s+rho\s*=\s*([+-]?[\d.]+)", stdout)
    if m:
        out["depth_spearman"] = float(m.group(1))

    # Per-rank separation lines in summary look like:
    #   Phylum              1.05x            POOR
    #   Class               1.20x        MODERATE
    for rank in ("Phylum", "Class", "Order", "Family"):
        m = re.search(rf"^{rank}\s+([\d.]+)x", stdout, re.MULTILINE)
        if m:
            out[rank.lower() + "_sep"] = float(m.group(1))
    return out


def run_one(tag: str, label: str, ckpt_filename: str) -> dict | None:
    """Run the analyzer on one checkpoint; return parsed metrics or None if file missing."""
    tag_dir = REPO_ROOT / "artifacts" / "tags" / tag
    ckpt = tag_dir / ckpt_filename
    if not ckpt.exists():
        print(f"  SKIP {tag}/{label}: {ckpt_filename} not found", file=sys.stderr)
        return None
    out_dir = tag_dir / f"analysis_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  RUN  {tag}/{label} → {out_dir.name}/", file=sys.stderr)
    proc = subprocess.run(
        [
            str(VENV_PY),
            str(ANALYZER),
            "--checkpoint", str(ckpt),
            "--mapping", str(MAPPING),
            "--ranks", "phylum", "class", "order", "family",
            "-o", str(out_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        print(f"    ANALYZER FAILED (rc {proc.returncode})", file=sys.stderr)
        print(f"    stderr: {proc.stderr[-500:]}", file=sys.stderr)
        return None
    parsed = parse_analyzer_stdout(proc.stdout)
    # Also save the full stdout next to the plots so future-Claude has the verbatim run.
    (out_dir / "analysis_stdout.txt").write_text(proc.stdout)
    return parsed


def main():
    # Optional argv tag-filter: `_sweep_diagnostic_analyses.py <tag> [<tag> ...]`.
    # When filtered to a single tag, write to a tag-specific TSV so the shared
    # diagnostics TSV (A/B/C history) is never overwritten/perturbed.
    filter_tags = sys.argv[1:] if len(sys.argv) > 1 else None
    summary_tsv = (REPO_ROOT / "artifacts" / "tags" / f"{filter_tags[0]}_summary.tsv"
                   if filter_tags and len(filter_tags) == 1 else SUMMARY_TSV)
    summary_tsv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    print("\n=== Sweep diagnostic analyses ===\n", file=sys.stderr)

    for tag, base_ckpts in TAG_TO_CHECKPOINTS.items():
        if filter_tags and tag not in filter_tags:
            continue
        tag_dir = REPO_ROOT / "artifacts" / "tags" / tag
        if not tag_dir.exists():
            print(f"SKIP {tag}: tag dir not found locally — pull artifacts first.", file=sys.stderr)
            continue

        # Build the list of (label, filename) for this tag.
        checkpoints = list(base_ckpts)
        if tag in MILESTONE_TAGS:
            checkpoints.extend(discover_milestones(tag_dir))

        print(f"\n{tag}:", file=sys.stderr)
        for label, fname in checkpoints:
            metrics = run_one(tag, label, fname)
            if metrics is None:
                continue
            metrics["run"] = tag
            metrics["checkpoint"] = label
            rows.append(metrics)

    # Write summary TSV.
    if rows:
        fieldnames = ["run", "checkpoint",
                      "depth_pearson", "depth_spearman",
                      "phylum_sep", "class_sep", "order_sep", "family_sep"]
        with summary_tsv.open("w") as fh:
            fh.write("\t".join(fieldnames) + "\n")
            for r in rows:
                fh.write("\t".join(
                    str(r.get(k, "")) if r.get(k) is not None else ""
                    for k in fieldnames
                ) + "\n")
        print(f"\nWrote summary TSV: {summary_tsv}", file=sys.stderr)

    # Print markdown table to stdout for PROJECT_STATE paste.
    print("\n## Sweep results — paste into PROJECT_STATE.md\n")
    print("| Run | Checkpoint | depth↔norm Pearson | phylum | class | order | family |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for r in rows:
        def fmt(v):
            return f"{v:+.3f}" if v is not None and isinstance(v, float) else "—"
        def fmtx(v):
            return f"{v:.2f}×" if v is not None and isinstance(v, float) else "—"
        print(f"| {r['run']} | {r['checkpoint']} | {fmt(r.get('depth_pearson'))} | "
              f"{fmtx(r.get('phylum_sep'))} | {fmtx(r.get('class_sep'))} | "
              f"{fmtx(r.get('order_sep'))} | {fmtx(r.get('family_sep'))} |")


if __name__ == "__main__":
    main()
