"""Unified CLI entrypoint for `taxembed` subcommands."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import taxopy
import torch

from taxembed.builders import build_clade_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # .../taxembed/src/taxembed/cli -> repo root
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "tags"

DEFAULT_OUTPUT_DIR = DATA_DIR / "taxopy"


def slugify_tag(tag: str) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "_", tag.lower()).strip("_")
    return slug or "run"


def ensure_dirs() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def resolve_taxid(identifier: str, taxdump_dir: Path) -> tuple[int, str]:
    nodes_path = taxdump_dir / "nodes.dmp"
    names_path = taxdump_dir / "names.dmp"
    merged_path = taxdump_dir / "merged.dmp"

    if nodes_path.exists() and names_path.exists():
        taxdb = taxopy.TaxDb(
            nodes_dmp=str(nodes_path),
            names_dmp=str(names_path),
            merged_dmp=str(merged_path) if merged_path.exists() else None,
            keep_files=True,
        )
    else:
        taxdb = taxopy.TaxDb(taxdb_dir=str(taxdump_dir))

    if identifier.isdigit():
        taxid = int(identifier)
    else:
        try:
            match = taxopy.taxid_from_name(identifier, taxdb)
        except Exception as exc:  # pragma: no cover - taxopy raises various errors
            raise SystemExit(f"❌ Failed to resolve '{identifier}': {exc}") from exc

        if not match:
            raise SystemExit(f"❌ No TaxID found for '{identifier}'")

        if isinstance(match, (set, list, tuple)):
            choices = list(match)
        else:
            choices = [match]

        try:
            taxid = int(choices[0])
        except (TypeError, ValueError) as exc:  # pragma: no cover
            raise SystemExit(
                f"❌ Failed to interpret TaxID for '{identifier}': {choices[0]!r}"
            ) from exc

    name = taxdb.taxid2name.get(str(taxid)) or taxdb.taxid2name.get(taxid, str(taxid))
    return taxid, name


def run_subprocess(cmd: list[str]) -> None:
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"❌ Command failed: {' '.join(cmd)}")


def handle_train(args: argparse.Namespace) -> None:
    ensure_dirs()

    slug = slugify_tag(args.as_tag)
    tag_dir = ARTIFACTS_DIR / slug
    tag_dir.mkdir(parents=True, exist_ok=True)

    dataset_record: dict[str, Any]
    training_data_path: Path
    mapping_path: Path

    if args.file:
        print("📦 Using provided dataset file")
        if not args.mapping:
            raise SystemExit("--mapping is required when using --file")
        training_data_path = Path(args.file).resolve()
        mapping_path = Path(args.mapping).resolve()
        if not training_data_path.exists():
            raise SystemExit(f"❌ Training data not found: {training_data_path}")
        if not mapping_path.exists():
            raise SystemExit(f"❌ Mapping file not found: {mapping_path}")
        dataset_record = {
            "type": "file",
            "data_path": str(training_data_path),
            "mapping_path": str(mapping_path),
        }
    else:
        if not args.identifier:
            raise SystemExit("Provide a taxonomic identifier or use --file")
        print(f"🔎 Resolving identifier '{args.identifier}'...")
        taxid, name = resolve_taxid(args.identifier, DATA_DIR)
        print(f"  ↳ TaxID {taxid} ({name})")
        print("🧱 Building clade dataset via TaxoPy (this may take a while)...")
        build_result = build_clade_dataset(
            taxid,
            dataset_name=slug,
            output_dir=DEFAULT_OUTPUT_DIR,
            taxdump_dir=DATA_DIR,
            max_depth=args.max_depth,
        )
        training_data_path = build_result.files["transitive_pickle"]
        mapping_path = build_result.files["mapping"]
        dataset_record = {
            "type": "taxopy",
            "root_taxid": build_result.root_taxid,
            "root_name": name,
            "dataset_name": build_result.dataset_name,
            "dataset_dir": str(build_result.output_dir),
            "files": {k: str(v) for k, v in build_result.files.items()},
            "max_depth": build_result.max_depth,
            "pairs": build_result.pairs_count,
        }
        print(
            f"  ✓ Dataset ready: {build_result.node_count:,} nodes, "
            f"{build_result.pairs_count:,} pairs (depth {build_result.max_depth})"
        )

    checkpoint_base = tag_dir / f"{slug}.pth"

    # Use the installed taxembed-train command from the package
    train_cmd = [
        sys.executable,
        "-m",
        "taxembed.cli.train",
        "--data",
        str(training_data_path),
        "--mapping",
        str(mapping_path),
        "--checkpoint",
        str(checkpoint_base),
        "--epochs",
        str(args.epochs),
        "--dim",
        str(args.dim),
        "--batch-size",
        str(args.batch_size),
        "--n-negatives",
        str(args.n_negatives),
        "--lr",
        str(args.lr),
        "--margin",
        str(args.margin),
        "--lambda-reg",
        str(args.lambda_reg),
        "--early-stopping",
        str(args.early_stopping),
        "--gpu",
        str(args.gpu),
    ]

    print(f"\n🧬 Training tag '{args.as_tag}' (slug '{slug}')")
    if args.identifier and not args.file:
        print(f"  ↳ Identifier: {args.identifier}")
    print(f"  ↳ Data: {training_data_path}")
    print(f"  ↳ Mapping: {mapping_path}")
    print(f"  ↳ Checkpoints: {checkpoint_base}")
    print(f"  ↳ Command: {' '.join(train_cmd)}")

    success = False
    try:
        run_subprocess(train_cmd)
        success = True
    except SystemExit:
        raise
    finally:
        # Always try to write metadata, even if training was interrupted,
        # so visualize can inspect partial checkpoints.
        best_checkpoint = checkpoint_base.with_name(f"{checkpoint_base.stem}_best.pth")
        if not best_checkpoint.exists():
            best_checkpoint = checkpoint_base

        metadata = {
            "tag": args.as_tag,
            "slug": slug,
            "created_at": datetime.now(UTC).isoformat(),
            "identifier": args.identifier,
            "dataset": dataset_record,
            "training": {
                "args": {
                    "epochs": args.epochs,
                    "dim": args.dim,
                    "batch_size": args.batch_size,
                    "n_negatives": args.n_negatives,
                    "lr": args.lr,
                    "margin": args.margin,
                    "lambda_reg": args.lambda_reg,
                    "early_stopping": args.early_stopping,
                    "gpu": args.gpu,
                    "max_depth": args.max_depth,
                },
                "paths": {
                    "data": str(training_data_path),
                    "mapping": str(mapping_path),
                    "checkpoint_base": str(checkpoint_base),
                    "best_checkpoint": str(best_checkpoint),
                },
                "command": train_cmd,
            },
        }

        meta_path = tag_dir / "run.json"
        meta_path.write_text(json.dumps(metadata, indent=2))

        if success:
            print(f"\n✅ Training complete. Metadata saved to {meta_path}")
            print(f"   Best checkpoint: {best_checkpoint}")
        else:
            print(f"\n⚠️ Training interrupted, metadata saved to {meta_path}")
            print(f"   Last checkpoint: {best_checkpoint}")


def handle_visualize(args: argparse.Namespace) -> None:
    slug = slugify_tag(args.tag)
    tag_dir = ARTIFACTS_DIR / slug
    meta_path = tag_dir / "run.json"
    if not meta_path.exists():
        raise SystemExit(f"❌ No run metadata for tag '{args.tag}' ({meta_path} missing)")

    metadata = json.loads(meta_path.read_text())
    dataset_meta = metadata.get("dataset", {})
    paths = metadata.get("training", {}).get("paths", {})

    checkpoint_path_str = (
        args.checkpoint or paths.get("best_checkpoint") or paths.get("checkpoint_base", "")
    )
    if not checkpoint_path_str:
        raise SystemExit(f"❌ No checkpoint path found in metadata for tag '{args.tag}'")

    checkpoint_path = Path(checkpoint_path_str)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (tag_dir / checkpoint_path_str).resolve()
    else:
        checkpoint_path = checkpoint_path.resolve()

    mapping_path = Path(args.mapping or paths.get("mapping", "")).resolve()

    if not checkpoint_path.exists():
        raise SystemExit(f"❌ Checkpoint not found: {checkpoint_path}")
    if not mapping_path.exists():
        raise SystemExit(f"❌ Mapping file not found: {mapping_path}")

    output_path = Path(args.output) if args.output else tag_dir / f"{slug}_umap.png"

    # Use the installed taxembed visualization module
    viz_cmd = [
        sys.executable,
        "-m",
        "taxembed.visualization.umap_viz",
        str(checkpoint_path),
        "--mapping",
        str(mapping_path),
        "--sample",
        str(args.sample),
        "--output",
        str(output_path),
    ]
    if args.names:
        viz_cmd.extend(["--names", str(args.names)])
    if args.nodes:
        viz_cmd.extend(["--nodes", str(args.nodes)])
    root_taxid = args.root_taxid or dataset_meta.get("root_taxid")
    if root_taxid is not None:
        viz_cmd.extend(["--root-taxid", str(root_taxid)])
    # Always pass children depth (default is 0 for immediate children)
    viz_cmd.extend(["--children", str(args.children)])

    # Extract title information from checkpoint and metadata
    clade_name = (
        dataset_meta.get("root_name") or dataset_meta.get("dataset_name") or args.tag.title()
    )
    epoch = None
    loss = None

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        epoch = ckpt.get("epoch", None)
        loss = ckpt.get("loss", None)
    except Exception:
        pass

    if clade_name:
        viz_cmd.extend(["--clade-name", str(clade_name)])
    if epoch is not None:
        viz_cmd.extend(["--epoch", str(epoch)])
    if loss is not None:
        viz_cmd.extend(["--loss", f"{loss:.6f}"])

    print(f"\n🎨 Visualizing tag '{args.tag}'")
    print(f"  ↳ Checkpoint: {checkpoint_path}")
    print(f"  ↳ Mapping: {mapping_path}")
    print(f"  ↳ Output: {output_path}")
    print(f"  ↳ Command: {' '.join(viz_cmd)}")

    run_subprocess(viz_cmd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="taxembed", description="Unified CLI for taxonomy embeddings"
    )
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Build dataset and train embeddings")
    train_parser.add_argument(
        "identifier", nargs="?", help="TaxID or clade name recognized by NCBI"
    )
    train_parser.add_argument(
        "-as", "--as-tag", required=True, help="Tag name used to reference this run"
    )
    train_parser.add_argument("--file", help="Path to prebuilt transitive dataset (.pkl)")
    train_parser.add_argument("--mapping", help="Mapping file (required with --file)")
    train_parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Limit descendant depth when building clades",
    )
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--dim", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--n-negatives", type=int, default=50)
    train_parser.add_argument("--lr", type=float, default=0.005)
    train_parser.add_argument("--margin", type=float, default=0.2)
    train_parser.add_argument("--lambda-reg", type=float, default=0.1)
    train_parser.add_argument("--early-stopping", type=int, default=5)
    train_parser.add_argument("--gpu", type=int, default=-1, help="GPU device index (-1 for CPU)")
    train_parser.set_defaults(func=handle_train)

    visualize_parser = subparsers.add_parser("visualize", help="Visualize a trained tag with UMAP")
    visualize_parser.add_argument("tag", help="Tag used during `taxembed train ... -as TAG`")
    visualize_parser.add_argument(
        "--sample", type=int, default=25000, help="Number of points for UMAP sampling"
    )
    visualize_parser.add_argument("--output", help="Output image path")
    visualize_parser.add_argument("--checkpoint", help="Override checkpoint path")
    visualize_parser.add_argument("--mapping", help="Override mapping path")
    visualize_parser.add_argument("--names", help="Override names.dmp path")
    visualize_parser.add_argument("--nodes", help="Override nodes.dmp path")
    visualize_parser.add_argument("--root-taxid", type=int, help="Override root TaxID for coloring")
    visualize_parser.add_argument(
        "--children",
        type=int,
        default=0,
        help="Depth level for coloring (0=children, 1=grandchildren, 2=great-grandchildren, etc.)",
    )
    visualize_parser.set_defaults(func=handle_visualize)

    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze hierarchy quality of embeddings"
    )
    analyze_parser.add_argument("checkpoint", help="Checkpoint file to analyze")
    analyze_parser.add_argument("--mapping", help="Override mapping path")
    analyze_parser.set_defaults(func=handle_analyze)

    return parser


def handle_analyze(args: argparse.Namespace) -> None:
    """Handle analyze subcommand."""
    from taxembed.analysis import analyze_hierarchy

    # For now, just call the main function
    # TODO: Update to accept checkpoint argument
    analyze_hierarchy()


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
