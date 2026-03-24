"""Unified CLI entrypoint for `taxembed` subcommands."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import taxopy

from taxembed.builders import build_clade_dataset
from taxembed.analysis.dimension import angular_packing_dim, participation_ratio, recommend_dim


PROJECT_ROOT = Path(__file__).resolve().parents[3]  # .../poincare-embeddings/src/taxembed/cli -> repo root
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
            raise SystemExit(f"❌ Failed to interpret TaxID for '{identifier}': {choices[0]!r}") from exc

    name = taxdb.taxid2name.get(str(taxid)) or taxdb.taxid2name.get(taxid, str(taxid))
    return taxid, name


def run_subprocess(cmd: list[str]) -> None:
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"❌ Command failed: {' '.join(cmd)}")


def _canonical_dataset_name(
    name: str, taxid: int, max_depth: int | None = None, clean: bool = False,
) -> str:
    """Canonical dataset name based on clade, not training tag."""
    base = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or str(taxid)
    suffix = f"_d{max_depth}" if max_depth is not None else ""
    clean_suffix = "_clean" if clean else ""
    return f"{base}_{taxid}{suffix}{clean_suffix}"


def _find_cached_dataset(
    dataset_name: str,
    output_dir: Path,
    max_pairs: int | None = None,
) -> Optional[Dict[str, Any]]:
    """Check if a compatible pre-built dataset exists and return its info."""
    dataset_dir = output_dir / dataset_name
    manifest_glob = list(dataset_dir.glob("*_manifest.json"))
    if not manifest_glob:
        return None

    manifest_path = manifest_glob[0]
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    # Check max_pairs compatibility: cached dataset must have >= requested pairs
    if max_pairs and manifest.get("transitive_pairs", 0) < max_pairs:
        return None

    # Find the data files
    npz_files = list(dataset_dir.glob("*_transitive.npz"))
    pkl_files = list(dataset_dir.glob("*_transitive.pkl"))
    mapping_files = list(dataset_dir.glob("*.mapping.tsv"))

    if not mapping_files or (not npz_files and not pkl_files):
        return None

    data_path = npz_files[0] if npz_files else pkl_files[0]
    return {
        "data_path": data_path,
        "mapping_path": mapping_files[0],
        "manifest": manifest,
        "dataset_dir": dataset_dir,
    }


def handle_build(args: argparse.Namespace) -> None:
    """Handle `taxembed build` — prebuild a clade dataset for reuse."""
    ensure_dirs()

    print(f"🔎 Resolving identifier '{args.identifier}'...")
    taxid, name = resolve_taxid(args.identifier, DATA_DIR)
    print(f"  ↳ TaxID {taxid} ({name})")

    use_clean = getattr(args, "clean", False)
    dataset_name = _canonical_dataset_name(name, taxid, args.max_depth, clean=use_clean)
    dataset_dir = DEFAULT_OUTPUT_DIR / dataset_name

    # Check cache
    cached = _find_cached_dataset(dataset_name, DEFAULT_OUTPUT_DIR, args.max_pairs)
    if cached and not args.force:
        m = cached["manifest"]
        print(f"  ✓ Dataset already exists: {dataset_dir}")
        print(f"    {m['nodes']:,} nodes, {m['transitive_pairs']:,} pairs (depth {m['max_depth_observed']})")
        print(f"  Use --force to rebuild.")
        return

    clean_msg = " (with noise filtering)" if use_clean else ""
    print(f"🧱 Building clade dataset '{dataset_name}'{clean_msg}...")
    build_result = build_clade_dataset(
        taxid,
        dataset_name=dataset_name,
        output_dir=DEFAULT_OUTPUT_DIR,
        taxdump_dir=DATA_DIR,
        max_depth=args.max_depth,
        max_pairs=args.max_pairs,
        clean=use_clean,
    )
    print(
        f"  ✓ Dataset ready: {build_result.node_count:,} nodes, "
        f"{build_result.pairs_count:,} pairs (depth {build_result.max_depth})"
    )
    print(f"  ↳ Path: {build_result.output_dir}")
    print(f"\n  Use with: taxembed train {args.identifier} -as <tag>")


def handle_train(args: argparse.Namespace) -> None:
    ensure_dirs()

    slug = slugify_tag(args.as_tag)
    tag_dir = ARTIFACTS_DIR / slug
    tag_dir.mkdir(parents=True, exist_ok=True)

    dataset_record: Dict[str, Any]
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
    elif args.dataset:
        # Use a named pre-built dataset
        cached = _find_cached_dataset(args.dataset, DEFAULT_OUTPUT_DIR)
        if cached is None:
            raise SystemExit(
                f"❌ Dataset '{args.dataset}' not found in {DEFAULT_OUTPUT_DIR}.\n"
                f"   Build it first: taxembed build <identifier>"
            )
        training_data_path = cached["data_path"]
        mapping_path = cached["mapping_path"]
        m = cached["manifest"]
        print(f"📦 Using pre-built dataset '{args.dataset}'")
        print(f"  ↳ {m['nodes']:,} nodes, {m['transitive_pairs']:,} pairs (depth {m['max_depth_observed']})")
        dataset_record = {
            "type": "cached",
            "dataset_name": args.dataset,
            "root_taxid": m["root_taxid"],
            "root_name": m.get("root_name", ""),
            "dataset_dir": str(cached["dataset_dir"]),
            "max_depth": m["max_depth_observed"],
            "pairs": m["transitive_pairs"],
        }
    else:
        if not args.identifier:
            raise SystemExit("Provide a taxonomic identifier, --dataset, or --file")
        print(f"🔎 Resolving identifier '{args.identifier}'...")
        taxid, name = resolve_taxid(args.identifier, DATA_DIR)
        print(f"  ↳ TaxID {taxid} ({name})")

        use_clean = getattr(args, "clean", False)

        # Check for cached dataset with canonical name
        canonical = _canonical_dataset_name(name, taxid, args.max_depth, clean=use_clean)
        cached = _find_cached_dataset(canonical, DEFAULT_OUTPUT_DIR, args.max_pairs)

        if cached:
            m = cached["manifest"]
            training_data_path = cached["data_path"]
            mapping_path = cached["mapping_path"]
            print(f"📦 Found cached dataset '{canonical}'")
            print(f"  ↳ {m['nodes']:,} nodes, {m['transitive_pairs']:,} pairs (depth {m['max_depth_observed']})")
            dataset_record = {
                "type": "cached",
                "dataset_name": canonical,
                "root_taxid": m["root_taxid"],
                "root_name": m.get("root_name", name),
                "dataset_dir": str(cached["dataset_dir"]),
                "max_depth": m["max_depth_observed"],
                "pairs": m["transitive_pairs"],
                "clean": m.get("clean", False),
            }
        else:
            clean_msg = " with noise filtering" if use_clean else ""
            print(f"🧱 Building clade dataset '{canonical}'{clean_msg} (this may take a while)...")
            build_result = build_clade_dataset(
                taxid,
                dataset_name=canonical,
                output_dir=DEFAULT_OUTPUT_DIR,
                taxdump_dir=DATA_DIR,
                max_depth=args.max_depth,
                max_pairs=args.max_pairs,
                clean=use_clean,
            )
            # Prefer .npz (columnar, ~30x smaller) over pickle when available
            training_data_path = build_result.files.get("transitive_npz") or build_result.files["transitive_pickle"]
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
    train_script = PROJECT_ROOT / "train_small.py"

    train_cmd = [
        sys.executable,
        str(train_script),
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
        "--optimizer",
        args.optimizer,
        "--burnin",
        str(args.burnin),
        "--burnin-multiplier",
        str(args.burnin_multiplier),
        "--radial-nudge",
        str(args.radial_nudge),
    ]
    if args.curriculum:
        train_cmd.append("--curriculum")
        train_cmd.extend(["--curriculum-phases", args.curriculum_phases])
    if args.epoch_fraction < 1.0:
        train_cmd.extend(["--epoch-fraction", str(args.epoch_fraction)])
    if args.depth_scale_margin:
        train_cmd.append("--depth-scale-margin")
        train_cmd.extend(["--margin-min", str(args.margin_min)])
        train_cmd.extend(["--margin-max", str(args.margin_max)])
    if args.radial_schedule != "linear":
        train_cmd.extend(["--radial-schedule", args.radial_schedule])
    if args.grad_accum_steps > 1:
        train_cmd.extend(["--grad-accum-steps", str(args.grad_accum_steps)])
    if args.amp:
        train_cmd.append("--amp")
    if args.tiered_negatives:
        train_cmd.append("--tiered-negatives")
    if args.class_balanced:
        train_cmd.append("--class-balanced")
    if args.class_weighted_loss:
        train_cmd.append("--class-weighted-loss")
    if args.euclidean_param:
        train_cmd.append("--euclidean-param")

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
            "created_at": datetime.now(timezone.utc).isoformat(),
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
                    "optimizer": args.optimizer,
                    "curriculum": args.curriculum,
                    "curriculum_phases": args.curriculum_phases if args.curriculum else None,
                    "burnin": args.burnin,
                    "burnin_multiplier": args.burnin_multiplier,
                    "radial_nudge": args.radial_nudge,
                    "epoch_fraction": args.epoch_fraction,
                    "max_pairs": args.max_pairs,
                    "depth_scale_margin": args.depth_scale_margin,
                    "margin_min": args.margin_min,
                    "margin_max": args.margin_max,
                    "radial_schedule": args.radial_schedule,
                    "tiered_negatives": args.tiered_negatives,
                    "class_balanced": args.class_balanced,
                    "class_weighted_loss": args.class_weighted_loss,
                    "euclidean_param": args.euclidean_param,
                    "clean": getattr(args, "clean", False),
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

    checkpoint_path_str = args.checkpoint or paths.get("best_checkpoint") or paths.get("checkpoint_base", "")
    if not checkpoint_path_str:
        raise SystemExit(f"❌ No checkpoint path found in metadata for tag '{args.tag}'")
    
    checkpoint_path = Path(checkpoint_path_str)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (tag_dir / checkpoint_path_str).resolve()
    else:
        checkpoint_path = checkpoint_path.resolve()

    mapping_path = Path(
        args.mapping or paths.get("mapping", "")
    ).resolve()

    if not checkpoint_path.exists():
        raise SystemExit(f"❌ Checkpoint not found: {checkpoint_path}")
    if not mapping_path.exists():
        raise SystemExit(f"❌ Mapping file not found: {mapping_path}")

    output_path = Path(args.output) if args.output else tag_dir / f"{slug}_umap.png"

    viz_script = PROJECT_ROOT / "visualize_multi_groups.py"
    viz_cmd = [
        sys.executable,
        str(viz_script),
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
    clade_name = dataset_meta.get("root_name") or dataset_meta.get("dataset_name") or args.tag.title()
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
    if args.metric != "poincare":
        viz_cmd.extend(["--metric", args.metric])

    print(f"\n🎨 Visualizing tag '{args.tag}'")
    print(f"  ↳ Checkpoint: {checkpoint_path}")
    print(f"  ↳ Mapping: {mapping_path}")
    print(f"  ↳ Output: {output_path}")
    print(f"  ↳ Command: {' '.join(viz_cmd)}")

    run_subprocess(viz_cmd)


def handle_dim(args: argparse.Namespace) -> None:
    """Handle `taxembed dim` subcommand — dimension analysis."""
    max_cosine = args.max_cosine
    embeddings = None
    n_nodes = None
    clade_name = None

    # If --tag is given, load embeddings from a trained checkpoint
    if args.tag:
        slug = slugify_tag(args.tag)
        tag_dir = ARTIFACTS_DIR / slug
        meta_path = tag_dir / "run.json"
        if not meta_path.exists():
            raise SystemExit(f"❌ No run metadata for tag '{args.tag}' ({meta_path} missing)")

        metadata = json.loads(meta_path.read_text())
        paths = metadata.get("training", {}).get("paths", {})
        dataset_meta = metadata.get("dataset", {})
        clade_name = dataset_meta.get("root_name") or args.tag

        checkpoint_path = Path(paths.get("best_checkpoint") or paths.get("checkpoint_base", ""))
        if not checkpoint_path.exists():
            raise SystemExit(f"❌ Checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        emb_weight = ckpt.get("embeddings") or ckpt.get("state_dict", {}).get("lt.weight")
        if emb_weight is None:
            raise SystemExit("❌ No embeddings found in checkpoint")

        embeddings = emb_weight.detach().numpy()
        n_nodes = embeddings.shape[0]

    # If identifier given (no tag), resolve node count from taxonomy
    if args.identifier and n_nodes is None:
        print(f"🔎 Resolving '{args.identifier}'...")
        taxid, name = resolve_taxid(args.identifier, DATA_DIR)
        clade_name = name
        print(f"  ↳ TaxID {taxid} ({name})")

        # Count nodes via TaxoPy
        taxdb = taxopy.TaxDb(taxdb_dir=str(DATA_DIR))
        parent_map = {int(c): int(p) for c, p in taxdb.taxid2parent.items()}
        from taxembed.builders.taxopy_clade import _build_children_index, _collect_clade
        children_map = _build_children_index(parent_map)
        depths, _ = _collect_clade(children_map, taxid)
        n_nodes = len(depths)

    if n_nodes is None:
        raise SystemExit("❌ Provide an identifier or --tag to analyze")

    rec = recommend_dim(n_nodes, max_cosine=max_cosine, embeddings=embeddings)

    print(f"\nDimension Analysis for {clade_name or 'taxonomy'} ({n_nodes:,} nodes)")
    print(f"  Angular packing (eps={max_cosine}): d >= {rec['angular_packing']}  [theoretical upper bound]")
    print(f"  Angular packing (eps=0.5): d >= {rec['angular_packing_relaxed']}   [relaxed]")
    print(f"  Recommended minimum: {rec['recommended']}")

    if embeddings is not None:
        pr = rec["participation_ratio"]
        cdim = rec["current_dim"]
        print(f"\n  PCA participation ratio: {pr:.1f} (current dim={cdim})")
        if pr < cdim * 0.5:
            print(f"  Suggestion: Current dim has headroom. Effective structure uses ~{pr:.0f} dimensions.")
        elif pr > cdim * 0.9:
            print(f"  Suggestion: Embeddings are saturating. Consider increasing dim to {int(pr * 1.5)}+.")
        else:
            print(f"  Suggestion: Current dim captures effective structure adequately.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="taxembed", description="Unified CLI for taxonomy embeddings")
    subparsers = parser.add_subparsers(dest="command")

    # --- build subcommand ---
    build_parser = subparsers.add_parser("build", help="Pre-build a clade dataset for reuse across training runs")
    build_parser.add_argument("identifier", help="TaxID or clade name recognized by NCBI")
    build_parser.add_argument("--max-depth", type=int, default=None, help="Limit descendant depth")
    build_parser.add_argument("--max-pairs", type=int, default=None, help="Cap total training pairs")
    build_parser.add_argument("--clean", action="store_true",
                              help="Filter taxonomy noise (sp., cf., environmental, etc.) via bottom-up leaf pruning")
    build_parser.add_argument("--force", action="store_true", help="Rebuild even if cached dataset exists")
    build_parser.set_defaults(func=handle_build)

    # --- train subcommand ---
    train_parser = subparsers.add_parser("train", help="Build dataset and train embeddings")
    train_parser.add_argument("identifier", nargs="?", help="TaxID or clade name recognized by NCBI")
    train_parser.add_argument("-as", "--as-tag", required=True, help="Tag name used to reference this run")
    train_parser.add_argument("--file", help="Path to prebuilt transitive dataset (.pkl)")
    train_parser.add_argument("--mapping", help="Mapping file (required with --file)")
    train_parser.add_argument("--dataset", help="Name of pre-built dataset (from `taxembed build`)")
    train_parser.add_argument("--max-depth", type=int, default=None, help="Limit descendant depth when building clades")
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--dim", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--n-negatives", type=int, default=50)
    train_parser.add_argument("--lr", type=float, default=0.005)
    train_parser.add_argument("--margin", type=float, default=0.2)
    train_parser.add_argument("--lambda-reg", type=float, default=0.1)
    train_parser.add_argument("--early-stopping", type=int, default=15)
    train_parser.add_argument("--gpu", type=int, default=-1, help="GPU device index (-1 for CPU)")
    train_parser.add_argument("--optimizer", choices=["radam", "adam"], default="adam",
                              help="Optimizer: adam (Euclidean, default) or radam (Riemannian Adam)")
    train_parser.add_argument("--curriculum", action="store_true",
                              help="Enable curriculum learning (shallow pairs first)")
    train_parser.add_argument("--curriculum-phases", default="1:1,20:3,50:None",
                              help='Curriculum schedule as epoch:max_depth_diff (default: "1:1,20:3,50:None")')
    train_parser.add_argument("--burnin", type=int, default=0,
                              help="Burn-in epochs with reduced LR (0 to disable)")
    train_parser.add_argument("--burnin-multiplier", type=float, default=0.1,
                              help="LR multiplier during burn-in (default: 0.1)")
    train_parser.add_argument("--radial-nudge", type=float, default=0.05,
                              help="Post-step radial correction strength (0 to disable)")
    train_parser.add_argument("--epoch-fraction", type=float, default=1.0,
                              help="Fraction of pairs to train per epoch (default: 1.0 = all)")
    train_parser.add_argument("--max-pairs", type=int, default=None,
                              help="Cap total training pairs via stratified subsampling")
    train_parser.add_argument("--depth-scale-margin", action="store_true",
                              help="Scale margin by depth_diff (tight local, strong global)")
    train_parser.add_argument("--margin-min", type=float, default=0.05,
                              help="Minimum margin for depth-scaled mode (default: 0.05)")
    train_parser.add_argument("--margin-max", type=float, default=1.0,
                              help="Maximum margin for depth-scaled mode (default: 1.0)")
    train_parser.add_argument("--radial-schedule", choices=["linear", "log"], default="linear",
                              help="Radial target schedule: linear (default) or log")
    train_parser.add_argument("--grad-accum-steps", type=int, default=1,
                              help="Gradient accumulation steps (default: 1)")
    train_parser.add_argument("--amp", action="store_true",
                              help="Enable mixed precision training (CUDA/MPS)")
    train_parser.add_argument("--tiered-negatives", action="store_true",
                              help="Use tiered negative sampling (hard/medium/easy); best for large clades")
    train_parser.add_argument("--class-balanced", action="store_true",
                              help="Class-balanced pair sampling: draw equal pairs from each top-level class per epoch")
    train_parser.add_argument("--class-weighted-loss", action="store_true",
                              help="Upweight minority class pair losses by inverse sqrt frequency")
    train_parser.add_argument("--euclidean-param", action="store_true",
                              help="Learn in R^d with tanh map to Poincare ball (fixes gradient vanishing)")
    train_parser.add_argument("--clean", action="store_true",
                              help="Filter taxonomy noise (sp., cf., environmental, etc.) via bottom-up leaf pruning")
    train_parser.set_defaults(func=handle_train)

    visualize_parser = subparsers.add_parser("visualize", help="Visualize a trained tag with UMAP")
    visualize_parser.add_argument("tag", help="Tag used during `taxembed train ... -as TAG`")
    visualize_parser.add_argument("--sample", type=int, default=25000, help="Number of points for UMAP sampling")
    visualize_parser.add_argument("--output", help="Output image path")
    visualize_parser.add_argument("--checkpoint", help="Override checkpoint path")
    visualize_parser.add_argument("--mapping", help="Override mapping path")
    visualize_parser.add_argument("--names", help="Override names.dmp path")
    visualize_parser.add_argument("--nodes", help="Override nodes.dmp path")
    visualize_parser.add_argument("--root-taxid", type=int, help="Override root TaxID for coloring")
    visualize_parser.add_argument("--children", type=int, default=0,
                                 help="Depth level for coloring (0=children, 1=grandchildren, 2=great-grandchildren, etc.)")
    visualize_parser.add_argument("--metric", choices=["euclidean", "poincare"], default="poincare",
                                 help="UMAP distance metric (default: poincare)")
    visualize_parser.set_defaults(func=handle_visualize)

    dim_parser = subparsers.add_parser("dim", help="Analyze embedding dimensionality")
    dim_parser.add_argument("identifier", nargs="?", help="TaxID or clade name (for pre-training estimate)")
    dim_parser.add_argument("--tag", help="Tag from a previous train run (for post-training analysis)")
    dim_parser.add_argument("--max-cosine", type=float, default=0.2,
                            help="Max cosine overlap for angular packing (default: 0.2)")
    dim_parser.set_defaults(func=handle_dim)

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()

