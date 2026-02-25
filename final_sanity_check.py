#!/usr/bin/env python3
"""
Final sanity check before commit.
Validates all core files, package structure, and data integrity.
"""

import os
import sys


def check_file_exists(path, description):
    """Check if file exists."""
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  ✅ {description}: {path} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"  ❌ {description} MISSING: {path}")
        return False


def check_dir_exists(path, description):
    """Check if directory exists."""
    if os.path.isdir(path):
        print(f"  ✅ {description}: {path}/")
        return True
    else:
        print(f"  ❌ {description} MISSING: {path}/")
        return False


def main():
    print("=" * 80)
    print("FINAL SANITY CHECK")
    print("=" * 80)
    print()

    all_passed = True

    # 1. Core training scripts (root-level entry points)
    print("1️⃣  Core Training Scripts")
    all_passed &= check_file_exists("train_small.py", "Main training script")
    all_passed &= check_file_exists("train_hierarchical.py", "Hierarchical model")
    all_passed &= check_file_exists("visualize_multi_groups.py", "Visualization")
    all_passed &= check_file_exists("build_transitive_closure.py", "Transitive closure builder")
    all_passed &= check_file_exists("prepare_taxonomy_data.py", "Data preparation")
    print()

    # 2. Documentation
    print("2️⃣  Documentation")
    all_passed &= check_file_exists("README.md", "Main README")
    all_passed &= check_file_exists("docs/QUICKSTART.md", "Quick start guide")
    all_passed &= check_file_exists("docs/TRAIN_SMALL_GUIDE.md", "Training guide")
    all_passed &= check_file_exists("docs/JOURNEY.md", "Development history")
    all_passed &= check_file_exists("docs/CLI_COMMANDS.md", "CLI reference")
    print()

    # 3. Package structure
    print("3️⃣  Package Structure")
    all_passed &= check_file_exists("pyproject.toml", "Package config")
    all_passed &= check_dir_exists("src/taxembed", "taxembed package")
    all_passed &= check_file_exists("src/taxembed/cli/main.py", "CLI entry point")
    all_passed &= check_dir_exists("tests", "Test suite")
    all_passed &= check_dir_exists("scripts", "User scripts")
    all_passed &= check_file_exists(
        "scripts/analyze_hierarchy_hyperbolic.py", "Hierarchy analysis"
    )
    print()

    # 4. Data files (only if data/ exists — gitignored, may not be present)
    print("4️⃣  Data Files")
    if os.path.isdir("data"):
        check_file_exists("data/names.dmp", "NCBI names")
        check_file_exists("data/nodes.dmp", "NCBI nodes")
        print("  (data/ is gitignored — missing files are OK on fresh clone)")
    else:
        print("  ⏭️  data/ not present (gitignored, run 'taxembed download' to fetch)")
    print()

    # 5. Verify no stale files remain
    print("5️⃣  Cleanup Verification")
    stale_files = [
        "embed.py",
        "nn_demo.py",
        "requirements.txt",
        "ruff.toml",
        "sanity_check.py",
        "remap_edges.py",
        "check_model.py",
        "check_dataset_composition.py",
        "analyze_hierarchy.py",
        "QUICKSTART.md",
        "PROGRESS.md",
        "TRAINING_LOG.md",
        "FINAL_STATUS.md",
        "CLEANUP_SUMMARY.md",
    ]
    stale_dirs = [
        "hype",
        "src/taxembed/manifolds",
        "src/taxembed/models",
        "src/taxembed/datasets",
    ]

    cleanup_ok = True
    for f in stale_files:
        if os.path.exists(f):
            print(f"  ⚠️  Stale file still present: {f}")
            cleanup_ok = False
    for d in stale_dirs:
        if os.path.isdir(d):
            print(f"  ⚠️  Stale directory still present: {d}/")
            cleanup_ok = False

    if cleanup_ok:
        print("  ✅ No stale files found (good!)")
    print()

    # Final verdict
    print("=" * 80)
    if all_passed and cleanup_ok:
        print("✅ ALL CHECKS PASSED - Repository is ready for commit!")
    else:
        print("❌ SOME CHECKS FAILED - Please review above")
        sys.exit(1)
    print("=" * 80)


if __name__ == "__main__":
    main()
