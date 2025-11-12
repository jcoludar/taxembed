#!/usr/bin/env python3
"""Visualize taxonomy embeddings with UMAP."""

import sys
import os
import argparse

# Add root to path to import existing script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from visualize_multi_groups import main as viz_main


def main():
    """Create UMAP visualization of embeddings."""
    viz_main()


if __name__ == "__main__":
    main()
