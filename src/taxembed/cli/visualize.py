#!/usr/bin/env python3
"""Visualize taxonomy embeddings with UMAP."""

import os
import sys

# Add root to path to import existing script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from visualize_multi_groups import main as viz_main


def main():
    """Create UMAP visualization of embeddings."""
    import sys as _sys
    print("WARNING: taxembed-visualize is deprecated. Use 'taxembed visualize' instead.", file=_sys.stderr)
    viz_main()


if __name__ == "__main__":
    main()
