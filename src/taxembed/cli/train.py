#!/usr/bin/env python3
"""Train hierarchical Poincaré embeddings."""

import os
import sys

# Add root to path to import existing script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from train_small import main as train_main


def main():
    """Train hierarchical model on taxonomy data."""
    import sys as _sys
    print("WARNING: taxembed-train is deprecated. Use 'taxembed train' instead.", file=_sys.stderr)
    train_main()


if __name__ == "__main__":
    main()
