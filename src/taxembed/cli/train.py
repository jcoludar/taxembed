#!/usr/bin/env python3
"""Train hierarchical Poincaré embeddings."""

import sys
import os

# Add root to path to import existing script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from train_small import main as train_main


def main():
    """Train hierarchical model on taxonomy data."""
    train_main()


if __name__ == "__main__":
    main()
