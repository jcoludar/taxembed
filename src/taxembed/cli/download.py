#!/usr/bin/env python3
"""Download and prepare NCBI taxonomy data."""

import sys
import os

# Add root to path to import existing script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from prepare_taxonomy_data import main as prepare_main


def main():
    """Download NCBI taxonomy data."""
    prepare_main()


if __name__ == "__main__":
    main()
