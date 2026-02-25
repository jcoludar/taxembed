#!/usr/bin/env python3
"""Download and prepare NCBI taxonomy data."""

import os
import sys

# Add root to path to import existing script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from prepare_taxonomy_data import main as prepare_main


def main():
    """Download NCBI taxonomy data."""
    import sys as _sys
    print("WARNING: taxembed-download is deprecated. Use 'taxembed download' instead.", file=_sys.stderr)
    prepare_main()


if __name__ == "__main__":
    main()
