#!/usr/bin/env python3
"""Build transitive closure for hierarchical training."""

import sys
import os

# Add root to path to import existing script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from build_transitive_closure import main as build_main


def main():
    """Build transitive closure from taxonomy edges."""
    build_main()


if __name__ == "__main__":
    main()
