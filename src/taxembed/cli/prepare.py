#!/usr/bin/env python3
"""Build transitive closure for hierarchical training."""

import os
import sys

# Add root to path to import existing script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from build_transitive_closure import main as build_main


def main():
    """Build transitive closure from taxonomy edges."""
    import sys as _sys
    print("WARNING: taxembed-prepare is deprecated. Use 'taxembed prepare' instead.", file=_sys.stderr)
    build_main()


if __name__ == "__main__":
    main()
