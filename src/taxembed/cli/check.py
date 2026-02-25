#!/usr/bin/env python3
"""Check and validate trained models."""

import os
import sys

# Add root to path to import existing script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from final_sanity_check import main as check_main


def main():
    """Run comprehensive sanity checks."""
    import sys as _sys
    print("WARNING: taxembed-check is deprecated. Use 'taxembed check' instead.", file=_sys.stderr)
    check_main()


if __name__ == "__main__":
    main()
