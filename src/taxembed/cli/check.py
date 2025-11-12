#!/usr/bin/env python3
"""Check and validate trained models."""

import sys
import os

# Add root to path to import existing script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from final_sanity_check import main as check_main


def main():
    """Run comprehensive sanity checks."""
    check_main()


if __name__ == "__main__":
    main()
