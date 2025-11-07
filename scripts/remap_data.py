#!/usr/bin/env python3
"""Remap taxonomy IDs in edge list.

Converts original taxonomy IDs to sequential indices for efficient training.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import and run the original script
from remap_edges import main

if __name__ == "__main__":
    main()
