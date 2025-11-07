#!/usr/bin/env python3
"""Prepare NCBI taxonomy data for embedding.

Downloads and processes NCBI taxonomy data into edge list format.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import and run the original script
from prepare_taxonomy_data import main

if __name__ == "__main__":
    main()
