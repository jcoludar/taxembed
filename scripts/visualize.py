#!/usr/bin/env python3
"""Visualize embeddings using UMAP.

Creates 2D projections of the learned embeddings for visualization.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import and run the original script
from evaluate_and_visualize import main

if __name__ == "__main__":
    main()
