#!/usr/bin/env python3
"""Evaluate trained embeddings.

Computes reconstruction metrics and other evaluation measures.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import and run the original script
from evaluate_full import main

if __name__ == "__main__":
    main()
