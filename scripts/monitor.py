#!/usr/bin/env python3
"""Monitor training progress in real-time.

Displays clustering quality metrics during training.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import and run the original script
from monitor_training import main

if __name__ == "__main__":
    main()
