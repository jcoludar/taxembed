"""Verify local .venv has all deps for analyze_hierarchy_hyperbolic.py."""
import importlib
import sys

mods = ["torch", "pandas", "numpy", "scipy", "taxopy", "matplotlib"]
missing = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", "?")
        print(f"  OK  {m} {ver}")
    except ImportError as e:
        missing.append(m)
        print(f"  MISS {m}: {e}")

if missing:
    print(f"\nMissing: {missing}")
    sys.exit(1)
print("\nAll deps present.")
