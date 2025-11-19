#!/usr/bin/env python3
"""Build transitive closure for hierarchical training."""

from taxembed.data.transitive import main as build_transitive_closure


def main():
    """Build transitive closure from taxonomy edges."""
    build_transitive_closure()


if __name__ == "__main__":
    main()
