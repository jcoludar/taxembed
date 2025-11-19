#!/usr/bin/env python3
"""Check and validate trained models."""

from taxembed.validation import run_checks


def main():
    """Run comprehensive sanity checks."""
    run_checks()


if __name__ == "__main__":
    main()
