#!/usr/bin/env python3
"""Download and prepare NCBI taxonomy data."""

from taxembed.data import download_taxonomy


def main():
    """Download NCBI taxonomy data."""
    download_taxonomy()


if __name__ == "__main__":
    main()
