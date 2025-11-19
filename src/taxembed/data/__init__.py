"""Data downloading and processing utilities."""

from .download import (
    ensure_taxdump,
    parse_nodes_dmp,
    parse_names_dmp,
    download_taxonomy,
)
from .transitive import build_transitive_closure

__all__ = [
    "ensure_taxdump",
    "parse_nodes_dmp",
    "parse_names_dmp",
    "download_taxonomy",
    "build_transitive_closure",
]
