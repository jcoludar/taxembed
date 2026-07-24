"""Taxonomy Bridge — align protein-language-model embeddings to the Poincaré taxonomy space.

Two complementary directions:

* **READ** — a linear ridge maps ProtT5 protein representations into the hyperbolic tangent space of
  a trained taxonomy embedding, then places (``exp0``) and retrieves the nearest taxonomy node.
* **CLEAN** — LEACE (concept erasure) removes the taxonomy-predictive subspace from the protein
  representations, and a principal-angle overlap measures whether READ and CLEAN act on the same
  directions.

The package is a flat module layout: library primitives (``core``, ``erase``, ``splits``,
``clusters``, ``h5_io``, ``prott5``, ``device``, ``taxdump``) plus the evaluation/driver scripts
(``read_eval``, ``clean_eval``, ``build_sp_panel``, ``run_all_life`` …). Paths are configured via
``config`` and the ``TAXEMBED_*`` environment variables. See ``README.md``.
"""
