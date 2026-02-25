# Legacy Facebook Research Code

This directory contains the original Poincar&eacute; embeddings implementation from
[Facebook Research](https://github.com/facebookresearch/poincare-embeddings)
(Nickel & Kiela, 2017).

**Status:** Archived. Not imported by modern training code (`train_small.py`,
`train_hierarchical.py`, or the `taxembed` CLI).

## Contents

- `hype/` &mdash; Core library: manifolds (Poincar&eacute;, Lorentz, Euclidean),
  graph datasets, Riemannian SGD, Cython extensions.
- `embed.py` &mdash; Legacy entry point that wraps `hype/`.

## Why archived

The current v9d/v10a architecture uses a custom training loop with Euclidean Adam,
radial nudge, depth-aware initialization, and transitive-closure training pairs.
None of these features exist in the original Facebook code. Keeping `hype/` at the
repo root confused linters and new contributors.

These files are preserved for historical reference and to credit the original work.
