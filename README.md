# Poincaré Embeddings for NCBI Taxonomy

Learning hierarchical embeddings of organisms using Poincaré geometry.

This project builds on the [PyTorch implementation of Poincaré Embeddings](https://github.com/facebookresearch/poincare-embeddings) by Nickel & Kiela (2017), adapted to embed the NCBI taxonomy dataset.

## Overview

This project learns low-dimensional embeddings of ~2.7M organisms from the NCBI taxonomy in hyperbolic space. Taxonomically related organisms (e.g., primates, mammals) are closer together in the embedding space, capturing the hierarchical structure of biological taxonomy.

## Key Features

- Embeddings of 2.7M+ organisms from NCBI taxonomy
- Hyperbolic geometry for natural hierarchy representation
- Real-time training monitoring with clustering quality metrics
- Per-epoch checkpointing for early stopping
- Fixed initialization and data loading for proper training

## Project Structure

```
taxembed/
├── src/
│   └── taxembed/              # Main package
│       ├── __init__.py
│       ├── manifolds/         # Hyperbolic manifold implementations
│       ├── models/            # Embedding models
│       ├── data/              # Data loading and processing
│       └── utils/             # Utility functions
├── scripts/                   # Standalone scripts
│   ├── prepare_taxonomy_data.py
│   ├── remap_edges.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py
├── data/                      # Data directory (gitignored)
├── tests/                     # Unit tests
├── pyproject.toml             # Project configuration (uv)
├── ruff.toml                  # Ruff linter configuration
└── README.md                  # This file
```

## Installation

### Prerequisites

- Python 3.8+
- [uv](https://github.com/astral-sh/uv) (fast Python package installer)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/jcoludar/taxembed.git
cd taxembed
```

2. Install dependencies using uv:
```bash
uv sync
```

3. Build C++ extensions:
```bash
uv run python setup.py build_ext --inplace
```

## Quick Start

### Prepare Data

```bash
uv run python scripts/prepare_taxonomy_data.py
uv run python scripts/remap_edges.py
```

### Train Model

```bash
uv run python scripts/train.py \
  --dataset data/taxonomy_edges.mapped.edgelist \
  --checkpoint taxonomy_model.pth \
  --dim 10 \
  --epochs 50 \
  --negs 50 \
  --burnin 10 \
  --batchsize 32 \
  --model distance \
  --manifold poincare \
  --lr 0.1 \
  --ndproc 1 \
  --eval-each 999999 \
  --fresh
```

### Monitor Training

In a separate terminal:
```bash
uv run python scripts/monitor_training.py
```

Shows real-time clustering quality (primate distances vs random pairs).

### Evaluate and Visualize

```bash
uv run python scripts/evaluate.py --checkpoint taxonomy_model.pth
uv run python scripts/visualize.py --checkpoint taxonomy_model.pth
```

## Development

### Code Quality

This project uses **ruff** for linting and code quality checks.

Run linting:
```bash
uv run ruff check src/ scripts/
```

Fix linting issues automatically:
```bash
uv run ruff check --fix src/ scripts/
```

Format code:
```bash
uv run ruff format src/ scripts/
```

### Running Tests

```bash
uv run pytest
```

With coverage:
```bash
uv run pytest --cov=src/taxembed
```

## Dependencies

### Core
- **PyTorch** - Deep learning framework
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **Cython** - Performance optimization
- **tqdm** - Progress bars

### Optional
- **UMAP** - Dimensionality reduction for visualization
- **Matplotlib** - Plotting and visualization
- **scikit-learn** - Machine learning utilities

### Development
- **Ruff** - Fast Python linter and formatter
- **Pytest** - Testing framework

## Configuration

### Ruff Configuration

Ruff is configured in `ruff.toml` with the following settings:

- **Line length**: 100 characters
- **Target Python version**: 3.8+
- **Enabled rules**: E, W, F, I, C, B, UP (pycodestyle, pyflakes, isort, comprehensions, bugbear, pyupgrade)
- **Ignored rules**: E501 (line too long), W503 (line break before binary operator)

### Project Configuration

All project metadata and dependencies are defined in `pyproject.toml` following PEP 518 standards.

## Documentation

- `IMPLEMENTATION_NOTES.md` - Detailed technical notes on fixes and improvements
- `TRAINING_SUMMARY.md` - Training results and metrics
- `FINAL_ASSESSMENT.md` - Quality assessment

## References

If you find this code useful for your research, please cite the following paper:

```bibtex
@incollection{nickel2017poincare,
  title = {Poincaré Embeddings for Learning Hierarchical Representations},
  author = {Nickel, Maximilian and Kiela, Douwe},
  booktitle = {Advances in Neural Information Processing Systems 30},
  editor = {I. Guyon and U. V. Luxburg and S. Bengio and H. Wallach and R. Fergus and S. Vishwanathan and R. Garnett},
  pages = {6341--6350},
  year = {2017},
  publisher = {Curran Associates, Inc.},
  url = {http://papers.nips.cc/paper/7213-poincare-embeddings-for-learning-hierarchical-representations.pdf}
}
```

## License

This code is licensed under [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
