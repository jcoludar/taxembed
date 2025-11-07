# Project Structure

This document describes the organization of the taxembed project following cookiecutter best practices.

## Directory Layout

```
taxembed/
├── src/
│   └── taxembed/                    # Main package (namespace package)
│       ├── __init__.py              # Package initialization
│       ├── manifolds/               # Hyperbolic manifold implementations
│       │   ├── __init__.py
│       │   ├── poincare.py          # Poincaré manifold
│       │   └── euclidean.py         # Euclidean manifold
│       ├── models/                  # Embedding models
│       │   ├── __init__.py
│       │   ├── base.py              # Base model class
│       │   └── distance.py          # Distance-based models
│       ├── datasets/                # Data loading and processing
│       │   ├── __init__.py
│       │   ├── graph.py             # Graph dataset utilities
│       │   └── loaders.py           # Data loaders
│       └── utils/                   # Utility functions
│           ├── __init__.py
│           ├── checkpoint.py        # Checkpoint management
│           ├── metrics.py           # Evaluation metrics
│           └── visualization.py     # Visualization utilities
│
├── scripts/                         # Standalone executable scripts
│   ├── train.py                     # Main training script
│   ├── prepare_data.py              # Data preparation
│   ├── remap_data.py                # ID remapping
│   ├── evaluate.py                  # Model evaluation
│   ├── monitor.py                   # Training monitoring
│   └── visualize.py                 # Visualization
│
├── tests/                           # Unit and integration tests
│   ├── __init__.py
│   ├── test_example.py              # Example test
│   ├── test_manifolds.py            # Manifold tests
│   ├── test_models.py               # Model tests
│   └── test_datasets.py             # Dataset tests
│
├── data/                            # Data directory (gitignored)
│   ├── taxonomy_edges.edgelist      # Raw edge list
│   ├── taxonomy_edges.mapped.edgelist # Remapped edge list
│   └── taxonomy_edges.mapping.tsv   # ID mapping
│
├── docs/                            # Documentation
│   ├── index.md                     # Main documentation
│   ├── installation.md              # Installation guide
│   ├── usage.md                     # Usage guide
│   └── api.md                       # API reference
│
├── pyproject.toml                   # Project metadata and dependencies (uv)
├── ruff.toml                        # Ruff linter configuration
├── setup.py                         # Setup script for C++ extensions
├── README.md                        # Project README
├── CONTRIBUTING.md                  # Contribution guidelines
├── STRUCTURE.md                     # This file
├── LICENSE                          # CC-BY-NC 4.0 license
└── .gitignore                       # Git ignore rules
```

## Key Directories

### `src/taxembed/`

The main Python package containing all source code. Using the `src/` layout provides several benefits:
- Prevents accidental imports of the package from the current directory
- Makes it clear what is part of the package vs. project configuration
- Follows Python packaging best practices

### `scripts/`

Standalone executable scripts for common tasks:
- **train.py** - Main training loop for embeddings
- **prepare_data.py** - Download and process NCBI taxonomy data
- **remap_data.py** - Convert taxonomy IDs to sequential indices
- **evaluate.py** - Compute reconstruction metrics
- **monitor.py** - Real-time training monitoring
- **visualize.py** - Create 2D UMAP projections

These scripts can be run with `uv run python scripts/script_name.py`.

### `tests/`

Unit and integration tests using pytest. Tests follow the naming convention `test_*.py` and are organized by module.

### `data/`

Data directory for datasets and processed files. This directory is gitignored to avoid committing large files.

## Configuration Files

### `pyproject.toml`

Project metadata and dependency management using uv. Includes:
- Project name, version, and description
- Core and optional dependencies
- Development tools configuration
- Build system specification

### `ruff.toml`

Linter and formatter configuration. Specifies:
- Line length (100 characters)
- Target Python version (3.8+)
- Enabled/disabled rules
- Per-file rule overrides

### `setup.py`

Build configuration for C++ extensions (Cython modules). Required for compiling performance-critical code.

## Development Workflow

### Installation

```bash
uv sync                              # Install dependencies
python setup.py build_ext --inplace  # Build C++ extensions
```

### Code Quality

```bash
uv run ruff check src/ scripts/       # Lint code
uv run ruff check --fix src/ scripts/ # Auto-fix issues
uv run ruff format src/ scripts/      # Format code
```

### Testing

```bash
uv run pytest                         # Run all tests
uv run pytest --cov=src/taxembed     # With coverage
```

### Running Scripts

```bash
uv run python scripts/train.py --help
uv run python scripts/prepare_data.py
uv run python scripts/evaluate.py --checkpoint model.pth
```

## Migration Notes

This structure was created from the original flat layout:
- Original `hype/` package remains in place for backward compatibility
- New `src/taxembed/` structure provides a cleaner organization
- Scripts moved from root to `scripts/` directory
- Configuration moved to `pyproject.toml` and `ruff.toml`
- Tests organized in `tests/` directory

## Best Practices

1. **Always use `uv run`** when executing Python scripts to ensure correct environment
2. **Run linting before committing** to maintain code quality
3. **Add tests for new features** in the `tests/` directory
4. **Update documentation** when changing APIs
5. **Keep imports organized** (standard library, third-party, local)
6. **Use type hints** for better code clarity and IDE support
