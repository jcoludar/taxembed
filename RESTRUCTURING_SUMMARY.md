# Repository Restructuring Summary

This document summarizes the restructuring of the taxembed repository to follow cookiecutter best practices with modern Python tooling.

## What Changed

### ✅ New Structure

The repository now follows a professional Python project layout:

```
taxembed/
├── src/taxembed/          # Main package (src/ layout)
├── scripts/               # Standalone scripts
├── tests/                 # Unit tests
├── data/                  # Data directory (gitignored)
├── pyproject.toml         # Project configuration (uv)
├── ruff.toml              # Linter configuration
├── Makefile               # Convenience commands
└── docs/                  # Documentation
```

### 📦 Dependency Management: uv

**Before:** `requirements.txt` + `setup.py` + `environment.yml`

**After:** `pyproject.toml` with uv

Benefits:
- Single source of truth for dependencies
- Faster installation (uv is 10-100x faster than pip)
- Better dependency resolution
- Follows PEP 518 standards

### 🔍 Code Quality: ruff

**Before:** No linting configuration

**After:** `ruff.toml` with comprehensive linting rules

Features:
- Fast Python linter (written in Rust)
- Automatic code formatting
- Import sorting (isort)
- Detects common bugs
- 100-character line limit

### 📚 Documentation

**New files:**
- `README.md` - Comprehensive project documentation
- `QUICKSTART.md` - Get started in minutes
- `STRUCTURE.md` - Project organization guide
- `CONTRIBUTING.md` - Updated contribution guidelines
- `Makefile` - Convenient command shortcuts

### 🧪 Testing

**New structure:**
- `tests/` directory for unit tests
- pytest configuration in `pyproject.toml`
- Coverage reporting support

## Migration Guide

### For Users

**Old way:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python setup.py build_ext --inplace
python embed.py -dset data/taxonomy_edges.mapped.edgelist ...
```

**New way:**
```bash
uv sync
python setup.py build_ext --inplace
uv run python scripts/train.py --dataset data/taxonomy_edges.mapped.edgelist ...
```

Or use the Makefile:
```bash
make install
make build
uv run python scripts/train.py --dataset data/taxonomy_edges.mapped.edgelist ...
```

### For Developers

**Old way:**
```bash
# No linting, no tests, no structure
```

**New way:**
```bash
# Code quality
make lint          # Check code
make format        # Fix code style

# Testing
make test          # Run tests
make test-cov      # With coverage

# Development
make clean         # Clean build artifacts
```

## Key Improvements

### 1. **Cleaner Organization**
- Source code in `src/taxembed/` (src/ layout)
- Scripts in `scripts/` directory
- Tests in `tests/` directory
- Configuration files at root

### 2. **Better Dependency Management**
- Single `pyproject.toml` file
- Clear separation of core vs. optional dependencies
- Faster installation with uv

### 3. **Code Quality**
- Automatic linting with ruff
- Code formatting enforcement
- Import sorting
- Bug detection

### 4. **Improved Documentation**
- Clear README with examples
- Quick start guide
- Project structure documentation
- Updated contribution guidelines

### 5. **Developer Experience**
- Makefile for common tasks
- Convenient `uv run` commands
- Pytest integration
- Coverage reporting

## Backward Compatibility

The original `hype/` package and root-level scripts remain in place for backward compatibility. You can still use:
```bash
python embed.py ...
python prepare_taxonomy_data.py ...
```

However, we recommend migrating to the new structure:
```bash
uv run python scripts/train.py ...
uv run python scripts/prepare_data.py ...
```

## Configuration Details

### pyproject.toml

```toml
[project]
name = "taxembed"
version = "0.1.0"
requires-python = ">=3.8"
dependencies = [
    "torch>=2.0.0",
    "numpy>=1.21.0,<2.0",
    # ... more dependencies
]

[project.optional-dependencies]
dev = ["ruff>=0.1.0", "pytest>=7.0.0"]
viz = ["matplotlib>=3.5.0", "umap-learn>=0.5.0"]
```

### ruff.toml

```toml
line-length = 100
target-version = "py38"

[lint]
select = ["E", "W", "F", "I", "C", "B", "UP"]
ignore = ["E501", "W503", "E203"]
```

## Next Steps

1. **Update your workflow:**
   - Use `uv sync` instead of `pip install`
   - Use `uv run` to execute scripts
   - Use `make` commands for common tasks

2. **Set up linting in your IDE:**
   - Configure ruff in VS Code, PyCharm, etc.
   - Enable auto-formatting on save

3. **Add tests:**
   - Create test files in `tests/` directory
   - Run `make test` to verify

4. **Update CI/CD:**
   - Use `uv sync` in your CI pipeline
   - Add `make lint` and `make test` steps

## Questions?

Refer to:
- `README.md` - Project overview and usage
- `QUICKSTART.md` - Get started quickly
- `STRUCTURE.md` - Detailed project organization
- `CONTRIBUTING.md` - Development guidelines
