# Contributing to taxembed

We want to make contributing to this project as easy and transparent as possible.

## Development Setup

1. Clone the repository:

```bash
git clone https://github.com/jcoludar/taxembed.git
cd taxembed
```

2. Install development dependencies using uv:

```bash
uv sync
```

## Pull Requests

We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests in the `tests/` directory.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes: `uv run pytest`
5. Make sure your code passes linting: `uv run ruff check src/ scripts/`
6. Format your code: `uv run ruff format src/ scripts/`

## Code Quality

This project maintains high code quality standards using modern Python tools:

- **Ruff**: Fast linting and formatting
- **MyPy**: Static type checking
- **Pytest**: Comprehensive test suite

### Linting with Ruff

Check for linting issues:

```bash
uv run ruff check .
```

Fix linting issues automatically:

```bash
# Safe fixes only
uv run ruff check --fix .

# Include unsafe fixes (e.g., unused imports)
uv run ruff check --fix --unsafe-fixes .
```

View detailed error explanations:

```bash
uv run ruff check --output-format=full .
```

### Formatting with Ruff

Format code to match project style:

```bash
# Format all Python files
uv run ruff format .

# Check formatting without making changes
uv run ruff format --check .
```

### Type Checking with MyPy

Run static type analysis:

```bash
# Check all source files
uv run mypy src/taxembed

# Check specific module
uv run mypy src/taxembed/models/

# Show more detailed error messages
uv run mypy --show-error-codes src/taxembed
```

**Note:** This project uses gradual typing. Most modules are currently exempt from strict type checking (see `pyproject.toml`). When adding type hints to a module, remove it from the `[[tool.mypy.overrides]]` section.

### Testing

Run the test suite:

```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=src/taxembed --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_models.py

# Run with verbose output
uv run pytest -v
```

### Complete Quality Check

Run all quality checks at once:

```bash
# Lint, format check, type check, and test
uv run ruff check . && \
uv run ruff format --check . && \
uv run mypy src/taxembed && \
uv run pytest
```

## Coding Style

- **Follow PEP 8 guidelines** (enforced by Ruff)
- **Use type hints** for all new code (function signatures and return types)
- **Write docstrings** for all public functions and classes (Google style preferred)
- **Keep lines under 100 characters** (enforced by Ruff formatter)
- **Use meaningful variable and function names** (avoid single letters except in loops)
- **Prefer explicit over implicit** (e.g., `zip(a, b, strict=True)`)

### Type Hints Guidelines

```python
# Good: Complete type hints
def process_data(
    input_path: Path,
    batch_size: int = 32,
    verbose: bool = False,
) -> dict[str, np.ndarray]:
    """Process data from file.

    Args:
        input_path: Path to input file
        batch_size: Number of items per batch
        verbose: Enable verbose logging

    Returns:
        Dictionary mapping names to arrays
    """
    ...

# Bad: Missing type hints
def process_data(input_path, batch_size=32, verbose=False):
    ...
```

## Issues

We use GitHub issues to track public bugs. Please ensure your description is clear and has sufficient instructions to reproduce the issue.

## License

By contributing to taxembed, you agree that your contributions will be licensed under the CC-BY-NC 4.0 license found in the LICENSE file in the root directory of this source tree.
