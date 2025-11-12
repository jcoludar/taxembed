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

3. Build C++ extensions:
```bash
uv run python setup.py build_ext --inplace
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

This project uses **ruff** for linting and code formatting.

### Linting

Check for linting issues:
```bash
uv run ruff check src/ scripts/
```

Fix linting issues automatically:
```bash
uv run ruff check --fix src/ scripts/
```

### Formatting

Format code to match project style:
```bash
uv run ruff format src/ scripts/
```

### Testing

Run the test suite:
```bash
uv run pytest
```

With coverage report:
```bash
uv run pytest --cov=src/taxembed
```

## Coding Style

- Follow PEP 8 guidelines
- Use type hints where possible
- Write docstrings for all public functions and classes
- Keep lines under 100 characters (enforced by ruff)
- Use meaningful variable and function names

## Issues

We use GitHub issues to track public bugs. Please ensure your description is clear and has sufficient instructions to reproduce the issue.

## License

By contributing to taxembed, you agree that your contributions will be licensed under the CC-BY-NC 4.0 license found in the LICENSE file in the root directory of this source tree.
