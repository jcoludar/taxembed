.PHONY: help install build lint format test clean

help:
	@echo "taxembed - Poincaré Embeddings for NCBI Taxonomy"
	@echo ""
	@echo "Available commands:"
	@echo "  make install      Install dependencies with uv"
	@echo "  make build        Build C++ extensions"
	@echo "  make lint         Check code with ruff"
	@echo "  make format       Format code with ruff"
	@echo "  make test         Run tests with pytest"
	@echo "  make clean        Remove build artifacts"
	@echo "  make help         Show this help message"

install:
	uv sync

build:
	python setup.py build_ext --inplace

lint:
	uv run ruff check src/ scripts/

lint-fix:
	uv run ruff check --fix src/ scripts/

format:
	uv run ruff format src/ scripts/

test:
	uv run pytest

test-cov:
	uv run pytest --cov=src/taxembed --cov-report=html

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.so" -delete

.DEFAULT_GOAL := help
