.PHONY: help install lint format test clean train check

help:
	@echo "taxembed - Hierarchical Poincaré Embeddings for Taxonomy"
	@echo ""
	@echo "Available commands:"
	@echo "  make install      Install dependencies with uv"
	@echo "  make lint         Check code with ruff"
	@echo "  make format       Format code with ruff"
	@echo "  make test         Run tests with pytest"
	@echo "  make train        Train small model (quick test)"
	@echo "  make check        Run sanity checks"
	@echo "  make clean        Remove build artifacts"
	@echo ""
	@echo "📖 See docs/ for detailed guides"

install:
	@echo "Installing dependencies with uv..."
	uv sync
	@echo "✅ Installation complete"

install-dev:
	@echo "Installing with dev dependencies..."
	uv sync --all-extras
	@echo "✅ Dev installation complete"

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

train:
	@echo "Training small model for 1 epoch (sanity check)..."
	python train_small.py --epochs 1
	@echo "✅ Quick training test complete"

check:
	@echo "Running sanity checks..."
	python final_sanity_check.py
	@echo "✅ Sanity checks passed"

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.so" -delete
	@echo "✅ Cleanup complete"

.DEFAULT_GOAL := help
