# Examples

This directory contains example scripts demonstrating how to use the taxembed package.

## Available Examples

### basic_training.py

Simple example showing how to train a Poincaré embedding model on taxonomy data.

```bash
uv run python examples/basic_training.py
```

### custom_dataset.py

Example of using taxembed with custom taxonomy data.

```bash
uv run python examples/custom_dataset.py --root-taxid 33208 --name animals
```

### visualize_groups.py

Demonstrates how to create visualizations highlighting specific taxonomic groups.

```bash
uv run python examples/visualize_groups.py --checkpoint model.pth
```

### nn_demo.py

Interactive demo for finding nearest neighbors in the embedding space.

```bash
uv run python examples/nn_demo.py
```

## Running Examples

All examples can be run using `uv`:

```bash
# Install dependencies first
uv sync

# Run any example
uv run python examples/<example_name>.py
```

## Learning Path

1. Start with `basic_training.py` to understand the training workflow
2. Try `custom_dataset.py` to work with specific taxonomic clades
3. Use `visualize_groups.py` to analyze your trained models
4. Explore `nn_demo.py` for interactive analysis
