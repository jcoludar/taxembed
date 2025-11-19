# User Guide

Comprehensive guide to using taxembed for hierarchical taxonomy embeddings.

## Installation

### Using uv (recommended)

```bash
git clone https://github.com/yourusername/taxembed.git
cd taxembed
uv sync
```

### Using pip

```bash
pip install -e .
```

## Quick Start

### 1. Download NCBI Taxonomy

```bash
taxembed-download
```

This downloads and extracts:

- `data/nodes.dmp` - Taxonomy structure
- `data/names.dmp` - Organism names
- `data/taxonomy_edges.edgelist` - Parent-child relationships

### 2. Prepare Training Data

```bash
taxembed-prepare
```

This builds the transitive closure (all ancestor-descendant pairs):

- Input: `data/taxonomy_edges_small.edgelist`
- Output: `data/taxonomy_edges_small_transitive.pkl` (975K training pairs)

### 3. Train Model

```bash
taxembed-train
```

Or with custom parameters:

```bash
taxembed-train --epochs 100 --dim 10 --lambda-reg 0.1
```

Training options:

- `--epochs`: Number of training epochs (default: 100)
- `--dim`: Embedding dimensionality (default: 10)
- `--batch-size`: Batch size (default: 64)
- `--lr`: Learning rate (default: 0.005)
- `--margin`: Ranking loss margin (default: 0.2)
- `--lambda-reg`: Regularization strength (default: 0.1)
- `--early-stopping`: Patience for early stopping (default: 5)

### 4. Visualize Results

```bash
taxembed-visualize taxonomy_model_small_best.pth
```

## CLI Commands

### Unified `taxembed` Command

The main entry point supports multiple subcommands:

```bash
# Train any clade by name or TaxID
taxembed train Cnidaria -as cnidaria --epochs 100

# Visualize trained model
taxembed visualize cnidaria --children 1

# Analyze hierarchy quality
taxembed analyze cnidaria_best.pth
```

### Legacy Commands

Individual commands are still available:

- `taxembed-download` - Download NCBI taxonomy
- `taxembed-prepare` - Build transitive closure
- `taxembed-train` - Train embeddings
- `taxembed-visualize` - Create UMAP visualizations
- `taxembed-check` - Run sanity checks

## Working with Custom Clades

### Build Custom Dataset

```python
from taxembed.builders import build_clade_dataset

result = build_clade_dataset(
    root_taxid=33208,  # Metazoa (animals)
    dataset_name="animals",
    output_dir="data/taxopy/animals"
)

print(f"Created dataset with {result.node_count:,} nodes")
```

### Train on Custom Data

```bash
taxembed train 33208 -as animals --epochs 100
```

Or using Python:

```python
from taxembed.cli.train import main
import sys

sys.argv = [
    'train',
    '--data', 'data/taxopy/animals/taxonomy_edges_animals_transitive.pkl',
    '--mapping', 'data/taxopy/animals/taxonomy_edges_animals.mapping.tsv',
    '--checkpoint', 'animals_model.pth',
    '--epochs', '100'
]

main()
```

## Understanding the Model

### Poincaré Ball Model

Taxembed uses hyperbolic geometry (Poincaré ball model) to represent hierarchies:

- **Center**: Root of taxonomy (Cellular organisms)
- **Boundary**: Leaf nodes (species/strains)
- **Distance from center**: Depth in hierarchy
- **Angular distance**: Similarity within level

### Training Features

1. **Transitive Closure**: Trains on ALL ancestor-descendant pairs, not just parent-child
2. **Depth-Aware Initialization**: Deeper nodes start closer to boundary
3. **Radial Regularization**: Encourages ||embedding|| ≈ f(depth)
4. **Hard Negative Sampling**: Samples cousins at same depth
5. **Depth Weighting**: Deeper pairs weighted more heavily

### Ball Constraint Enforcement

Three-layer strategy ensures 100% valid embeddings:

1. **Gradient clipping**: Prevents large updates
2. **Selective projection**: Projects updated embeddings after each batch
3. **Full projection**: Projects all embeddings at epoch end

## Advanced Usage

### Custom Training Loop

```python
import torch
from taxembed.models import HierarchicalPoincareEmbedding
from taxembed.training import HierarchicalDataLoader, train_model

# Load data
with open('data/training_data.pkl', 'rb') as f:
    training_data = pickle.load(f)

# Create model
model = HierarchicalPoincareEmbedding(
    n_nodes=10000,
    dim=10,
    max_depth=38,
    init_depth_data=idx_to_depth
)

# Create data loader
dataloader = HierarchicalDataLoader(
    training_data=training_data,
    n_nodes=10000,
    batch_size=64,
    n_negatives=50
)

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
train_model(
    model, dataloader, optimizer,
    n_epochs=100,
    idx_to_depth=idx_to_depth,
    max_depth=38,
    device=torch.device('cpu'),
    checkpoint_base='my_model.pth'
)
```

### Loading Trained Embeddings

```python
import torch
import pandas as pd

# Load checkpoint
ckpt = torch.load('taxonomy_model_small_best.pth')
embeddings = ckpt['embeddings']  # Shape: (n_nodes, dim)

# Load TaxID mapping
mapping = pd.read_csv('data/taxonomy_edges_small.mapping.tsv',
                     sep='\t', header=None, names=['idx', 'taxid'])

# Get embedding for specific TaxID
taxid = 9606  # Homo sapiens
idx = mapping[mapping['taxid'] == str(taxid)]['idx'].iloc[0]
human_embedding = embeddings[idx]
```

### Nearest Neighbors

```python
import torch

def find_nearest_neighbors(query_idx, embeddings, model, k=10):
    """Find k nearest neighbors in hyperbolic space."""
    query_emb = embeddings[query_idx].unsqueeze(0)
    all_embs = embeddings

    # Compute Poincaré distances
    distances = model.poincare_distance(
        query_emb.expand(len(embeddings), -1),
        all_embs
    )

    # Get top k
    _, indices = torch.topk(distances, k, largest=False)
    return indices, distances[indices]
```

## Troubleshooting

### Out of Memory

Reduce batch size:

```bash
taxembed-train --batch-size 32
```

### Training Too Slow

Increase batch size and reduce negatives:

```bash
taxembed-train --batch-size 128 --n-negatives 25
```

### Poor Hierarchy Quality

- Train longer (100+ epochs)
- Increase regularization: `--lambda-reg 0.2`
- Try larger embedding dimension: `--dim 20`

### Embeddings Outside Ball

This shouldn't happen with current implementation. If it does:

- Check for NaN/Inf in data
- Reduce learning rate: `--lr 0.001`
- Increase regularization

## Best Practices

1. **Start with small dataset**: Test on `taxonomy_edges_small` first
2. **Monitor metrics**: Watch for decreasing loss and stable norms
3. **Use early stopping**: Prevents overfitting (default: 5 epochs patience)
4. **Save checkpoints**: Models save best checkpoint automatically
5. **Validate results**: Use `taxembed-check` to verify data quality

## Performance

### Small Dataset (111K nodes)

- Training time: ~3 min/epoch on M3 Mac CPU
- Recommended epochs: 50-100
- Expected loss: 0.47 after 28 epochs

### Full Dataset (2.7M nodes)

- Training time: ~60 min/epoch on M3 Mac CPU
- Recommended epochs: 20-50
- Memory: ~8GB RAM

## Development

### Code Quality Tools

The project uses modern Python tooling to maintain high code quality:

#### Linting with Ruff

Ruff provides fast linting and formatting:

```bash
# Check for linting issues
uv run ruff check .

# Auto-fix safe issues
uv run ruff check --fix .

# Auto-fix including unsafe fixes (e.g., remove unused imports)
uv run ruff check --fix --unsafe-fixes .

# Show detailed error messages
uv run ruff check --output-format=full .
```

#### Formatting with Ruff

Ensure consistent code style:

```bash
# Format all Python files
uv run ruff format .

# Check formatting without applying changes
uv run ruff format --check .
```

#### Type Checking with MyPy

Static type analysis catches bugs early:

```bash
# Check all source code
uv run mypy src/taxembed

# Check specific module
uv run mypy src/taxembed/models/

# Show error codes for better understanding
uv run mypy --show-error-codes src/taxembed
```

The project uses **gradual typing**, meaning type hints are being added incrementally. Most modules are currently exempt from strict type checking (see `pyproject.toml`).

#### Testing with Pytest

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

#### Complete Quality Check

Run all checks at once before committing:

```bash
uv run ruff check . && \
uv run ruff format --check . && \
uv run mypy src/taxembed && \
uv run pytest
```

### Configuration

All tools are configured in `pyproject.toml`:

- **Ruff**: Line length 100, Python 3.11+, comprehensive rule set
- **MyPy**: Strict typing with gradual adoption
- **Pytest**: Coverage reporting enabled

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed development guidelines.

---

## References

- [Poincaré Embeddings Paper](https://arxiv.org/abs/1705.08039)
- [NCBI Taxonomy](https://www.ncbi.nlm.nih.gov/taxonomy)
- [Examples](../examples/)
