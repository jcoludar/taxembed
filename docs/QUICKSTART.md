# Quick Start Guide

Get started with hierarchical Poincare taxonomy embeddings in minutes.

## Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/jcoludar/taxembed.git
cd poincare-embeddings
```

2. **Install dependencies:**
```bash
make install
# or: uv sync
```

That's it! No compilation needed.

## Train a Model

The unified CLI handles everything: taxonomy download, dataset building, and training.

```bash
# Train any clade by name or TaxID
taxembed train Echinodermata -as echino_v4 --epochs 100

# For large clades (>30K nodes), increase capacity
taxembed train Mollusca -as mollusca_v5 --dim 20 --curriculum --n-negatives 100 --epochs 200
```

**Key training options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--dim` | 10 | Embedding dimensions (increase for large clades) |
| `--epochs` | 100 | Maximum training epochs |
| `--radial-nudge` | 0.05 | Post-step norm correction strength |
| `--curriculum` | off | Teach shallow pairs first (helps large trees) |
| `--n-negatives` | 50 | Negative samples per positive pair |
| `--tiered-negatives` | off | 50% hard (cousins), 30% medium (same class), 20% easy negatives |
| `--euclidean-param` | off | Parametrize in R^d with tanh map to ball (smoother gradients) |
| `--class-weighted-loss` | off | Upweight minority class pairs for balanced gradient signal |
| `--optimizer` | adam | `adam` (default, good angular clustering) or `radam` (Riemannian) |
| `--early-stopping` | 15 | Patience epochs before stopping |

## Analyze Results

```bash
# Check hierarchy quality (depth-norm correlation, class/order separation)
python scripts/analyze_hierarchy_hyperbolic.py --tag echino_v4
```

## Visualize

```bash
# Euclidean UMAP (default)
taxembed visualize echino_v4 --children 2

# Poincare distance UMAP (respects hyperbolic geometry)
taxembed visualize echino_v4 --children 2 --metric poincare
```

**Visualization options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--children` | 0 | Coloring depth (0=children, 1=grandchildren, 2=great-grandchildren) |
| `--metric` | euclidean | UMAP distance: `euclidean` or `poincare` |
| `--sample` | 25000 | Points to sample for UMAP |
| `--output` | auto | Custom output path |

## Use Pre-trained Embeddings

```python
import torch
import pandas as pd

# Load embeddings from any trained tag
ckpt = torch.load('artifacts/tags/echino_v4/echino_v4_best.pth', map_location='cpu')
embeddings = ckpt['embeddings']  # Shape: (n_nodes, dim)
epoch = ckpt['epoch']
loss = ckpt['loss']

# Load TaxID mapping
mapping = pd.read_csv(
    'data/taxopy/echino_v4/taxonomy_edges_echino_v4.mapping.tsv',
    sep='\t', header=None, names=['idx', 'taxid']
)
```

## Sizing Guide

| Clade Size | Recommended `--dim` | Notes |
|-----------|-------------------|-------|
| <10K nodes | 10 (default) | Works well out of the box |
| 10K-50K nodes | 20-50 | Add `--curriculum --tiered-negatives` |
| 50K-200K nodes | 50-100 | Add `--curriculum --tiered-negatives --n-negatives 100 --epochs 200` |
| 200K+ nodes | 100+ | Full config: `--dim 100 --curriculum --tiered-negatives --n-negatives 100 --epochs 300 --class-weighted-loss` |

## Development

```bash
make lint        # Check with ruff
make format      # Format code
make test        # Run tests
```

## Troubleshooting

### Missing taxonomy data
The `taxembed train` command auto-downloads NCBI taxonomy to `data/`. If it fails, download manually:
```bash
python prepare_taxonomy_data.py
```

### Module not found
```bash
make install
# or: uv sync
```

## Next Steps

- See **README.md** for architecture details and experimental results
- Check `artifacts/tags/<tag>/run.json` for full metadata on any trained model
- Run `python scripts/analyze_hierarchy_hyperbolic.py --tag <tag>` for detailed quality metrics
