# Quick Start Guide

Get started with hierarchical Poincaré taxonomy embeddings in minutes.

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

## Use Pre-trained Model

A production-ready model is included in `small_model_28epoch/`:

```python
import torch
import pandas as pd

# Load embeddings
ckpt = torch.load('small_model_28epoch/taxonomy_model_small_best.pth')
embeddings = ckpt['embeddings']  # 92,290 organisms × 10 dimensions

# Load TaxID mapping
mapping = pd.read_csv('data/taxonomy_edges_small.mapping.tsv', 
                      sep='\t', header=None, names=['idx', 'taxid'])
```

## Train New Model

### Step 1: Prepare Data (if needed)

Download NCBI taxonomy:
```bash
taxembed-download
# or: python prepare_taxonomy_data.py
```

Build transitive closure (975K training pairs):
```bash
taxembed-prepare
# or: python build_transitive_closure.py
```

### Step 2: Train

```bash
taxembed-train
# or: python train_small.py
```

Training takes ~2.5 hours on M3 Mac CPU. The script includes:
- Real-time progress bars
- Early stopping (patience=5)
- Automatic best model saving

### Step 3: Visualize

```bash
taxembed-visualize small_model_28epoch/taxonomy_model_small_best.pth
# or: python visualize_multi_groups.py small_model_28epoch/taxonomy_model_small_best.pth
```

Generates UMAP visualization with key taxonomic groups highlighted.

### Step 4: Verify

```bash
taxembed-check
# or: python final_sanity_check.py
```

Runs comprehensive validation of models and data files.

## Development

### Check Code Quality

```bash
make lint        # Check with ruff
make format      # Format code
make test        # Run tests
```

### Quick Sanity Check

```bash
make check       # Run final_sanity_check.py
make train       # Train for 1 epoch (test)
```

## Common Commands

| Task | Command |
|------|---------|
| Install dependencies | `make install` |
| Train (1 epoch test) | `make train` |
| Check code quality | `make lint` |
| Format code | `make format` |
| Run tests | `make test` |
| Sanity check | `make check` |
| Clean artifacts | `make clean` |
| Show help | `make help` |

## Troubleshooting

### Missing data files

If `train_small.py` fails with "Training data not found":
```bash
python build_transitive_closure.py
```

### Module not found

Make sure you've installed dependencies:
```bash
make install
```

Or activate the virtual environment:
```bash
source venv311/bin/activate  # if using venv
```

## Next Steps

- See **docs/** for detailed guides:
  - `docs/TRAIN_SMALL_GUIDE.md` - Training documentation
  - `docs/JOURNEY.md` - Development history
  - `docs/FINAL_STATUS.md` - Production status
- Review **README.md** for architecture details
- Check **small_model_28epoch/** for production model

## Support

For questions, see the documentation in **docs/** or open a GitHub issue.
