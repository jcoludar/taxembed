# Quick Start Guide

Get started with taxembed in minutes.

## Prerequisites

- Python 3.8 or higher
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/jcoludar/taxembed.git
cd taxembed
```

2. **Install dependencies:**
```bash
make install
# or: uv sync
```

3. **Build C++ extensions:**
```bash
make build
# or: python setup.py build_ext --inplace
```

## Training

### Step 1: Prepare Data

Download and process NCBI taxonomy data:
```bash
uv run python scripts/prepare_data.py
```

Remap taxonomy IDs to sequential indices:
```bash
uv run python scripts/remap_data.py
```

### Step 2: Train Model

Start training with default parameters:
```bash
uv run python scripts/train.py \
  --dataset data/taxonomy_edges.mapped.edgelist \
  --checkpoint taxonomy_model.pth \
  --dim 10 \
  --epochs 50 \
  --negs 50 \
  --burnin 10 \
  --batchsize 32 \
  --model distance \
  --manifold poincare \
  --lr 0.1 \
  --ndproc 1 \
  --eval-each 999999 \
  --fresh
```

### Step 3: Monitor Training (Optional)

In another terminal, monitor training progress:
```bash
uv run python scripts/monitor.py
```

### Step 4: Evaluate Results

After training completes:
```bash
uv run python scripts/evaluate.py --checkpoint taxonomy_model.pth
```

### Step 5: Visualize Embeddings

Create 2D visualizations:
```bash
uv run python scripts/visualize.py --checkpoint taxonomy_model.pth
```

## Code Quality

### Check Code Style

```bash
make lint
# or: uv run ruff check src/ scripts/
```

### Fix Code Style Issues

```bash
make format
# or: uv run ruff format src/ scripts/
```

### Run Tests

```bash
make test
# or: uv run pytest
```

## Common Commands

| Task | Command |
|------|---------|
| Install dependencies | `make install` |
| Build extensions | `make build` |
| Check code quality | `make lint` |
| Format code | `make format` |
| Run tests | `make test` |
| Clean build artifacts | `make clean` |
| Show help | `make help` |

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'hype'"

**Solution:** Make sure you've installed dependencies and built extensions:
```bash
make install
make build
```

### Issue: "ruff: command not found"

**Solution:** Use `uv run` to execute commands in the project environment:
```bash
uv run ruff check src/
```

### Issue: "CUDA out of memory"

**Solution:** Use CPU training with `-gpu -1`:
```bash
uv run python scripts/train.py ... -gpu -1
```

### Issue: "Slow data loading"

**Solution:** Reduce number of data loading processes:
```bash
uv run python scripts/train.py ... -ndproc 1
```

## Next Steps

- Read [STRUCTURE.md](STRUCTURE.md) for project organization
- Check [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines
- Review [README.md](README.md) for detailed documentation

## Support

For issues and questions, please open a GitHub issue or refer to the documentation.
