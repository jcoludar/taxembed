# Scripts Guide

This document describes all scripts in the repository and their purposes.

## Core Training Scripts

### `embed.py`
**Purpose:** Main training script for Poincaré embeddings

**Usage:**
```bash
python embed.py \
  -dset data/taxonomy_edges_small.mapped.edgelist \
  -checkpoint model.pth \
  -dim 10 -epochs 50 -negs 50 -burnin 10 \
  -batchsize 32 -model distance -manifold poincare \
  -lr 0.1 -gpu -1 -ndproc 1 -train_threads 1 \
  -eval_each 999999 -fresh
```

**Key Parameters:**
- `-dset`: Input edge list file
- `-checkpoint`: Output checkpoint file
- `-dim`: Embedding dimension
- `-epochs`: Number of training epochs
- `-fresh`: Start fresh training (don't resume)

## Data Preparation Scripts

### `prepare_taxonomy_data.py`
**Purpose:** Parse NCBI taxonomy and create edge lists

**Usage:**
```bash
python prepare_taxonomy_data.py
```

**Output:**
- `data/taxonomy_edges.csv` - CSV format with header
- `data/taxonomy_edges.edgelist` - Edgelist format (no header)

### `remap_edges.py`
**Purpose:** Remap TaxIDs to sequential indices for training

**Usage:**
```bash
python remap_edges.py data/taxonomy_edges.edgelist
```

**Output:**
- `data/taxonomy_edges.mapped.edgelist` - Sequential indices
- `data/taxonomy_edges.mapping.tsv` - TaxID to index mapping

## Visualization & Analysis

### `scripts/visualize_embeddings.py` ⭐
**Purpose:** Universal visualization tool for any checkpoint

**Usage:**
```bash
# Basic visualization
python scripts/visualize_embeddings.py model.pth

# Highlight primates
python scripts/visualize_embeddings.py model.pth --highlight primates

# Only show mammals
python scripts/visualize_embeddings.py model.pth --only mammals

# Custom sample size
python scripts/visualize_embeddings.py model.pth --highlight bacteria --sample 50000
```

**Features:**
- Works with any checkpoint
- Highlight taxonomic groups (primates, mammals, bacteria, etc.)
- Nearest neighbor analysis
- UMAP projections
- Automatic output naming

**Supported groups:**
- primates, mammals, vertebrates
- bacteria, archaea, fungi, plants
- insects, rodents

### `monitor_training.py`
**Purpose:** Real-time training monitoring

**Usage:**
```bash
# In separate terminal during training
python monitor_training.py
```

**Output:** Shows clustering quality metrics in real-time

## Evaluation Scripts

### `evaluate_full.py`
**Purpose:** Evaluate embeddings and compute metrics

**Usage:**
```bash
python evaluate_full.py <checkpoint.pth> <mapping.tsv>
```

**Output:**
- Nearest neighbors for key organisms
- UMAP projection
- Reconstruction metrics

### `evaluate_and_visualize.py`
**Purpose:** Combined evaluation and visualization

**Usage:**
```bash
python evaluate_and_visualize.py --checkpoint model.pth
```

## Utility Scripts

### `scripts/validate_data.py` ⭐
**Purpose:** Validate data quality

**Usage:**
```bash
python scripts/validate_data.py small   # Validate small dataset
python scripts/validate_data.py full    # Validate full dataset
```

**Checks:**
- No header lines in edgelists
- All values are numeric
- Mapping consistency
- Sequential indices

### `scripts/regenerate_data.sh`
**Purpose:** Regenerate all data files from NCBI taxonomy

**Usage:**
```bash
./scripts/regenerate_data.sh
```

**Steps:**
1. Parse NCBI taxonomy
2. Create small subset
3. Remap edges (full and small)
4. Validate all data

### `scripts/cleanup_repo.sh` ⭐
**Purpose:** Clean up repository (remove checkpoints, logs, etc.)

**Usage:**
```bash
./scripts/cleanup_repo.sh
```

**Removes:**
- All checkpoint files (*.pth, *.pth.*)
- Log files (*.log)
- Visualization files (*.png)
- Redundant scripts
- Temporary files

### `nn_demo.py`
**Purpose:** Quick demo of nearest neighbors

**Usage:**
```bash
python nn_demo.py
```

## Deprecated Scripts (Removed by Cleanup)

These scripts were consolidated into `scripts/visualize_embeddings.py`:
- ❌ `visualize_primates.py`
- ❌ `visualize_primates_proper.py`
- ❌ `visualize_primates_small_only.py`
- ❌ `visualize_by_taxonomy.py`
- ❌ `visualize_trained_small_dataset.py`

Old shell scripts (replaced by proper scripts):
- ❌ `train-mammals.sh`
- ❌ `train-nouns.sh`
- ❌ `train_taxonomy.sh`
- ❌ `train_taxonomy_quick.sh`

## Recommended Workflow

### 1. Initial Setup
```bash
# Install dependencies
make install
make build

# Validate data
python scripts/validate_data.py small
```

### 2. Training
```bash
# Train on small dataset
python embed.py \
  -dset data/taxonomy_edges_small.mapped.edgelist \
  -checkpoint model_small.pth \
  -dim 10 -epochs 50 -negs 50 -burnin 10 \
  -batchsize 32 -model distance -manifold poincare \
  -lr 0.1 -gpu -1 -ndproc 1 -train_threads 1 \
  -eval_each 999999 -fresh
```

### 3. Visualization
```bash
# Highlight primates
python scripts/visualize_embeddings.py model_small.pth --highlight primates

# Only show mammals
python scripts/visualize_embeddings.py model_small.pth --only mammals
```

### 4. Cleanup (when done)
```bash
# Remove checkpoints and temp files
./scripts/cleanup_repo.sh
```

## Script Organization

```
taxembed/
├── embed.py                          # Main training (root)
├── prepare_taxonomy_data.py          # Data prep (root)
├── remap_edges.py                    # Data remapping (root)
├── monitor_training.py               # Training monitor (root)
├── evaluate_full.py                  # Evaluation (root)
├── evaluate_and_visualize.py         # Combined eval (root)
├── nn_demo.py                        # Quick demo (root)
│
└── scripts/                          # Organized utility scripts
    ├── visualize_embeddings.py       # ⭐ Universal visualization
    ├── validate_data.py              # ⭐ Data validation
    ├── regenerate_data.sh            # Data regeneration
    ├── cleanup_repo.sh               # ⭐ Repository cleanup
    ├── prepare_data.py               # Wrapper
    ├── remap_data.py                 # Wrapper
    ├── monitor.py                    # Wrapper
    ├── evaluate.py                   # Wrapper
    └── train.py                      # Wrapper (needs fixing)
```

## Quick Reference

| Task | Command |
|------|---------|
| **Train model** | `python embed.py -dset <edgelist> -checkpoint <output.pth> ...` |
| **Visualize** | `python scripts/visualize_embeddings.py <checkpoint.pth>` |
| **Highlight group** | `python scripts/visualize_embeddings.py <checkpoint.pth> --highlight primates` |
| **Validate data** | `python scripts/validate_data.py small` |
| **Clean repo** | `./scripts/cleanup_repo.sh` |
| **Regenerate data** | `./scripts/regenerate_data.sh` |

## Tips

1. **Always validate data** before training: `python scripts/validate_data.py small`
2. **Use the universal visualization tool**: `scripts/visualize_embeddings.py` works with any checkpoint
3. **Clean up regularly**: Remove old checkpoints with `./scripts/cleanup_repo.sh`
4. **Monitor training**: Use `monitor_training.py` in a separate terminal
5. **Check nearest neighbors**: Add `--nearest 10` to visualization commands
