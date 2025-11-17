# CLI Commands Reference

After installing with `uv sync`, the following CLI commands are available:

## Available Commands

### 📥 `taxembed-download`
Download and prepare NCBI taxonomy data.

```bash
taxembed-download
```

**What it does:**
- Downloads `taxdump.tar.gz` from NCBI FTP
- Extracts `nodes.dmp` and `names.dmp`
- Creates small dataset (111K organisms)
- Creates full dataset (2.7M organisms)

**Output files:**
- `data/nodes.dmp` - Taxonomy structure
- `data/names.dmp` - Organism names
- `data/taxonomy_edges_small.edgelist` - Small dataset
- `data/taxonomy_edges.edgelist` - Full dataset

---

### 🔧 `taxembed-prepare`
Build transitive closure for hierarchical training.

```bash
taxembed-prepare
```

**What it does:**
- Loads small dataset edges
- Computes all ancestor-descendant pairs
- Calculates depth information
- Saves training data in multiple formats

**Input:**
- `data/taxonomy_edges_small.mapping.tsv`
- `data/nodes.dmp`

**Output:**
- `data/taxonomy_edges_small_transitive.pkl` - 975K training pairs
- `data/taxonomy_edges_small_transitive.tsv` - TSV format
- `data/taxonomy_edges_small_transitive.edgelist` - Edge list format

**Processing time:** ~2-5 minutes

---

### 🚂 `taxembed-train`
Train hierarchical Poincaré embeddings.

```bash
taxembed-train [OPTIONS]
```

**Options:**
- `--data PATH` - Training data (default: `data/taxonomy_edges_small_transitive.pkl`)
- `--checkpoint PATH` - Output model path (default: `taxonomy_model_small.pth`)
- `--dim INT` - Embedding dimension (default: 10)
- `--epochs INT` - Max epochs (default: 100)
- `--early-stopping INT` - Patience (default: 5)
- `--batch-size INT` - Batch size (default: 64)
- `--n-negatives INT` - Negative samples (default: 50)
- `--lr FLOAT` - Learning rate (default: 0.005)
- `--margin FLOAT` - Ranking margin (default: 0.2)
- `--lambda-reg FLOAT` - Regularization (default: 0.1)

**Example:**
```bash
# Default settings
taxembed-train

# Custom parameters
taxembed-train --epochs 50 --batch-size 128 --lr 0.003
```

**What it does:**
- Loads transitive closure data
- Initializes model with depth-aware embeddings
- Trains with ranking loss + regularization
- Shows real-time progress and metrics
- Saves checkpoints and best model
- Early stopping when loss plateaus

**Training time:** ~2.5 hours (M3 Mac CPU)

**Output:**
- `taxonomy_model_small_epoch*.pth` - Periodic checkpoints
- `taxonomy_model_small_best.pth` - Best model (lowest loss)

---

### 📊 `taxembed-visualize`
Create UMAP visualization of embeddings.

```bash
taxembed-visualize MODEL_PATH [OPTIONS]
```

**Arguments:**
- `MODEL_PATH` - Path to trained model checkpoint

**Example:**
```bash
taxembed-visualize small_model_28epoch/taxonomy_model_small_best.pth
```

**What it does:**
- Loads embeddings from checkpoint
- Finds key taxonomic groups (Mammals, Birds, Insects, etc.)
- Performs stratified sampling (50K points)
- Runs UMAP dimensionality reduction
- Creates colored scatter plot

**Output:**
- `taxonomy_embeddings_multi_groups.png` - Visualization

**Processing time:** ~2-3 minutes

---

### ✅ `taxembed-check`
Run comprehensive validation checks.

```bash
taxembed-check
```

**What it checks:**
1. **Core Scripts** - Training and visualization scripts present
2. **Documentation** - README and guides available
3. **Small Model** - Production model valid (92K organisms)
4. **Animals Model** - Reference model valid (1M organisms)
5. **Data Files** - Training data and NCBI files present
6. **Cleanup** - No intermediate files remaining

**Example output:**
```
✅ Core scripts present
✅ Small model validated (loss 0.472, 100% in ball)
✅ Data files intact
✅ All checks passed
```

---

## Quick Workflow

### First Time Setup
```bash
# 1. Install
make install

# 2. Download data
taxembed-download

# 3. Prepare training data
taxembed-prepare
```

### Train and Visualize
```bash
# 4. Train model (~2.5 hours)
taxembed-train

# 5. Visualize results
taxembed-visualize taxonomy_model_small_best.pth

# 6. Verify everything
taxembed-check
```

---

## Unified Pipeline (`taxembed`)

### `taxembed train`

```bash
uv run taxembed train <TaxID-or-name> -as <tag> [options]
```

- Resolves `identifier` as either a numeric TaxID or an NCBI-recognized clade name
- Builds the dataset on the fly via TaxoPy (unless `--file` and `--mapping` are supplied)
- Invokes `train_small.py` with customizable hyperparameters (`--epochs`, `--lr`, `--margin`, `--lambda-reg`, etc.)
- Stores checkpoints and metadata under `artifacts/tags/<tag>/run.json`

Examples:

```bash
uv run taxembed train 33208 -as animals --epochs 40 --lr 0.003
uv run taxembed train "Fungi" -as fungi
uv run taxembed train --file data/custom_transitive.pkl \
    --mapping data/custom.mapping.tsv \
    -as custom_run
```

### `taxembed visualize`

```bash
uv run taxembed visualize <tag> [--sample N] [--output PATH]
```

- Reuses metadata recorded during `taxembed train`
- Runs `visualize_multi_groups.py` with the correct checkpoint + mapping
- Supports overrides for sample size, output path, or even custom checkpoint/mapping if needed

```bash
uv run taxembed visualize animals --sample 15000 --output animals_umap.png
```

- Colors default to the immediate child taxa of the trained clade (legend included automatically). Use `--root-taxid` to choose a different root, or `--names/--nodes` to point at custom taxonomy dumps.

---

## Alternative: Direct Python Scripts

All CLI commands are thin wrappers around Python scripts. You can also run:

```bash
python prepare_taxonomy_data.py       # = taxembed-download
python build_transitive_closure.py    # = taxembed-prepare
python train_small.py                 # = taxembed-train
python visualize_multi_groups.py ...  # = taxembed-visualize
python final_sanity_check.py          # = taxembed-check
python scripts/build_clade_dataset.py --root-taxid 33208 --dataset-name animals
```

`scripts/build_clade_dataset.py` taps into TaxoPy so you can materialize on-demand datasets for any clade (output stored under `data/taxopy/<name>/` with manifest + transitive closure files).

---

## Package Structure

CLI commands are defined in `pyproject.toml`:

```toml
[project.scripts]
taxembed = "taxembed.cli.main:main"
taxembed-download = "taxembed.cli.download:main"
taxembed-prepare = "taxembed.cli.prepare:main"
taxembed-train = "taxembed.cli.train:main"
taxembed-visualize = "taxembed.cli.visualize:main"
taxembed-check = "taxembed.cli.check:main"
```

Implementation files:
```
src/taxembed/cli/
├── __init__.py
├── download.py      # Wrapper for prepare_taxonomy_data.py
├── prepare.py       # Wrapper for build_transitive_closure.py
├── train.py         # Wrapper for train_small.py
├── visualize.py     # Wrapper for visualize_multi_groups.py
└── check.py         # Wrapper for final_sanity_check.py
```

---

## Benefits

1. **Discoverable** - Commands start with `taxembed-`
2. **Autocomplete** - Tab completion works in shell
3. **Consistent** - Uniform naming convention
4. **Documented** - `--help` for each command
5. **Installed** - Available after `uv sync`

---

## Makefile Shortcuts

For even faster access:

```makefile
make install         # uv sync
make train           # Quick 1-epoch test
make check           # Run taxembed-check
```

---

*See QUICKSTART.md for step-by-step usage guide.*
