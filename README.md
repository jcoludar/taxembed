# Hierarchical Taxonomy Embeddings with Poincaré Geometry

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Production Ready](https://img.shields.io/badge/status-production%20ready-brightgreen.svg)](https://github.com/jcoludar/taxembed)

**Learn hierarchical embeddings of NCBI's biological taxonomy in hyperbolic space.**

✅ **Production model included:** 92K organisms, loss 0.472, epoch 28
📊 **Validated:** 100% ball constraint compliance, clear hierarchical clustering
📁 **Location:** `small_model_28epoch/`

This project extends Facebook Research's Poincaré embeddings with hierarchical features specifically designed for deep taxonomic hierarchies (38 levels, 2.7M organisms).

---

## ✨ Features

- **Hyperbolic Geometry**: Embeddings in Poincaré ball model (ideal for hierarchies)
- **Transitive Closure Training**: 975K ancestor-descendant pairs (not just parent-child)
- **Depth-Aware Features**: Initialization, regularization, and weighting by taxonomic depth
- **Hard Negative Sampling**: Cousin sampling at same depth level
- **Ball Constraint Enforcement**: 3-layer strategy ensures 100% valid embeddings
- **Performance Optimized**: 1000x faster regularizer, selective projection
- **Comprehensive Validation**: Automated sanity checks and quality metrics

---

## 🚀 Quick Start

### **Production Model Available** ⭐

A pre-trained model is included in `small_model_28epoch/`:

- **92,290 organisms** embedded in 10 dimensions
- **Best epoch:** 28, **Loss:** 0.472
- **100% ball constraint** compliance
- Ready for immediate use!

### **Installation**

```bash
# Clone the repository
git clone https://github.com/jcoludar/taxembed.git
cd taxembed

# Install with uv
uv sync

```

After installation, the unified CLI is available:

- `taxembed train <clade> -as <tag>` - Train model for any clade (auto-builds dataset)
- `taxembed visualize <tag>` - Visualize results with automatic best checkpoint
- `taxembed-download` - Download NCBI taxonomy (legacy, auto-handled by train)
- `taxembed-prepare` - Build transitive closure (legacy, auto-handled by train)
- `taxembed-check` - Validate installation

📖 See [docs/CLI_COMMANDS.md](docs/CLI_COMMANDS.md) for detailed usage.

### **Using Pre-trained Model**

```python
import torch
import pandas as pd

# Load embeddings
ckpt = torch.load('small_model_28epoch/taxonomy_model_small_best.pth')
embeddings = ckpt['embeddings']  # Shape: (92290, 10)

# Load TaxID mapping
mapping = pd.read_csv('data/taxonomy_edges_small.mapping.tsv',
                      sep='\t', header=None, names=['idx', 'taxid'])
```

### **Train New Model**

**Using unified CLI** (recommended - easiest):

```bash
# Train any clade by name or TaxID (auto-builds dataset, downloads taxonomy if needed)
taxembed train Cnidaria -as cnidaria --epochs 100 --lambda 0.1
taxembed train 6073 -as echinoderms --epochs 50

# Visualize results (automatically uses best checkpoint)
taxembed visualize cnidaria
taxembed visualize echinoderms --children 1  # Color by grandchildren

# All artifacts saved to artifacts/tags/<tag>/
```

**Using legacy CLI commands**:

```bash
# 1. Download NCBI taxonomy
taxembed-download

# 2. Build transitive closure (975K training pairs)
taxembed-prepare

# 3. Train model (~2.5 hours on M3 Mac CPU)
taxembed-train

# 4. Visualize results
taxembed-visualize taxonomy_model_small_best.pth
```

**Using Python scripts directly**:

```bash
python prepare_taxonomy_data.py       # Download
python build_transitive_closure.py    # Prepare
python train_small.py                 # Train

# With custom parameters:
python train_small.py \
    --epochs 50 \
    --batch-size 128 \
    --lr 0.003
```

### **Analyze Results**

```bash
# Check hierarchy quality
python analyze_hierarchy_hyperbolic.py

# Visualize embeddings
python scripts/visualize_embeddings.py my_model.pth --highlight mammals
```

---

## 📊 What's Different from Facebook's Implementation?

| Feature               | Facebook          | This Project                                         |
| --------------------- | ----------------- | ---------------------------------------------------- |
| **Training Data**     | Parent-child only | All ancestor-descendant pairs (9.8x more)            |
| **Initialization**    | Random            | Depth-aware (root near center, leaves near boundary) |
| **Regularization**    | None              | Radial penalty to enforce depth → radius mapping     |
| **Negative Sampling** | Random            | Hard negatives (cousins at same taxonomic level)     |
| **Loss Weighting**    | Uniform           | Depth-weighted (deeper pairs more important)         |
| **Ball Constraints**  | Soft projection   | 3-layer enforcement (100% compliance)                |
| **Performance**       | Baseline          | 1000x faster regularizer, 30x faster projection      |

---

## 📁 Project Structure

```
taxembed/
├── src/taxembed/              # Main package
│   ├── models/                # Poincaré embedding models
│   ├── training/              # Training loop and data loaders
│   ├── data/                  # Data downloading and processing
│   ├── visualization/         # UMAP and plotting utilities
│   ├── analysis/              # Hierarchy quality analysis
│   ├── validation/            # Sanity checks and validation
│   ├── builders/              # Dataset builders (TaxoPy)
│   └── cli/                   # Command-line interface
│
├── tests/                     # Comprehensive test suite
│   ├── test_models.py
│   ├── test_training.py
│   ├── test_data.py
│   └── conftest.py
│
├── examples/                  # Example scripts
│   ├── basic_training.py
│   ├── custom_dataset.py
│   └── nn_demo.py
│
├── docs/                      # Documentation
│   ├── user-guide.md          # Comprehensive usage guide
│   ├── theory.md              # Mathematical background
│   └── CLI_COMMANDS.md        # CLI reference
│
├── _vendor/                   # Facebook's original code (backup)
│   ├── hype/
│   └── embed.py
│
├── data/                      # Data files (gitignored)
├── artifacts/                 # Training outputs (gitignored)
├── pyproject.toml             # Package configuration
├── CONTRIBUTING.md            # Contribution guidelines
└── README.md                  # This file
```

---

## 🎯 Current Status

### **What Works ✅**

- ✅ Clean data pipeline with validation
- ✅ Transitive closure computation (975K pairs)
- ✅ Hierarchical training features implemented
- ✅ Perfect ball constraint enforcement (100% inside)
- ✅ Stable training (~3 min/epoch on M3 Mac CPU)
- ✅ Automatic checkpointing and early stopping

### **What Needs Work ⚠️**

- ⚠️ Hierarchy quality is poor after limited training (2 epochs)
- ⚠️ Depth-norm correlation ~0 (should be >0.5)
- ⚠️ Taxonomic separation ratios <1.1x (should be >1.5x)
- ⚠️ Needs hyperparameter tuning or longer training

**See [JOURNEY.md](JOURNEY.md) for full development history and current challenges.**

---

## 🔧 Key Scripts

### **Training**

```bash
# Hierarchical training with all features
python train_hierarchical.py --help

# Simple training (Facebook's original)
python embed.py -dset data/taxonomy_edges.mapped.edgelist ...
```

### **Analysis**

```bash
# Validate data quality
python sanity_check.py

# Check hierarchy quality
python analyze_hierarchy_hyperbolic.py

# Visualize specific groups
python scripts/visualize_embeddings.py model.pth --highlight primates
```

### **Data Preparation**

```bash
# Download NCBI taxonomy
python prepare_taxonomy_data.py

# Build transitive closure
python build_transitive_closure.py

# Validate data
python scripts/validate_data.py small
```

### **Unified CLI: Train & Visualize Any Clade**

The `taxembed` command provides a streamlined workflow:

```bash
# Train any clade by name or TaxID (auto-builds dataset, downloads taxonomy if needed)
taxembed train Cnidaria -as cnidaria --epochs 100 --lambda 0.1
taxembed train 6073 -as echinoderms --epochs 50

# Visualize results (automatically uses best checkpoint for the tag)
taxembed visualize cnidaria
taxembed visualize echinoderms --children 1  # Color by grandchildren (--children 0 = children, 1 = grandchildren, etc.)

# All artifacts saved to artifacts/tags/<tag>/
# - run.json: metadata (config, paths, dataset info)
# - <tag>.pth: checkpoints
# - <tag>_best.pth: best checkpoint
# - <tag>_umap.png: visualizations
```

**Features:**

- **Automatic dataset building**: Uses [TaxoPy](https://pypi.org/project/taxopy/) to query NCBI taxonomy and build datasets on-the-fly
- **Smart checkpoint selection**: Visualization automatically uses the best checkpoint for each tag
- **Hierarchical coloring**: `--children` flag controls depth (0=children, 1=grandchildren, 2=great-grandchildren, etc.)
- **Informative titles**: Plots show clade name, children level, epochs, and loss
- **Organized artifacts**: All outputs stored in `artifacts/tags/<tag>/` with full metadata

**Advanced usage:**

```bash
# Use pre-built dataset files
taxembed train --file data/my_transitive.pkl --mapping data/my.mapping.tsv -as custom_tag

# Override visualization settings
taxembed visualize cnidaria --sample 15000 --output custom_plot.png --root-taxid 6072
```

### **Build Custom Clade Datasets (Standalone)**

```bash
# Example: build the Metazoa (animals) subset with automatic mapping
uv run python scripts/build_clade_dataset.py \
    --root-taxid 33208 \
    --dataset-name animals
```

This leverages [TaxoPy](https://pypi.org/project/taxopy/) to:

- Query NCBI taxonomy for all descendants of the requested root
- Emit raw and remapped edgelists (`data/taxopy/<name>/taxonomy_edges_<name>.edgelist`)
- Write mapping + manifest files for reproducible provenance
- Generate transitive-closure datasets ready for `train_small.py`

Use `--max-depth` to truncate deep subtrees or point `--taxdump-dir` at an alternate taxonomy download.

---

## 📖 Documentation

- **[docs/user-guide.md](docs/user-guide.md)** - Comprehensive usage guide
- **[docs/theory.md](docs/theory.md)** - Mathematical background and theory
- **[docs/CLI_COMMANDS.md](docs/CLI_COMMANDS.md)** - Command-line reference
- **[examples/](examples/)** - Example scripts and tutorials
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - How to contribute

---

## 🧪 Validation

Before training, run the comprehensive sanity check:

```bash
python sanity_check.py
```

This validates:

- ✅ Mapping file integrity (no duplicates, continuous indices)
- ✅ Transitive closure data (valid indices, no self-loops)
- ✅ Projection logic (keeps embeddings in ball)
- ✅ Hyperbolic distance (correct formula)
- ✅ Initialization (proper depth-based radii)
- ✅ Sibling map (hard negatives at same depth)
- ✅ Regularizer targets (all < 1.0)
- ✅ Training configuration (reasonable batch sizes)

**Expected: 10/10 checks passed**

---

## 📈 Performance

### **Optimizations Applied**

- **Regularizer**: Vectorized (1000x faster, 1.7B → 111K ops/epoch)
- **Projection**: Selective (30x faster, only updated embeddings)
- **Tensor Creation**: Pre-allocated arrays (10-100x faster)
- **Device**: CPU-only on macOS (stable, no MPS hanging)

### **Training Speed**

- Small dataset (111K organisms): ~3 min/epoch on M3 Mac
- Full dataset (2.7M organisms): ~60 min/epoch on M3 Mac

---

## 🔬 Experimental Results

### **Ball Constraint Enforcement**

| Version         | Max Norm | Outside Ball | Status     |
| --------------- | -------- | ------------ | ---------- |
| v1 (weak reg)   | 2.18     | 54%          | ❌ Broken  |
| v2 (strong reg) | 1.45     | 2.2%         | ⚠️ Better  |
| v3 (3-layer)    | 1.00     | 0%           | ✅ Perfect |

### **Hierarchy Quality** (After 2 epochs)

| Metric          | Target | Actual | Status  |
| --------------- | ------ | ------ | ------- |
| Depth-norm corr | >0.5   | +0.003 | ❌ Poor |
| Phylum sep      | >1.5x  | 1.08x  | ❌ Poor |
| Class sep       | >1.5x  | 0.99x  | ❌ Poor |

**Conclusion:** Constraints work perfectly, but hierarchy learning needs more time or tuning.

---

## 🚧 Known Issues & Future Work

### **Current Limitations**

1. **Poor hierarchy quality** - Only 2 epochs completed, needs more training
2. **Data imbalance** - 94% deep ancestors, 6% parent-child (may need balanced sampling)
3. **Regularization trade-off** - λ=0.1 enforces constraints but may limit expressiveness
4. **No curriculum learning** - Trains on all pairs at once (may need progressive training)

### **Future Directions**

1. Train longer with increased patience (50-100 epochs)
2. Implement balanced sampling (equal parent-child, grandparent, deep)
3. Progressive training (parent-child → grandparent → all ancestors)
4. Try Riemannian optimizer (respects manifold natively)
5. Experiment with margin schedules (increase margin with depth)

---

## 🛠️ Development

### **Code Quality Tools**

This project maintains high code quality using modern Python tooling:

```bash
# Linting and formatting with Ruff
uv run ruff check .                    # Check for issues
uv run ruff check --fix .              # Auto-fix issues
uv run ruff format .                   # Format code

# Static type checking with MyPy
uv run mypy src/taxembed               # Type check source

# Testing with Pytest
uv run pytest                          # Run test suite
uv run pytest --cov=src/taxembed       # With coverage

# Complete quality check
uv run ruff check . && uv run mypy src/taxembed && uv run pytest
```

**Configuration:**

- All tools configured in `pyproject.toml`
- Ruff: 100 char lines, Python 3.11+, comprehensive rules
- MyPy: Strict typing with gradual adoption strategy
- Pytest: Comprehensive test suite with coverage reporting

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### **Priority Areas**

- Hyperparameter tuning for better hierarchy quality
- Balanced/curriculum sampling strategies
- Alternative hyperbolic models (Lorentz, Klein)
- Evaluation metrics for taxonomic hierarchies
- Scalability to full 2.7M organism dataset

---

## 📚 References

### **Original Papers**

- Nickel & Kiela (2017). "Poincaré Embeddings for Learning Hierarchical Representations" [[PDF](https://arxiv.org/abs/1705.08039)]
- Facebook Research implementation: [[GitHub](https://github.com/facebookresearch/poincare-embeddings)]

### **Data**

- NCBI Taxonomy: https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/
- Taxonomy documentation: https://www.ncbi.nlm.nih.gov/taxonomy

### **Related Work**

- Hyperbolic Neural Networks
- Lorentz Embeddings
- Box Embeddings for Hierarchies

---

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- Based on Facebook Research's Poincaré embeddings
- Extended for hierarchical taxonomy by @jcoludar
- Development history in [JOURNEY.md](JOURNEY.md)

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/jcoludar/taxembed/issues)
- **Documentation**: See [JOURNEY.md](JOURNEY.md) and [docs/](docs/)
- **Quick Help**: See [QUICKSTART.md](QUICKSTART.md)

---

**⭐ If you find this useful, please star the repository!**

_Last Updated: December 2025_
