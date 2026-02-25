# Hierarchical Taxonomy Embeddings with Poincaré Geometry

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Production Ready](https://img.shields.io/badge/status-production%20ready-brightgreen.svg)](https://github.com/jcoludar/taxembed)

**Learn hierarchical embeddings of NCBI's biological taxonomy in hyperbolic space.**

✅ **v10a architecture:** Euclidean Adam + radial nudge + tiered negatives + class-weighted loss
📊 **Validated:** Echinodermata r=+0.990 (1.68x sep), Mollusca r=+0.65, clear UMAP separation
📁 **Models:** `artifacts/tags/<tag>/` (run `taxembed train <clade> -as <tag>`)

This project extends Facebook Research's Poincaré embeddings with hierarchical features specifically designed for deep taxonomic hierarchies (38 levels, 2.7M organisms).

---

## ✨ Features

- **Hyperbolic Geometry**: Embeddings in Poincaré ball model (ideal for hierarchies)
- **Transitive Closure Training**: Ancestor-descendant pairs (not just parent-child)
- **Depth-Aware Features**: Initialization, regularization, and weighting by taxonomic depth
- **Hard Negative Sampling**: Cousin sampling at same depth level (vectorized for scale)
- **Vectorized Negative Sampling**: Depth-grouped batch operations instead of per-sample loops
- **Ball Constraint Enforcement**: Per-batch projection ensures 100% valid embeddings
- **Radial Nudge**: Post-step norm correction preserves angular clustering while enforcing depth-radius mapping
- **Dual UMAP Metrics**: Euclidean and Poincaré distance UMAP visualizations
- **Curriculum Learning**: Optional shallow-pairs-first training for large trees
- **Performance Optimized**: 1000x faster regularizer, selective projection
- **Comprehensive Validation**: `analyze_hierarchy_hyperbolic.py` for depth-norm correlation and taxonomic separation

---

## 🚀 Quick Start

### **Trained Models** ⭐

Models are stored in `artifacts/tags/<tag>/` with full metadata in `run.json`:

| Tag | Clade | Nodes | Depth-Norm r | Class Sep | Status |
|-----|-------|-------|-------------|-----------|--------|
| `echino_v9d` | Echinodermata | 7,833 | +0.990 | 1.68x | Production |
| `echino_v4` | Echinodermata | 7,833 | +0.950 | 1.21x | Production |
| `mollusca_v4` | Mollusca | 53,720 | +0.650 | 1.11x | Experimental |

Legacy model in `small_model_28epoch/` (92K organisms, pre-v4 architecture).

### **Installation**

```bash
# Clone the repository
git clone https://github.com/jcoludar/taxembed.git
cd poincare-embeddings

# Install with uv (recommended)
make install
# or: uv sync
```

After installation, the unified CLI is available:
- `taxembed train <clade> -as <tag>` - Train model for any clade (auto-builds dataset)
- `taxembed visualize <tag>` - Visualize results with automatic best checkpoint
- `taxembed visualize <tag> --metric poincare` - Poincaré distance UMAP
- `taxembed-download` - Download NCBI taxonomy (legacy, auto-handled by train)
- `taxembed-prepare` - Build transitive closure (legacy, auto-handled by train)
- `taxembed-check` - Validate installation

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
# v4 defaults: Euclidean Adam + radial nudge (0.05) + lambda_reg 0.1
taxembed train Echinodermata -as echino_v4 --epochs 100
taxembed train Mollusca -as mollusca_v4 --epochs 100

# For large clades (>30K nodes), increase capacity:
taxembed train Mollusca -as mollusca_v5 --dim 20 --curriculum --n-negatives 100 --epochs 200

# Visualize results (automatically uses best checkpoint)
taxembed visualize echino_v4 --children 2
taxembed visualize echino_v4 --children 2 --metric poincare  # Poincaré distance UMAP

# Analyze hierarchy quality
python scripts/analyze_hierarchy_hyperbolic.py --tag echino_v4

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
python scripts/analyze_hierarchy_hyperbolic.py

# Visualize embeddings
python scripts/visualize_embeddings.py my_model.pth --highlight mammals
```

---

## 📊 What's Different from Facebook's Implementation?

| Feature | Facebook | This Project (v10a) |
|---------|----------|-------------------|
| **Training Data** | Parent-child only | All ancestor-descendant pairs (transitive closure) |
| **Optimizer** | SGD | Euclidean Adam (preserves angular gradients) |
| **Initialization** | Random | Depth-aware (root near center, leaves near boundary) |
| **Regularization** | None | Radial penalty + post-step radial nudge (norm-only correction) |
| **Negative Sampling** | Random | Hard negatives (cousins at same taxonomic level) |
| **Loss Weighting** | Uniform | Depth-weighted (deeper pairs more important) |
| **Ball Constraints** | Soft projection | Per-batch unconditional projection (100% compliance) |
| **Visualization** | None | UMAP with Euclidean or Poincaré distance metric |
| **Performance** | Baseline | 1000x faster regularizer, selective projection |

---

## 📁 Project Structure

```
poincare-embeddings/
├── README.md                       # This file
├── pyproject.toml                  # Package config + ruff + pytest
├── Makefile                        # Common tasks (install, test, lint)
├── Dockerfile                      # Container deployment
│
├── train_hierarchical.py           # Core: model, dataloader, loss, vectorized sampling
├── train_small.py                  # Training orchestrator (called by CLI)
├── visualize_multi_groups.py       # UMAP visualization (called by CLI)
├── build_transitive_closure.py     # Transitive closure builder (CLI dep)
├── prepare_taxonomy_data.py        # NCBI taxonomy downloader (CLI dep)
├── final_sanity_check.py           # Validation checks (CLI dep)
│
├── src/taxembed/                   # Installable package
│   ├── cli/                        # Unified `taxembed` CLI
│   ├── analysis/                   # Dimension analysis
│   ├── builders/                   # TaxoPy clade dataset builder
│   ├── optim/                      # Riemannian optimizer
│   └── utils/                      # Data validation
│
├── scripts/                        # Standalone tools
│   ├── analyze_hierarchy_hyperbolic.py  # Post-training quality analysis
│   ├── build_clade_dataset.py      # Standalone clade dataset builder
│   ├── validate_data.py            # Data validation utility
│   └── train_lrz.sh               # HPC/Slurm training script
│
├── tests/                          # Test suite
├── data/                           # Data files (gitignored)
├── artifacts/                      # Trained models (gitignored)
│
└── docs/                           # Documentation
    ├── QUICKSTART.md               # 5-minute guide
    ├── JOURNEY.md                  # Development history
    ├── CLI_COMMANDS.md             # CLI reference
    ├── TRAIN_*_GUIDE.md            # Training guides
    └── archive/                    # Historical dev docs + legacy code
```

---

## 🎯 Current Status (v10a — February 2026)

### **Architecture (v10a)**
The v10a architecture combines Euclidean Adam (proven angular clustering) with a post-step **radial nudge**, **tiered negative sampling**, and **class-weighted loss**. This achieves both angular class separation AND radial hierarchy simultaneously.

Key components:
- **Euclidean Adam optimizer** — preserves angular gradients (unlike RiemannianAdam which crushes boundary gradients via conformal factor)
- **Radial nudge** (`--radial-nudge 0.05`) — after each batch, nudges norms toward depth-based targets: `new_norm = (1 - α) * norm + α * target_norm`
- **Tiered negative sampling** (`--tiered-negatives`) — 50% hard (cousins), 30% medium (same class), 20% easy
- **Vectorized sampling** — depth-grouped batch operations, O(unique_depths) instead of O(batch_size)
- **Per-batch projection** — unconditionally projects embeddings back into the Poincaré ball
- **λ_reg = 0.1** — full regularization strength (no auto-reduction)

### **Results**

| Clade | Nodes | Depth-Norm r | Class Sep | Order Sep | Status |
|-------|-------|-------------|-----------|-----------|--------|
| Echinodermata (v9d) | 7,833 | +0.990 | 1.68x (STRONG) | — | ✅ Excellent |
| Echinodermata (v4) | 7,833 | +0.950 | 1.21x (MODERATE) | 1.32x (MODERATE) | ✅ Good |
| Mollusca (v4) | 53,720 | +0.650 | 1.11x (POOR) | 1.12x (POOR) | ⚠️ Needs tuning |

### **What Works ✅**
- ✅ Depth-norm correlation consistently positive (+0.65 to +0.95)
- ✅ Clear UMAP clustering visible for major taxonomic groups
- ✅ Both Euclidean and Poincaré distance UMAP supported
- ✅ Unified CLI (`taxembed train/visualize`) with automatic dataset building
- ✅ Full metadata tracking in `run.json` per tag
- ✅ Curriculum learning support for large trees

### **What Needs Work ⚠️**
- ⚠️ Large clades (>30K nodes) need higher dimensionality (`--dim 20+`)
- ⚠️ Imbalanced trees (e.g., Gastropoda = 70% of Mollusca) reduce class separation
- ⚠️ Default hyperparameters optimized for ~10K nodes; larger trees need tuning

---

## 🔧 Key Scripts

### **Training**
```bash
# Recommended: use the unified CLI
taxembed train Echinodermata -as echino_v10 --epochs 100 --tiered-negatives

# Or directly:
python train_hierarchical.py --help
```

### **Analysis**
```bash
# Validate data quality
python final_sanity_check.py

# Check hierarchy quality
python scripts/analyze_hierarchy_hyperbolic.py --tag echino_v10
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
# v4 defaults: optimizer=adam, radial-nudge=0.05, lambda-reg=0.1, dim=10
taxembed train Echinodermata -as echino_v4 --epochs 100
taxembed train Mollusca -as mollusca_v4 --epochs 100

# For large clades (>30K nodes), scale up:
taxembed train Mollusca -as mollusca_v5 --dim 20 --curriculum --n-negatives 100 --epochs 200

# Visualize results (automatically uses best checkpoint for the tag)
taxembed visualize echino_v4 --children 2
taxembed visualize echino_v4 --children 2 --metric poincare  # Poincaré distance UMAP

# Analyze hierarchy quality
python scripts/analyze_hierarchy_hyperbolic.py --tag echino_v4

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
- **Dual UMAP metrics**: `--metric euclidean` (default) or `--metric poincare` for hyperbolic distance
- **Radial nudge**: `--radial-nudge 0.05` (default) gently enforces depth-radius mapping without disturbing angular structure
- **Curriculum learning**: `--curriculum` teaches shallow pairs first, then progressively deeper
- **Informative titles**: Plots show clade name, children level, epochs, and loss
- **Organized artifacts**: All outputs stored in `artifacts/tags/<tag>/` with full metadata

**Advanced usage:**
```bash
# Use pre-built dataset files
taxembed train --file data/my_transitive.pkl --mapping data/my.mapping.tsv -as custom_tag

# Override visualization settings
taxembed visualize echino_v4 --sample 15000 --output custom_plot.png --root-taxid 7586

# Use Riemannian Adam (alternative optimizer, good radial hierarchy but weaker angular clustering)
taxembed train Cnidaria -as cnidaria_radam --optimizer radam --burnin 10
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

- **[QUICKSTART.md](docs/QUICKSTART.md)** - Get started in 5 minutes
- **[JOURNEY.md](JOURNEY.md)** - Full development history from Facebook's code to now
- **[SESSION_SUMMARY_NOV8.md](SESSION_SUMMARY_NOV8.md)** - Latest session summary with findings
- **[docs/archive/](docs/archive/)** - Intermediate development documents

---

## 🧪 Validation

Before training, run the comprehensive sanity check:

```bash
python final_sanity_check.py
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

## 🔬 Experimental History

### **Architecture Evolution**
| Version | Optimizer | Radial Control | Depth-Norm r | Angular Clustering |
|---------|-----------|----------------|-------------|-------------------|
| v1-v2 | Euclidean Adam | Regularizer only | -0.074 | 0.65 (good) |
| v3 | RiemannianAdam | Conformal factor | +0.936 | 0.065 (destroyed) |
| v4 | Euclidean Adam + radial nudge | Nudge + regularizer | +0.950 | Visible UMAP clusters |
| **v9d/v10a** | **Eucl. Adam + nudge + tiered negs** | **Nudge + reg + class weight** | **+0.990** | **1.68x class sep** |

**Key insight:** RiemannianAdam's conformal factor `((1-||p||²)²/4)` gives 110x gradient reduction at norm 0.9, crushing angular gradients for deep nodes. The v4 radial nudge achieves the same radial ordering without touching directions.

### **Scaling Analysis (v4)**
| Metric | Echinodermata (7.8K) | Mollusca (53.7K) | Ratio |
|--------|---------------------|------------------|-------|
| Nodes/dim | 783 | 5,372 | 6.9x |
| Pairs/node | 8.6 | 9.0 | ~same |
| Updates/node (total) | ~948 | ~287 | 0.30x |
| Best loss | 0.169 | 0.295 | 1.75x |
| Depth-norm r | +0.950 | +0.650 | 0.68x |

**Conclusion:** Larger clades need proportionally more capacity (dim) and training (epochs/lr). The default dim=10 is optimal for ~10K nodes but insufficient for 50K+.

---

## 🚧 Known Issues & Next Steps

### **Current Limitations**
1. **Large-clade scaling** — Default dim=10 insufficient for >30K nodes (Mollusca: 5,372 nodes/dim)
2. **Class imbalance** — Dominant subclades (e.g., Gastropoda = 70%) consume angular space
3. **Undertrained large models** — Early stopping triggers before sufficient updates/node for large trees

### **Next Run: Mollusca v5 (Earmarked)**
```bash
# Retrain Mollusca with tuned hyperparameters for large clades
VIRTUAL_ENV= uv run taxembed train Mollusca -as mollusca_v5 \
    --dim 20 \
    --curriculum \
    --n-negatives 100 \
    --epochs 200 \
    --early-stopping 25

# Then analyze and visualize
VIRTUAL_ENV= uv run python scripts/analyze_hierarchy_hyperbolic.py --tag mollusca_v5
VIRTUAL_ENV= uv run taxembed visualize mollusca_v5 --children 2
VIRTUAL_ENV= uv run taxembed visualize mollusca_v5 --children 2 --metric poincare
```

**Rationale:**
- `--dim 20`: Doubles capacity from 5,372 to 2,686 nodes/dim (closer to Echinodermata's 783)
- `--curriculum`: Teaches shallow structure first, critical for large trees with deep hierarchies
- `--n-negatives 100`: Stronger gradient signal per batch (2x default)
- `--epochs 200 --early-stopping 25`: More room to converge before plateau detection

### **Future Directions**
1. Adaptive dimensionality heuristic based on node count
2. Learning rate scheduling (warmup + cosine decay) instead of fixed lr
3. Class-balanced negative sampling to counteract dominant subtrees
4. Multi-scale evaluation: per-rank separation metrics at every level

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

*Last Updated: February 2026*
