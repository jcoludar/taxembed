# Hierarchical Taxonomy Embeddings with Poincaré Geometry

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Learn hierarchical embeddings of NCBI's biological taxonomy in hyperbolic space.**

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

### **Installation**

```bash
# Clone the repository
git clone https://github.com/jcoludar/taxembed.git
cd taxembed

# Create virtual environment
python3.11 -m venv venv311
source venv311/bin/activate  # or venv311\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### **Download Data**

```bash
# Download and prepare NCBI taxonomy
python prepare_taxonomy_data.py

# This creates:
# - data/taxonomy_edges_small.edgelist (111K organisms)
# - data/taxonomy_edges.edgelist (2.7M organisms)
# - data/nodes.dmp, names.dmp (NCBI taxonomy files)
```

### **Train Model (Small Dataset)**

```bash
# Build transitive closure (ancestor-descendant pairs)
python build_transitive_closure.py

# Train hierarchical model
bash run_hierarchical_training.sh

# Or with custom parameters:
python train_hierarchical.py \
    --data data/taxonomy_edges_small_transitive.pkl \
    --checkpoint my_model.pth \
    --dim 10 \
    --epochs 100 \
    --early-stopping 10 \
    --lr 0.005 \
    --lambda-reg 0.1
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

| Feature | Facebook | This Project |
|---------|----------|--------------|
| **Training Data** | Parent-child only | All ancestor-descendant pairs (9.8x more) |
| **Initialization** | Random | Depth-aware (root near center, leaves near boundary) |
| **Regularization** | None | Radial penalty to enforce depth → radius mapping |
| **Negative Sampling** | Random | Hard negatives (cousins at same taxonomic level) |
| **Loss Weighting** | Uniform | Depth-weighted (deeper pairs more important) |
| **Ball Constraints** | Soft projection | 3-layer enforcement (100% compliance) |
| **Performance** | Baseline | 1000x faster regularizer, 30x faster projection |

---

## 📁 Project Structure

```
poincare-embeddings/
├── train_hierarchical.py          # Main hierarchical training script
├── build_transitive_closure.py    # Generate ancestor-descendant pairs
├── analyze_hierarchy_hyperbolic.py # Evaluate hierarchy quality
├── sanity_check.py                 # Comprehensive validation
├── prepare_taxonomy_data.py        # Download NCBI taxonomy
├── remap_edges.py                  # Map TaxIDs to indices
│
├── data/                           # Data files (gitignored)
│   ├── taxonomy_edges_small.edgelist
│   ├── taxonomy_edges_small_transitive.pkl
│   └── taxonomy_edges_small.mapping.tsv
│
├── scripts/                        # Utility scripts
│   ├── visualize_embeddings.py
│   ├── validate_data.py
│   └── ...
│
├── hype/                           # Original Facebook implementation
│   ├── graph.py
│   ├── manifolds/
│   └── ...
│
├── docs/                           # Documentation
│   └── archive/                    # Intermediate development docs
│
├── JOURNEY.md                      # Development history
├── QUICKSTART.md                   # 5-minute guide
└── README.md                       # This file
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

---

## 📖 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 5 minutes
- **[JOURNEY.md](JOURNEY.md)** - Full development history from Facebook's code to now
- **[SESSION_SUMMARY_NOV8.md](SESSION_SUMMARY_NOV8.md)** - Latest session summary with findings
- **[docs/archive/](docs/archive/)** - Intermediate development documents

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
| Version | Max Norm | Outside Ball | Status |
|---------|----------|--------------|--------|
| v1 (weak reg) | 2.18 | 54% | ❌ Broken |
| v2 (strong reg) | 1.45 | 2.2% | ⚠️ Better |
| v3 (3-layer) | 1.00 | 0% | ✅ Perfect |

### **Hierarchy Quality** (After 2 epochs)
| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Depth-norm corr | >0.5 | +0.003 | ❌ Poor |
| Phylum sep | >1.5x | 1.08x | ❌ Poor |
| Class sep | >1.5x | 0.99x | ❌ Poor |

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

*Last Updated: November 8, 2025*
