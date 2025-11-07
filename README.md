# Poincaré Embeddings for NCBI Taxonomy (TaxEmbed)

**Learn hierarchical embeddings of 2.7M+ organisms using hyperbolic geometry.**

This project learns embeddings of the complete NCBI taxonomy in **hyperbolic (Poincaré) space**, where taxonomically related organisms (primates, mammals, bacteria) naturally cluster together, preserving the hierarchical tree structure.

Built on [Nickel & Kiela (2017)](http://papers.nips.cc/paper/7213-poincare-embeddings-for-learning-hierarchical-representations.pdf).

## 🚀 Quick Start (5 Minutes)

### Installation

```bash
# 1. Clone repository
git clone https://github.com/jcoludar/taxembed.git
cd taxembed

# 2. Create environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Build C++ extensions
python setup.py build_ext --inplace
```

### Train Your First Model (Small Dataset)

```bash
# Train for 50 epochs on 111K organisms
python embed.py \
  -dset data/taxonomy_edges_small.mapped.edgelist \
  -checkpoint my_model.pth \
  -dim 10 -epochs 50 -negs 50 -burnin 10 \
  -batchsize 32 -model distance -manifold poincare \
  -lr 0.1 -gpu -1 -ndproc 1 -train_threads 1 \
  -eval_each 999999 -fresh
```

Training takes ~5 minutes. You'll see:
```
Epoch 0: loss=3.94
Epoch 10: loss=3.21
Epoch 50: loss=2.32  ✓ Model learned!
```

### Visualize Results

```bash
# Highlight primates in UMAP projection
python scripts/visualize_embeddings.py my_model.pth --highlight primates

# Output: umap_my_model_primates_highlighted.png
# Shows all organisms with primates (394) colored red
```

### Check Nearest Neighbors

The visualization automatically shows nearest neighbors:
```
Homo sapiens (Human):
  1. TaxID 63221 (distance: 0.0007)  ← Other primate!
  2. TaxID 67983 (distance: 0.223)
```

**That's it!** You've trained embeddings and visualized them. 🎉

## 📖 What Are Poincaré Embeddings?

### The Problem
Traditional embeddings use **Euclidean space** (flat), but hierarchies grow exponentially:
- Root: 1 node
- Level 1: 10 nodes  
- Level 2: 100 nodes
- Level 3: 1,000 nodes

You need exponentially growing dimensions to fit this in flat space!

### The Solution
**Hyperbolic space** (Poincaré disk) has:
- Exponentially growing space as you move from center
- Perfect for trees: root near center, leaves near boundary
- Can represent exponential growth in constant dimensions

**Visual:**
```
Poincaré Disk (hyperbolic space)
         ___________
       /             \
      |  (Primates)  |  ← Organisms cluster by taxonomy
      |              |
      | Mammals      |  ← Related groups near each other
      |              |
      |    Bacteria →|  ← Distance = taxonomic distance
       \            /
         ----------
     center = root
     boundary = leaves
```

### Your Data
**Input:** Parent-child relationships from NCBI taxonomy
```
Homo sapiens (9606) → Homo (9605)
Homo (9605) → Homininae (207598)
Homininae → Hominidae → Primates → Mammalia → ...
```

**Output:** One 10-dimensional vector per organism
```
embed[Homo_sapiens]  = [0.15, 0.23, ..., 0.45]
embed[other_primate] = [0.16, 0.24, ..., 0.44]  ← Very close!
embed[E_coli]        = [-0.80, 0.02, ..., 0.10] ← Far away!
```

**Training:** Make related organisms close, unrelated organisms far
- Parent-child distance: ~0.1-0.5 (small)
- Random pair distance: ~1.0-2.0 (large)

**See [`POINCARE_EMBEDDINGS_EXPLAINED.md`](POINCARE_EMBEDDINGS_EXPLAINED.md) for full technical explanation.**

## 🎯 Usage Examples

### Basic Training

```bash
# Small dataset (111K organisms, ~5 min)
python embed.py \
  -dset data/taxonomy_edges_small.mapped.edgelist \
  -checkpoint model_small.pth \
  -dim 10 -epochs 50 -negs 50 -burnin 10 \
  -batchsize 32 -model distance -manifold poincare \
  -lr 0.1 -gpu -1 -ndproc 1 -train_threads 1 \
  -eval_each 999999 -fresh

# Full dataset (2.7M organisms, ~1 hour)
python embed.py \
  -dset data/taxonomy_edges.mapped.edgelist \
  -checkpoint model_full.pth \
  -dim 10 -epochs 200 -negs 50 -burnin 10 \
  -batchsize 32 -model distance -manifold poincare \
  -lr 0.1 -gpu -1 -ndproc 1 -train_threads 1 \
  -eval_each 999999 -fresh
```

### Visualization Options

```bash
# Highlight any taxonomic group
python scripts/visualize_embeddings.py model.pth --highlight primates
python scripts/visualize_embeddings.py model.pth --highlight mammals
python scripts/visualize_embeddings.py model.pth --highlight bacteria

# Show only specific group
python scripts/visualize_embeddings.py model.pth --only primates

# Custom sample size with nearest neighbors
python scripts/visualize_embeddings.py model.pth --sample 50000 --nearest 10

# Supported groups: primates, mammals, vertebrates, bacteria, archaea, fungi, plants, insects, rodents
```

### Data Validation

```bash
# Always validate data before training
python scripts/validate_data.py small
python scripts/validate_data.py full
```

## 📁 Project Structure

```
taxembed/
├── embed.py                   # ⭐ Main training script
├── prepare_taxonomy_data.py   # Data preparation
├── remap_edges.py             # TaxID remapping
├── evaluate_full.py           # Evaluation
│
├── scripts/                   # Utility scripts
│   ├── visualize_embeddings.py  # ⭐ Universal visualization
│   ├── validate_data.py         # ⭐ Data validation
│   ├── cleanup_repo.sh          # Repository cleanup
│   └── regenerate_data.sh       # Data regeneration
│
├── src/taxembed/              # Source package
│   ├── manifolds/             # Hyperbolic geometry
│   ├── models/                # Embedding models
│   ├── datasets/              # Data loaders
│   └── utils/                 # Utilities
│
├── hype/                      # Original package (backward compat)
├── tests/                     # Unit tests
├── data/                      # Data files (gitignored)
│
├── pyproject.toml             # Project config
├── ruff.toml                  # Linter config
├── Makefile                   # Convenience commands
└── [12 documentation files]   # See below
```

## 📚 Documentation

### For Users
- **[README.md](README.md)** ⭐ This file - quick start and overview
- **[QUICKSTART.md](QUICKSTART.md)** - Detailed quick start guide
- **[SCRIPTS_GUIDE.md](SCRIPTS_GUIDE.md)** ⭐ Complete script reference
- **[GETTING_STARTED.md](GETTING_STARTED.md)** - Comprehensive setup guide

### Understanding Poincaré Embeddings
- **[POINCARE_EMBEDDINGS_EXPLAINED.md](POINCARE_EMBEDDINGS_EXPLAINED.md)** ⭐ Technical explanation
- **[TRAINING_EXPLAINED_SIMPLE.md](TRAINING_EXPLAINED_SIMPLE.md)** ⭐ Simple explanation with examples

### Project Information
- **[STRUCTURE.md](STRUCTURE.md)** - Project organization
- **[DATA_FIXES_SUMMARY.md](DATA_FIXES_SUMMARY.md)** - Data bug fixes
- **[CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md)** - Repository cleanup
- **[REPOSITORY_STATUS.md](REPOSITORY_STATUS.md)** - Current status
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines

## 🔬 Technical Details

### Training Parameters

| Parameter | Small Dataset | Full Dataset | Meaning |
|-----------|--------------|--------------|---------|
| `--dim` | 10 | 10-50 | Embedding dimension |
| `--epochs` | 50 | 200+ | Training epochs |
| `--negs` | 50 | 50 | Negative samples per positive |
| `--burnin` | 10 | 10 | Burn-in epochs (low LR) |
| `--batchsize` | 32 | 32-128 | Batch size |
| `--lr` | 0.1 | 0.1 | Learning rate |

### Model Architecture

- **Manifold:** Poincaré ball model of hyperbolic space
- **Distance:** Poincaré distance (not Euclidean)
- **Optimizer:** Riemannian SGD (respects curved geometry)
- **Loss:** Margin-based ranking loss
- **Initialization:** Small random vectors (norm ~ 0.1)

### Data Format

**Edge List (parent-child):**
```
0 1      # TaxID 2 → TaxID 131567
2 3      # TaxID 6 → TaxID 335928
...
```

**Mapping File:**
```
taxid   idx
2       0
131567  1
6       2
...
```

## 🛠️ Advanced Usage

### Custom Dimensions
```bash
# Try different embedding dimensions
python embed.py -dset data/... -checkpoint model_d5.pth -dim 5 ...
python embed.py -dset data/... -checkpoint model_d20.pth -dim 20 ...
python embed.py -dset data/... -checkpoint model_d50.pth -dim 50 ...
```

### Different Manifolds
```bash
# Poincaré (default, recommended)
python embed.py ... -manifold poincare

# Lorentz (alternative hyperbolic model)
python embed.py ... -manifold lorentz
```

### GPU Training
```bash
# Use GPU if available
python embed.py ... -gpu 0
```

### Resume Training
```bash
# Remove -fresh flag to resume from checkpoint
python embed.py ... -checkpoint model.pth  # (no -fresh)
```

## 🧪 Validation & Testing

### Data Validation
```bash
# Check data quality
python scripts/validate_data.py small
python scripts/validate_data.py full
```

### Run Tests
```bash
# Unit tests
python -m pytest tests/

# With coverage
python -m pytest --cov=src/taxembed tests/
```

### Lint Code
```bash
# Check style
ruff check src/ scripts/

# Auto-fix
ruff check --fix src/ scripts/

# Format
ruff format src/ scripts/
```

## 🚧 Future Extensions (Roadmap)

### 1. Species Names (Text Integration)
Add text encoder to learn from species names alongside graph structure.

**Use Cases:**
- Text-based queries: "find species like 'sapiens'"
- Handle synonyms and typos
- Cross-lingual support

**Implementation:**
- Add BERT/BioBERT encoder
- Joint loss: graph structure + text similarity
- See [POINCARE_EMBEDDINGS_EXPLAINED.md](POINCARE_EMBEDDINGS_EXPLAINED.md) for details

### 2. Protein Embeddings
Incorporate protein sequence embeddings for each organism.

**Use Cases:**
- Find organisms by protein function
- Cluster by proteome similarity
- Better organism disambiguation

**Implementation:**
- Use ESM/ProtT5 protein embeddings
- Aggregate proteins per organism
- Multi-modal embedding space

### 3. Additional Features
Add organism features: genome size, GC content, habitat, etc.

**Use Cases:**
- Predict missing features
- Find organisms by phenotype
- Better downstream predictions

**Implementation:**
- Feature vectors per organism
- Graph Neural Network architecture
- Joint embedding of structure + features

### 4. Word Descriptions
Add natural language descriptions of organisms.

**Use Cases:**
- Semantic search
- Generate organism descriptions
- Link to literature

**Implementation:**
- Sentence embeddings (Sentence-BERT)
- Align with taxonomy embeddings
- Enable text-to-organism mapping

**Status:** Currently, we focus on graph structure (works excellently for hierarchy). Extensions will be added based on use cases.

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## 🎓 Citation

If you find this code useful for your research, please cite the following paper:

```bibtex
@incollection{nickel2017poincare,
  title = {Poincaré Embeddings for Learning Hierarchical Representations},
  author = {Nickel, Maximilian and Kiela, Douwe},
  booktitle = {Advances in Neural Information Processing Systems 30},
  editor = {I. Guyon and U. V. Luxburg and S. Bengio and H. Wallach and R. Fergus and S. Vishwanathan and R. Garnett},
  pages = {6341--6350},
  year = {2017},
  publisher = {Curran Associates, Inc.},
  url = {http://papers.nips.cc/paper/7213-poincare-embeddings-for-learning-hierarchical-representations.pdf}
}
```

## License

This code is licensed under [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
