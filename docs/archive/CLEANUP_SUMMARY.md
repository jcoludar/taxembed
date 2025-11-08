# Repository Cleanup Summary

## What Was Removed

### Checkpoint Files
- **Removed:** 569 checkpoint files
- **File types:** `*.pth`, `*.pth.*`
- **Total size freed:** ~100+ GB

### Log Files
- **Removed:** All log files
- **Files:** `training.log`, `training_full.log`, `nohup.out`

### Visualization Files
- **Removed:** All generated PNG files
- **Files:** `umap_*.png`, `umap_projection.png`, etc.

### Redundant Scripts
**Consolidated into `scripts/visualize_embeddings.py`:**
- ❌ `visualize_primates.py`
- ❌ `visualize_primates_proper.py`
- ❌ `visualize_primates_small_only.py`
- ❌ `visualize_by_taxonomy.py`
- ❌ `visualize_trained_small_dataset.py`

**Old shell scripts removed:**
- ❌ `train-mammals.sh`
- ❌ `train-nouns.sh`
- ❌ `train_taxonomy.sh`
- ❌ `train_taxonomy_quick.sh`

## New Universal Tools Created

### 1. `scripts/visualize_embeddings.py` ⭐
**Purpose:** One script to visualize any checkpoint

**Features:**
- Works with any checkpoint file
- Highlight any taxonomic group (primates, mammals, bacteria, etc.)
- Show only specific groups
- Nearest neighbor analysis
- Automatic output naming
- Configurable sampling

**Usage:**
```bash
# Basic
python scripts/visualize_embeddings.py model.pth

# Highlight primates
python scripts/visualize_embeddings.py model.pth --highlight primates

# Only show mammals
python scripts/visualize_embeddings.py model.pth --only mammals

# Custom sample size
python scripts/visualize_embeddings.py model.pth --sample 50000
```

### 2. `scripts/cleanup_repo.sh`
**Purpose:** Automated repository cleanup

**Features:**
- Interactive confirmation
- Removes checkpoints, logs, visualizations
- Removes redundant scripts
- Reports what will be deleted

**Usage:**
```bash
./scripts/cleanup_repo.sh
```

### 3. `scripts/validate_data.py`
**Purpose:** Data quality validation

**Features:**
- Validates edgelist format
- Checks mapping consistency
- Verifies sequential indices
- Detects header bugs

**Usage:**
```bash
python scripts/validate_data.py small
python scripts/validate_data.py full
```

## Repository Structure (After Cleanup)

```
taxembed/
├── Core Scripts (Root)
│   ├── embed.py                      # Main training
│   ├── prepare_taxonomy_data.py      # Data preparation
│   ├── remap_edges.py                # Data remapping
│   ├── monitor_training.py           # Training monitor
│   ├── evaluate_full.py              # Evaluation
│   ├── evaluate_and_visualize.py     # Combined eval
│   ├── nn_demo.py                    # Quick demo
│   └── reconstruction.py             # Reconstruction eval
│
├── src/taxembed/                     # Source code
│   ├── manifolds/                    # Hyperbolic manifolds
│   ├── models/                       # Embedding models
│   ├── datasets/                     # Data loading
│   └── utils/                        # Utilities
│
├── scripts/                          # Organized utilities
│   ├── visualize_embeddings.py       # ⭐ Universal visualization
│   ├── validate_data.py              # ⭐ Data validation
│   ├── cleanup_repo.sh               # ⭐ Repository cleanup
│   ├── regenerate_data.sh            # Data regeneration
│   ├── prepare_data.py               # Wrappers
│   ├── remap_data.py
│   ├── monitor.py
│   ├── evaluate.py
│   └── train.py
│
├── tests/                            # Unit tests
│   ├── __init__.py
│   └── test_example.py
│
├── hype/                             # Original package (backward compat)
│
├── Configuration
│   ├── pyproject.toml                # Project config (uv)
│   ├── ruff.toml                     # Linter config
│   ├── Makefile                      # Convenience commands
│   ├── setup.py                      # C++ extensions
│   ├── requirements.txt              # Legacy requirements
│   └── .gitignore                    # Git ignore rules
│
└── Documentation
    ├── README.md                     # Main documentation
    ├── QUICKSTART.md                 # Quick start guide
    ├── GETTING_STARTED.md            # Getting started
    ├── STRUCTURE.md                  # Project structure
    ├── SCRIPTS_GUIDE.md              # ⭐ Script documentation
    ├── CONTRIBUTING.md               # Contribution guide
    ├── DATA_FIXES_SUMMARY.md         # Data bug fixes
    ├── DATA_HANDLING_REVIEW.md       # Data analysis
    ├── CLEANUP_SUMMARY.md            # This file
    ├── RESTRUCTURING_SUMMARY.md      # Restructuring notes
    ├── RESTRUCTURING_COMPLETE.md     # Restructuring completion
    ├── PROJECT_TREE.txt              # Visual tree
    ├── TRAINING_SUMMARY.md           # Training notes
    ├── IMPLEMENTATION_NOTES.md       # Implementation notes
    ├── FINAL_ASSESSMENT.md           # Quality assessment
    └── CODE_OF_CONDUCT.md            # Code of conduct
```

## Updated .gitignore

Now properly ignores:
- Checkpoints: `*.pth`, `*.pth.*`
- Logs: `*.log`, `training*.log`, `nohup.out`
- Visualizations: `*.png`, `*.jpg`
- Data: `data/`
- Build artifacts: `build/`, `dist/`, `*.so`
- Python cache: `__pycache__/`, `*.pyc`
- Virtual environments: `venv/`, `venv311/`
- IDE files: `.idea/`, `.vscode/`

## Benefits of Cleanup

### Before Cleanup
- 569 checkpoint files (~100+ GB)
- 8 redundant visualization scripts
- 4 old shell scripts
- Numerous log and PNG files
- Confusing script organization

### After Cleanup
- ✅ Clean repository
- ✅ 1 universal visualization tool (replaces 5 scripts)
- ✅ Clear script organization
- ✅ Comprehensive documentation
- ✅ Proper .gitignore
- ✅ Easy to maintain

## Workflow Examples

### Training a Model
```bash
python embed.py \
  -dset data/taxonomy_edges_small.mapped.edgelist \
  -checkpoint my_model.pth \
  -dim 10 -epochs 50 -negs 50 -burnin 10 \
  -batchsize 32 -model distance -manifold poincare \
  -lr 0.1 -gpu -1 -ndproc 1 -train_threads 1 \
  -eval_each 999999 -fresh
```

### Visualizing Results
```bash
# Highlight primates
python scripts/visualize_embeddings.py my_model.pth --highlight primates

# Only show mammals
python scripts/visualize_embeddings.py my_model.pth --only mammals --sample 30000

# Basic visualization with nearest neighbors
python scripts/visualize_embeddings.py my_model.pth --nearest 10
```

### Validating Data
```bash
python scripts/validate_data.py small
```

### Cleaning Up
```bash
./scripts/cleanup_repo.sh
```

## Key Improvements

### 1. Consolidation
- **Before:** 5 separate visualization scripts, each hardcoded for specific use cases
- **After:** 1 universal tool that works with any checkpoint and any taxonomic group

### 2. Documentation
- **Before:** Minimal script documentation
- **After:** Comprehensive `SCRIPTS_GUIDE.md` with usage examples

### 3. Organization
- **Before:** Scripts scattered in root directory
- **After:** Organized in `scripts/` directory with clear purposes

### 4. Maintenance
- **Before:** Hard to understand which scripts to use
- **After:** Clear documentation and single universal tool

### 5. Disk Space
- **Before:** 100+ GB of old checkpoints
- **After:** Clean repository, generate files as needed

## Future Maintenance

### When Training
1. Train model: `python embed.py ...`
2. Visualize: `python scripts/visualize_embeddings.py <checkpoint> --highlight <group>`
3. Clean up: `./scripts/cleanup_repo.sh` (when done)

### When Adding Features
- Add to `scripts/` directory
- Update `SCRIPTS_GUIDE.md`
- Follow naming convention: `<action>_<noun>.py`

### When Sharing Code
- Repository is now clean and presentable
- Clear documentation for users
- No large binary files
- Professional organization

## Recommendations

1. **Use the universal visualization tool** for all embedding visualizations
2. **Clean up regularly** with `./scripts/cleanup_repo.sh`
3. **Validate data** before training with `scripts/validate_data.py`
4. **Follow the scripts guide** for standard workflows
5. **Keep documentation updated** when adding new scripts

## Summary

✅ **Removed:** 569 checkpoints, 8 redundant scripts, numerous temp files
✅ **Created:** Universal visualization tool, cleanup script, comprehensive documentation
✅ **Organized:** Scripts in proper directories, clear naming, good documentation
✅ **Professional:** Clean repo ready for production use and sharing

The repository is now **production-ready** with a clean, maintainable structure! 🎉
