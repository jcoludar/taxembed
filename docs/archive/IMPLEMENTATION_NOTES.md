# Poincaré Embeddings for NCBI Taxonomy - Implementation Notes

## Overview
This project implements Poincaré embeddings to learn hierarchical representations of the NCBI taxonomy dataset. The goal is to embed organisms in hyperbolic space such that taxonomically related organisms are closer together.

## Key Modifications Made

### 1. Data Loading & Batch Processing
- **Fixed**: `BatchedDataset` requires `num_workers > 0` to start worker threads
  - Previous: `-ndproc 0` disabled threading, resulting in empty batches
  - Solution: Use `-ndproc 1` to enable single worker thread
  - File: `hype/graph_dataset.pyx`

### 2. Training Loop Improvements
- **Added**: Per-epoch checkpoint saving for early stopping
  - Saves `taxonomy_model_full_fixed_epochN.pth` after each epoch
  - Enables monitoring and early stopping without losing progress
  - File: `hype/train.py` (lines 96-104)

- **Fixed**: `UnboundLocalError` for `elapsed` variable
  - Initialized before batch loop
  - File: `hype/train.py`

### 3. Embedding Initialization (CRITICAL FIX)
- **Problem**: Embeddings initialized with scale=1e-4 ([-1e-4, 1e-4])
  - Initial norm ≈ 3e-4 (essentially zero)
  - Gradients too small to learn
  - Loss stayed constant at 3.931
  
- **Solution**: Changed initialization scale to 0.1
  - Initial norm ≈ 0.316 (reasonable)
  - Gradients large enough for learning
  - File: `hype/manifolds/manifold.py` (line 26)

### 4. Data Format Support
- **Added**: Support for `.edgelist` file format
  - Auto-detects file format (CSV vs whitespace-separated)
  - File: `embed.py`

### 5. Model Checkpointing
- **Added**: Final model save when evaluation is disabled
  - Prevents loss of trained model
  - File: `embed.py` (lines 309-314)

## Training Parameters

### Recommended Settings
```bash
python embed.py \
  -dset data/taxonomy_edges.mapped.edgelist \
  -checkpoint taxonomy_model_full_fixed.pth \
  -dim 10 \
  -epochs 50 \
  -negs 50 \
  -burnin 10 \
  -batchsize 32 \
  -model distance \
  -manifold poincare \
  -lr 0.1 \
  -gpu -1 \
  -ndproc 1 \
  -train_threads 1 \
  -eval_each 999999 \
  -fresh
```

### Learning Rate Schedule
- **Burn-in (epochs 0-9)**: lr = 0.1 × 0.01 = 0.001 (stable initialization)
- **Normal (epochs 10+)**: lr = 0.1 (faster learning)

## Monitoring Training

### Real-time Clustering Quality
```bash
python monitor_training.py
```

Shows per-epoch:
- Loss (should decrease)
- Primate distance (within-group)
- Random distance (all organisms)
- Ratio (random/primate) - should be > 1.5 for good clustering

### Training Progress
```bash
tail -f training_full.log
```

## Dataset

### NCBI Taxonomy Data
- **Nodes**: 2,705,747 organisms
- **Edges**: 2,705,745 parent-child relationships
- **Format**: Whitespace-separated edgelist with TaxID remapping
- **Files**:
  - `data/taxonomy_edges.mapped.edgelist`: Full dataset
  - `data/taxonomy_edges.mapping.tsv`: TaxID ↔ index mapping
  - `data/nodes.dmp`: NCBI taxonomy node information

### Data Preparation
```bash
python remap_edges.py
python prepare_taxonomy_data.py
```

## Known Issues & Limitations

1. **Training Speed**: CPU training is slow (~40 min per epoch on 2.7M edges)
   - Consider using GPU with `-gpu 0` (requires CUDA)
   - Or use smaller dataset for testing

2. **Hierarchical Learning**: Model requires many epochs to learn taxonomy
   - Burn-in phase helps but adds overhead
   - Consider reducing epochs for faster iteration

3. **Memory Usage**: Full model checkpoint is ~206MB
   - Each epoch saves separate checkpoint
   - Clean up old checkpoints if storage is limited

## Files Modified

- `embed.py`: Main training script
- `hype/train.py`: Training loop with per-epoch checkpointing
- `hype/manifolds/manifold.py`: Fixed embedding initialization
- `hype/graph_dataset.pyx`: Batch generation (no changes needed, but requires `ndproc > 0`)
- `hype/graph.py`: Data loading utilities
- `requirements.txt`: Dependencies

## New Files Added

- `monitor_training.py`: Real-time clustering quality monitoring
- `TRAINING_SUMMARY.md`: Training results and metrics
- `FINAL_ASSESSMENT.md`: Quality assessment
- `evaluate_full.py`: Model evaluation and UMAP visualization
- `visualize_primates_proper.py`: Primate-specific visualization
- `prepare_taxonomy_data.py`: Data preparation utilities
- `remap_edges.py`: TaxID remapping utility

## Next Steps

1. **GPU Training**: Use GPU for faster training
   - Modify `-gpu 0` parameter
   - Set `export PYTORCH_ENABLE_MPS_FALLBACK=1` for macOS

2. **Hyperparameter Tuning**:
   - Experiment with embedding dimension (currently 10)
   - Adjust learning rate and burn-in multiplier
   - Try different negative sampling counts

3. **Evaluation**:
   - Run `python evaluate_full.py` to get UMAP visualizations
   - Check nearest neighbors for sample organisms
   - Measure reconstruction quality

4. **Production Deployment**:
   - Save best epoch checkpoint
   - Create inference script for embedding new organisms
   - Benchmark performance on downstream tasks
