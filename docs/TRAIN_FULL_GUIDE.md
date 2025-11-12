# Training on Full Dataset - Complete Guide

## Overview

Train hierarchical Poincaré embeddings on the **full NCBI taxonomy** (2.7M organisms) using the same successful approach from the small dataset.

---

## Step 1: Build Transitive Closure

**⏱️  Time Required: 30-60 minutes**

```bash
python build_transitive_closure_full.py
```

### What This Does:
- Loads full NCBI taxonomy (2.7M+ organisms)
- Computes all ancestor-descendant pairs (not just parent-child)
- Creates training data with depth metadata
- Generates ~100M+ training pairs

### Output Files:
- `data/taxonomy_edges_transitive.pkl` - Training data with metadata
- `data/taxonomy_edges_transitive.tsv` - Human-readable format
- `data/taxonomy_edges_transitive.edgelist` - Edge list format

---

## Step 2: Train Model

**⏱️  Time Required: 3-5 hours on M3 Mac CPU**

```bash
python train_full.py
```

### Default Configuration:
Based on successful small dataset training:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Dimensions | 10 | Same as small dataset |
| Batch Size | 128 | Larger for efficiency |
| Learning Rate | 0.005 | Proven effective |
| Margin | 0.2 | Ranking loss margin |
| Regularization (λ) | 0.1 | Ball constraint |
| Negative Samples | 50 | Hard negatives |
| Early Stopping | 10 epochs | Higher patience for large dataset |
| Max Epochs | 100 | Will likely stop earlier |

### What You'll See:

```
======================================================================================================
 Epoch |       Loss |      ΔLoss |  Improve |      Reg |  MaxNorm | Outside |     Status
======================================================================================================
Epoch   1/100:  45%|████████████▌              | 6845/15248 batches [05:23<06:42]
     1 | 0.987654 |        --- |      --- | 0.012345 |   1.0000 |   0.00% |      FIRST
       └─ Best: 0.987654 @ epoch 1 | Norms: [0.0889, 0.6147, 1.0000]
```

---

## Step 3: Monitor Training

Training will automatically:
- ✅ Display real-time progress bar per epoch
- ✅ Show loss improvements with color coding
- ✅ Save last 5 epoch checkpoints
- ✅ Save best model automatically
- ✅ Stop early when no improvement

### Expected Timeline:
- Epoch 1-5: ~5-10 min/epoch (initial learning)
- Epoch 6-20: ~5-10 min/epoch (convergence)
- Epoch 20+: Likely early stopping

---

## Parameter Tuning

### For Faster Training (Less Accuracy):
```bash
python train_full.py \
  --batch-size 256 \
  --n-negatives 30 \
  --early-stopping 5
```

### For Better Quality (Slower):
```bash
python train_full.py \
  --batch-size 64 \
  --n-negatives 100 \
  --early-stopping 15 \
  --lr 0.003
```

### For Different Checkpoint Name:
```bash
python train_full.py --checkpoint taxonomy_model_full_v2.pth
```

---

## Output Files

After training:
- `taxonomy_model_full_best.pth` - Best model (lowest loss)
- `taxonomy_model_full_epoch{N}.pth` - Last 5 epoch checkpoints
- `taxonomy_model_full.pth` - Final model

Each checkpoint contains:
- Embeddings (2.7M × 10 dimensions)
- Training metadata (epoch, loss, etc.)
- Model state for resuming

---

## Comparison: Small vs Full

| Metric | Small Dataset | Full Dataset |
|--------|--------------|--------------|
| Organisms | 111K | 2.7M |
| Training Pairs | 975K | ~100M+ |
| Model Size | 3.5 MB | ~210 MB |
| Training Time | 2-3 hours | 3-5 hours |
| Epoch Time | ~3 min | ~5-10 min |
| Best Epoch | 28 | TBD |
| Final Loss | 0.472 | TBD |

---

## Troubleshooting

### "Training data not found"
Run `python build_transitive_closure_full.py` first. This takes 30-60 minutes.

### Out of Memory
```bash
python train_full.py --batch-size 64
```

### Training Too Slow
```bash
python train_full.py --batch-size 256 --n-negatives 30
```

### Want to Use GPU
```bash
python train_full.py --gpu 0
```
(Requires CUDA-enabled GPU)

---

## After Training

### 1. Check Results
```bash
python check_model.py
```

### 2. Plot Analysis
```bash
python plot_best_epoch.py
```

### 3. Visualize Embeddings
```bash
python visualize_multi_groups.py taxonomy_model_full_best.pth
```

### 4. Analyze Hierarchy
```bash
python analyze_hierarchy_hyperbolic.py
```

---

## Expected Results

Based on small dataset success, the full model should achieve:
- ✅ Loss reduction: ~50% improvement (0.98 → 0.47)
- ✅ All embeddings inside ball (0% outside)
- ✅ Proper hierarchical clustering
- ✅ Meaningful nearest neighbors
- ✅ Clear taxonomic group separation

---

## Key Differences from Small Dataset

1. **Batch Size**: 128 (vs 64) - larger for efficiency
2. **Early Stopping**: 10 epochs (vs 5) - more patience for complex dataset
3. **Training Time**: 3-5 hours (vs 2-3 hours)
4. **Model Size**: 210 MB (vs 3.5 MB)

All other parameters are identical to the successful small dataset training!

---

## Pro Tips

1. **Run overnight**: Full training takes 3-5 hours
2. **Monitor early epochs**: If loss doesn't decrease in first 3 epochs, something's wrong
3. **Save intermediate checkpoints**: Last 5 epochs auto-saved
4. **Check max norm**: Should always be ≤ 1.0 (ball constraint)
5. **Compare to small**: Similar convergence pattern expected

---

## Resume Training (If Interrupted)

Training will need to be restarted from scratch. The current implementation doesn't support resuming. Consider:
- Using a screen/tmux session
- Running overnight when uninterrupted
- Monitoring the first few epochs to ensure proper training

---

## Next Steps After Full Training

1. Compare full vs small model performance
2. Test on downstream tasks (e.g., taxonomic prediction)
3. Evaluate embedding quality metrics
4. Visualize major taxonomic groups
5. Query nearest neighbors for validation
