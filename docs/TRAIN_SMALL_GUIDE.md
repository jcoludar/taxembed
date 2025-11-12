# Training on Small Dataset - Quick Guide

## Quick Start

```bash
python train_small.py
```

That's it! The script is pre-configured for the small dataset with optimal defaults.

## What You'll See

The script displays a **real-time metrics table** showing:

```
======================================================================================================
 Epoch |       Loss |     ΔLoss |  Improve |      Reg | MaxNorm | Outside |     Status
======================================================================================================
     1 | 0.234567 |       --- |      --- | 0.045678 |  0.9876 |   0.00% |      FIRST
     2 | 0.198765 |  -0.035802 |   -15.27% | 0.043210 |  0.9654 |   0.00% |  ✓ BETTER
     3 | 0.187654 |  -0.011111 |    -5.59% | 0.041234 |  0.9543 |   0.00% |  ✓ BETTER
     4 | 0.195432 |  +0.007778 |    +4.14% | 0.042567 |  0.9678 |   0.00% |  ✗ WORSE
```

### Columns Explained:
- **Epoch**: Current epoch number
- **Loss**: Training loss for this epoch
- **ΔLoss**: Change from previous epoch (negative = better)
- **Improve**: Percentage improvement (green ✓ = better, red ✗ = worse)
- **Reg**: Regularization loss (keeps embeddings in ball)
- **MaxNorm**: Maximum embedding norm (should be < 1.0)
- **Outside**: Percentage of embeddings outside ball (should be 0%)
- **Status**: Visual indicator of improvement

## Custom Parameters

### Faster Training (Fewer Epochs)
```bash
python train_small.py --epochs 50 --early-stopping 3
```

### Stronger Regularization (Keep embeddings tighter in ball)
```bash
python train_small.py --lambda-reg 0.2
```

### Larger Batches (Faster, less precise)
```bash
python train_small.py --batch-size 128
```

### Higher Learning Rate (Faster convergence, less stable)
```bash
python train_small.py --lr 0.01
```

### Save to Custom Location
```bash
python train_small.py --checkpoint models/my_model.pth
```

## Full Parameter List

```bash
python train_small.py --help
```

Available options:
- `--data`: Training data path (default: small transitive closure)
- `--checkpoint`: Output model path
- `--dim`: Embedding dimension (default: 10)
- `--epochs`: Maximum epochs (default: 100)
- `--early-stopping`: Patience before stopping (default: 5)
- `--batch-size`: Batch size (default: 64)
- `--n-negatives`: Negative samples per positive (default: 50)
- `--lr`: Learning rate (default: 0.005)
- `--margin`: Ranking loss margin (default: 0.2)
- `--lambda-reg`: Regularization strength (default: 0.1)
- `--gpu`: GPU device or -1 for CPU (default: -1)

## Output Files

After training, you'll get:
- `taxonomy_model_small_best.pth` - Best model (lowest loss)
- `taxonomy_model_small_epoch{N}.pth` - Last 5 epoch checkpoints
- `taxonomy_model_small.pth` - Final model

## Prerequisites

Make sure you've built the transitive closure first:
```bash
python build_transitive_closure.py
```

This creates `data/taxonomy_edges_small_transitive.pkl` which the training script needs.

## Next Steps

After training:

1. **Analyze hierarchy quality:**
   ```bash
   python analyze_hierarchy_hyperbolic.py
   ```

2. **Visualize embeddings:**
   ```bash
   python scripts/visualize_embeddings.py taxonomy_model_small_best.pth --highlight mammals
   ```

## Troubleshooting

### "Training data not found"
Run `python build_transitive_closure.py` first.

### Training is too slow
- Reduce batch size: `--batch-size 32`
- Reduce negative samples: `--n-negatives 20`
- Use fewer epochs: `--epochs 50`

### Embeddings escape the ball (Outside > 0%)
- Increase regularization: `--lambda-reg 0.2`
- Decrease learning rate: `--lr 0.001`

### Loss not improving
- Increase learning rate: `--lr 0.01`
- Increase margin: `--margin 0.3`
- Train longer: `--epochs 200 --early-stopping 10`
