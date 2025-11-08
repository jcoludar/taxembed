# Checkpoint Management

## Overview

To prevent disk space issues during training, the repository includes automatic checkpoint management that keeps only the **20 most recent checkpoints** per model.

## Automatic Management During Training

The `train_with_early_stopping.py` script includes built-in checkpoint management:

```bash
python train_with_early_stopping.py
```

**Features:**
- ✅ Monitors training in real-time
- ✅ Implements early stopping (patience=6 epochs)
- ✅ Automatically keeps only 20 most recent checkpoints
- ✅ Deletes old checkpoints as new ones are created
- ✅ Shows space-saving notifications

**Example Output:**
```
💾 Checkpoint saved: taxonomy_model_small_early_stop_epoch12.pth (keeping 12/20)
💾 Checkpoint saved: taxonomy_model_small_early_stop_epoch13.pth (keeping 13/20)
...
💾 Checkpoint saved: taxonomy_model_small_early_stop_epoch21.pth (keeping 20/20)
🗑️  Deleted old checkpoint: taxonomy_model_small_early_stop_epoch1.pth
💾 Checkpoint saved: taxonomy_model_small_early_stop_epoch22.pth (keeping 20/20)
🗑️  Deleted old checkpoint: taxonomy_model_small_early_stop_epoch2.pth
```

## Manual Cleanup

If you have old checkpoints from previous training runs, use the cleanup script:

```bash
# See what would be deleted (dry run)
python cleanup_old_checkpoints.py --dry-run

# Delete old checkpoints, keep 20 most recent
python cleanup_old_checkpoints.py

# Keep only 10 most recent
python cleanup_old_checkpoints.py --keep 10

# Clean specific directory
python cleanup_old_checkpoints.py --directory /path/to/checkpoints
```

## Checkpoint Naming Formats

The system recognizes these checkpoint patterns:

1. **Epoch-based:** `model_name_epoch123.pth`
2. **Dot-notation:** `model_name.pth.123`

Both formats are automatically detected and managed.

## How It Works

### During Training
1. Training saves checkpoints each epoch: `model_epoch0.pth`, `model_epoch1.pth`, etc.
2. The monitoring script tracks all saved checkpoints in a queue (max 20)
3. When a 21st checkpoint is saved, the oldest (epoch 0) is automatically deleted
4. This continues throughout training, maintaining exactly 20 checkpoints

### Manual Cleanup
1. Script scans directory for checkpoint files
2. Groups checkpoints by model name
3. Sorts by epoch number
4. Keeps N most recent, deletes the rest

## Disk Space Savings

**Example for small dataset:**
- Each checkpoint: ~50 MB
- Without management: 200 epochs × 50 MB = **10 GB**
- With management: 20 checkpoints × 50 MB = **1 GB**
- **Savings: 9 GB (90%)**

**Example for full dataset:**
- Each checkpoint: ~219 MB
- Without management: 200 epochs × 219 MB = **43.8 GB**
- With management: 20 checkpoints × 219 MB = **4.4 GB**
- **Savings: 39.4 GB (90%)**

## Configuration

### Change Maximum Checkpoints

Edit `train_with_early_stopping.py`:

```python
# Keep 30 checkpoints instead of 20
summary = train_with_monitoring(cmd, patience=6, max_checkpoints=30, ...)
```

### Disable Auto-Cleanup

If you want to keep all checkpoints, set a very high limit:

```python
summary = train_with_monitoring(cmd, patience=6, max_checkpoints=9999, ...)
```

## Best Practices

1. **Use early stopping:** Training typically converges in 20-50 epochs, so keeping 20 checkpoints is sufficient
2. **Run cleanup before training:** Clear out old experiments to start fresh
3. **Monitor disk space:** Use `df -h` to check available space
4. **Keep final model:** The script always saves the final model separately at the end

## Files Involved

- `train_with_early_stopping.py` - Training with auto-cleanup
- `cleanup_old_checkpoints.py` - Manual cleanup utility
- `hype/train.py` - Core training loop (saves per-epoch checkpoints)

## Troubleshooting

### Checkpoints Not Being Deleted

Check if checkpoint pattern matches. The script looks for:
- `Saved checkpoint: <filename>` in training output

### Running Out of Disk Space

1. Stop training (Ctrl+C)
2. Run manual cleanup: `python cleanup_old_checkpoints.py --keep 5`
3. Check space: `df -h`
4. Resume training

### Accidentally Deleted Important Checkpoint

The final model is always saved separately as `taxonomy_model_small_early_stop.pth` (without epoch number), which is never deleted.

## Summary

✅ **Automatic:** Training script manages checkpoints automatically  
✅ **Efficient:** Saves 90% disk space  
✅ **Safe:** Always keeps the 20 most recent checkpoints  
✅ **Manual option:** Cleanup script for existing files  
✅ **Configurable:** Adjust limits as needed  
