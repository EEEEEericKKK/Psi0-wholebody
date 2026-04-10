# Action Accuracy Evaluation Script

## Overview

The `scripts/eval_action_accuracy.py` script evaluates action prediction accuracy on the training/validation dataset. It can compute data preprocessing statistics and, when extended, evaluate trained model predictions.

## Current Features

✅ **Data Statistics**: Computes statistics on preprocessed actions  
✅ **14D Format Support**: Works with hands_only (14D) format  
✅ **36D Format Support**: Works with full humanoid (36D) format  
✅ **JSON Output**: Saves detailed metrics to JSON file  
⚠️ **Model Evaluation**: Requires implementation for trained model inference  

## Usage

### Basic Usage (Data Statistics)

```bash
# Evaluate 14D format data
python scripts/eval_action_accuracy.py \
  --num-samples 1000 \
  --action-format hands_only \
  --action-dim 14 \
  --output results_14d.json

# Evaluate 36D format data
python scripts/eval_action_accuracy.py \
  --num-samples 1000 \
  --action-format full \
  --action-dim 36 \
  --output results_36d.json
```

### With Checkpoint (Future)

```bash
# When model loading is implemented
python scripts/eval_action_accuracy.py \
  --checkpoint path/to/checkpoint.pth \
  --num-samples 1000 \
  --action-format hands_only \
  --action-dim 14 \
  --output model_accuracy.json
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--checkpoint` | str | None | Path to model checkpoint (optional) |
| `--config` | str | posttrain_he_psi0_config | Configuration name |
| `--data-root` | str | $DATA_HOME/HE_RAW | Data root directory |
| `--split` | str | val | Dataset split (train/val) |
| `--num-samples` | int | 1000 | Number of samples to evaluate |
| `--batch-size` | int | 8 | Batch size for evaluation |
| `--action-dim` | int | 14 | Action dimension (14 or 36) |
| `--output` | str | action_accuracy_results.json | Output file path |
| `--device` | str | cuda | Device (cuda/cpu) |
| `--action-format` | str | hands_only | Format (hands_only/full) |

## Output Format

The script generates a JSON file with the following structure:

```json
{
  "config": {
    "checkpoint": "None",
    "split": "val",
    "num_samples": 100,
    "action_dim": 14,
    "action_format": "hands_only"
  },
  "data_statistics": {
    "action_mean": [/* 14 values */],
    "action_std": [/* 14 values */],
    "action_min": [/* 14 values */],
    "action_max": [/* 14 values */],
    "per_dim_stats": [
      {
        "dimension": 0,
        "name": "left_x",
        "mean": 0.00844,
        "std": 0.01122,
        "min": -0.00916,
        "max": 0.04675
      },
      /* ... more dimensions ... */
    ]
  }
}
```

## Example Output (14D Format)

After running on 100 samples:

```
📊 Results Summary:
   Samples processed: 100
   Action shape: (1700, 14)
   Action range: [-0.093, 0.103]
   Action mean: 0.001 ± 0.017
```

**Per-dimension statistics** (hands_only format):
- `left_x`, `left_y`, `left_z`: Left hand position
- `left_roll`, `left_pitch`, `left_yaw`: Left hand rotation
- `right_x`, `right_y`, `right_z`: Right hand position
- `right_roll`, `right_pitch`, `right_yaw`: Right hand rotation
- `head_height`: Head/camera height
- `discrete_token`: Discrete action token

## Implementation Details

### Current Functionality

The script currently:
1. Loads the HE_RAW dataset
2. Applies data transforms (repack, normalization, model preprocessing)
3. Collects preprocessed actions
4. Computes statistics across all samples

### Data Flow

```
Raw Dataset
    ↓
HEPosttrainRepackTransform (extract 14D/36D actions)
    ↓
ActionStateTransform (normalize with bounds_q99)
    ↓
Psi0ModelTransform (image preprocessing)
    ↓
Statistics Computation
    ↓
JSON Output
```

### Extending for Model Evaluation

To add trained model evaluation:

1. **Load Model from Checkpoint**:
```python
from psi.models.psi0 import Psi0Model
from psi.config.model_psi0 import Psi0ModelConfig

config = Psi0ModelConfig(action_dim=14, odim=14)
model = Psi0Model(config)
checkpoint = torch.load(args.checkpoint)
model.load_state_dict(checkpoint["model"])
model.eval()
```

2. **Run Inference**:
```python
with torch.no_grad():
    pred_actions = model(batch)
```

3. **Compute Accuracy Metrics**:
```python
evaluator = ActionAccuracyEvaluator(model, dataset, device)
metrics = evaluator.evaluate(num_samples=1000)
```

The `ActionAccuracyEvaluator` class in the script already includes methods for computing:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Position accuracy within 5cm threshold
- Rotation accuracy within 5° threshold
- Per-dimension metrics

## Requirements

- Python 3.10+
- PyTorch
- Transformers
- NumPy
- tqdm

Install from project environment:
```bash
source .venv-psi/bin/activate
```

## Troubleshooting

### Error: DATA_HOME not set

**Solution**: Set the environment variable or use `--data-root`:
```bash
export DATA_HOME=/path/to/data
# OR
python scripts/eval_action_accuracy.py --data-root /path/to/data/HE_RAW ...
```

### Error: Out of memory

**Solution**: Reduce `--num-samples` or `--batch-size`:
```bash
python scripts/eval_action_accuracy.py --num-samples 100 --batch-size 4 ...
```

### Error: Checkpoint not found

**Solution**: Run without checkpoint for data-only mode or provide valid checkpoint path.

## Future Enhancements

- [ ] Add model checkpoint loading
- [ ] Add inference loop
- [ ] Compute prediction vs ground truth metrics
- [ ] Add visualization of per-dimension errors
- [ ] Support for test set evaluation
- [ ] Add confidence intervals
- [ ] Per-robot-type breakdown
- [ ] Temporal consistency metrics

## Related Documentation

- [Action Format 14D README](./ACTION_FORMAT_14D_README.md)
- [Training Verification](./TRAINING_VERIFICATION.md)
- [Modification Summary](./MODIFICATION_SUMMARY.md)

---

**Created**: 2026-04-02  
**Status**: Data statistics mode working, model evaluation pending
