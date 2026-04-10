# Action Expert Modification Summary

## What Was Changed

The action expert has been modified to output **14 dimensions** instead of 36:
- **Left hand 6DOF**: position (xyz) + rotation (rpy) = 6 dims
- **Right hand 6DOF**: position (xyz) + rotation (rpy) = 6 dims  
- **Head camera height**: 1 dim
- **Discrete token**: 1 dim

## Files Modified

1. **`src/psi/config/transform.py`**
   - Modified `HEPosttrainRepackTransform` class
   - Added `action_format` parameter: "full" (36D, default) or "hands_only" (14D)
   - Added extraction methods for wrist poses, head height, and discrete token

2. **`assets/stats/he_raw_rel_stats_14d.json`** (NEW)
   - Normalization statistics for 14D action space
   - Bounds for hand positions, rotations, head height, and token

3. **`scripts/train/psi0/posttrain-he-psi0-14d.sh`** (NEW)
   - Training script configured for 14D format
   - Sets `--model.action-dim=14` and `--data.transform.repack.action-format=hands_only`

4. **`ACTION_FORMAT_14D_README.md`** (NEW)
   - Comprehensive documentation

## How to Use

### Training
```bash
# Use the new 14D training script
bash scripts/train/psi0/posttrain-he-psi0-14d.sh
```

### Key Training Arguments
```bash
--data.transform.repack.action-format=hands_only   # Enable 14D mode
--data.transform.repack.pad-action-dim=14
--data.transform.field.stat-path=assets/stats/he_raw_rel_stats_14d.json
--model.action-dim=14
--model.odim=14
```

## Model Architecture Impact

**No architecture changes needed!** The model automatically adapts:
- `ActionProjectionOut` layer projects to `action_dim` (configurable)
- Works with any dimension: 7D, 14D, 36D, etc.

## Backward Compatibility

✅ **Original 36D format still works**
- Use `action_format="full"` (default)
- Original script `posttrain-he-psi0.sh` unchanged

## Quick Verification

```bash
# Check syntax
python3 -m py_compile src/psi/config/transform.py

# Verify stats file exists
ls -lh assets/stats/he_raw_rel_stats_14d.json

# Check training script
bash -n scripts/train/psi0/posttrain-he-psi0-14d.sh
```

## Action Format Details

```
Index  | Component           | Range
-------|---------------------|------------------
0-2    | Left hand XYZ       | ±0.15m (delta)
3-5    | Left hand RPY       | ±0.8 rad (delta)
6-8    | Right hand XYZ      | ±0.15m (delta)
9-11   | Right hand RPY      | ±0.8 rad (delta)
12     | Head height         | ±0.05m (delta)
13     | Discrete token      | ±0.5 (delta)
```

## Notes

- Using **delta actions** by default (change from frame to frame)
- RPY rotations use Euler angles (roll-pitch-yaw)
- Head height and token extraction use fallback logic from available state data
- For production: consider implementing proper forward kinematics for wrist poses

## Testing Before Full Training

1. Load a data sample and verify action shape is (T, 14)
2. Run model forward pass with 14D config
3. Short training run (100 steps) to verify pipeline

See `ACTION_FORMAT_14D_README.md` for detailed testing instructions.
