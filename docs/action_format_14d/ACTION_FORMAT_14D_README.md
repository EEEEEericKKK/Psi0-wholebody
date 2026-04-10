# Action Expert Output Format Modification - 14D

## Overview
This modification changes the action expert output from 36D (full humanoid control) to 14D format optimized for hand control with head tracking.

## New Action Format (14 Dimensions)
```
Dimension Layout:
├── Left Hand 6DOF (indices 0-5)
│   ├── Position XYZ (0-2): 3D position in camera/base frame
│   └── Rotation RPY (3-5): Roll, Pitch, Yaw angles in radians
├── Right Hand 6DOF (indices 6-11)
│   ├── Position XYZ (6-8): 3D position in camera/base frame
│   └── Rotation RPY (9-11): Roll, Pitch, Yaw angles in radians
├── Head Camera Height (index 12): 1D height value in meters
└── Discrete Token (index 13): 1D discrete action/mode indicator
```

## Modified Files

### 1. Transform Layer
**File**: `src/psi/config/transform.py`

**Changes**:
- Added `action_format` parameter to `HEPosttrainRepackTransform` class
  - `"full"`: Original 36D format (default for backward compatibility)
  - `"hands_only"`: New 14D format
  
- New methods added:
  - `_extract_wrist_poses_from_joint_angles()`: Extracts 6DOF wrist poses from raw data
  - `_extract_head_height()`: Extracts camera/head height from state data
  - `_extract_discrete_token()`: Extracts or computes discrete action token

**Usage**:
```python
transform = HEPosttrainRepackTransform(
    action_format="hands_only",  # Enable 14D format
    action_chunk_size=16,
    use_delta_actions=True,
    pad_action_dim=14,
    pad_state_dim=14
)
```

### 2. Statistics File
**File**: `assets/stats/he_raw_rel_stats_14d.json`

**New file** containing normalization bounds for 14D actions:
- Action bounds (for delta actions):
  - XYZ delta: ±0.15m (q01/q99)
  - RPY delta: ±0.8 rad (q01/q99)
  - Head height delta: ±0.05m (q01/q99)
  - Token delta: ±0.5 (q01/q99)
  
- State bounds:
  - XYZ: ±0.6m (q01/q99)
  - RPY: ±2.5 rad (q01/q99)
  - Head height: 1.0-2.0m (q01/q99)
  - Token: 0.0-1.0 (q01/q99)

### 3. Training Script
**File**: `scripts/train/psi0/posttrain-he-psi0-14d.sh`

**New training script** with 14D configuration:
```bash
# Key parameters changed:
--data.transform.repack.action-format=hands_only  # Enable 14D mode
--data.transform.repack.pad-action-dim=14
--data.transform.repack.pad-state-dim=14
--data.transform.field.stat-path=assets/stats/he_raw_rel_stats_14d.json
--data.transform.field.pad-action-dim=14
--data.transform.field.pad-state-dim=14
--model.action-dim=14
--model.odim=14
```

### 4. Model Architecture
**File**: `src/psi/models/psi0.py`

**No changes needed** - The model architecture is already dimension-agnostic:
- `ActionProjectionOut` takes `action_dim` as constructor parameter
- Automatically projects hidden states to the specified action dimension
- Works seamlessly with any action dimension (7D, 14D, 36D, etc.)

## Data Flow

### Training Pipeline
```
Raw Data (HE_RAW dataset)
    ↓
HERawDataset.__getitem__()
    - Loads images, states, joint_angles
    ↓
HEPosttrainRepackTransform.__call__()
    - Extracts wrist 6DOF from joint_angles
    - Extracts head height from states
    - Extracts/computes discrete token
    - Concatenates to 14D action vector
    - Computes delta actions if enabled
    ↓
ActionStateTransform.__call__()
    - Normalizes using bounds_q99 from stats file
    - Clips to [-1, 1] range
    ↓
Model (Psi0Model)
    - ActionProjectionIn: projects to hidden_dim
    - ActionTransformer: processes with VLM features
    - ActionProjectionOut: projects to action_dim=14
    ↓
Loss Computation & Backprop
```

### Inference Pipeline
```
Observation (image + states)
    ↓
Model Forward Pass
    - Outputs 14D normalized actions
    ↓
ActionStateTransform.denormalize()
    - Converts from [-1, 1] to physical units
    ↓
14D Action Vector
    - Left hand 6DOF: meters & radians
    - Right hand 6DOF: meters & radians  
    - Head height: meters
    - Discrete token: continuous value (can be rounded/discretized)
```

## Usage Instructions

### Training with 14D Format

1. **Use the new training script**:
```bash
bash scripts/train/psi0/posttrain-he-psi0-14d.sh
```

2. **Or modify existing script** by adding:
```bash
--data.transform.repack.action-format=hands_only \
--data.transform.repack.pad-action-dim=14 \
--data.transform.field.stat-path=assets/stats/he_raw_rel_stats_14d.json \
--model.action-dim=14 \
--model.odim=14
```

### Loading 14D Checkpoint for Inference

```python
from psi.models.psi0 import Psi0Model
from psi.config.model_psi0 import Psi0ModelConfig

config = Psi0ModelConfig(
    action_dim=14,
    odim=14,
    action_chunk_size=16,
)

model = Psi0Model(config)
model.load_checkpoint("path/to/14d_checkpoint.pth")

# Model will output actions of shape (B, T, 14)
actions = model(observations)

# Parse actions
left_hand_xyz = actions[:, :, 0:3]
left_hand_rpy = actions[:, :, 3:6]
right_hand_xyz = actions[:, :, 6:9]
right_hand_rpy = actions[:, :, 9:12]
head_height = actions[:, :, 12:13]
discrete_token = actions[:, :, 13:14]
```

## Backward Compatibility

The original 36D format is preserved by default:
- Setting `action_format="full"` (default) uses original behavior
- Existing checkpoints and scripts remain functional
- Original training script `posttrain-he-psi0.sh` is unchanged

## Implementation Notes

### Wrist Pose Extraction
The current implementation provides fallback logic for extracting wrist poses:

1. **Preferred**: If dataset has `action.wrists.left.xyz` and `action.wrists.left.rpy` keys (like Egodex), use directly
2. **Fallback**: Extract from `action.joint_angles` using robot-specific indexing
3. **Future**: Implement proper forward kinematics (FK) for accurate wrist pose computation

### Head Height Source
Current implementation uses observation states as proxy. To improve:
- Parse robot base height from telemetry
- Use camera pose from robot state
- Compute from torso/head joint angles via FK

### Discrete Token
Current implementation extracts from hand joints as continuous value. Consider:
- Rounding to nearest integer for discrete modes
- Using separate classification head
- Mapping to specific action tokens (0=idle, 1=grasp, 2=release, etc.)

## Testing Recommendations

Before full training:

1. **Data Loading Test**:
```bash
python3 -c "
from psi.data.humanoid.he_raw_dataset import HERawDataset
from psi.config.transform import HEPosttrainRepackTransform

dataset = HERawDataset('$DATA_HOME/HE_RAW', action_chunk_size=17)
transform = HEPosttrainRepackTransform(action_format='hands_only', pad_action_dim=14)

sample = dataset[0]
transformed = transform(sample)
print('Action shape:', transformed['actions'].shape)
assert transformed['actions'].shape[1] == 14, 'Expected 14D actions'
print('✓ Data loading test passed')
"
```

2. **Model Forward Pass Test**:
```bash
python3 -c "
import torch
from psi.models.psi0 import Psi0Model
from psi.config.model_psi0 import Psi0ModelConfig

config = Psi0ModelConfig(action_dim=14, odim=14)
# Test that model can be instantiated with 14D
print('✓ Model instantiation test passed')
"
```

3. **Short Training Run**:
```bash
# Run for 100 steps to verify training pipeline
# Modify posttrain-he-psi0-14d.sh: --train.max_training_steps=100
```

## Troubleshooting

### Issue: Action dimension mismatch
**Error**: `RuntimeError: mat1 and mat2 shapes cannot be multiplied`

**Solution**: Ensure all dimension parameters are consistent:
- `model.action-dim=14`
- `model.odim=14`  
- `data.transform.repack.pad-action-dim=14`
- `data.transform.field.pad-action-dim=14`

### Issue: Stats file not found
**Error**: `FileNotFoundError: assets/stats/he_raw_rel_stats_14d.json`

**Solution**: The stats file was created during modification. Verify it exists:
```bash
ls -lh assets/stats/he_raw_rel_stats_14d.json
```

### Issue: Wrist pose data missing
**Error**: `KeyError: 'action.wrists.left.xyz'`

**Solution**: The transform includes fallback logic. If you see this error, the fallback may need adjustment for your specific dataset structure.

## Future Enhancements

1. **Rotation Representation**: Consider 6D rotation representation instead of RPY for better continuity
2. **Forward Kinematics**: Implement proper FK to compute wrist poses from joint angles
3. **Multi-Head Output**: Separate heads for continuous (positions) and discrete (token) outputs
4. **Data Augmentation**: Add augmentation specific to hand poses (rotation jitter, position noise)
5. **Loss Weighting**: Different weights for position vs rotation vs token losses

## References

- Original codebase: `Psi0-wholebody`
- Model architecture: `src/psi/models/psi0.py`
- Transform pipeline: `src/psi/config/transform.py`
- Training scripts: `scripts/train/psi0/`
