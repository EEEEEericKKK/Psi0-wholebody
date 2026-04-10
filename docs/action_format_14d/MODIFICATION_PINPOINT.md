# Modification Locations - Pinpoint Guide

This document lists the **exact locations** of all modifications for the 14D action format.

## 1. Transform Layer Modifications

### File: `src/psi/config/transform.py`

**Location: Class definition (line ~705)**
```python
class HEPosttrainRepackTransform(RepackTransform):
```

**ADDED: New parameter (line ~712)**
```python
action_format: str = "full"  # Options: "full", "hands_only"
```

**ADDED: Three new methods (lines ~714-850)**

#### Method 1: Extract wrist poses
```python
def _extract_wrist_poses_from_joint_angles(self, data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract wrist 6DOF poses from joint angles or action data.
    Returns: (left_wrist_6dof, right_wrist_6dof) each of shape (T, 6) [xyz, rpy]
    """
```
- **Purpose**: Extracts 6DOF pose for each wrist from raw joint angles
- **Returns**: Two arrays of shape (T, 6) for left and right hands

#### Method 2: Extract head height
```python
def _extract_head_height(self, data: dict[str, Any]) -> np.ndarray:
    """
    Extract head/camera height from state data.
    Returns: array of shape (T, 1)
    """
```
- **Purpose**: Extracts head/camera height from observation states
- **Returns**: Array of shape (T, 1)

#### Method 3: Extract discrete token
```python
def _extract_discrete_token(self, data: dict[str, Any]) -> np.ndarray:
    """
    Extract or compute discrete action token.
    Returns: array of shape (T, 1)
    """
```
- **Purpose**: Extracts or computes discrete action/mode indicator
- **Returns**: Array of shape (T, 1)

**MODIFIED: Main __call__ method (line ~713)**
- **Added**: New branch for `if self.action_format == "hands_only":`
- **Logic**: 
  1. Calls the three new extraction methods
  2. Concatenates results into 14D action vector
  3. Applies delta actions if enabled
  4. Returns transformed data dict

**Original code**: Preserved in `else:` branch (lines ~808-832)

---

## 2. Statistics File

### File: `assets/stats/he_raw_rel_stats_14d.json` (NEW FILE)

**Created**: New file with normalized statistics for 14D actions

**Structure**:
```json
{
  "states": {
    "min": [14 values],
    "max": [14 values],
    "q01": [14 values],  // 1st percentile
    "q99": [14 values],  // 99th percentile
    "mean": [14 values],
    "std": [14 values]
  },
  "action": {
    "min": [14 values],
    "max": [14 values],
    "q01": [14 values],
    "q99": [14 values],
    "mean": [14 values],
    "std": [14 values]
  }
}
```

**Dimension order**:
```
[0-2]   Left hand XYZ
[3-5]   Left hand RPY
[6-8]   Right hand XYZ
[9-11]  Right hand RPY
[12]    Head height
[13]    Discrete token
```

---

## 3. Training Script

### File: `scripts/train/psi0/posttrain-he-psi0-14d.sh` (NEW FILE)

**Based on**: `scripts/train/psi0/posttrain-he-psi0.sh`

**Line-by-line changes**:

**Line 17**: Changed experiment name
```bash
--exp=posttrain_14d \
```

**Line 19**: Changed training name
```bash
--train.name=posttrain_14d \
```

**Line 42**: ADDED - Enable 14D format
```bash
--data.transform.repack.action-format=hands_only \
```

**Line 43**: Changed action dimension
```bash
--data.transform.repack.pad-action-dim=14 \
```

**Line 44**: Changed state dimension
```bash
--data.transform.repack.pad-state-dim=14 \
```

**Line 46**: Changed stats file path
```bash
--data.transform.field.stat-path=assets/stats/he_raw_rel_stats_14d.json
```

**Line 48**: Changed action dimension
```bash
--data.transform.field.pad-action-dim=14 \
```

**Line 49**: Changed state dimension
```bash
--data.transform.field.pad-state-dim=14 \
```

**Line 56**: Changed model action dimension
```bash
--model.action-dim=14 \
```

**Line 59**: Changed model observation dimension
```bash
--model.odim=14 \
```

---

## 4. Model Architecture

### File: `src/psi/models/psi0.py`

**NO CHANGES REQUIRED**

The model is already dimension-agnostic:

**Line ~784**: `ActionProjectionOut` class
```python
class ActionProjectionOut(nn.Module):
    def __init__(self, hidden_size, action_dim):
        super().__init__()
        self.linear = nn.Linear(hidden_size, action_dim, bias=True)
```
- The `action_dim` parameter is passed from config
- Automatically creates output layer with correct dimensions

**Line ~1049**: `ActionTransformerModel` class
```python
self.action_proj_out = ActionProjectionOut(
    self.config.hidden_dim, 
    self.config.action_dim  # ← Uses config value (14 or 36)
)
```

---

## 5. Model Configuration

### File: `src/psi/config/model_psi0.py`

**NO CHANGES REQUIRED**

Defaults remain at 7D (line 17):
```python
action_dim: int = 7
```

Training script overrides via command-line args:
```bash
--model.action-dim=14
```

---

## 6. Data Loading

### File: `src/psi/data/humanoid/he_raw_dataset.py`

**NO CHANGES REQUIRED**

The dataset returns raw data:
- `action.joint_angles`
- `observation.hand_joints`
- `observation.arm_joints`

Transform layer handles extraction of 14D format.

---

## Summary of File Modifications

| File | Status | Lines Changed |
|------|--------|---------------|
| `src/psi/config/transform.py` | **MODIFIED** | ~150 lines added |
| `assets/stats/he_raw_rel_stats_14d.json` | **NEW** | 102 lines |
| `scripts/train/psi0/posttrain-he-psi0-14d.sh` | **NEW** | 67 lines |
| `src/psi/models/psi0.py` | No change | 0 |
| `src/psi/config/model_psi0.py` | No change | 0 |
| `src/psi/data/humanoid/he_raw_dataset.py` | No change | 0 |

---

## Verification Checklist

Before training, verify:

- [ ] `src/psi/config/transform.py` has `action_format` parameter
- [ ] `src/psi/config/transform.py` has three new extraction methods
- [ ] `assets/stats/he_raw_rel_stats_14d.json` exists and has 14 dimensions
- [ ] `scripts/train/psi0/posttrain-he-psi0-14d.sh` sets `action-format=hands_only`
- [ ] All dimension parameters in training script set to 14
- [ ] Python syntax check passes: `python3 -m py_compile src/psi/config/transform.py`
- [ ] Bash syntax check passes: `bash -n scripts/train/psi0/posttrain-he-psi0-14d.sh`

---

## Quick Access Commands

```bash
# View transform modifications
git diff src/psi/config/transform.py

# Check new stats file
cat assets/stats/he_raw_rel_stats_14d.json | head -30

# Compare training scripts
diff scripts/train/psi0/posttrain-he-psi0.sh scripts/train/psi0/posttrain-he-psi0-14d.sh

# Verify syntax
python3 -m py_compile src/psi/config/transform.py
bash -n scripts/train/psi0/posttrain-he-psi0-14d.sh
```

---

## Rollback Instructions

To revert to original 36D format:

1. **Option A**: Use original training script
   ```bash
   bash scripts/train/psi0/posttrain-he-psi0.sh
   ```

2. **Option B**: Set action_format to "full"
   ```bash
   --data.transform.repack.action-format=full
   ```

3. **Option C**: Revert transform.py changes
   ```bash
   git checkout src/psi/config/transform.py
   ```

The original functionality is preserved by default (`action_format="full"`).
