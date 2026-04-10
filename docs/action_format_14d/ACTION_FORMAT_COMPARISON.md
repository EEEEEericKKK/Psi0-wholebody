# Action Format Comparison

## Before: 36D Format (Full Humanoid)
```
┌─────────────────────────────────────────────────────────────┐
│                    36D Action Vector                         │
├─────────────────────────────────────────────────────────────┤
│ Left Hand Fingers (7D)    ← thumb, index, middle, ring, ... │
│ Right Hand Fingers (7D)   ← thumb, index, middle, ring, ... │
│ Left Arm Joints (7D)      ← shoulder to wrist joints        │
│ Right Arm Joints (7D)     ← shoulder to wrist joints        │
│ Additional States (8D)    ← torso, head, etc.               │
└─────────────────────────────────────────────────────────────┘
         Padded from 28D to 36D
```

## After: 14D Format (Hand-Centric)
```
┌─────────────────────────────────────────────────────────────┐
│                    14D Action Vector                         │
├─────────────────────────────────────────────────────────────┤
│ Left Hand 6DOF (6D)                                         │
│   ├─ Position XYZ (3D)    ← world/camera frame position    │
│   └─ Rotation RPY (3D)    ← roll, pitch, yaw angles        │
├─────────────────────────────────────────────────────────────┤
│ Right Hand 6DOF (6D)                                        │
│   ├─ Position XYZ (3D)    ← world/camera frame position    │
│   └─ Rotation RPY (3D)    ← roll, pitch, yaw angles        │
├─────────────────────────────────────────────────────────────┤
│ Head/Camera Height (1D)   ← height in meters               │
├─────────────────────────────────────────────────────────────┤
│ Discrete Token (1D)       ← mode/gripper/action type       │
└─────────────────────────────────────────────────────────────┘
         No padding needed (native 14D)
```

## Data Flow Diagram

### Original Pipeline (36D)
```
┌──────────────┐
│ Raw Dataset  │ action.joint_angles (26D or 28D)
└──────┬───────┘
       │
       v
┌──────────────────────────────┐
│ HEPosttrainRepackTransform   │
│ action_format="full"         │
└──────┬───────────────────────┘
       │ Repack & pad fingers
       v
┌──────────────┐
│ Actions (36D)│ Full humanoid control
└──────┬───────┘
       │
       v
┌──────────────────────────────┐
│ ActionStateTransform         │
│ Normalize with stats (36D)   │
└──────┬───────────────────────┘
       │
       v
┌──────────────┐
│ Model Input  │ Normalized [-1, 1]
└──────┬───────┘
       │
       v
┌──────────────────────────────┐
│ ActionProjectionOut          │
│ Linear(hidden_dim → 36)      │
└──────┬───────────────────────┘
       │
       v
┌──────────────┐
│ Output (36D) │ Predicted actions
└──────────────┘
```

### Modified Pipeline (14D)
```
┌──────────────┐
│ Raw Dataset  │ action.joint_angles + states
└──────┬───────┘
       │
       v
┌─────────────────────────────────────────┐
│ HEPosttrainRepackTransform              │
│ action_format="hands_only"              │
│  ├─ Extract left wrist 6DOF             │
│  ├─ Extract right wrist 6DOF            │
│  ├─ Extract head height                 │
│  └─ Extract discrete token              │
└──────┬──────────────────────────────────┘
       │ Concatenate
       v
┌──────────────┐
│ Actions (14D)│ Hands + head + token
└──────┬───────┘
       │
       v
┌──────────────────────────────┐
│ ActionStateTransform         │
│ Normalize with stats (14D)   │
└──────┬───────────────────────┘
       │
       v
┌──────────────┐
│ Model Input  │ Normalized [-1, 1]
└──────┬───────┘
       │
       v
┌──────────────────────────────┐
│ ActionProjectionOut          │
│ Linear(hidden_dim → 14)      │
└──────┬───────────────────────┘
       │
       v
┌──────────────┐
│ Output (14D) │ Predicted actions
└──────────────┘
```

## Configuration Changes

### Training Script Arguments

**Before (36D)**:
```bash
--model.action-dim=36
--model.odim=36
--data.transform.repack.pad-action-dim=36
--data.transform.field.pad-action-dim=36
--data.transform.field.stat-path=assets/stats/he_raw_rel_stats_combined_no_static.json
```

**After (14D)**:
```bash
--data.transform.repack.action-format=hands_only  # ← NEW
--model.action-dim=14
--model.odim=14
--data.transform.repack.pad-action-dim=14
--data.transform.field.pad-action-dim=14
--data.transform.field.stat-path=assets/stats/he_raw_rel_stats_14d.json
```

## Benefits of 14D Format

✅ **Simpler action space**
   - Focuses on task-relevant end-effector control
   - No need to learn finger kinematics

✅ **Faster training**
   - Smaller output dimension
   - Fewer parameters in final projection layer

✅ **Better generalization**
   - Position + orientation is more transferable
   - Less sensitive to robot-specific joint configs

✅ **Easier deployment**
   - 6DOF targets map directly to inverse kinematics
   - Can control different robot morphologies

✅ **Explicit mode control**
   - Discrete token for state machine control
   - Head height for body pose awareness

## Trade-offs

⚠️ **Loss of finger dexterity**
   - No individual finger control
   - Need separate gripper control

⚠️ **Requires IK solver**
   - 6DOF targets → joint angles via IK
   - May need per-robot IK implementation

⚠️ **Simplified body model**
   - No torso/leg control
   - Assumes fixed base or separate control

## When to Use Each Format

**Use 36D (full)** when:
- Need fine-grained finger control
- Training robot-specific behaviors
- Have complete joint-level demonstrations

**Use 14D (hands_only)** when:
- Focus on bimanual manipulation
- Want cross-robot generalization
- Using high-level task specifications
- Have Cartesian space demonstrations
