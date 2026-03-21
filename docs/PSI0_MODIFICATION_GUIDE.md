# Psi0 Model Modification Guide

This guide explains the Psi0 model structure and how to modify it for custom action representations.

---

## 1. Psi0 Model Architecture Overview

### 1.1 High-Level Structure

The Psi0 model consists of two main components:

```
Psi0Model
├── vlm_model (Qwen3VLForConditionalGeneration)  # Vision-Language backbone
│   └── Processes images + text instructions → hidden states
│
└── action_header (ActionTransformerModel)       # Action prediction head
    └── Processes VLM features + proprioception + noisy actions → action output
```

**Key File**: `src/psi/models/psi0.py`

### 1.2 Psi0Model Class (Lines 1480-1600)

```python
class Psi0Model(nn.Module):
    def __init__(self, model_cfg, vlm_model: Qwen3VLForConditionalGeneration):
        # Creates action_header based on config
        if model_cfg.use_dit:
            self.action_header = DiTActionTransformerModel(...)
        else:
            self.action_header = ActionTransformerModel(...)

        self.vlm_model = vlm_model  # Frozen during post-training
```

**Two Execution Modes**:
- **Training**: `forward()` computes diffusion loss
- **Inference**: `predict_action()` iteratively denoises actions

---

## 2. Action Expert Architecture (ActionTransformerModel)

The action expert is a **diffusion-based transformer** that predicts actions conditioned on VLM features.

### 2.1 Component Overview

**File**: `src/psi/models/psi0.py`, Lines 1008-1208

```
ActionTransformerModel
├── time_ins_embed      # Timestep embedding (diffusion step)
├── obs_proj            # ObservationProjection - processes VLM hidden states + proprio
├── action_proj_in      # ActionProjectionIn - encodes noisy actions
├── transformer_blocks  # List of VLATransformerBlock (joint attention)
└── action_proj_out     # ActionProjectionOut - final action prediction
```

### 2.2 Key Components Detail

#### 2.2.1 ObservationProjection (Lines 423-745)

Projects VLM features and proprioception into observation tokens.

```python
class ObservationProjection(nn.Module):
    def __init__(self, ..., odim: int, view_feature_dim: int, ...):
        # views_proj: Projects VLM hidden states
        self.views_proj = nn.Linear(view_feature_dim, output_dim)  # 1920 → hidden_dim

        # _obs_proc: Projects proprioception
        self._obs_proc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(odim, output_dim)  # odim → hidden_dim
        )
```

**Inputs**:
- `views`: VLM hidden states `(B, V, S, 1920)`
- `obs`: Proprioception `(B, 1, odim)`

**Output**: Observation tokens `(B, S+1, hidden_dim)`

#### 2.2.2 ActionProjectionIn (Lines 746-782)

Encodes noisy actions for diffusion process.

```python
class ActionProjectionIn(nn.Module):
    def __init__(self, action_pred_horizon, action_dim, output_dim):
        self.ac_proj = nn.Sequential(
            nn.Linear(action_dim, action_dim),        # action_dim → action_dim
            nn.GELU(approximate="tanh"),
            nn.Linear(action_dim, output_dim),        # action_dim → hidden_dim
        )
        # Learnable positional encoding for action horizon
        self.dec_pos = nn.Parameter(torch.empty(action_pred_horizon, output_dim))
```

**Input**: Noisy actions `(B, Tp, action_dim)`
**Output**: Action tokens `(B, Tp, hidden_dim)`

#### 2.2.3 VLATransformerBlock (Lines 814-1006)

Joint attention block between action and observation tokens.

```python
class VLATransformerBlock(nn.Module):
    def forward(self, action_hidden_states, obs_hidden_states, temb, ...):
        # 1. AdaLN normalization with timestep
        norm_action = self.norm1_act(action_hidden_states, emb=temb)
        norm_obs = self.norm1_obs(obs_hidden_states, emb=temb)

        # 2. Joint cross-attention
        act_attn, obs_attn = self.attn(norm_action, norm_obs)

        # 3. Feed-forward networks
        action_out = self.ff_act(norm_action)
        obs_out = self.ff_obs(norm_obs)

        return action_out, obs_out
```

#### 2.2.4 ActionProjectionOut (Lines 784-810)

Final layer that outputs action predictions.

```python
class ActionProjectionOut(nn.Module):
    def __init__(self, hidden_size, action_dim):
        self.norm_final = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, action_dim)  # hidden_dim → action_dim
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size)
        )

    def forward(self, x, t):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=-1)
        x = x * scale + shift
        return self.linear(x)  # (B, Tp, action_dim)
```

### 2.3 Forward Pass Flow

```
Input:
  - VLM hidden states (B, 1, seq_len, 1920)
  - Proprioception (B, 1, odim)
  - Noisy actions (B, Tp, action_dim)
  - Timestep (B,)

1. time_ins_embed(timestep) → temb (B, hidden_dim)

2. obs_proj(views, obs) → obs_tokens (B, S+1, hidden_dim)

3. action_proj_in(noisy_actions) → action_tokens (B, Tp, hidden_dim)

4. For each VLATransformerBlock:
   action_tokens, obs_tokens = block(action_tokens, obs_tokens, temb)

5. action_proj_out(action_tokens, temb) → predicted_velocity (B, Tp, action_dim)
```

---

## 3. Modifying Action Dimensions

### 3.1 Configuration Changes

**File**: `src/psi/config/model_psi0.py`

```python
class Psi0ModelConfig(ModelConfig):
    action_dim: int = 7              # ← CHANGE THIS: Your action dimension
    action_chunk_size: int = 6       # ← Prediction horizon (Tp)
    action_exec_horizon: int = 6     # ← Execution horizon (Ta)
    odim: int = 15                   # ← Proprioception/state dimension
    hidden_dim: int = 1536           # Action expert hidden dimension
    num_blocks: int = 6              # Number of transformer blocks
    nhead: int = 24                  # Attention heads
```

### 3.2 Components Affected by action_dim

When you change `action_dim`, these components automatically adapt:

| Component | Parameter | Effect |
|-----------|-----------|--------|
| `ActionProjectionIn` | `action_dim` | Input MLP dimension |
| `ActionProjectionOut` | `action_dim` | Output linear layer |
| `ObservationProjection` | `action_dim` | (stored but not used directly) |

### 3.3 Creating a Custom Config

Create a new training config file:

**File**: `src/psi/config/train/your_custom_config.py`

```python
from psi.config.config import LaunchConfig
from psi.config.model_psi0 import Psi0ModelConfig
from psi.config.data_he import HERawDataConfig  # or your data config
from psi.config import transform as pt
from psi.config.transform import DataTransform

class CustomModelConfig(Psi0ModelConfig):
    action_dim: int = 27           # Your custom action dimension
    action_chunk_size: int = 16    # Your prediction horizon
    odim: int = 27                 # Your proprioception dimension

class DynamicDataTransform(DataTransform):
    repack: pt.YourRepackTransform     # Custom repack transform
    field: pt.ActionStateTransform      # Normalization
    model: pt.Psi0ModelTransform        # VLM input preparation

class DynamicDataConfig(HERawDataConfig):
    transform: DynamicDataTransform

class DynamicLaunchConfig(LaunchConfig):
    data: DynamicDataConfig
    model: CustomModelConfig
```

### 3.4 Loss Weight Configuration

For different action dimensions, update the loss weights:

```python
class Psi0ModelConfig:
    # Default: [xyz_weight, rpy_weight, gripper_weight] for 7-DOF
    loss_w: List[float] = [0.1, 0.2, 0.1]
```

For custom action dimensions, the trainer uses uniform weights:

**File**: `src/psi/trainers/posttrain.py`, Lines 55-63

```python
if self.model_cfg.action_dim == 7:
    w_xyz, w_rpy, w_gripper = self.model_cfg.loss_w
    self.loss_w = torch.tensor([w_xyz]*3 + [w_rpy]*3 + [w_gripper])
else:
    # Uniform weights for non-7-DOF actions
    self.loss_w = torch.tensor([1.0 / action_dim] * action_dim)
```

To customize, you can:
1. Modify `Psi0ModelConfig.loss_w` to accept variable-length lists
2. Override the weight computation in your trainer subclass

---

## 4. Modifying the Data Loader

The data pipeline has three transform stages:

```
Raw Dataset → RepackTransform → FieldTransform → ModelTransform → Training
```

### 4.1 Transform Pipeline Overview

**File**: `src/psi/config/transform.py`

```python
class DataTransform(BaseModel):
    repack: RepackTransform    # Restructures raw data format
    model: ModelTransform      # Prepares VLM inputs (images, text)
    field: FieldTransform      # Normalizes actions to [-1, 1]

    def __call__(self, data, **kwargs):
        data = self.repack(data, **kwargs)
        data = self.field(data, **kwargs)
        data = self.model(data, **kwargs)
        return data
```

### 4.2 Creating a Custom RepackTransform

The RepackTransform converts raw dataset format to a standardized dict.

**Required Output Format**:
```python
{
    "observations": List[PIL.Image],      # List of images
    "states": np.ndarray,                 # Shape: (To, state_dim) or (state_dim,)
    "actions": np.ndarray,                # Shape: (Tp, action_dim)
    "instruction": str,                   # Task description
    "dataset": str,                       # Dataset name
    "actions_mask": np.ndarray,           # Shape: (Tp, action_dim), optional
}
```

**Example Custom Transform**:

```python
# File: src/psi/config/transform.py (add to existing file)

class YourCustomRepackTransform(RepackTransform):
    dataset_name: str = "your_dataset"
    num_past_frames: int = 0
    action_chunk_size: int = 16

    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        # Extract and reshape your data

        # Example: Convert your end-effector representation
        # From: xyz(3) + quaternion(4) = 7
        # To: xyz(3) + 6D rotation(6) = 9

        raw_actions = data["actions"]  # (T, 7)

        xyz = raw_actions[:, :3]
        quat = raw_actions[:, 3:7]

        # Convert quaternion to 6D rotation
        rot6d = self.quat_to_rot6d(quat)  # Implement this

        actions = np.concatenate([xyz, rot6d], axis=1)  # (T, 9)

        return {
            "observations": [Image.fromarray(data["image"])],
            "states": data["proprioception"].astype(np.float32),
            "actions": actions.astype(np.float32),
            "instruction": data.get("task", ""),
            "dataset": self.dataset_name,
            "actions_mask": np.ones_like(actions, dtype=bool),
        }

    def quat_to_rot6d(self, quat):
        """Convert quaternion (w, x, y, z) to 6D rotation representation."""
        from scipy.spatial.transform import Rotation as R
        rot_mat = R.from_quat(quat[:, [1,2,3,0]]).as_matrix()  # scipy uses (x,y,z,w)
        return rot_mat[:, :, :2].reshape(-1, 6)
```

### 4.3 ActionStateTransform (Normalization)

**File**: `src/psi/config/transform.py`, Lines 408-573

Normalizes actions to [-1, 1] range using precomputed statistics.

**Creating Statistics File**:

```json
// File: assets/stats/your_dataset_stats.json
{
    "action": {
        "min": [/* action_dim values */],
        "max": [/* action_dim values */],
        "mean": [/* optional */],
        "std": [/* optional */]
    },
    "states": {
        "min": [/* state_dim values */],
        "max": [/* state_dim values */]
    }
}
```

**Usage in Config**:

```python
class DynamicDataTransform(DataTransform):
    field: pt.ActionStateTransform = Field(
        default_factory=lambda: pt.ActionStateTransform(
            stat_path="assets/stats/your_dataset_stats.json",
            action_norm_type="bounds",  # or "bounds_q99"
            stat_action_key="action",
            stat_state_key="states",
            normalize_state=False,  # Set True if normalizing states
        )
    )
```

### 4.4 Psi0ModelTransform (VLM Input Preparation)

**File**: `src/psi/config/transform.py`, Lines 792-883

Prepares images and text for Qwen3-VL processing.

```python
class Psi0ModelTransform(ModelTransform):
    resize: ResizeImage
    center_crop: CenterCrop
    img_aug: bool = False

    def __call__(self, data, vlm_processor=None, **kwargs):
        # 1. Image preprocessing
        images = [transform(img) for img in data["observations"]]

        # 2. Build Qwen VL inputs
        inputs = self.build_qwenvl_inputs(vlm_processor, images, instruction)

        # 3. Attach action/state data
        inputs['actions'] = data["actions"]
        inputs['states'] = data["states"]
        return inputs
```

**Customization Options**:

```python
class YourModelTransform(Psi0ModelTransform):
    # Custom image size
    resize: ResizeImage = Field(default_factory=lambda: ResizeImage(size=(224, 224)))

    # Enable/disable augmentation
    img_aug: bool = True
```

### 4.5 Complete Custom Data Config Example

```python
# File: src/psi/config/train/your_config.py

from pydantic import Field
from psi.config.config import LaunchConfig, DataConfig
from psi.config.model_psi0 import Psi0ModelConfig
from psi.config import transform as pt
from psi.config.transform import DataTransform
from psi.config.augmentation import ResizeImage, CenterCrop, ColorJitter

class YourRepackTransform(pt.RepackTransform):
    dataset_name: str = "your_dataset"
    action_chunk_size: int = 16
    # ... your implementation

class YourDataTransform(DataTransform):
    repack: YourRepackTransform = Field(
        default_factory=lambda: YourRepackTransform(
            action_chunk_size=16,
        )
    )
    field: pt.ActionStateTransform = Field(
        default_factory=lambda: pt.ActionStateTransform(
            stat_path="assets/stats/your_stats.json",
            action_norm_type="bounds",
        )
    )
    model: pt.Psi0ModelTransform = Field(
        default_factory=lambda: pt.Psi0ModelTransform(
            resize=ResizeImage(size=(270, 480)),
            center_crop=CenterCrop(size=(270, 480)),
            img_aug=True,
        )
    )

class YourDataConfig(DataConfig):
    dataset_name: str = "your_dataset"
    dataset_path: str = "/path/to/your/data"
    transform: YourDataTransform = Field(default_factory=YourDataTransform)

    # Implement create_dataset() method
    def create_dataset(self):
        from your_module import YourRawDataset
        return YourRawDataset(self.dataset_path)

class YourModelConfig(Psi0ModelConfig):
    action_dim: int = 27
    action_chunk_size: int = 16
    odim: int = 27

class YourLaunchConfig(LaunchConfig):
    data: YourDataConfig
    model: YourModelConfig
```

---

## 5. Summary: Minimal Changes Checklist

### For Action Dimension Changes Only:

1. **Config** (`model_psi0.py` or custom config):
   - Set `action_dim` to your value
   - Set `odim` to your proprioception dimension
   - Optionally adjust `action_chunk_size`

2. **Statistics file** (`assets/stats/`):
   - Create JSON with min/max for your action space

3. **RepackTransform** (if action format differs):
   - Create custom transform to output correct action shape

### For End-Effector Representation Changes:

1. All of the above, plus:

2. **RepackTransform**:
   - Convert your representation (e.g., quaternion → 6D rotation)
   - Ensure output matches `action_dim`

3. **Inference post-processing** (if needed):
   - Convert predicted actions back to your robot's format

---

## 6. Architecture Diagram

```
                          ┌─────────────────────────────────────┐
                          │           Psi0Model                 │
                          │                                     │
┌───────────┐             │  ┌──────────────────┐               │
│  Images   │────────────►│  │   Qwen3-VL       │  VLM hidden   │
│  + Text   │             │  │   (frozen)       │──────────────►│
└───────────┘             │  └──────────────────┘               │
                          │                                     │
                          │  ┌────────────────────────────────┐ │
                          │  │      ActionTransformerModel    │ │
                          │  │                                │ │
┌───────────┐             │  │  ┌──────────────┐              │ │
│  Proprio  │────────────►│  │  │ obs_proj     │              │ │
└───────────┘             │  │  └──────────────┘              │ │
                          │  │         │                      │ │
                          │  │         ▼                      │ │
┌───────────┐             │  │  ┌──────────────┐              │ │
│  Noisy    │────────────►│  │  │ action_proj  │              │ │
│  Actions  │             │  │  │ _in          │              │ │
└───────────┘             │  │  └──────────────┘              │ │
                          │  │         │                      │ │
                          │  │         ▼                      │ │
┌───────────┐             │  │  ┌──────────────┐              │ │
│ Timestep  │────────────►│  │  │ VLATransform │ × N blocks   │ │
└───────────┘             │  │  │ erBlock      │              │ │
                          │  │  └──────────────┘              │ │
                          │  │         │                      │ │
                          │  │         ▼                      │ │
                          │  │  ┌──────────────┐              │ │
                          │  │  │ action_proj  │──────────────┼─┼──► Action
                          │  │  │ _out         │  (B,Tp,Da)   │ │    Output
                          │  │  └──────────────┘              │ │
                          │  │                                │ │
                          │  └────────────────────────────────┘ │
                          │                                     │
                          └─────────────────────────────────────┘
```
