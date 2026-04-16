# Psi0 Real-World Observation/Action Spec and Modification Plan

This document summarizes:
1. Current Psi0 real-world data/IO format in this repo.
2. What observation/action signals are required/optional.
3. Visual resolution/FPS constraints.
4. How to adapt Psi0 for more/fewer observation modalities.
5. A no-code, implementation-ready plan for your target setup.

---

## 1) Current real-world data and model IO format

### 1.1 Raw real-world capture format (teleop recorder)

The teleop worker writes per-frame records in `real/teleop/worker.py`:
- `time`
- `robot_type`
- `states`:
  - `arm_state` (G1: 14)
  - `leg_state` (G1: 15)
  - `hand_state` (G1: 14)
  - `hand_pressure_state` (G1 tactile summary)
  - `imu` (`quaternion`, `accelerometer`, `gyroscope`, `rpy`)
  - `odometry` (`position`, `velocity`, `rpy`, `quat`)
- `actions` (filled after IK merge)
- `image` path
- `depth` path

References:
- `real/teleop/constants.py`
- `real/teleop/worker.py`
- `real/teleop/merger.py`

### 1.2 Training/inference repack format expected by Psi0 pipeline

For real fine-tuning, `RealRepackTransform` expects configurable keys:
- image keys: `image_keys` (default: `["observation.images.egocentric"]`)
- state key: `state_key` (default: `"states"`)
- action key: `action_key` (default: `"action"`)
- instruction key: `instruction_key` (default: `"task"`)

It emits:
- `observations`: list of PIL images
- `states`: float32 array
- `actions`: float32 array
- `instruction`: lowercased string
- `actions_mask`

References:
- `src/psi/config/transform.py` (`RealRepackTransform`)
- `src/psi/config/train/finetune_real_psi0_config.py`

### 1.3 Psi0 online serving request/response format

Request payload in server helpers:
- `image` (dict of camera arrays)
- `instruction` (natural language)
- `state` (dict; server currently reads `state["states"]`)
- `history`
- `condition`
- `gt_action`
- `dataset_name`
- `timestamp`

Response:
- `action`: `(Ta, Da)` action chunk
- `err`
- `traj_image`

References:
- `src/psi/deploy/helpers.py`
- `src/psi/deploy/psi0_serve_simple.py`
- `real/deploy/psi-inference.py`
- `real/deploy/psi-inference_rtc.py`

---

## 2) Observation signals: what is needed today

### 2.1 Required (for current Psi0 serving path)

- **Images**: at least one RGB image (currently egocentric path assumed in several places).
- **State vector**: server expects pre-flattened `state["states"]` with dimension matching model config `odim`.
- **Instruction**: natural language string.

References:
- `src/psi/deploy/psi0_serve_simple.py` (`state_dict["states"]`, image dict iteration)
- `src/psi/config/model_psi0.py` (`odim`)

### 2.2 Available robot signals from G1 capture stack

G1 dimensions in constants:
- leg 15, arm 14, hand 14, IMU quat 4, IMU accel 3, IMU gyro 3, IMU rpy 3, odom pos 3, odom vel 3, odom rpy 3, odom quat 4, hand pressure 216.

References:
- `real/teleop/constants.py`
- `real/teleop/worker.py`

### 2.3 Notes on modality flexibility

The pipeline is already parameterized for:
- configurable image key list (`image_keys`)
- configurable state/action/instruction keys
- configurable state/action padding (`pad_state_dim`, `pad_action_dim`)
- configurable `odim`, `action_dim`, horizons in model config

References:
- `src/psi/config/transform.py`
- `src/psi/config/model_psi0.py`

---

## 3) Action format today

### 3.1 Model-side action shape

Core model predicts `(B, action_chunk_size, action_dim)` and serving returns first `action_exec_horizon`.

References:
- `src/psi/config/model_psi0.py` (`action_dim`, `action_chunk_size`, `action_exec_horizon`)
- `src/psi/models/psi0.py` (`predict_action*`)
- `src/psi/deploy/psi0_serve_simple.py`

### 3.2 Existing real deployment convention (whole-body 36D control path)

`real/deploy/psi-inference*.py` uses a 36D convention:
- `[0:14]` hand command (`q_hand`)
- `[14:28]` arm command
- `[28:32]` torso roll/pitch/yaw/height
- `[32:36]` base/control terms (`vx`, `vy`, `vyaw`, `target_yaw`)

References:
- `real/deploy/psi-inference.py`
- `real/deploy/psi-inference_rtc.py`

### 3.3 Existing partial support related to your requested output

`HEPosttrainRepackTransform` already includes a `hands_only` action format:
- 16D = left wrist 6DoF + right wrist 6DoF + head xyz(3) + discrete token(1)

This is close in spirit, but **not** the same as “q-hand + head position + token”.

Reference:
- `src/psi/config/transform.py` (`HEPosttrainRepackTransform`, `action_format="hands_only"`)

---

## 4) Visual requirements/constraints (resolution + FPS)

### 4.1 Hard constraints in code

No strict hard-coded FPS requirement for Psi0 model itself; timing is mainly handled via dataset `fps` -> `delta_timestamps`.

References:
- `src/psi/data/lerobot/lerobot_ext.py` (reads dataset metadata fps)
- `src/psi/config/transform.py` (`delta_timestamps`)

### 4.2 Current defaults and preprocessing behavior

- RealSense server currently configured to **640x480 @ 30fps**.
- Teleop recording loop targets **30Hz**.
- Psi0 model transform resizes/crops images to **224x224** by default (or adaptive dataset-specific sizes).
- Qwen3VL transform defaults include dataset sizes (HE 240x320, EgoDex 270x480) and uses Qwen vision tokenization.

References:
- `real/teleop/image_server/realsense_server.py`
- `real/teleop/worker.py`
- `src/psi/config/transform.py` (`Psi0ModelTransform`, `Qwen3vlModelTransform`)
- `src/psi/config/model_psi0.py` (`min_pixels`, `max_pixels`)

### 4.3 Implication for your target (3 cameras, 320x240, 50fps)

- 320x240 is compatible with current transform pipeline (it will be resized/cropped as configured).
- 50fps is not blocked by model code; but dataset/loader timestamping and deployment throughput must be aligned (especially for action horizon and RTC delay settings).

---

## 5) Your requested target setup (spec to implement)

### 5.1 Inputs to support

1. Everything G1 can collect (full proprio + IMU + odometry + tactile optional).
2. 3 camera streams at 50fps, 320x240.
3. Natural language description.
4. One discrete token representing whole-body pose.

### 5.2 Outputs to support

VLA action expert output should be:
- original Psi0 `q_hand` component
- `head_position`
- the discrete whole-body pose token from data

---

## 6) Delicate no-code plan for modifications

## Phase A — Lock schema and naming (single source of truth)

1. Define canonical observation schema for training + serving:
   - `observation.images.cam0/cam1/cam2`
   - `observation.state.*` (or flattened `states`)
   - `instruction`
   - `observation.pose_token` (discrete)
2. Define canonical action schema:
   - `action.q_hand` (explicit dimension and ordering)
   - `action.head_position` (xyz)
   - `action.pose_token` (discrete id/logit target)
3. Freeze ordering and dimensions in one config/dataclass.

## Phase B — Data ingestion + repack layer

1. Extend real-data repack to ingest 3 cameras (configurable `image_keys`).
2. Add deterministic state-construction function for “everything G1 can collect”.
3. Add pose token extraction path from dataset sample.
4. Add output action construction for `q_hand + head_position + token`.
5. Keep backward-compatible mode flag (old action format vs new compact format).

## Phase C — Model shape/config plumbing

1. Update config defaults for:
   - `n_cams=3`
   - new `odim` for expanded state
   - new `action_dim` for target output
2. Ensure all projection modules consume updated dims:
   - observation projection input (`odim`)
   - action in/out projections (`action_dim`)
   - normalization stats dimensions
3. Add explicit config switch for optional modalities (less/more observations), e.g.:
   - `use_imu`, `use_odom`, `use_tactile`, `use_pose_token`.

## Phase D — Serving and runtime IO contract

1. Update request validator/assembler to accept 3 images + expanded state + pose token.
2. Update response payload contract to include:
   - `q_hand`
   - `head_position`
   - `pose_token` (predicted or copied mode, per design)
3. Keep a compatibility adapter for old 36D actuator consumers.

## Phase E — Camera/FPS pipeline alignment

1. Update real camera server and client to run 3 streams at 320x240@50.
2. Align loop rates and timestamps (capture, send, infer, execute).
3. Tune `action_chunk_size`, `action_exec_horizon`, and RTC delay bounds for 50fps budget.

## Phase F — Normalization and stats

1. Regenerate/extend normalization stats for new state/action fields.
2. Add token handling policy:
   - numeric normalized channel, or
   - separate categorical head/loss.

## Phase G — Training/eval rollout

1. Create new train preset config for the target schema.
2. Keep legacy presets untouched.
3. Add schema assertions at dataloader boundary to fail fast on mismatched dimensions.

## Phase H — Validation gates

1. Dataset sample gate: all required keys present for new schema.
2. Model gate: forward pass with 3 cams and new dims.
3. Serve gate: request/response contract check with target payload.
4. Runtime gate: control bridge accepts `q_hand + head_position + token` output mode.

---

## 7) Key risks and design decisions to resolve before coding

1. **`q_hand` exact definition**: confirm dimension and ordering (likely 14D from current real deploy split).
2. **Pose token behavior in output**: copied/pass-through vs predicted classification/regression head.
3. **State flattening policy**: full G1 raw concatenation vs selected subset for stability.
4. **3-camera semantic mapping**: which view is `cam0/cam1/cam2` and synchronization policy.
5. **50fps feasibility**: if full pipeline cannot sustain 50Hz inference, decide decimation/interpolation strategy.

---

## 8) Practical starting point for implementation (when you decide to code)

First-impact files:
- `src/psi/config/transform.py` (repack + model transforms)
- `src/psi/config/model_psi0.py` (dims/horizons/camera count knobs)
- `src/psi/models/psi0.py` (projection/action dimension plumbing)
- `src/psi/deploy/helpers.py` and `src/psi/deploy/psi0_serve_simple.py` (serve contract)
- `real/deploy/psi-inference.py` / `psi-inference_rtc.py` (runtime consumer mapping)
- `real/teleop/image_server/realsense_server.py` + real client capture paths (3-cam 50fps pipeline)

