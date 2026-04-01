import argparse
import json
import logging
import lzma
import math
import os
import re
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import imageio.v3 as iio
import numpy as np
# import open3d as o3d
import pandas as pd
import pinocchio as pin
from datasets import Dataset, Features, Sequence, Value
from datasets.utils.logging import set_verbosity_error
from huggingface_hub import create_repo, create_tag, upload_large_folder
from tqdm import tqdm

CODE_VERSION = "v2.1"
FPS = 30

set_verbosity_error()
logging.getLogger("pyarrow").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)


LEFT_HAND_JOINTS_G1 = [
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
]
RIGHT_HAND_JOINTS_G1 = [
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
]
LEFT_ARM_JOINTS_G1 = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]
RIGHT_ARM_JOINTS_G1 = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

WRIST_FRAMES_G1 = {
    "left": "left_wrist_yaw_link",
    "right": "right_wrist_yaw_link",
}
HAND_FRAMES_G1 = {
    "left_thumb": "left_hand_thumb_2_link",
    "left_index": "left_hand_index_1_link",
    "left_middle": "left_hand_middle_1_link",
    "right_thumb": "right_hand_thumb_2_link",
    "right_index": "right_hand_index_1_link",
    "right_middle": "right_hand_middle_1_link",
}

OFFSETS_G1 = {
    "left_thumb": np.array([0.0, -0.0458, 0.0]),
    "left_index": np.array([0.0458, 0.0, 0.0]),
    "left_middle": np.array([0.0458, 0.0, 0.0]),
    "right_thumb": np.array([0.0, 0.0458, 0.0]),
    "right_index": np.array([0.0458, 0.0, 0.0]),
    "right_middle": np.array([0.0458, 0.0, 0.0]),
    "left_wrist": np.array([0.05, 0.0, 0.0]),
    "right_wrist": np.array([0.05, 0.0, 0.0]),
}


LEFT_HAND_JOINTS_H1 = [
    "L_pinky_proximal_joint",
    "L_ring_proximal_joint",
    "L_middle_proximal_joint",
    "L_index_proximal_joint",
    "L_thumb_proximal_pitch_joint",
    "L_thumb_proximal_yaw_joint",
]
RIGHT_HAND_JOINTS_H1 = [
    "R_pinky_proximal_joint",
    "R_ring_proximal_joint",
    "R_middle_proximal_joint",
    "R_index_proximal_joint",
    "R_thumb_proximal_pitch_joint",
    "R_thumb_proximal_yaw_joint",
]

LEFT_ARM_JOINTS_H1 = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_elbow_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]
RIGHT_ARM_JOINTS_H1 = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

MIMIC_JOINTS_H1 = [
    ("L_thumb_intermediate_joint", "L_thumb_proximal_pitch_joint", 1.6, 0.0),
    ("L_thumb_distal_joint", "L_thumb_proximal_pitch_joint", 2.4, 0.0),
    ("L_index_intermediate_joint", "L_index_proximal_joint", 1.0, 0.0),
    ("L_middle_intermediate_joint", "L_middle_proximal_joint", 1.0, 0.0),
    ("L_ring_intermediate_joint", "L_ring_proximal_joint", 1.0, 0.0),
    ("L_pinky_intermediate_joint", "L_pinky_proximal_joint", 1.0, 0.0),
    ("R_thumb_intermediate_joint", "R_thumb_proximal_pitch_joint", 1.6, 0.0),
    ("R_thumb_distal_joint", "R_thumb_proximal_pitch_joint", 2.4, 0.0),
    ("R_index_intermediate_joint", "R_index_proximal_joint", 1.0, 0.0),
    ("R_middle_intermediate_joint", "R_middle_proximal_joint", 1.0, 0.0),
    ("R_ring_intermediate_joint", "R_ring_proximal_joint", 1.0, 0.0),
    ("R_pinky_intermediate_joint", "R_pinky_proximal_joint", 1.0, 0.0),
]

WRIST_FRAMES_H1 = {
    "left": "left_wrist_yaw_link",
    "right": "right_wrist_yaw_link",
}
HAND_FRAMES_H1 = {
    "left_thumb": "L_thumb_tip",
    "left_index": "L_index_tip",
    "left_middle_finger": "L_middle_tip",
    "left_ring_finger": "L_ring_tip",
    "left_little_finger": "L_pinky_tip",
    "right_thumb": "R_thumb_tip",
    "right_index": "R_index_tip",
    "right_middle_finger": "R_middle_tip",
    "right_ring_finger": "R_ring_tip",
    "right_little_finger": "R_pinky_tip",
}

HAND_BASE_FRAMES_H1 = {
    "left_hand_base": "L_hand_base_link",
    "right_hand_base": "R_hand_base_link",
}


ACTION_KEYS_G1 = [
    "action.joint_angles",
    "action.wrists.left.xyz",
    "action.wrists.left.rpy",
    "action.wrists.right.xyz",
    "action.wrists.right.rpy",
    "action.hands.left_thumb.xyz",
    "action.hands.left_thumb.rpy",
    "action.hands.left_index.xyz",
    "action.hands.left_index.rpy",
    "action.hands.left_middle.xyz",
    "action.hands.left_middle.rpy",
    "action.hands.right_thumb.xyz",
    "action.hands.right_thumb.rpy",
    "action.hands.right_index.xyz",
    "action.hands.right_index.rpy",
    "action.hands.right_middle.xyz",
    "action.hands.right_middle.rpy",
]

ACTION_KEYS_H1 = [
    "action.joint_angles",
    "action.wrists.left.xyz",
    "action.wrists.left.rpy",
    "action.wrists.right.xyz",
    "action.wrists.right.rpy",
    "action.hands.left_thumb.xyz",
    "action.hands.left_thumb.rpy",
    "action.hands.left_index.xyz",
    "action.hands.left_index.rpy",
    "action.hands.left_middle_finger.xyz",
    "action.hands.left_middle_finger.rpy",
    "action.hands.left_ring_finger.xyz",
    "action.hands.left_ring_finger.rpy",
    "action.hands.left_little_finger.xyz",
    "action.hands.left_little_finger.rpy",
    "action.hands.right_thumb.xyz",
    "action.hands.right_thumb.rpy",
    "action.hands.right_index.xyz",
    "action.hands.right_index.rpy",
    "action.hands.right_middle_finger.xyz",
    "action.hands.right_middle_finger.rpy",
    "action.hands.right_ring_finger.xyz",
    "action.hands.right_ring_finger.rpy",
    "action.hands.right_little_finger.xyz",
    "action.hands.right_little_finger.rpy",
]


def action_keys_for_robot(robot_type: str) -> List[str]:
    if robot_type == "g1":
        return ACTION_KEYS_G1
    if robot_type == "h1":
        return ACTION_KEYS_H1
    if robot_type == "both":
        seen = set()
        merged = []
        for key in ACTION_KEYS_G1 + ACTION_KEYS_H1:
            if key not in seen:
                merged.append(key)
                seen.add(key)
        return merged
    raise ValueError(f"Unsupported robot type: {robot_type}")


def detect_robot_type(ep_dir: Path) -> str:
    return HE2WePretrainConverter.get_robot_type(ep_dir)


@dataclass
class InfoDict:
    codebase_version: str
    robot_type: str
    total_episodes: int
    total_frames: int
    total_tasks: int
    total_videos: int
    total_chunks: int
    chunks_size: int
    fps: int
    data_path: str
    video_path: str
    features: Dict[str, Any]


def load_done_episodes(meta_dir: Path) -> set[int]:
    done = set()
    stats_file = meta_dir / "episodes_stats.jsonl"
    if stats_file.exists():
        with open(stats_file, "r") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add(rec["episode_index"])
                except Exception:
                    continue
    return done


def episode_complete(ep_index: int, work_dir: Path, chunks_size: int, done_eps: set[int]) -> bool:
    chunk_id = ep_index // chunks_size
    data_ok = (work_dir / "data" / f"chunk-{chunk_id:03d}" / f"episode_{ep_index:06d}.parquet").exists()
    vid_ok = (work_dir / "videos" / f"chunk-{chunk_id:03d}" / "egocentric" / f"episode_{ep_index:06d}.mp4").exists()
    stats_ok = ep_index in done_eps
    return data_ok and vid_ok and stats_ok


def append_jsonl_line_atomic(path: Path, obj: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n"
    fd = os.open(str(path), os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o644)
    try:
        with os.fdopen(fd, "a", encoding="utf-8") as f:
            try:
                import fcntl

                fcntl.flock(f, fcntl.LOCK_EX)
            except Exception:
                pass
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
    finally:
        pass


def read_json_list(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)
    assert isinstance(data, list), f"Expected a list in {path}, got {type(data)}"
    return data


def iter_tasks(data_root: Path) -> Iterator[Tuple[str, Path, str, str]]:
    for cat_dir in sorted([p for p in data_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        for task_dir in sorted([p for p in cat_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
            yield f"{cat_dir.name}/{task_dir.name}", task_dir, cat_dir.name, task_dir.name


def matrix_to_rpy(rot: np.ndarray) -> np.ndarray:
    sy = np.sqrt(rot[0, 0] ** 2 + rot[1, 0] ** 2)
    singular = sy < 1e-6
    if singular:
        roll = np.arctan2(-rot[1, 2], rot[1, 1])
        pitch = np.arctan2(-rot[2, 0], sy)
        yaw = 0.0
    else:
        roll = np.arctan2(rot[2, 1], rot[2, 2])
        pitch = np.arctan2(-rot[2, 0], sy)
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
    return np.array([roll, pitch, yaw], dtype=float)


def rot_x(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def rot_y(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def matrix_to_wxyz(rot: np.ndarray) -> np.ndarray:
    quat = pin.Quaternion(rot)
    xyzw = quat.coeffs()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=float)


def wxyz_to_rpy(wxyz: np.ndarray) -> np.ndarray:
    quat = pin.Quaternion(wxyz[1], wxyz[2], wxyz[3], wxyz[0])
    return matrix_to_rpy(quat.toRotationMatrix())


def convert_h1_hand(qpos: List[float]) -> List[float]:
    out = [1.7 - qpos[i] for i in [4, 6, 2, 0]]
    out.append(1.2 - qpos[8])
    out.append(0.5 - qpos[9])
    return [float(x) for x in out]


def action_to_joint_cfg_g1(action: np.ndarray) -> Dict[str, float]:
    action = np.asarray(action).reshape(-1)
    if action.size != 28:
        raise ValueError(f"Expected 28D action, got {action.size}")

    joint_cfg: Dict[str, float] = {}
    joint_cfg.update(zip(LEFT_HAND_JOINTS_G1, action[0:7]))
    joint_cfg.update(zip(RIGHT_HAND_JOINTS_G1, action[7:14]))
    joint_cfg.update(zip(LEFT_ARM_JOINTS_G1, action[14:21]))
    joint_cfg.update(zip(RIGHT_ARM_JOINTS_G1, action[21:28]))
    return joint_cfg


def action_to_joint_cfg_h1(action: np.ndarray) -> Dict[str, float]:
    action = np.asarray(action).reshape(-1)
    if action.size != 26:
        raise ValueError(f"Expected 26D action, got {action.size}")

    joint_cfg: Dict[str, float] = {}
    joint_cfg.update(zip(LEFT_HAND_JOINTS_H1, action[0:6]))
    joint_cfg.update(zip(RIGHT_HAND_JOINTS_H1, action[6:12]))
    joint_cfg.update(zip(LEFT_ARM_JOINTS_H1, action[12:19]))
    joint_cfg.update(zip(RIGHT_ARM_JOINTS_H1, action[19:26]))

    for target, source, multiplier, offset in MIMIC_JOINTS_H1:
        if source in joint_cfg:
            joint_cfg[target] = joint_cfg[source] * multiplier + offset
    return joint_cfg


def joints_to_q(model: pin.Model, joint_dict: Dict[str, float]) -> np.ndarray:
    q = pin.neutral(model)
    for name, val in joint_dict.items():
        jid = model.getJointId(name)
        if jid == 0:
            continue
        j = model.joints[jid]
        if j.nq == 1:
            q[j.idx_q] = float(val)
    return q


def compute_pose_g1(
    model: pin.Model,
    q: np.ndarray,
    frame_name: str,
    offset: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    data = model.createData()
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    fid = model.getFrameId(frame_name)
    if fid >= len(data.oMf):
        raise RuntimeError(f"Frame {frame_name} id={fid} >= nframes={len(data.oMf)}")
    oMf = data.oMf[fid]
    pos = oMf.translation
    if offset is not None:
        pos = pos + oMf.rotation @ offset
    return {"xyz": pos, "rpy": matrix_to_rpy(oMf.rotation)}


def compute_poses_h1(
    model: pin.Model,
    q: np.ndarray,
    frame_names: Dict[str, str],
) -> Dict[str, Dict[str, np.ndarray]]:
    data = model.createData()
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for label, frame_name in frame_names.items():
        fid = model.getFrameId(frame_name)
        if fid >= len(data.oMf):
            raise RuntimeError(f"Frame {frame_name} id={fid} >= nframes={len(data.oMf)}")
        oMf = data.oMf[fid]
        out[label] = {"xyz": oMf.translation, "rot": oMf.rotation}
    return out


def build_action_pose_g1(model: pin.Model, action: np.ndarray) -> Dict[str, Any]:
    joint_cfg = action_to_joint_cfg_g1(action)
    q = joints_to_q(model, joint_cfg)

    left_wrist = compute_pose_g1(model, q, WRIST_FRAMES_G1["left"], offset=OFFSETS_G1["left_wrist"])
    right_wrist = compute_pose_g1(model, q, WRIST_FRAMES_G1["right"], offset=OFFSETS_G1["right_wrist"])
    left_thumb = compute_pose_g1(model, q, HAND_FRAMES_G1["left_thumb"], offset=OFFSETS_G1["left_thumb"])
    left_index = compute_pose_g1(model, q, HAND_FRAMES_G1["left_index"], offset=OFFSETS_G1["left_index"])
    left_middle = compute_pose_g1(model, q, HAND_FRAMES_G1["left_middle"], offset=OFFSETS_G1["left_middle"])
    right_thumb = compute_pose_g1(model, q, HAND_FRAMES_G1["right_thumb"], offset=OFFSETS_G1["right_thumb"])
    right_index = compute_pose_g1(model, q, HAND_FRAMES_G1["right_index"], offset=OFFSETS_G1["right_index"])
    right_middle = compute_pose_g1(model, q, HAND_FRAMES_G1["right_middle"], offset=OFFSETS_G1["right_middle"])

    return {
        "action.joint_angles": action.astype(np.float32),
        "action.wrists.left.xyz": left_wrist["xyz"].astype(np.float32),
        "action.wrists.left.rpy": left_wrist["rpy"].astype(np.float32),
        "action.wrists.right.xyz": right_wrist["xyz"].astype(np.float32),
        "action.wrists.right.rpy": right_wrist["rpy"].astype(np.float32),
        "action.hands.left_thumb.xyz": left_thumb["xyz"].astype(np.float32),
        "action.hands.left_thumb.rpy": left_thumb["rpy"].astype(np.float32),
        "action.hands.left_index.xyz": left_index["xyz"].astype(np.float32),
        "action.hands.left_index.rpy": left_index["rpy"].astype(np.float32),
        "action.hands.left_middle.xyz": left_middle["xyz"].astype(np.float32),
        "action.hands.left_middle.rpy": left_middle["rpy"].astype(np.float32),
        "action.hands.right_thumb.xyz": right_thumb["xyz"].astype(np.float32),
        "action.hands.right_thumb.rpy": right_thumb["rpy"].astype(np.float32),
        "action.hands.right_index.xyz": right_index["xyz"].astype(np.float32),
        "action.hands.right_index.rpy": right_index["rpy"].astype(np.float32),
        "action.hands.right_middle.xyz": right_middle["xyz"].astype(np.float32),
        "action.hands.right_middle.rpy": right_middle["rpy"].astype(np.float32),
    }


def build_action_pose_h1(model: pin.Model, action: np.ndarray) -> Dict[str, Any]:
    joint_cfg = action_to_joint_cfg_h1(action)
    q = joints_to_q(model, joint_cfg)
    wrists = compute_poses_h1(model, q, WRIST_FRAMES_H1)
    tips = compute_poses_h1(model, q, HAND_FRAMES_H1)
    hand_bases = compute_poses_h1(model, q, HAND_BASE_FRAMES_H1)

    left_thumb_rpy = wxyz_to_rpy(
        matrix_to_wxyz(
            hand_bases["left_hand_base"]["rot"]
            @ (
                rot_y(float(-joint_cfg.get("L_thumb_proximal_yaw_joint", 0.0)))
                @ rot_x(float(joint_cfg.get("L_thumb_proximal_pitch_joint", 0.0) * 5.0))
            )
        )
    )
    right_thumb_rpy = wxyz_to_rpy(
        matrix_to_wxyz(
            hand_bases["right_hand_base"]["rot"]
            @ (
                rot_y(float(joint_cfg.get("R_thumb_proximal_yaw_joint", 0.0)))
                @ rot_x(float(-joint_cfg.get("R_thumb_proximal_pitch_joint", 0.0) * 5.0))
            )
        )
    )

    return {
        "action.joint_angles": action.astype(np.float32),
        "action.wrists.left.xyz": wrists["left"]["xyz"].astype(np.float32),
        "action.wrists.left.rpy": matrix_to_rpy(wrists["left"]["rot"]).astype(np.float32),
        "action.wrists.right.xyz": wrists["right"]["xyz"].astype(np.float32),
        "action.wrists.right.rpy": matrix_to_rpy(wrists["right"]["rot"]).astype(np.float32),
        "action.hands.left_thumb.xyz": tips["left_thumb"]["xyz"].astype(np.float32),
        "action.hands.left_thumb.rpy": left_thumb_rpy.astype(np.float32),
        "action.hands.left_index.xyz": tips["left_index"]["xyz"].astype(np.float32),
        "action.hands.left_index.rpy": matrix_to_rpy(tips["left_index"]["rot"]).astype(np.float32),
        "action.hands.left_middle_finger.xyz": tips["left_middle_finger"]["xyz"].astype(np.float32),
        "action.hands.left_middle_finger.rpy": matrix_to_rpy(tips["left_middle_finger"]["rot"]).astype(np.float32),
        "action.hands.left_ring_finger.xyz": tips["left_ring_finger"]["xyz"].astype(np.float32),
        "action.hands.left_ring_finger.rpy": matrix_to_rpy(tips["left_ring_finger"]["rot"]).astype(np.float32),
        "action.hands.left_little_finger.xyz": tips["left_little_finger"]["xyz"].astype(np.float32),
        "action.hands.left_little_finger.rpy": matrix_to_rpy(tips["left_little_finger"]["rot"]).astype(np.float32),
        "action.hands.right_thumb.xyz": tips["right_thumb"]["xyz"].astype(np.float32),
        "action.hands.right_thumb.rpy": right_thumb_rpy.astype(np.float32),
        "action.hands.right_index.xyz": tips["right_index"]["xyz"].astype(np.float32),
        "action.hands.right_index.rpy": matrix_to_rpy(tips["right_index"]["rot"]).astype(np.float32),
        "action.hands.right_middle_finger.xyz": tips["right_middle_finger"]["xyz"].astype(np.float32),
        "action.hands.right_middle_finger.rpy": matrix_to_rpy(tips["right_middle_finger"]["rot"]).astype(np.float32),
        "action.hands.right_ring_finger.xyz": tips["right_ring_finger"]["xyz"].astype(np.float32),
        "action.hands.right_ring_finger.rpy": matrix_to_rpy(tips["right_ring_finger"]["rot"]).astype(np.float32),
        "action.hands.right_little_finger.xyz": tips["right_little_finger"]["xyz"].astype(np.float32),
        "action.hands.right_little_finger.rpy": matrix_to_rpy(tips["right_little_finger"]["rot"]).astype(np.float32),
    }


def extract_action_joints(frame: Dict[str, Any], robot_type: str) -> np.ndarray:
    actions = frame.get("actions", {}) or {}
    r = actions.get("right_angles")
    l = actions.get("left_angles")
    hand_joints: List[float] = []

    if r is not None and l is not None:
        if robot_type == "h1" and len(r) == 12 and len(l) == 12:
            hand_joints.extend(convert_h1_hand(l))
            hand_joints.extend(convert_h1_hand(r))
        else:
            hand_joints.extend([float(x) for x in l])
            hand_joints.extend([float(x) for x in r])
    else:
        raise ValueError("Missing left/right hand angles in actions.")

    sq = actions.get("sol_q")
    if sq is None:
        raise ValueError("Missing sol_q in actions.")
    body_joints = [float(x) for x in sq]
    if len(body_joints) != 14:
        raise ValueError(f"Expected sol_q to be 14D, got {len(body_joints)}")
    arm_joints = body_joints
    action = np.array(hand_joints + arm_joints, dtype=np.float32)
    expected = 28 if robot_type == "g1" else 26
    if action.size != expected:
        raise ValueError(f"Expected {expected}D action for {robot_type}, got {action.size}")
    return action


def fill_missing_action_fields(fields: Dict[str, Any], action_keys: List[str]) -> Dict[str, Any]:
    for key in action_keys:
        if key in fields:
            continue
        if key == "action.joint_angles":
            continue
        if key.endswith(".xyz") or key.endswith(".rpy"):
            fields[key] = np.full((3,), np.nan, dtype=np.float32)
    return fields


@lru_cache(maxsize=None)
def load_model(urdf_path: str) -> pin.Model:
    mesh_dir = os.path.dirname(urdf_path)
    return pin.RobotWrapper.BuildFromURDF(urdf_path, mesh_dir).model


class HE2WePretrainConverter:
    def __init__(self, robot_type: str, urdf_map: Dict[str, str]):
        self.robot_type = robot_type
        self.urdf_map = urdf_map
        self.action_keys = action_keys_for_robot(robot_type)

        action_features = {key: Sequence(Value("float32")) for key in self.action_keys}

        self.features = Features(
            {
                "observation.arm_joints": Sequence(Value("float32")),
                "observation.leg_joints": Sequence(Value("float32")),
                "observation.hand_joints": Sequence(Value("float32")),
                **action_features,
                "timestamp": Value("float32"),
                "frame_index": Value("int64"),
                "episode_index": Value("int64"),
                "index": Value("int64"),
                "task_index": Value("int64"),
                "next.done": Value("bool"),
            }
        )

        self.task_description_dict: Dict[str, str] = {}
        self.kept_records: List[Tuple[int, int, Path, str, str]] = []
        self.lengths_by_episode: Dict[int, int] = {}
        self.tasks_meta: Dict[int, Dict[str, Any]] = {}
        self.chunks_size: int = 1000

    @staticmethod
    def get_robot_type(ep_dir: Path) -> str:
        data_list = read_json_list(ep_dir / "data.json")
        if not data_list:
            return "h1"
        for frame in data_list:
            st = frame.get("states", {})
            if isinstance(st, dict) and "robot_type" in st:
                try:
                    return str(st["robot_type"]).lower()
                except Exception:
                    return "h1"
            if "robot_type" in frame:
                try:
                    return str(frame["robot_type"]).lower()
                except Exception:
                    return "h1"
        return "h1"

    def load_depth(self, depth_lzma_path: Path) -> Optional[np.ndarray]:
        try:
            with open(depth_lzma_path, "rb") as f:
                decompressed = lzma.decompress(f.read())
            depth_u16 = np.frombuffer(decompressed, dtype=np.uint16).reshape((480, 640))
            return depth_u16.astype(np.float32)
        except Exception:
            return None

    def load_lidar(self, pcd_path: Path) -> Optional[np.ndarray]:
        try:

            def pad_to_six(m):
                whole, dec = m.group("whole"), m.group("dec")
                return f"{whole}.{dec.ljust(6, '0')}"

            pcd_path_str = re.sub(r"(?P<whole>\d+)\.(?P<dec>\d{1,6})(?=\.pcd$)", pad_to_six, str(pcd_path))
            pcd = o3d.io.read_point_cloud(pcd_path_str)
            pts = np.asarray(pcd.points, dtype=np.float32)
            pts = pts[~np.all(pts == 0, axis=1)]
            if pts.size == 0:
                return None
            return pts
        except Exception:
            return None

    def build_obs(
        self, prev_rpy_height: Dict[str, Any], frame: Dict[str, Any], depth_arr: np.ndarray, pts: np.ndarray
    ) -> Dict[str, Any]:
        states = frame.get("states", {}) or {}

        arm_joints = [float(x) for x in states.get("arm_state", [])]
        leg_joints = [float(x) for x in states.get("leg_state", [])]
        hand_joints = [float(x) for x in states.get("hand_state", [])]

        return {
            "observation.leg_joints": leg_joints,
            "observation.arm_joints": arm_joints,
            "observation.hand_joints": hand_joints,
        }

    def build_action_fields(self, robot_type: str, action: np.ndarray) -> Dict[str, Any]:
        if robot_type == "g1":
            model = load_model(self.urdf_map["g1"])
            fields = build_action_pose_g1(model, action)
        elif robot_type == "h1":
            model = load_model(self.urdf_map["h1"])
            fields = build_action_pose_h1(model, action)
        else:
            raise ValueError(f"Unsupported robot type: {robot_type}")

        if self.robot_type == "both":
            fields = fill_missing_action_fields(fields, self.action_keys)
        return fields

    def make_one_episode(
        self,
        task_index: int,
        episode_index: int,
        episode_dir: Path,
        out_base: Path,
        chunks_size: int,
        robot_type: str,
    ) -> Tuple[int, int, Dict[str, Any]]:
        try:
            chunk_path = out_base / f"chunk-{episode_index // chunks_size:03d}"
            chunk_path.mkdir(parents=True, exist_ok=True)
            parquet_path = chunk_path / f"episode_{episode_index:06d}.parquet"

            vid_chunk_dir = out_base.parent / "videos" / f"chunk-{episode_index // chunks_size:03d}" / "egocentric"
            vid_chunk_dir.mkdir(parents=True, exist_ok=True)
            vid_path = vid_chunk_dir / f"episode_{episode_index:06d}.mp4"

            data_list = read_json_list(episode_dir / "data.json")
            assert len(data_list) > 0, f"data.json malformed in {episode_dir}"

            def safe_path(episode_dir, f, key):
                p = f.get(key)
                return (episode_dir / p).resolve() if p else None

            rgb_paths = [safe_path(episode_dir, f, "image") for f in data_list]
            depth_paths = [safe_path(episode_dir, f, "depth") for f in data_list]
            lidar_paths = [safe_path(episode_dir, f, "lidar") for f in data_list]

            def iter_depths():
                for p in depth_paths:
                    yield self.load_depth(p) if p else np.full((480, 640), np.nan, np.float32)

            def iter_lidars():
                for p in lidar_paths:
                    yield self.load_lidar(p) if p else np.zeros((0, 3), np.float32)

            rows: List[Dict[str, Any]] = []
            prev_rpy_height = {"torso_rpy": [0, 0, 0], "torso_height": 0.75}

            for i, (frame, depth_arr, lidar_pts) in enumerate(zip(data_list, iter_depths(), iter_lidars())):
                obs = self.build_obs(prev_rpy_height, frame, depth_arr, lidar_pts)
                action = extract_action_joints(frame, robot_type)
                action_fields = self.build_action_fields(robot_type, action)

                rows.append(
                    {
                        **obs,
                        **action_fields,
                        "timestamp": i * (1.0 / FPS),
                        "frame_index": i,
                        "episode_index": episode_index,
                        "index": i,
                        "task_index": task_index,
                        "next.done": (i == len(data_list) - 1),
                    }
                )

            assert rows, f"No valid rows in episode {episode_index}"

            stats = None
            for r in rows:
                a = np.array(r["action.joint_angles"], dtype=np.float32)
                if stats is None:
                    stats = {"min": a.copy(), "max": a.copy(), "sum": a.copy(), "sumsq": a**2, "count": 1}
                else:
                    stats["min"] = np.minimum(stats["min"], a)
                    stats["max"] = np.maximum(stats["max"], a)
                    stats["sum"] += a
                    stats["sumsq"] += a**2
                    stats["count"] += 1

            assert stats is not None, f"No valid actions in episode {episode_index}"
            stats = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in stats.items()}

            tmp_dir = out_base / f"_tmp_ep_{episode_index:06d}"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            parquet_tmp = tmp_dir / "episode.parquet"
            video_tmp = tmp_dir / "episode.mp4"

            ds = Dataset.from_list(rows, features=self.features)
            ds.to_parquet(str(parquet_tmp))

            def frame_iter():
                for p in rgb_paths:
                    yield iio.imread(p)

            iio.imwrite(video_tmp, list(frame_iter()), fps=FPS, codec="libx264")
            os.replace(parquet_tmp, parquet_path)
            os.replace(video_tmp, vid_path)
            shutil.rmtree(tmp_dir)

            action_mean = (np.array(stats["sum"]) / stats["count"]).tolist()
            action_std = (
                np.sqrt(
                    np.maximum(
                        np.array(stats["sumsq"]) / stats["count"]
                        - np.square(np.array(stats["sum"]) / stats["count"]),
                        0,
                    )
                )
            ).tolist()

            episode_stats = {
                "episode_index": episode_index,
                "stats": {
                    "action": {
                        "min": stats["min"],
                        "max": stats["max"],
                        "mean": action_mean,
                        "std": action_std,
                        "count": [len(rows)],
                    },
                    "timestamp": {
                        "min": [0.0],
                        "max": [(len(rows) - 1) / FPS],
                        "mean": [((len(rows) - 1) / 2) / FPS],
                        "std": [len(rows) / (2 * FPS * math.sqrt(3))],
                        "count": [len(rows)],
                    },
                },
            }
            meta_dir = out_base.parent / "meta"
            meta_dir.mkdir(parents=True, exist_ok=True)
            append_jsonl_line_atomic(meta_dir / "episodes_stats.jsonl", episode_stats)
        except Exception as e:
            print(f"Error processing episode {episode_index} in {episode_dir}: {e}")
            exit(-1)

        return episode_index, len(rows), stats

    def run(self, data_root: Path, work_dir: Path, chunks_size: int, num_workers: int, robot_type: str):
        self.chunks_size = chunks_size
        tdd = Path("task_description_dict.json")
        if not tdd.is_file():
            self.task_description_dict = {}
        else:
            self.task_description_dict = json.load(open(tdd))

        data_dir = work_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        self.episode_sources: list[tuple[int, Path, str, str, str, str]] = []
        task_index = 0
        self.tasks_meta = {}
        ep_index = 0

        all_ep_dirs: List[Tuple[int, Path, str, str, str]] = []
        for task_name, task_dir, cat_name, leaf_name in iter_tasks(data_root):
            desc = self.task_description_dict.get(leaf_name, "")
            ep_dirs = [p for p in task_dir.iterdir() if p.is_dir() and re.match(r"episode_\d+", p.name)]
            ep_dirs = sorted(ep_dirs, key=lambda p: int(re.findall(r"\d+", p.name)[0]))
            for ep_dir in ep_dirs:
                all_ep_dirs.append((task_index, ep_dir, desc, task_name, cat_name))
            self.tasks_meta[task_index] = {"name": task_name, "category": cat_name, "description": desc}
            task_index += 1

        print(f"Detecting robot types for {len(all_ep_dirs)} episodes...")
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            ep_dirs_only = [t[1] for t in all_ep_dirs]
            robot_types = list(
                tqdm(
                    ex.map(detect_robot_type, ep_dirs_only),
                    total=len(ep_dirs_only),
                    desc="Scanning robot types",
                    unit="ep",
                )
            )

        meta_dir = work_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        done_eps = load_done_episodes(meta_dir)

        filtered = []
        for (meta, rtype) in zip(all_ep_dirs, robot_types):
            task_idx, ep_dir, desc, tname, _cat_name = meta
            if robot_type == "both" or rtype == robot_type:
                filtered.append((task_idx, ep_dir, ep_index, desc, tname, rtype))
                ep_index += 1

        filtered = [
            (task_idx, ep_dir, ep_index, desc, tname, rtype)
            for (task_idx, ep_dir, ep_index, desc, tname, rtype) in filtered
            if not episode_complete(ep_index, work_dir, chunks_size, done_eps)
        ]

        self.episode_sources = filtered

        if not self.episode_sources:
            print(f"No episodes matched robot type '{robot_type}'.")
            return

        print(f"Resuming: {len(filtered)} new episodes (skipped {len(done_eps)})")

        total = len(self.episode_sources)
        data_stats: list[tuple[int, int, dict[str, any]]] = []
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = [
                ex.submit(self.make_one_episode, task_idx, i, ep_dir, data_dir, chunks_size, rtype)
                for i, (task_idx, ep_dir, _, _, _, rtype) in enumerate(self.episode_sources)
            ]
            for fut in tqdm(as_completed(futures), total=total, desc="Processing new episodes", unit="ep"):
                ep_idx, n_frames, stats = fut.result()
                if n_frames <= 0:
                    continue
                data_stats.append((ep_idx, n_frames, stats))

                self.lengths_by_episode[ep_idx] = n_frames
                self.num_episodes = len(self.lengths_by_episode)
                self.total_frames = sum(self.lengths_by_episode.values())

        self.lengths_by_episode = {ep_idx: n for ep_idx, n, _ in data_stats}
        self.num_episodes = len(self.episode_sources)
        self.total_frames = sum(self.lengths_by_episode.values())

        print(f"Now total episodes: {self.num_episodes}, frames: {self.total_frames}")

    def scan_meta_only(self, data_root: Path, chunks_size: int, num_workers: int, robot_type: str):
        self.chunks_size = chunks_size
        tdd = Path("task_description_dict.json")
        if not tdd.is_file():
            self.task_description_dict = {}
        else:
            self.task_description_dict = json.load(open(tdd))

        all_ep_dirs: List[Tuple[int, Path, str, str, str]] = []
        task_index = 0
        self.tasks_meta = {}
        for task_name, task_dir, cat_name, leaf_name in iter_tasks(data_root):
            desc = self.task_description_dict.get(leaf_name, "")
            ep_dirs = [p for p in task_dir.iterdir() if p.is_dir() and re.match(r"episode_\d+", p.name)]
            ep_dirs = sorted(ep_dirs, key=lambda p: int(re.findall(r"\d+", p.name)[0]))
            for ep_dir in ep_dirs:
                all_ep_dirs.append((task_index, ep_dir, desc, task_name, cat_name))
            self.tasks_meta[task_index] = {"name": task_name, "category": cat_name, "description": desc}
            task_index += 1

        print(f"Detecting robot types for {len(all_ep_dirs)} episodes...")
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            ep_dirs_only = [t[1] for t in all_ep_dirs]
            robot_types = list(
                tqdm(
                    ex.map(detect_robot_type, ep_dirs_only),
                    total=len(ep_dirs_only),
                    desc="Scanning robot types",
                    unit="ep",
                )
            )

        filtered = []
        ep_index = 0
        for (meta, rtype) in zip(all_ep_dirs, robot_types):
            task_idx, ep_dir, desc, tname, _cat_name = meta
            if robot_type == "both" or rtype == robot_type:
                filtered.append((task_idx, ep_dir, ep_index, desc, tname, rtype))
                ep_index += 1

        self.episode_sources = filtered
        self.lengths_by_episode = {}
        for _task_idx, ep_dir, ep_index, _desc, _tname, _rtype in self.episode_sources:
            try:
                data_list = read_json_list(ep_dir / "data.json")
                n = len(data_list)
            except Exception:
                n = 0
            if n > 0:
                self.lengths_by_episode[ep_index] = n

        self.num_episodes = len(self.lengths_by_episode)
        self.total_frames = sum(self.lengths_by_episode.values())
        print(f"Meta-only scan complete: {self.num_episodes} episodes, {self.total_frames} frames")

    def write_meta(self, out_dir: Path):
        meta_dir = out_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        dataset_cursor = 0
        ep_rows_meta = []

        for (task_idx, ep_dir, ep_index, task_dsc, _tname, _rtype) in sorted(
            self.episode_sources, key=lambda x: x[0]
        ):
            n = self.lengths_by_episode.get(ep_index, 0)
            if n <= 0:
                continue
            ep_rows_meta.append(
                {
                    "episode_index": ep_index,
                    "tasks": [task_idx],
                    "length": n,
                    "dataset_from_index": dataset_cursor,
                    "dataset_to_index": dataset_cursor + (n - 1),
                    "robot_type": self.get_robot_type(ep_dir),
                    "instruction": task_dsc,
                }
            )
            dataset_cursor += n

        episodes_df = pd.DataFrame(ep_rows_meta).sort_values("episode_index").reset_index(drop=True)

        task_rows = []
        for ti, meta in self.tasks_meta.items():
            task_rows.append(
                {
                    "task_index": ti,
                    "task": meta.get("name", f"task_{ti:04d}"),
                    "category": meta.get("category", ""),
                    "description": meta.get("description", ""),
                }
            )
        tasks_df = pd.DataFrame(task_rows).sort_values("task_index").reset_index(drop=True)

        features_meta = {
            "observation.images.egocentric": {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": float(FPS),
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.arm_joints": {"dtype": "float32", "shape": [-1]},
            "observation.leg_joints": {"dtype": "float32", "shape": [-1]},
            "observation.hand_joints": {"dtype": "float32", "shape": [-1]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "next.done": {"dtype": "bool", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        }

        for key in self.action_keys:
            if key == "action.joint_angles":
                features_meta[key] = {"dtype": "float32", "shape": [-1]}
            else:
                features_meta[key] = {"dtype": "float32", "shape": [3]}

        robot_types = set(episodes_df["robot_type"].tolist()) if not episodes_df.empty else {"h1"}
        global_robot_type = list(robot_types)[0] if len(robot_types) == 1 else "mixed"

        info = InfoDict(
            codebase_version=CODE_VERSION,
            robot_type=global_robot_type,
            total_episodes=self.num_episodes,
            total_frames=self.total_frames,
            total_tasks=len(self.tasks_meta),
            total_videos=self.num_episodes,
            total_chunks=math.ceil(self.num_episodes / self.chunks_size),
            chunks_size=self.chunks_size,
            fps=FPS,
            data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            video_path="videos/chunk-{episode_chunk:03d}/egocentric/episode_{episode_index:06d}.mp4",
            features=features_meta,
        )

        (meta_dir / "info.json").write_text(json.dumps(asdict(info), indent=4))

        with open(meta_dir / "tasks.jsonl", "w") as f_tasks:
            for row in tasks_df.to_dict(orient="records"):
                json.dump(row, f_tasks)
                f_tasks.write("\n")

        with open(meta_dir / "episodes.jsonl", "w") as f_eps:
            for row in episodes_df.to_dict(orient="records"):
                json.dump(row, f_eps)
                f_eps.write("\n")

        stats_path = meta_dir / "episodes_stats.jsonl"
        if not stats_path.exists():
            stats_path.write_text("")

        print(
            f"\nWrote meta (info.json, tasks.jsonl, episodes.jsonl, episodes_stats.jsonl) and {self.num_episodes} episode(s) into: {out_dir}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--work-dir", type=str, default="_lerobot_build")
    parser.add_argument("--repo-id", type=str)
    parser.add_argument("--chunks-size", type=int, default=1000)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--repo-exist-ok", action="store_true")
    parser.add_argument("--num-workers", type=int, default=os.cpu_count(), help="Max parallel workers (default: all CPUs)")
    parser.add_argument(
        "--robot-type",
        type=str,
        choices=["h1", "g1", "both"],
        default="both",
        help="Filter episodes by robot type (h1, g1, or both)",
    )
    parser.add_argument("--meta-only", action="store_true", help="Skip data/video generation and only write meta files")
    parser.add_argument("--urdf-g1", type=str, default="./real/assets/g1/g1_body29_hand14.urdf")
    parser.add_argument("--urdf-h1", type=str, default="./real/assets/h1_inspire/urdf/h1_inspire.urdf")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    work_dir = Path(args.work_dir).resolve()
    for d in [work_dir / "data", work_dir / "videos", work_dir / "meta"]:
        d.mkdir(parents=True, exist_ok=True)

    urdf_map = {
        "g1": str(Path(args.urdf_g1).expanduser().resolve()),
        "h1": str(Path(args.urdf_h1).expanduser().resolve()),
    }

    pipeline = HE2WePretrainConverter(args.robot_type, urdf_map)
    if args.meta_only:
        pipeline.scan_meta_only(data_root, args.chunks_size, args.num_workers, args.robot_type)
    else:
        pipeline.run(data_root, work_dir, args.chunks_size, args.num_workers, args.robot_type)
    pipeline.write_meta(work_dir)

    if args.push:
        if not args.repo_id:
            raise ValueError("--repo-id is required when --push is set")
        create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=args.repo_exist_ok)
        upload_large_folder(repo_id=args.repo_id, repo_type="dataset", folder_path=str(work_dir))
        create_tag(args.repo_id, tag=CODE_VERSION, repo_type="dataset")
        print(f"\n✅ Uploaded to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
