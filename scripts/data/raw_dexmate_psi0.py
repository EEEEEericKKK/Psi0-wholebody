"""
Convert dexmate teleoperation recordings to LeRobot dataset format.

Input structure:
  <data-root>/
    episode_0/
      data.json   # list of frames with states/actions/image/lidar
      images/
        frame_000000.jpg ...
    episode_1/
      ...

Output (LeRobot v2.1):
  <out-dir>/
    data/chunk-000/episode_000000.parquet
    videos/chunk-000/egocentric/episode_000000.mp4
    meta/info.json
    meta/tasks.jsonl
    meta/episodes.jsonl
    meta/episodes_stats.jsonl
"""

import argparse
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import imageio.v3 as iio
import numpy as np
from PIL import Image
import pandas as pd
from datasets import Dataset, Features, Sequence, Value

FPS = 30
CHUNKS_SIZE = 1000
CODE_VERSION = "v2.1"

FEATURES = Features(
    {
        "states": Sequence(Value("float32")),
        "action": Sequence(Value("float32")),
        "timestamp": Value("float32"),
        "frame_index": Value("int64"),
        "episode_index": Value("int64"),
        "index": Value("int64"),
        "task_index": Value("int64"),
        "next.done": Value("bool"),
    }
)


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


def load_data_json(path: Path) -> List[Dict]:
    with open(path, encoding="latin-1") as f:
        return json.load(f)


def build_states(frame: Dict) -> List[float]:
    s = frame["states"]
    return (
        [float(x) for x in s["hand_joints"]]
        + [float(x) for x in s["arm_joints"]]
        + [float(x) for x in s["head"]]
        + [float(x) for x in s["torso"]]
    )


def build_action(frame: Dict) -> List[float]:
    a = frame["actions"]
    return (
        [float(x) for x in a["hand_joints"]]
        + [float(x) for x in a["arm_joints"]]
        + [float(x) for x in a["head"]]
        + [float(x) for x in a["torso"]]
    )


def process_episode(
    ep_dir: Path,
    episode_index: int,
    task_index: int,
    out_dir: Path,
    task_description: str,
) -> int:
    data_list = load_data_json(ep_dir / "data.json")
    assert data_list, f"Empty data.json in {ep_dir}"
    data_list = data_list[:450]

    # --- parquet ---
    chunk_id = episode_index // CHUNKS_SIZE
    parquet_dir = out_dir / "data" / f"chunk-{chunk_id:03d}"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = parquet_dir / f"episode_{episode_index:06d}.parquet"

    rows = []
    for i, frame in enumerate(data_list):
        rows.append(
            {
                "states": build_states(frame),
                "action": build_action(frame),
                "timestamp": float(i) / FPS,
                "frame_index": i,
                "episode_index": episode_index,
                "index": i,
                "task_index": task_index,
                "next.done": (i == len(data_list) - 1),
            }
        )

    ds = Dataset.from_list(rows, features=FEATURES)
    ds.to_parquet(str(parquet_path))

    # --- video ---
    vid_dir = out_dir / "videos" / f"chunk-{chunk_id:03d}" / "egocentric"
    vid_dir.mkdir(parents=True, exist_ok=True)
    vid_path = vid_dir / f"episode_{episode_index:06d}.mp4"

    frames = [np.array(Image.fromarray(iio.imread(ep_dir / f["image"])).resize((640, 480))) for f in data_list]
    iio.imwrite(str(vid_path), frames, fps=FPS, codec="libx264")

    # --- per-episode stats ---
    actions = np.array([r["action"] for r in rows], dtype=np.float32)
    n = len(rows)
    ep_stats = {
        "episode_index": episode_index,
        "stats": {
            "action": {
                "min": actions.min(0).tolist(),
                "max": actions.max(0).tolist(),
                "mean": actions.mean(0).tolist(),
                "std": actions.std(0).tolist(),
                "count": [n],
            },
            "timestamp": {
                "min": [0.0],
                "max": [(n - 1) / FPS],
                "mean": [((n - 1) / 2) / FPS],
                "std": [n / (2 * FPS * math.sqrt(3))],
                "count": [n],
            },
        },
    }
    stats_path = out_dir / "meta" / "episodes_stats.jsonl"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "a") as f:
        f.write(json.dumps(ep_stats) + "\n")

    return n


def write_meta(
    out_dir: Path,
    episode_lengths: Dict[int, int],
    task_description: str,
    robot_type: str = "dexmate",
):
    meta_dir = out_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    total_frames = sum(episode_lengths.values())
    num_episodes = len(episode_lengths)

    # episodes.jsonl
    cursor = 0
    ep_rows = []
    for ep_idx in sorted(episode_lengths):
        n = episode_lengths[ep_idx]
        ep_rows.append(
            {
                "episode_index": ep_idx,
                "tasks": [0],
                "length": n,
                "dataset_from_index": cursor,
                "dataset_to_index": cursor + n - 1,
            }
        )
        cursor += n

    with open(meta_dir / "episodes.jsonl", "w") as f:
        for row in ep_rows:
            f.write(json.dumps(row) + "\n")

    # tasks.jsonl
    with open(meta_dir / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": task_description}) + "\n")

    # info.json
    state_dim = 12 + 14 + 3 + 3  # hand + arm + head + torso
    action_dim = state_dim
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
        "states": {"dtype": "float32", "shape": [state_dim]},
        "action": {"dtype": "float32", "shape": [action_dim]},
        "timestamp": {"dtype": "float32", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "next.done": {"dtype": "bool", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
    }

    info = InfoDict(
        codebase_version=CODE_VERSION,
        robot_type=robot_type,
        total_episodes=num_episodes,
        total_frames=total_frames,
        total_tasks=1,
        total_videos=num_episodes,
        total_chunks=math.ceil(num_episodes / CHUNKS_SIZE),
        chunks_size=CHUNKS_SIZE,
        fps=FPS,
        data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        video_path="videos/chunk-{episode_chunk:03d}/egocentric/episode_{episode_index:06d}.mp4",
        features=features_meta,
    )
    (meta_dir / "info.json").write_text(json.dumps(asdict(info), indent=4))

    print(f"Wrote {num_episodes} episodes, {total_frames} frames → {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert dexmate recordings to LeRobot format")
    parser.add_argument("--data-root", required=True, help="Path to task session dir (contains episode_N/ subdirs)")
    parser.add_argument("--out-dir", required=True, help="Output directory for LeRobot dataset")
    parser.add_argument("--task", default="manipulation task", help="Task description string")
    parser.add_argument("--robot-type", default="dexmate", help="Robot type label")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()

    def find_episode_dirs(root: Path) -> List[Path]:
        """Find episode_N dirs directly under root, or one level deeper in subdirs."""
        direct = [p for p in root.iterdir() if p.is_dir() and re.match(r"episode_\d+", p.name)]
        if direct:
            return sorted(direct, key=lambda p: int(re.search(r"\d+", p.name).group()))
        # collect from all immediate subdirs, sorted by subdir name then episode number
        result = []
        for subdir in sorted(root.iterdir(), key=lambda p: p.name):
            if subdir.is_dir():
                result.extend(
                    sorted(
                        [p for p in subdir.iterdir() if p.is_dir() and re.match(r"episode_\d+", p.name)],
                        key=lambda p: int(re.search(r"\d+", p.name).group()),
                    )
                )
        return result

    ep_dirs = find_episode_dirs(data_root)
    assert ep_dirs, f"No episode_N directories found in {data_root}"
    print(f"Found {len(ep_dirs)} episodes in {data_root}")

    episode_lengths = {}
    for ep_index, ep_dir in enumerate(ep_dirs):
        print(f"  Processing {ep_dir.name} → episode_{ep_index:06d} ...", end=" ", flush=True)
        n = process_episode(ep_dir, ep_index, task_index=0, out_dir=out_dir, task_description=args.task)
        episode_lengths[ep_index] = n
        print(f"{n} frames")

    write_meta(out_dir, episode_lengths, args.task, args.robot_type)


if __name__ == "__main__":
    main()
