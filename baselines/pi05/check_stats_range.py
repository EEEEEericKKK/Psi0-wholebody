"""Check if q01 and q99 are nearly identical for each dimension in a stats JSON file.

A dimension is flagged if |q99[i] - q01[i]| < threshold (default 1e-5),
meaning the feature has almost no variance across the dataset.

Usage:
    python check_stats_range.py <stats_json_path> [--threshold 1e-5]
"""

import json
import argparse
from pathlib import Path


def check_stats(path: str, threshold: float = 1e-5) -> None:
    with open(path) as f:
        stats = json.load(f)

    any_flagged = False
    for key, fields in stats.items():
        q01 = fields.get("q01")
        q99 = fields.get("q99")
        if q01 is None or q99 is None:
            continue

        # Scalar case: wrap in list
        if not isinstance(q01, list):
            q01 = [q01]
            q99 = [q99]

        flagged = [(i, q01[i], q99[i]) for i in range(len(q01)) if abs(q99[i] - q01[i]) < threshold]
        if flagged:
            any_flagged = True
            print(f"\n[{key}] {len(flagged)}/{len(q01)} dimensions have |q99-q01| < {threshold:.0e}:")
            for i, lo, hi in flagged:
                print(f"  dim {i:3d}: q01={lo:.6e}  q99={hi:.6e}  diff={abs(hi-lo):.6e}")

    if not any_flagged:
        print(f"All dimensions have |q99-q01| >= {threshold:.0e}. No issues found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to stats JSON file")
    parser.add_argument("--threshold", type=float, default=1e-5)
    args = parser.parse_args()
    check_stats(args.path, args.threshold)
