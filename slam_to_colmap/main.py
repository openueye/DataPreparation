from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deprecated: the old SLAM-pose/LiDAR-points COLMAP-compatible export "
            "is disabled. Use `python -m data_preparation hybrid-sfm-lidar` instead."
        )
    )
    return parser.parse_args()


def main() -> None:
    parse_args()
    raise SystemExit(
        "Deprecated command disabled: *_slam_compat used SLAM camera poses and is not a clean "
        "LiDAR-initialization ablation. Use `python -m data_preparation hybrid-sfm-lidar` to build "
        "a COLMAP-camera/LiDAR-seed scene."
    )


if __name__ == "__main__":
    main()
