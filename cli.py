from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List

from data_preparation.hybrid import main as hybrid
from data_preparation.sfm import main as sfm
from data_preparation.slam import main as slam
from data_preparation.shared.route_helpers import invoke_module
from data_preparation.workflows import WORKFLOWS, run_workflow


@dataclass(frozen=True)
class Command:
    module: str
    help: str


LEGACY_COMMANDS: Dict[str, Command] = {
    "rosbag-inspect": Command(
        "data_preparation.rosbag_to_3dgs.inspect_rosbags_for_3dgs",
        "Inspect ROS2 bags and write inventory/feasibility reports.",
    ),
    "rosbag-extract-images": Command(
        "data_preparation.rosbag_to_colmap.extract_rosbag_images",
        "Extract image-only ROS2 bag frames for a pure visual COLMAP SfM baseline.",
    ),
    "hybrid-sfm-lidar": Command(
        "data_preparation.slam_to_colmap.filtered_scene_main",
        "Build a COLMAP-camera scene with transformed SLAM/LiDAR seed points.",
    ),
    "slam-to-colmap": Command(
        "data_preparation.slam_to_colmap.main",
        "Convert a pure-headerstamp SLAM/LiDAR scene into a COLMAP text model.",
    ),
    "rectify-fishpoly": Command(
        "data_preparation.rectification.fishpoly_to_pinhole",
        "Rectify a FishPoly SLAM/LiDAR scene into undistorted pinhole images.",
    ),
    "video2colmap": Command(
        "data_preparation.video2colmap.preprocess_video_to_colmap",
        "Extract video frames and run a COLMAP preprocessing pipeline.",
    ),
}


FORMAL_COMMAND_HELP: Dict[str, str] = {
    "sfm": "Run COLMAP/SfM from an existing images_rectified/ directory.",
    "hybrid": "Keep SfM cameras/poses and replace points3D with aligned SLAM/LiDAR points.",
    "slam": "Convert SLAM poses and a colored SLAM point cloud into a COLMAP text model.",
}

FORMAL_COMMANDS: Dict[str, Callable[[List[str]], int]] = {
    "sfm": sfm.run,
    "hybrid": hybrid.run,
    "slam": slam.run,
}

COMMANDS: Dict[str, Command] = LEGACY_COMMANDS


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m data_preparation",
        description="Build COLMAP-compatible scenes through the formal SFM, hybrid, and SLAM routes.",
        epilog=(
            "Formal routes: sfm, hybrid, slam. "
            "Legacy/debug commands remain callable directly but are hidden from the default summary."
        ),
    )
    parser.add_argument("command", nargs="?", help="Command to run.")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the selected command.")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command is None:
        print("Formal routes:")
        for name in ("sfm", "hybrid", "slam"):
            print(f"  {name:<12} {FORMAL_COMMAND_HELP[name]}")
        print("\nUse: python -m data_preparation <sfm|hybrid|slam> --help")
        print("Legacy/debug commands remain callable directly but are hidden from this summary.")
        return 0

    if args.command in FORMAL_COMMANDS:
        try:
            return FORMAL_COMMANDS[args.command](args.args)
        except ValueError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 2

    if args.command in WORKFLOWS:
        return run_workflow(args.command, args.args)

    if args.command not in LEGACY_COMMANDS:
        print(f"[ERROR] Unknown command: {args.command}", file=sys.stderr)
        print("Use: python -m data_preparation <sfm|hybrid|slam> --help", file=sys.stderr)
        return 2

    command = LEGACY_COMMANDS[args.command]
    invoke_module(command.module, args.command, args.args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
