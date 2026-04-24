from __future__ import annotations

import argparse
import runpy
import sys
from dataclasses import dataclass
from typing import Dict, List

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
    "rosbag-convert": Command(
        "data_preparation.rosbag_to_3dgs.convert_rosbag_to_3dgs",
        "Convert one ROS2 bag into a 3DGS-ready LiDAR/RGB scene.",
    ),
    "rosbag-validate-projection": Command(
        "data_preparation.rosbag_to_3dgs.validate_extrinsic_projection",
        "Validate raw bag LiDAR-to-image projection with extrinsic overlays.",
    ),
    "slam-to-colmap": Command(
        "data_preparation.slam_to_colmap.main",
        "Convert an exported SLAM/LiDAR scene to COLMAP text layout.",
    ),
    "rectify-fishpoly": Command(
        "data_preparation.rectification.fishpoly_to_pinhole",
        "Rectify a FishPoly SLAM/LiDAR scene into undistorted pinhole images.",
    ),
    "video2colmap": Command(
        "data_preparation.video2colmap.preprocess_video_to_colmap",
        "Extract video frames and run a COLMAP preprocessing pipeline.",
    ),
    "projection-check": Command(
        "data_preparation.data_quality.projection_overlay",
        "Check processed scene LiDAR/RGB projection quality.",
    ),
    "colorize-lidar": Command(
        "data_preparation.data_quality.colorize_lidar_map",
        "Colorize a global LiDAR map using synchronized RGB images.",
    ),
}


WORKFLOW_HELP: Dict[str, str] = {
    "inspect": "Inspect raw inputs for a scene and write standardized reports.",
    "prepare": "Prepare a training-ready scene from rosbag or video inputs.",
    "validate": "Validate a prepared scene and write projection diagnostics.",
    "colorize": "Colorize a LiDAR map using synchronized RGB images.",
    "export": "Export derived scene formats such as COLMAP-compatible layout.",
    "run": "Run the default prepare + validate workflow for a scene.",
}

COMMANDS: Dict[str, Command] = LEGACY_COMMANDS


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m data_preparation",
        description="Data preprocessing toolkit for ROS bags, SLAM exports, videos, and quality checks.",
    )
    parser.add_argument("command", nargs="?", choices=sorted([*WORKFLOWS, *LEGACY_COMMANDS]), help="Command to run.")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the selected command.")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command is None:
        print("Workflow commands:")
        for name in ("inspect", "prepare", "validate", "colorize", "export", "run"):
            print(f"  {name:<28} {WORKFLOW_HELP[name]}")
        print("\nLegacy commands:")
        for name in sorted(LEGACY_COMMANDS):
            print(f"  {name:<28} {LEGACY_COMMANDS[name].help}")
        print("\nUse: python -m data_preparation <command> --help")
        return 0

    if args.command in WORKFLOWS:
        return run_workflow(args.command, args.args)

    command = LEGACY_COMMANDS[args.command]
    sys.argv = [args.command, *args.args]
    runpy.run_module(command.module, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
