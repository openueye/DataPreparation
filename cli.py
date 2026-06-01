from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Dict, List

from data_preparation.hybrid import main as hybrid
from data_preparation.sfm import main as sfm
from data_preparation.slam import main as slam
from data_preparation.shared.route_helpers import invoke_module


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
    "video2colmap": Command(
        "data_preparation.video2colmap.preprocess_video_to_colmap",
        "Extract video frames and run a COLMAP preprocessing pipeline.",
    ),
}


COMMANDS: Dict[str, Command] = LEGACY_COMMANDS
DEPTH_PRIOR_COMMAND = Command(
    "data_preparation.depth_prior.project",
    "Generate synchronized raw LiDAR depth priors.",
)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m data_preparation",
        description="Build COLMAP-compatible scenes through the formal SFM, hybrid, and SLAM routes.",
        epilog=(
            "Formal routes: sfm, hybrid, slam. "
            "Remaining legacy/debug commands are hidden from the default summary."
        ),
    )
    parser.add_argument("command", nargs="?", help="Command to run.")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the selected command.")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command is None:
        print(
            "Formal routes:\n"
            "  sfm          Organize a ROS bag and run COLMAP/SfM on rectified images.\n"
            "  hybrid       Organize a ROS bag, run SfM, and align SLAM/LiDAR points.\n"
            "  slam         Organize a ROS bag and convert SLAM poses/points to COLMAP text.\n"
            "  depth-prior-project\n"
            "               Generate scene/depths/*.npy metric depth priors.\n"
            "\nUse: python -m data_preparation <sfm|hybrid|slam|depth-prior-project> --help"
        )
        print("Remaining legacy/debug commands are hidden from this summary.")
        return 0

    if args.command == "sfm":
        try:
            return sfm.run(args.args)
        except ValueError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 2

    if args.command == "hybrid":
        try:
            return hybrid.run(args.args)
        except ValueError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 2

    if args.command == "slam":
        try:
            return slam.run(args.args)
        except ValueError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 2

    if args.command == "depth-prior-project":
        invoke_module(DEPTH_PRIOR_COMMAND.module, args.command, args.args)
        return 0

    if args.command not in LEGACY_COMMANDS:
        print(f"[ERROR] Unknown command: {args.command}", file=sys.stderr)
        print("Use: python -m data_preparation <sfm|hybrid|slam|depth-prior-project> --help", file=sys.stderr)
        return 2

    command = LEGACY_COMMANDS[args.command]
    invoke_module(command.module, args.command, args.args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
