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


ADAPTER_COMMANDS: Dict[str, Command] = {
    "lsgslam-export": Command(
        "data_preparation.slam.lsgslam_adapter",
        "Export a COLMAP-compatible scene to LSG-SLAM EuRoC-style RGB-D layout.",
    ),
    "video2colmap": Command(
        "data_preparation.video2colmap.preprocess_video_to_colmap",
        "Deprecated direct backend. Prefer the sfm route unless debugging video-only preprocessing.",
    ),
}


COMMANDS: Dict[str, Command] = ADAPTER_COMMANDS
DEPTH_PRIOR_COMMAND = Command(
    "data_preparation.depth_prior.project",
    "Generate synchronized raw LiDAR depth priors.",
)
DEPTH_PRIOR_COMMANDS: Dict[str, Command] = {
    "depth-prior-project": DEPTH_PRIOR_COMMAND,
    "depth-prior-edge-masks": Command(
        "data_preparation.depth_prior.edge_masks",
        "Create edge-gated masks for completed depth priors.",
    ),
    "depth-prior-apply-mask": Command(
        "data_preparation.depth_prior.edge_masks",
        "Apply a depth-prior mask by zeroing rejected completed pixels.",
    ),
    "depth-prior-sidecars": Command(
        "data_preparation.depth_prior.sidecars",
        "Generate confidence and source-label sidecars for completed depth priors.",
    ),
}


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m data_preparation",
        description="Build COLMAP-compatible scenes through the formal SFM, hybrid, and SLAM routes.",
        epilog=(
            "Formal routes: sfm, hybrid, slam. "
            "Use depth-prior-* commands for depth-prior artifacts and lsgslam-export for LSG-SLAM layout export."
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
            "  depth-prior-edge-masks\n"
            "               Create edge-gated masks for completed depth priors.\n"
            "  depth-prior-apply-mask\n"
            "               Apply a depth-prior mask by zeroing rejected pixels.\n"
            "  depth-prior-sidecars\n"
            "               Generate confidence/source-label sidecars.\n"
            "  lsgslam-export\n"
            "               Export COLMAP scene + depth priors to LSG-SLAM RGB-D layout.\n"
            "\nUse: python -m data_preparation <command> --help"
        )
        print("Direct backend command video2colmap is kept for compatibility but is deprecated.")
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

    if args.command in DEPTH_PRIOR_COMMANDS:
        invoke_module(DEPTH_PRIOR_COMMANDS[args.command].module, args.command, args.args)
        return 0

    if args.command not in COMMANDS:
        print(f"[ERROR] Unknown command: {args.command}", file=sys.stderr)
        print(
            "Use: python -m data_preparation <sfm|hybrid|slam|depth-prior-project|depth-prior-edge-masks|depth-prior-apply-mask|depth-prior-sidecars|lsgslam-export> --help",
            file=sys.stderr,
        )
        return 2

    command = COMMANDS[args.command]
    invoke_module(command.module, args.command, args.args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
