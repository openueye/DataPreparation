from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from data_preparation.shared.route_helpers import (
    append_passthrough,
    invoke_module,
    prepare_scene_from_rosbag,
    resolve_prepared_scene_output,
    resolve_route_output,
    scene_name_from_rosbag_dir,
)


def run(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m data_preparation slam",
        description="Organize an Odin1 ROS bag directory and convert its SLAM poses/points into a COLMAP text model.",
    )
    parser.add_argument("--rosbag-dir", required=True, type=Path, help="Odin1 ROS bag directory containing a .db3 file.")
    parser.add_argument("--scene", default=None, help="Scene name used for the default 04_ProcessedData/slam output.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Explicit COLMAP scene output directory.")
    parser.add_argument("--prepared-output-dir", type=Path, default=None, help="Optional organized intermediate scene output directory.")
    parser.add_argument("--overwrite-prepared", action="store_true", help="Regenerate the organized intermediate scene.")
    parser.add_argument("--max-points", type=int, default=3_000_000, help="Maximum point count after voxel downsampling. Use 0 for all.")
    parser.add_argument(
        "--keep-static-poses",
        action="store_true",
        help="Keep consecutive duplicate/near-duplicate pose frames in the COLMAP export.",
    )
    parser.add_argument(
        "--min-pose-translation-m",
        type=float,
        default=1e-9,
        help="Translation threshold for dropping consecutive static poses.",
    )
    parser.add_argument(
        "--min-pose-rotation-deg",
        type=float,
        default=1e-6,
        help="Rotation threshold for dropping consecutive static poses.",
    )
    parser.add_argument("--repo-root", type=Path, default=None, help="Optional 00_Baselines repo root.")
    parser.add_argument("--thesis-root", type=Path, default=None, help="Optional explicit Thesis root.")
    parser.add_argument("passthrough", nargs=argparse.REMAINDER, help="Advanced args passed after -- to SLAM backend.")
    args = parser.parse_args(argv)
    rosbag_dir = args.rosbag_dir.expanduser().resolve()
    scene_name = scene_name_from_rosbag_dir(rosbag_dir, args.scene)
    prepared_dir = resolve_prepared_scene_output(
        scene=scene_name,
        output_dir=args.prepared_output_dir,
        repo_root=args.repo_root,
        thesis_root=args.thesis_root,
    )
    prepare_scene_from_rosbag(
        rosbag_dir=rosbag_dir,
        scene=scene_name,
        output_dir=prepared_dir,
        overwrite=args.overwrite_prepared,
    )
    output_dir = resolve_route_output(
        scene=scene_name,
        output_dir=args.output_dir,
        repo_root=args.repo_root,
        thesis_root=args.thesis_root,
        category="slam",
    )
    backend_argv = ["--input-dir", str(prepared_dir), "--output-dir", str(output_dir)]
    backend_argv.extend(["--max-points", str(args.max_points)])
    if args.keep_static_poses:
        backend_argv.append("--keep-static-poses")
    backend_argv.extend(["--min-pose-translation-m", str(args.min_pose_translation_m)])
    backend_argv.extend(["--min-pose-rotation-deg", str(args.min_pose_rotation_deg)])
    invoke_module(
        "data_preparation.slam.export_colmap",
        "slam",
        append_passthrough(backend_argv, args.passthrough),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1:]))
