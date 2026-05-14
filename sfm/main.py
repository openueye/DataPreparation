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
        prog="python -m data_preparation sfm",
        description="Organize an Odin1 ROS bag directory and run COLMAP/SfM on its rectified images.",
    )
    parser.add_argument("--rosbag-dir", required=True, type=Path, help="Odin1 ROS bag directory containing a .db3 file.")
    parser.add_argument("--scene", default=None, help="Scene name used for the default 04_ProcessedData/sfm output.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Explicit COLMAP scene output directory.")
    parser.add_argument("--prepared-output-dir", type=Path, default=None, help="Optional organized intermediate scene output directory.")
    parser.add_argument("--overwrite-prepared", action="store_true", help="Regenerate the organized intermediate scene.")
    parser.add_argument("--repo-root", type=Path, default=None, help="Optional 00_Baselines repo root.")
    parser.add_argument("--thesis-root", type=Path, default=None, help="Optional explicit Thesis root.")
    parser.add_argument("passthrough", nargs=argparse.REMAINDER, help="Advanced args passed after -- to COLMAP backend.")
    args = parser.parse_args(argv)
    rosbag_dir = args.rosbag_dir.expanduser().resolve()
    scene = scene_name_from_rosbag_dir(rosbag_dir, args.scene)
    prepared_dir = resolve_prepared_scene_output(
        scene=scene,
        output_dir=args.prepared_output_dir,
        repo_root=args.repo_root,
        thesis_root=args.thesis_root,
    )
    prepare_scene_from_rosbag(
        rosbag_dir=rosbag_dir,
        scene=scene,
        output_dir=prepared_dir,
        overwrite=args.overwrite_prepared,
    )
    output_dir = resolve_route_output(
        scene=scene,
        output_dir=args.output_dir,
        repo_root=args.repo_root,
        thesis_root=args.thesis_root,
        category="sfm",
    )
    backend_argv = ["--image-dir", str(prepared_dir / "images_rectified"), "--output-dir", str(output_dir)]
    invoke_module(
        "data_preparation.video2colmap.preprocess_video_to_colmap",
        "sfm",
        append_passthrough(backend_argv, args.passthrough),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1:]))
