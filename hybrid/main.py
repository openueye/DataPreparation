from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from data_preparation.shared.route_helpers import append_passthrough, invoke_module, resolve_route_output


def run(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m data_preparation hybrid",
        description="Build a COLMAP scene with SfM cameras/poses and aligned SLAM/LiDAR points.",
    )
    parser.add_argument("--scene-dir", required=True, type=Path, help="SLAM/reference scene root.")
    parser.add_argument("--sfm-scene-dir", required=True, type=Path, help="SfM COLMAP scene root.")
    parser.add_argument("--scene", default=None, help="Scene variant name used for the default hybrid output.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Explicit hybrid scene output directory.")
    parser.add_argument("--points-ply", type=Path, default=None, help="Colored SLAM point cloud PLY.")
    parser.add_argument("--poses-csv", type=Path, default=None, help="SLAM camera poses CSV.")
    parser.add_argument("--repo-root", type=Path, default=None, help="Optional 00_Baselines repo root.")
    parser.add_argument("--thesis-root", type=Path, default=None, help="Optional explicit Thesis root.")
    parser.add_argument("passthrough", nargs=argparse.REMAINDER, help="Advanced args passed after -- to hybrid backend.")
    args = parser.parse_args(argv)
    scene_name = args.scene or args.scene_dir.expanduser().resolve().name
    output_dir = resolve_route_output(
        scene=scene_name,
        output_dir=args.output_dir,
        repo_root=args.repo_root,
        thesis_root=args.thesis_root,
        category="hybrid",
    )
    backend_argv = [
        "--scene-dir",
        str(args.scene_dir.expanduser().resolve()),
        "--sfm-scene-dir",
        str(args.sfm_scene_dir.expanduser().resolve()),
        "--output-dir",
        str(output_dir),
        "--scene-name",
        scene_name,
    ]
    if args.points_ply is not None:
        backend_argv.extend(["--points-ply", str(args.points_ply.expanduser().resolve())])
    if args.poses_csv is not None:
        backend_argv.extend(["--poses-csv", str(args.poses_csv.expanduser().resolve())])
    invoke_module(
        "data_preparation.slam_to_colmap.filtered_scene_main",
        "hybrid",
        append_passthrough(backend_argv, args.passthrough),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1:]))

