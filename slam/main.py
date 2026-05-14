from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from data_preparation.shared.route_helpers import append_passthrough, invoke_module, resolve_route_output


REFERENCE_ORGANIZER = "Thesis/06_Referrance/DataprePython/organize_odin_scene_headerstamp.py"
REFERENCE_COLMAP_EXPORT = "Thesis/06_Referrance/DataprePython/export_pure_headerstamp_to_colmap.py"


def run(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m data_preparation slam",
        description="Convert SLAM poses and a colored SLAM point cloud into a COLMAP text model.",
    )
    parser.add_argument("--input-dir", "--scene-dir", dest="input_dir", required=True, type=Path)
    parser.add_argument("--scene", default=None, help="Scene name used for the default 04_ProcessedData/slam_compat output.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Explicit COLMAP scene output directory.")
    parser.add_argument("--points-ply", type=Path, default=None, help="Override lidar/global_map_slam_odom.ply.")
    parser.add_argument("--max-points", type=int, default=3_000_000, help="Maximum point count after voxel downsampling. Use 0 for all.")
    parser.add_argument("--repo-root", type=Path, default=None, help="Optional 00_Baselines repo root.")
    parser.add_argument("--thesis-root", type=Path, default=None, help="Optional explicit Thesis root.")
    parser.add_argument("passthrough", nargs=argparse.REMAINDER, help="Advanced args passed after -- to SLAM backend.")
    args = parser.parse_args(argv)
    input_dir = args.input_dir.expanduser().resolve()
    scene_name = args.scene or input_dir.name
    output_dir = resolve_route_output(
        scene=scene_name,
        output_dir=args.output_dir,
        repo_root=args.repo_root,
        thesis_root=args.thesis_root,
        category="slam",
    )
    backend_argv = ["--input-dir", str(input_dir), "--output-dir", str(output_dir)]
    if args.points_ply is not None:
        backend_argv.extend(["--points-ply", str(args.points_ply.expanduser().resolve())])
    backend_argv.extend(["--max-points", str(args.max_points)])
    invoke_module(
        "data_preparation.slam.export_colmap",
        "slam",
        append_passthrough(backend_argv, args.passthrough),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1:]))
