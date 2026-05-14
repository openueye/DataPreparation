from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from data_preparation.shared.route_helpers import append_passthrough, invoke_module, resolve_route_output


def run(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m data_preparation sfm",
        description="Run COLMAP/SfM using an existing rectified image directory.",
    )
    parser.add_argument("--image-dir", required=True, type=Path, help="Prepared rectified images directory.")
    parser.add_argument("--scene", default=None, help="Scene name used for the default 04_ProcessedData/sfm output.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Explicit COLMAP scene output directory.")
    parser.add_argument("--repo-root", type=Path, default=None, help="Optional 00_Baselines repo root.")
    parser.add_argument("--thesis-root", type=Path, default=None, help="Optional explicit Thesis root.")
    parser.add_argument("passthrough", nargs=argparse.REMAINDER, help="Advanced args passed after -- to COLMAP backend.")
    args = parser.parse_args(argv)
    output_dir = resolve_route_output(
        scene=args.scene,
        output_dir=args.output_dir,
        repo_root=args.repo_root,
        thesis_root=args.thesis_root,
        category="sfm",
    )
    backend_argv = ["--image-dir", str(args.image_dir.expanduser().resolve()), "--output-dir", str(output_dir)]
    invoke_module(
        "data_preparation.video2colmap.preprocess_video_to_colmap",
        "sfm",
        append_passthrough(backend_argv, args.passthrough),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1:]))

