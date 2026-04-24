from __future__ import annotations

import argparse
from pathlib import Path

try:
    from data_preparation.slam_to_colmap.converter import convert_scene
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from data_preparation.slam_to_colmap.converter import convert_scene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a processed SLAM/LiDAR scene into COLMAP-compatible text files.")
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--points-ply", type=Path, default=None)
    parser.add_argument("--max-points", type=int, default=300_000)
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument(
        "--allow-pinhole-approximation",
        action="store_true",
        help=(
            "Allow exporting non-pinhole or still-distorted images with K_like PINHOLE intrinsics. "
            "Use only for smoke tests; high-quality 3DGS training needs rectified/undistorted images."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = convert_scene(
        args.scene_dir,
        args.output_dir,
        points_ply=args.points_ply,
        max_points=args.max_points,
        copy_images=args.copy_images,
        allow_pinhole_approximation=args.allow_pinhole_approximation,
    )
    print(f"[INFO] COLMAP-compatible scene written: {report['output_scene']}")
    print(f"[INFO] images={report['num_images']} points={report['written_points']} source={report['points_source']}")
    print(f"[INFO] report={Path(report['output_scene']) / 'slam_to_colmap_report.json'}")


if __name__ == "__main__":
    main()
