from __future__ import annotations

import argparse
from pathlib import Path

try:
    from data_preparation.slam.converter import convert_scene
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from data_preparation.slam.converter import convert_scene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a pure-headerstamp Odin1 export into a COLMAP text-model package "
            "using rectified images, camera poses, and the lidar SLAM global map."
        )
    )
    parser.add_argument("--input-dir", "--scene-dir", dest="input_dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--images-subdir", default="images")
    parser.add_argument("--points-ply", type=Path, default=None)
    parser.add_argument("--max-points", type=int, default=3_000_000, help="Maximum point count after voxel downsampling. Use 0 for all.")
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--keep-static-poses",
        action="store_true",
        help="Keep consecutive duplicate/near-duplicate pose frames. Default drops them before COLMAP export.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = convert_scene(
        args.input_dir,
        args.output_dir,
        points_ply=args.points_ply,
        max_points=args.max_points,
        copy_images=args.copy_images,
        overwrite=args.overwrite,
        images_subdir=args.images_subdir,
        drop_static_poses=not args.keep_static_poses,
        min_pose_translation_m=args.min_pose_translation_m,
        min_pose_rotation_deg=args.min_pose_rotation_deg,
    )
    source_scene = report.get("source_scene_dir") or report.get("source_scene") or str(args.input_dir)
    print(f"[INFO] scene_name={report.get('scene_name', Path(source_scene).name)}")
    print(f"[INFO] input_dir={args.input_dir.resolve()}")
    print(f"[INFO] output_dir={args.output_dir.resolve()}")
    print(f"[INFO] images_mode={report.get('images', {}).get('mode', 'directory')}")
    if "slam_matched_frame_count" in report:
        print(f"[INFO] slam_matched_frame_count={report['slam_matched_frame_count']}")
    static_filter = report.get("poses", {}).get("static_pose_filter") or report.get("static_pose_filter") or {}
    if static_filter:
        print(f"[INFO] static_pose_filter_dropped={static_filter.get('dropped_count', 0)}")
    print(f"[INFO] image_count={report.get('poses', {}).get('frame_count', report.get('num_images'))}")
    print(f"[INFO] point_count={report.get('points3D', {}).get('written_points', report.get('written_points'))}")
    if report.get("points3D", {}).get("points3D_ply"):
        print(f"[INFO] points_ply={report['points3D']['points3D_ply']}")


if __name__ == "__main__":
    main()
