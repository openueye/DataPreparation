import argparse
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from data_preparation.shared.camera_models import project_world_to_pinhole
    from data_preparation.shared.io import find_image_path, load_csv_rows_by_key, load_json, safe_float, write_json
    from data_preparation.shared.pointcloud import read_ply_xyz, write_colorized_ply
    from data_preparation.shared.poses import world_from_camera_from_row
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from data_preparation.shared.camera_models import project_world_to_pinhole
    from data_preparation.shared.io import find_image_path, load_csv_rows_by_key, load_json, safe_float, write_json
    from data_preparation.shared.pointcloud import read_ply_xyz, write_colorized_ply
    from data_preparation.shared.poses import world_from_camera_from_row


DEFAULT_PREVIEW_FRAMES = ["000000", "000500", "001000", "001500", "002000", "002500", "003000"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Colorize an exported global LiDAR map by projecting points into synchronized RGB images."
    )
    parser.add_argument("--scene-dir", type=str, required=True, help="Path to an exported SLAM/LiDAR 3DGS scene.")
    parser.add_argument("--input-ply", type=str, default="", help="Input PLY. Defaults to <scene-dir>/lidar/global_map.ply.")
    parser.add_argument(
        "--output-ply",
        type=str,
        default="",
        help="Output PLY. Defaults to <scene-dir>/lidar/global_map_colorized.ply, or *_preview.ply when sampling.",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default="",
        help="JSON report path. Defaults to <scene-dir>/metadata/lidar_colorization_report.json.",
    )
    parser.add_argument(
        "--frames",
        nargs="+",
        default=None,
        help="Explicit frame ids to use. If omitted, frames are selected by --frame-stride and --max-frames.",
    )
    parser.add_argument("--frame-stride", type=int, default=32, help="Use every Nth associated frame when --frames is omitted.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=64,
        help="Maximum selected frames when --frames is omitted. Use 0 for no limit.",
    )
    parser.add_argument(
        "--sample-points",
        type=int,
        default=0,
        help="Randomly colorize only this many global map points for preview. Use 0 for all points.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed used with --sample-points.")
    parser.add_argument("--batch-size", type=int, default=200000, help="Point batch size for projection.")
    parser.add_argument(
        "--fusion",
        choices=("nearest", "average"),
        default="nearest",
        help="Multi-view color fusion strategy.",
    )
    parser.add_argument(
        "--fallback-gray",
        type=float,
        default=0.5,
        help="Fallback grayscale value in [0,1] for uncolored points.",
    )
    return parser.parse_args()


def load_associations(path: Path):
    return load_csv_rows_by_key(path, "frame_id")


def load_pose_rows(path: Path):
    return load_csv_rows_by_key(path, "frame_id")


def pose_matrix_from_row(row):
    return world_from_camera_from_row(row)


def select_frame_ids(associations, poses, explicit_frames, stride, max_frames):
    if explicit_frames:
        return [frame_id for frame_id in explicit_frames if frame_id in poses]

    stride = max(1, int(stride))
    frame_ids = sorted(frame_id for frame_id in associations.keys() if frame_id in poses)
    frame_ids = frame_ids[::stride]
    if max_frames and max_frames > 0:
        frame_ids = frame_ids[:max_frames]
    return frame_ids


def project_world_batch(points_world, world_from_camera, K):
    return project_world_to_pinhole(points_world, world_from_camera, K)


def sample_image_rgb(image_array, uv):
    px = np.rint(uv[:, 0]).astype(np.int64)
    py = np.rint(uv[:, 1]).astype(np.int64)
    px = np.clip(px, 0, image_array.shape[1] - 1)
    py = np.clip(py, 0, image_array.shape[0] - 1)
    return image_array[py, px, :3].astype(np.uint16)


def main(args):
    scene_dir = Path(args.scene_dir).expanduser().resolve()
    images_dir = scene_dir / "images"
    metadata_dir = scene_dir / "metadata"
    input_ply = Path(args.input_ply).expanduser().resolve() if args.input_ply else scene_dir / "lidar" / "global_map.ply"

    if args.output_ply:
        output_ply = Path(args.output_ply).expanduser().resolve()
    elif args.sample_points and args.sample_points > 0:
        output_ply = scene_dir / "lidar" / "global_map_colorized_preview.ply"
    else:
        output_ply = scene_dir / "lidar" / "global_map_colorized.ply"

    report_path = (
        Path(args.report_path).expanduser().resolve()
        if args.report_path
        else metadata_dir / "lidar_colorization_report.json"
    )

    camera_data = load_json(scene_dir / "intrinsics" / "camera.json")
    K = np.asarray(camera_data["K_like"], dtype=np.float64)
    width = int(camera_data["width"])
    height = int(camera_data["height"])
    associations = load_associations(metadata_dir / "associations.csv")
    poses = load_pose_rows(scene_dir / "poses" / "poses.csv")
    frame_ids = select_frame_ids(associations, poses, args.frames, args.frame_stride, args.max_frames)
    if not frame_ids:
        raise ValueError("No usable frames selected for colorization.")

    xyz_all = read_ply_xyz(input_ply)
    source_point_count = int(xyz_all.shape[0])
    finite_mask = np.isfinite(xyz_all).all(axis=1)
    xyz_all = xyz_all[finite_mask]

    sample_indices = None
    if args.sample_points and args.sample_points > 0 and xyz_all.shape[0] > args.sample_points:
        rng = np.random.default_rng(args.seed)
        sample_indices = np.sort(rng.choice(xyz_all.shape[0], size=args.sample_points, replace=False))
        xyz = xyz_all[sample_indices]
    else:
        xyz = xyz_all

    point_count = int(xyz.shape[0])
    fallback = int(np.clip(args.fallback_gray, 0.0, 1.0) * 255.0)
    rgb = np.full((point_count, 3), fallback, dtype=np.uint8)
    observation_count = np.zeros(point_count, dtype=np.uint16)

    if args.fusion == "nearest":
        best_depth = np.full(point_count, np.inf, dtype=np.float32)
    else:
        color_sum = np.zeros((point_count, 3), dtype=np.uint32)

    batch_size = max(1, int(args.batch_size))
    frame_reports = []
    for frame_index, frame_id in enumerate(frame_ids, start=1):
        image_path = find_image_path(images_dir, frame_id)
        image = np.asarray(Image.open(image_path).convert("RGB"))
        world_from_camera = pose_matrix_from_row(poses[frame_id])

        inside_points = 0
        in_front_points = 0
        sampled_points = 0
        depth_values = []
        for start in range(0, point_count, batch_size):
            end = min(start + batch_size, point_count)
            uv, depth, in_front = project_world_batch(xyz[start:end], world_from_camera, K)
            inside = (
                in_front
                & (uv[:, 0] >= 0)
                & (uv[:, 0] < width)
                & (uv[:, 1] >= 0)
                & (uv[:, 1] < height)
            )
            in_front_points += int(np.count_nonzero(in_front))
            inside_count = int(np.count_nonzero(inside))
            inside_points += inside_count
            if inside_count == 0:
                continue

            rgb_obs = sample_image_rgb(image, uv[inside])
            local_indices = np.flatnonzero(inside)
            global_indices = start + local_indices
            observation_count[global_indices] = np.minimum(
                observation_count[global_indices].astype(np.uint32) + 1,
                np.iinfo(np.uint16).max,
            ).astype(np.uint16)
            sampled_points += inside_count
            depth_inside = depth[inside]
            depth_values.append(depth_inside.astype(np.float32))

            if args.fusion == "nearest":
                improve = depth_inside < best_depth[global_indices]
                if np.any(improve):
                    update_indices = global_indices[improve]
                    rgb[update_indices] = rgb_obs[improve].astype(np.uint8)
                    best_depth[update_indices] = depth_inside[improve].astype(np.float32)
            else:
                color_sum[global_indices] += rgb_obs.astype(np.uint32)

        if args.fusion == "average":
            colored_now = observation_count > 0
            rgb[colored_now] = np.clip(
                color_sum[colored_now] / observation_count[colored_now, None],
                0,
                255,
            ).astype(np.uint8)

        if depth_values:
            depths = np.concatenate(depth_values)
            median_depth = safe_float(np.median(depths))
        else:
            median_depth = None

        frame_report = {
            "frame_id": frame_id,
            "image_path": str(image_path),
            "in_front_points": int(in_front_points),
            "inside_image_points": int(inside_points),
            "coverage_ratio": safe_float(inside_points / max(in_front_points, 1)),
            "median_projected_depth": median_depth,
        }
        frame_reports.append(frame_report)
        colored_points = int(np.count_nonzero(observation_count))
        print(
            f"[INFO] frame {frame_index}/{len(frame_ids)} {frame_id}: "
            f"inside={inside_points:,} cumulative_colored={colored_points:,}/{point_count:,}"
        )

    colored_mask = observation_count > 0
    colored_points = int(np.count_nonzero(colored_mask))
    write_colorized_ply(output_ply, xyz, rgb)

    observations = observation_count[colored_mask].astype(np.float32)
    report = {
        "scene_dir": str(scene_dir),
        "input_ply": str(input_ply),
        "output_ply": str(output_ply),
        "source_point_count": source_point_count,
        "finite_source_points": int(xyz_all.shape[0]),
        "processed_points": point_count,
        "sample_points": int(args.sample_points),
        "sampled": bool(sample_indices is not None),
        "colored_points": colored_points,
        "uncolored_points": int(point_count - colored_points),
        "color_coverage_ratio": safe_float(colored_points / max(point_count, 1)),
        "mean_observations_per_colored_point": safe_float(observations.mean()) if observations.size else 0.0,
        "median_observations_per_colored_point": safe_float(np.median(observations)) if observations.size else 0.0,
        "num_images_used": len(frame_ids),
        "frame_ids": frame_ids,
        "fusion": args.fusion,
        "projection_model": "K_like_pinhole_from_FishPoly_export",
        "frame_stride": int(args.frame_stride),
        "max_frames": int(args.max_frames),
        "batch_size": batch_size,
        "camera": {
            "camera_name": camera_data.get("camera_name"),
            "camera_model": camera_data.get("camera_model"),
            "width": width,
            "height": height,
            "K_like": K.tolist(),
        },
        "frames": frame_reports,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, report)

    print(f"[INFO] Colorized PLY written: {output_ply}")
    print(f"[INFO] Colorization report written: {report_path}")
    print(f"[INFO] Colored points: {colored_points:,}/{point_count:,} ({report['color_coverage_ratio']:.3f})")


if __name__ == "__main__":
    main(parse_args())
