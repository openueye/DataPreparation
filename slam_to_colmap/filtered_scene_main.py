from __future__ import annotations

import argparse
import csv
import os
import shutil
import struct
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from data_preparation.shared.io import write_json
    from data_preparation.shared.poses import quaternion_xyzw_to_matrix
    from data_preparation.slam_to_colmap.converter import link_or_copy_images
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from data_preparation.shared.io import write_json
    from data_preparation.shared.poses import quaternion_xyzw_to_matrix
    from data_preparation.slam_to_colmap.converter import link_or_copy_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a hybrid COLMAP scene: use RGB/SfM cameras and poses for pixel alignment, "
            "then transform filtered SLAM/LiDAR points into the SfM coordinate frame for 3DGS initialization."
        )
    )
    parser.add_argument("--scene-dir", required=True, type=Path, help="Filtered scene root, e.g. Downtown1_filtered.")
    parser.add_argument("--sfm-scene-dir", required=True, type=Path, help="Reference COLMAP SfM scene root.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-points", type=int, default=3_000_000, help="Maximum LiDAR points to export. Use 0 for all.")
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of symlinking to the SfM images.")
    parser.add_argument("--poses-csv", type=Path, default=None)
    parser.add_argument("--slam-frames-dir", type=Path, default=None)
    return parser.parse_args()


def read_next_bytes(handle, num_bytes: int, fmt: str):
    data = handle.read(num_bytes)
    if len(data) != num_bytes:
        raise EOFError("Unexpected end of COLMAP binary file.")
    return struct.unpack("<" + fmt, data)


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    return quaternion_xyzw_to_matrix(np.asarray([qvec[1], qvec[2], qvec[3], qvec[0]], dtype=np.float64))


def read_colmap_image_centers(images_bin: Path) -> Dict[str, np.ndarray]:
    centers = {}
    with images_bin.open("rb") as handle:
        num_images = read_next_bytes(handle, 8, "Q")[0]
        for _ in range(num_images):
            props = read_next_bytes(handle, 64, "idddddddi")
            qvec = np.asarray(props[1:5], dtype=np.float64)
            tvec = np.asarray(props[5:8], dtype=np.float64)
            image_name = ""
            char = read_next_bytes(handle, 1, "c")[0]
            while char != b"\x00":
                image_name += char.decode("utf-8")
                char = read_next_bytes(handle, 1, "c")[0]

            num_points2d = read_next_bytes(handle, 8, "Q")[0]
            handle.seek(24 * num_points2d, os.SEEK_CUR)

            rotation_cw = qvec_to_rotmat(qvec)
            centers[Path(image_name).stem] = -rotation_cw.T @ tvec
    return centers


def load_slam_camera_centers(poses_csv: Path) -> Dict[str, np.ndarray]:
    centers = {}
    with poses_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if "T_odom_from_camera_00" in row:
                matrix = np.asarray(
                    [float(row[f"T_odom_from_camera_{r}{c}"]) for r in range(4) for c in range(4)],
                    dtype=np.float64,
                ).reshape(4, 4)
                centers[row["frame_id"]] = matrix[:3, 3]
            else:
                centers[row["frame_id"]] = np.asarray(
                    [float(row["tx"]), float(row["ty"]), float(row["tz"])],
                    dtype=np.float64,
                )
    return centers


def similarity_sfm_to_slam(sfm_centers: Dict[str, np.ndarray], slam_centers: Dict[str, np.ndarray]):
    common_frames = sorted(set(sfm_centers) & set(slam_centers))
    if len(common_frames) < 3:
        raise ValueError(f"Need at least 3 common frames for similarity alignment, got {len(common_frames)}.")

    source = np.stack([sfm_centers[frame_id] for frame_id in common_frames], axis=0)
    target = np.stack([slam_centers[frame_id] for frame_id in common_frames], axis=0)
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean

    covariance = (target_centered.T @ source_centered) / len(common_frames)
    u_mat, singular_values, vt_mat = np.linalg.svd(covariance)
    sign = np.eye(3)
    if np.linalg.det(u_mat @ vt_mat) < 0.0:
        sign[-1, -1] = -1.0
    rotation = u_mat @ sign @ vt_mat
    source_variance = np.mean(np.sum(source_centered * source_centered, axis=1))
    scale = float(np.trace(np.diag(singular_values) @ sign) / source_variance)
    translation = target_mean - scale * rotation @ source_mean

    aligned = (scale * (rotation @ source.T)).T + translation
    residuals = np.linalg.norm(aligned - target, axis=1)
    stats = {
        "common_frames": len(common_frames),
        "frame_range": [common_frames[0], common_frames[-1]],
        "sfm_to_slam_scale": scale,
        "center_residual_median": float(np.median(residuals)),
        "center_residual_mean": float(np.mean(residuals)),
        "center_residual_p90": float(np.percentile(residuals, 90)),
        "center_residual_max": float(np.max(residuals)),
    }
    return scale, rotation, translation, stats


def npz_point_count(path: Path) -> int:
    with np.load(path) as data:
        if "xyz" not in data:
            raise KeyError(f"{path} does not contain an 'xyz' array.")
        return int(data["xyz"].shape[0])


def select_frame_ranges(counts: List[int], max_points: int) -> Tuple[int, np.ndarray, List[Tuple[int, int]]]:
    source_points = int(sum(counts))
    written_points = source_points if max_points <= 0 else min(int(max_points), source_points)
    sample_indices = (
        np.arange(source_points, dtype=np.int64)
        if written_points == source_points
        else np.linspace(0, source_points - 1, num=written_points, dtype=np.int64)
    )
    starts = np.cumsum(np.asarray([0] + counts[:-1], dtype=np.int64))
    ends = starts + np.asarray(counts, dtype=np.int64)
    ranges = [
        (
            int(np.searchsorted(sample_indices, start, side="left")),
            int(np.searchsorted(sample_indices, end, side="left")),
        )
        for start, end in zip(starts, ends)
    ]
    return source_points, sample_indices, ranges


def write_lidar_points_in_sfm_frame(
    path: Path,
    npz_files: List[Path],
    counts: List[int],
    max_points: int,
    sfm_to_slam_scale: float,
    sfm_to_slam_rotation: np.ndarray,
    sfm_to_slam_translation: np.ndarray,
) -> Tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    source_points, sample_indices, ranges = select_frame_ranges(counts, max_points)
    frame_starts = np.cumsum(np.asarray([0] + counts[:-1], dtype=np.int64))
    written = 0

    with path.open("w", encoding="utf-8") as handle:
        handle.write("# 3D point list with one line of data per point:\n")
        handle.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        handle.write(f"# Number of points: {sample_indices.shape[0]}, mean track length: 0\n")

        for npz_path, frame_start, (range_start, range_end) in zip(npz_files, frame_starts, ranges):
            if range_start == range_end:
                continue
            local_indices = sample_indices[range_start:range_end] - frame_start
            with np.load(npz_path) as data:
                xyz_slam = np.asarray(data["xyz"], dtype=np.float64)[local_indices]
                rgb = (
                    np.asarray(data["rgb"])[local_indices]
                    if "rgb" in data
                    else np.full((xyz_slam.shape[0], 3), 128, dtype=np.uint8)
                )

            finite = np.isfinite(xyz_slam).all(axis=1)
            xyz_slam = xyz_slam[finite]
            rgb = np.clip(rgb[finite], 0, 255).astype(np.uint8)

            xyz_sfm = (sfm_to_slam_rotation.T @ ((xyz_slam - sfm_to_slam_translation) / sfm_to_slam_scale).T).T
            for point, color in zip(xyz_sfm, rgb):
                written += 1
                handle.write(
                    f"{written} {point[0]:.17g} {point[1]:.17g} {point[2]:.17g} "
                    f"{int(color[0])} {int(color[1])} {int(color[2])} 0\n"
                )

    return source_points, written


def copy_sfm_pose_model(sfm_sparse0: Path, output_sparse0: Path) -> None:
    output_sparse0.mkdir(parents=True, exist_ok=True)
    for filename in ("cameras.bin", "images.bin"):
        source = sfm_sparse0 / filename
        if not source.exists():
            raise FileNotFoundError(f"Missing required SfM model file: {source}")
        shutil.copy2(source, output_sparse0 / filename)

    # Force the 3DGS loader to use the hybrid LiDAR points3D.txt instead of stale SfM points.
    for stale in ("points3D.bin", "points3D.ply"):
        stale_path = output_sparse0 / stale
        if stale_path.exists() or stale_path.is_symlink():
            stale_path.unlink()


def convert_filtered_scene(args: argparse.Namespace) -> Dict[str, object]:
    scene_dir = args.scene_dir.expanduser().resolve()
    sfm_scene_dir = args.sfm_scene_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    poses_csv = (args.poses_csv or scene_dir / "poses" / "camera_poses.csv").expanduser().resolve()
    slam_frames_dir = (args.slam_frames_dir or scene_dir / "lidar" / "slam_frames").expanduser().resolve()
    sfm_sparse0 = sfm_scene_dir / "sparse" / "0"
    output_sparse0 = output_dir / "sparse" / "0"

    for required in (scene_dir, sfm_scene_dir / "images", sfm_sparse0 / "images.bin", sfm_sparse0 / "cameras.bin", poses_csv, slam_frames_dir):
        if not required.exists():
            raise FileNotFoundError(f"Missing required input: {required}")

    sfm_centers = read_colmap_image_centers(sfm_sparse0 / "images.bin")
    slam_centers = load_slam_camera_centers(poses_csv)
    scale, rotation, translation, alignment_stats = similarity_sfm_to_slam(sfm_centers, slam_centers)

    output_dir.mkdir(parents=True, exist_ok=True)
    copy_sfm_pose_model(sfm_sparse0, output_sparse0)
    link_or_copy_images(sfm_scene_dir / "images", output_dir / "images", copy_images=args.copy_images)

    npz_files = sorted(slam_frames_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz SLAM frames found under {slam_frames_dir}")
    counts = [npz_point_count(path) for path in npz_files]
    source_points, written_points = write_lidar_points_in_sfm_frame(
        output_sparse0 / "points3D.txt",
        npz_files,
        counts,
        max_points=args.max_points,
        sfm_to_slam_scale=scale,
        sfm_to_slam_rotation=rotation,
        sfm_to_slam_translation=translation,
    )

    metadata = {
        "source_scene": str(scene_dir),
        "sfm_scene": str(sfm_scene_dir),
        "output_scene": str(output_dir),
        "format": "Hybrid COLMAP model: SfM RGB cameras/images + transformed SLAM/LiDAR points3D.txt",
        "pose_source": str(sfm_sparse0 / "images.bin"),
        "camera_source": str(sfm_sparse0 / "cameras.bin"),
        "image_source": str(sfm_scene_dir / "images"),
        "points_source": str(slam_frames_dir / "*.npz"),
        "source_lidar_frames": len(npz_files),
        "source_points": int(source_points),
        "written_points": int(written_points),
        "num_sfm_images": len(sfm_centers),
        "num_slam_pose_frames": len(slam_centers),
        "copy_images": bool(args.copy_images),
        "alignment": {
            **alignment_stats,
            "direction": "LiDAR points in SLAM coordinates are transformed into SfM coordinates using inverse(SfM->SLAM similarity).",
            "sfm_to_slam_rotation": rotation.tolist(),
            "sfm_to_slam_translation": translation.tolist(),
        },
        "tracks": {
            "written": False,
            "note": "Current 3DGS loader uses cameras/images poses and points3D xyz/rgb only; synthetic tracks are intentionally omitted.",
        },
        "baseline_type": "RGB/SfM pose-aligned training with SLAM/LiDAR geometric initialization.",
    }
    write_json(output_dir / "hybrid_sfm_lidar_report.json", metadata)
    return metadata


def main() -> None:
    args = parse_args()
    report = convert_filtered_scene(args)
    print(f"[INFO] Hybrid COLMAP scene written: {report['output_scene']}")
    print(f"[INFO] SfM images={report['num_sfm_images']} LiDAR points={report['written_points']}")
    print(
        "[INFO] alignment center residual "
        f"median={report['alignment']['center_residual_median']:.4f}m "
        f"p90={report['alignment']['center_residual_p90']:.4f}m"
    )
    print(f"[INFO] report={Path(report['output_scene']) / 'hybrid_sfm_lidar_report.json'}")


if __name__ == "__main__":
    main()
