from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image

try:
    from data_preparation.shared.io import write_json
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from data_preparation.shared.io import write_json

BASELINES_ROOT = Path(__file__).resolve().parents[2]
BASELINE_02_ROOT = BASELINES_ROOT / "02baseline"
if str(BASELINE_02_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_02_ROOT))

from common.colmap_io import (  # noqa: E402
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synchronized raw LiDAR metric .npy depth priors for an existing COLMAP-compatible scene."
    )
    parser.add_argument("--scene-dir", required=True, type=Path, help="COLMAP-compatible scene root.")
    parser.add_argument(
        "--prepared-scene-dir",
        type=Path,
        default=None,
        help="Prepared pure-headerstamp scene containing associations, raw LiDAR frames, and calib/tf_chain.json.",
    )
    parser.add_argument("--output-depths-dir", type=Path, default=None, help="Output directory. Defaults to <scene-dir>/depths.")
    parser.add_argument("--chunk-size", type=int, default=1_000_000, help="Point projection chunk size.")
    parser.add_argument("--min-depth", type=float, default=0.0, help="Minimum projected camera z-depth to keep, in meters.")
    parser.add_argument("--max-depth", type=float, default=0.0, help="Maximum projected camera z-depth to keep, in meters. Use <=0 to disable.")
    parser.add_argument(
        "--max-raw-cloud-dt-ms",
        type=float,
        default=20.0,
        help="Reject raw-frame projection if image/raw-cloud timestamp delta exceeds this threshold.",
    )
    parser.add_argument(
        "--method",
        choices=["raw-frame", "local-fused"],
        default="raw-frame",
        help="Depth prior generation method. Default preserves the historical single raw-frame route.",
    )
    parser.add_argument(
        "--fusion-window",
        type=int,
        default=5,
        help="Odd local source-frame window for --method local-fused. Default: 5.",
    )
    parser.add_argument(
        "--fusion-mode",
        choices=["centered"],
        default="centered",
        help="Neighbor selection policy for --method local-fused.",
    )
    parser.add_argument("--output-masks-dir", type=Path, default=None, help="Mask output directory for local-fused mode.")
    parser.add_argument(
        "--output-confidence-dir",
        type=Path,
        default=None,
        help="Confidence output directory for local-fused mode.",
    )
    parser.add_argument(
        "--write-confidence",
        action="store_true",
        help="Write per-pixel source-count confidence maps for local-fused mode.",
    )
    parser.add_argument(
        "--overlay-count",
        type=int,
        default=0,
        help="Write this many lightweight local-fused depth overlays under the scene directory. Default: 0.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .npy depth files.")
    return parser.parse_args(argv)


def load_colmap_model(scene_dir: Path):
    sparse0 = scene_dir / "sparse" / "0"
    if (sparse0 / "images.bin").exists() and (sparse0 / "cameras.bin").exists():
        return read_extrinsics_binary(str(sparse0 / "images.bin")), read_intrinsics_binary(str(sparse0 / "cameras.bin"))
    if (sparse0 / "images.txt").exists() and (sparse0 / "cameras.txt").exists():
        return read_extrinsics_text(str(sparse0 / "images.txt")), read_intrinsics_text(str(sparse0 / "cameras.txt"))
    raise FileNotFoundError(f"Missing COLMAP cameras/images under {sparse0}")


def intrinsics_to_params(intr) -> Tuple[int, int, float, float, float, float]:
    width = int(intr.width)
    height = int(intr.height)
    if intr.model == "SIMPLE_PINHOLE":
        fx = fy = float(intr.params[0])
        cx = float(intr.params[1])
        cy = float(intr.params[2])
    elif intr.model == "PINHOLE":
        fx = float(intr.params[0])
        fy = float(intr.params[1])
        cx = float(intr.params[2])
        cy = float(intr.params[3])
    else:
        raise ValueError(f"Unsupported COLMAP camera model for depth projection: {intr.model}")
    return width, height, fx, fy, cx, cy


def _empty_zbuffer_for_intrinsics(intr):
    width, height, fx, fy, cx, cy = intrinsics_to_params(intr)
    return np.full((height, width), np.inf, dtype=np.float32), width, height, fx, fy, cx, cy


def _empty_zbuffer_for_camera_params(width: int, height: int):
    return np.full((int(height), int(width)), np.inf, dtype=np.float32)


def project_depth_for_camera_points(
    points_camera_input: np.ndarray,
    intr,
    *,
    chunk_size: int,
    min_depth: float = 0.0,
    max_depth: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, object]]:
    zbuffer, width, height, fx, fy, cx, cy = _empty_zbuffer_for_intrinsics(intr)
    input_points = int(points_camera_input.shape[0])
    front_points = 0
    depth_range_points = 0
    projected_points = 0
    min_depth = max(float(min_depth), 0.0)
    max_depth = float(max_depth)

    for start in range(0, input_points, max(1, int(chunk_size))):
        points_camera = points_camera_input[start : start + max(1, int(chunk_size))].astype(np.float64, copy=False)
        z = points_camera[:, 2]
        front = z > 1e-6
        if not np.any(front):
            continue
        points_camera = points_camera[front]
        z = z[front]
        front_points += int(z.shape[0])
        in_depth_range = z > min_depth
        if max_depth > 0.0:
            in_depth_range = in_depth_range & (z <= max_depth)
        if not np.any(in_depth_range):
            continue
        points_camera = points_camera[in_depth_range]
        z = z[in_depth_range]
        depth_range_points += int(z.shape[0])
        u = np.rint((fx * points_camera[:, 0]) / z + cx).astype(np.int64)
        v = np.rint((fy * points_camera[:, 1]) / z + cy).astype(np.int64)
        inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        if not np.any(inside):
            continue
        projected_points += int(np.count_nonzero(inside))
        np.minimum.at(zbuffer, (v[inside], u[inside]), z[inside].astype(np.float32))

    valid = np.isfinite(zbuffer)
    depth = np.zeros_like(zbuffer, dtype=np.float32)
    depth[valid] = zbuffer[valid]
    valid_count = int(np.count_nonzero(valid))
    pixel_count = int(width * height)
    valid_depths = depth[valid]
    return depth, {
        "width": width,
        "height": height,
        "input_points": input_points,
        "front_points": front_points,
        "depth_range_points": depth_range_points,
        "projected_points": projected_points,
        "valid_pixels": valid_count,
        "pixel_count": pixel_count,
        "valid_ratio": float(valid_count / max(pixel_count, 1)),
        "valid_depth_min": float(valid_depths.min()) if valid_count else None,
        "valid_depth_max": float(valid_depths.max()) if valid_count else None,
        "valid_depth_mean": float(valid_depths.mean()) if valid_count else None,
        "min_depth_filter": float(min_depth),
        "max_depth_filter": float(max_depth),
    }


def find_duplicate_image_stems(extrinsics) -> Dict[str, list]:
    stems: Dict[str, list] = {}
    for image_id in sorted(extrinsics):
        name = extrinsics[image_id].name
        stems.setdefault(Path(name).stem, []).append(name)
    return {stem: names for stem, names in stems.items() if len(names) > 1}


def validate_image_dimensions(scene_dir: Path, extrinsics, intrinsics) -> None:
    images_dir = scene_dir / "images"
    for image_id in sorted(extrinsics):
        extr = extrinsics[image_id]
        intr = intrinsics[extr.camera_id]
        image_path = images_dir / extr.name
        if not image_path.is_file():
            raise FileNotFoundError(f"COLMAP image file is missing for depth projection: {image_path}")
        with Image.open(image_path) as image:
            image_width, image_height = image.size
        intr_width = int(intr.width)
        intr_height = int(intr.height)
        if (image_width, image_height) != (intr_width, intr_height):
            raise ValueError(
                "COLMAP camera dimensions do not match image file for "
                f"{extr.name}: camera={intr_width}x{intr_height}, image={image_width}x{image_height}"
            )


def read_csv_dicts(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json_dict(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def infer_prepared_scene_dir(scene_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        prepared = explicit.expanduser().resolve()
        if not prepared.exists():
            raise FileNotFoundError(f"Missing prepared scene dir: {prepared}")
        return prepared
    manifest_path = scene_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(
            "--prepared-scene-dir is required for --method raw-frame when scene/manifest.json is missing."
        )
    manifest = load_json_dict(manifest_path)
    source_scene = manifest.get("source_scene_dir")
    if not source_scene:
        raise ValueError(f"manifest.json does not contain source_scene_dir: {manifest_path}")
    prepared = Path(str(source_scene)).expanduser().resolve()
    if not prepared.exists():
        raise FileNotFoundError(f"Prepared scene dir from manifest is missing: {prepared}")
    return prepared


def resolve_associations_path(scene_dir: Path, prepared_scene_dir: Path) -> Path:
    candidates = [
        scene_dir / "metadata" / "frame_associations.csv",
        prepared_scene_dir / "associations" / "frame_associations.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Missing frame associations. Checked:\n" + "\n".join(str(path) for path in candidates)
    )


def load_associations_by_frame(path: Path) -> Dict[str, dict]:
    rows = read_csv_dicts(path)
    required = {"frame_id", "raw_cloud_id", "image_to_raw_cloud_dt_ns"}
    if not rows:
        raise ValueError(f"Frame associations are empty: {path}")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Frame associations missing required columns {sorted(missing)}: {path}")
    by_frame: Dict[str, dict] = {}
    for row in rows:
        frame_id = row["frame_id"]
        if frame_id in by_frame:
            raise ValueError(f"Duplicate frame_id in frame associations: {frame_id}")
        by_frame[frame_id] = row
    return by_frame


def load_associations_sequence(path: Path) -> list[dict]:
    rows = read_csv_dicts(path)
    required = {"frame_id", "raw_cloud_id", "image_timestamp_ns", "raw_cloud_timestamp_ns", "image_to_raw_cloud_dt_ns"}
    if not rows:
        raise ValueError(f"Frame associations are empty: {path}")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Frame associations missing required columns {sorted(missing)}: {path}")
    return rows


def load_camera_poses_by_frame(prepared_scene_dir: Path) -> Dict[str, np.ndarray]:
    poses_path = prepared_scene_dir / "poses" / "camera_poses.csv"
    if not poses_path.exists():
        raise FileNotFoundError(f"Missing camera poses for local-fused projection: {poses_path}")
    rows = read_csv_dicts(poses_path)
    if not rows:
        raise ValueError(f"Camera poses are empty: {poses_path}")
    pose_keys = [f"T_odom_from_camera_{r}{c}" for r in range(4) for c in range(4)]
    missing = {"frame_id", *pose_keys} - set(rows[0])
    if missing:
        raise ValueError(f"camera_poses.csv missing required columns {sorted(missing)}: {poses_path}")
    poses: Dict[str, np.ndarray] = {}
    for row in rows:
        frame_id = row["frame_id"]
        transform = np.asarray([float(row[key]) for key in pose_keys], dtype=np.float64).reshape(4, 4)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError(f"Invalid T_odom_from_camera for frame {frame_id}: {poses_path}")
        if not np.allclose(transform[3], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-9):
            raise ValueError(f"T_odom_from_camera must have homogeneous last row for frame {frame_id}: {poses_path}")
        if frame_id in poses:
            raise ValueError(f"Duplicate frame_id in camera poses: {frame_id}")
        poses[frame_id] = transform
    return poses


def load_rectified_camera_params(prepared_scene_dir: Path) -> Tuple[int, int, float, float, float, float, Dict[str, object]]:
    camera_path = prepared_scene_dir / "calib" / "camera_rectified.json"
    if not camera_path.exists():
        raise FileNotFoundError(f"Missing rectified camera intrinsics for local-fused projection: {camera_path}")
    payload = load_json_dict(camera_path)
    k_like = np.asarray(payload.get("K_like"), dtype=np.float64)
    if k_like.shape != (3, 3) or not np.isfinite(k_like).all():
        raise ValueError(f"camera_rectified.json must contain finite 3x3 K_like: {camera_path}")
    width = int(payload["width"])
    height = int(payload["height"])
    if width <= 0 or height <= 0:
        raise ValueError(f"camera_rectified.json width/height must be positive: {camera_path}")
    fx = float(k_like[0, 0])
    fy = float(k_like[1, 1])
    cx = float(k_like[0, 2])
    cy = float(k_like[1, 2])
    return width, height, fx, fy, cx, cy, {"intrinsics_path": str(camera_path), "camera_rectified": payload}


def load_camera_from_lidar_transform(prepared_scene_dir: Path) -> Tuple[np.ndarray, Dict[str, object]]:
    tf_path = prepared_scene_dir / "calib" / "tf_chain.json"
    if not tf_path.exists():
        raise FileNotFoundError(f"Missing prepared-scene calibration transform: {tf_path}")
    payload = load_json_dict(tf_path)
    if "T_camera_from_lidar" not in payload:
        raise ValueError(f"tf_chain.json must contain T_camera_from_lidar for raw-frame projection: {tf_path}")
    transform = np.asarray(payload["T_camera_from_lidar"], dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"T_camera_from_lidar must be a finite 4x4 matrix: {tf_path}")
    if not np.allclose(transform[3], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-9):
        raise ValueError(f"T_camera_from_lidar must have homogeneous last row [0, 0, 0, 1]: {tf_path}")
    return transform, {
        "transform_path": str(tf_path),
        "selected_direction": payload.get("selected_direction"),
        "T_camera_from_lidar": transform.tolist(),
        "lidar_frame_assumption": payload.get("lidar_frame_assumption"),
    }


def transform_lidar_to_camera(points_lidar: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points_lidar = np.asarray(points_lidar, dtype=np.float64)
    if points_lidar.ndim != 2 or points_lidar.shape[1] < 3:
        raise ValueError(f"Raw LiDAR cloud must be Nx3 or wider, got {points_lidar.shape}")
    points_lidar = points_lidar[:, :3]
    finite = np.isfinite(points_lidar).all(axis=1)
    points_lidar = points_lidar[finite]
    hom = np.concatenate([points_lidar, np.ones((points_lidar.shape[0], 1), dtype=np.float64)], axis=1)
    return (transform @ hom.T).T[:, :3].astype(np.float32)


def load_raw_lidar_xyz(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing synchronized raw LiDAR cloud: {path}")
    with np.load(path) as data:
        if "xyz" not in data:
            raise ValueError(f"Raw LiDAR NPZ must contain key 'xyz': {path}")
        return np.asarray(data["xyz"], dtype=np.float32)


def export_raw_frame_depth_priors(
    args: argparse.Namespace,
    *,
    scene_dir: Path,
    depths_dir: Path,
    extrinsics,
    intrinsics,
) -> Dict[str, object]:
    prepared_scene_dir = infer_prepared_scene_dir(scene_dir, getattr(args, "prepared_scene_dir", None))
    associations_path = resolve_associations_path(scene_dir, prepared_scene_dir)
    associations = load_associations_by_frame(associations_path)
    transform, transform_meta = load_camera_from_lidar_transform(prepared_scene_dir)
    max_dt_ns = int(round(float(getattr(args, "max_raw_cloud_dt_ms", 20.0)) * 1_000_000.0))
    min_depth = float(getattr(args, "min_depth", 0.0) or 0.0)
    max_depth = float(getattr(args, "max_depth", 0.0) or 0.0)
    chunk_size = int(getattr(args, "chunk_size", 1_000_000) or 1_000_000)

    per_image = []
    for image_id in sorted(extrinsics):
        extr = extrinsics[image_id]
        intr = intrinsics[extr.camera_id]
        frame_id = Path(extr.name).stem
        if frame_id not in associations:
            raise ValueError(f"Missing raw LiDAR association for image/frame_id '{frame_id}'.")
        association = associations[frame_id]
        raw_cloud_id = association["raw_cloud_id"]
        dt_ns = int(association["image_to_raw_cloud_dt_ns"])
        if dt_ns > max_dt_ns:
            raise ValueError(
                f"Raw LiDAR sync delta for {frame_id} is {dt_ns / 1e6:.3f} ms, "
                f"exceeds --max-raw-cloud-dt-ms={float(getattr(args, 'max_raw_cloud_dt_ms', 20.0)):.3f}."
            )
        raw_path = prepared_scene_dir / "lidar" / "raw_frames" / f"{raw_cloud_id}.npz"
        points_camera = transform_lidar_to_camera(load_raw_lidar_xyz(raw_path), transform)
        depth, stats = project_depth_for_camera_points(
            points_camera,
            intr,
            chunk_size=chunk_size,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        stats.update(
            {
                "image_name": extr.name,
                "image_stem": frame_id,
                "raw_cloud_id": raw_cloud_id,
                "raw_cloud_path": str(raw_path),
                "image_timestamp_ns": int(association["image_timestamp_ns"]) if association.get("image_timestamp_ns") else None,
                "raw_cloud_timestamp_ns": int(association["raw_cloud_timestamp_ns"]) if association.get("raw_cloud_timestamp_ns") else None,
                "image_to_raw_cloud_dt_ns": dt_ns,
                "image_to_raw_cloud_dt_ms": float(dt_ns / 1e6),
            }
        )
        depth_path = depths_dir / f"{frame_id}.npy"
        np.save(depth_path, depth.astype(np.float32, copy=False))
        stats["depth_path"] = str(depth_path)
        per_image.append(stats)

    valid_ratios = np.asarray([item["valid_ratio"] for item in per_image], dtype=np.float64)
    report = {
        "scene_dir": str(scene_dir),
        "depths_dir": str(depths_dir),
        "method": "raw-frame",
        "prepared_scene_dir": str(prepared_scene_dir),
        "associations_path": str(associations_path),
        "convention": "metric",
        "unit": "meter",
        "z_buffer_rule": "nearest",
        "depth_format": "float32_npy",
        "depth_filter": {
            "min_depth": min_depth,
            "max_depth": max_depth,
            "max_depth_disabled_when_leq_zero": True,
        },
        "sync": {"max_raw_cloud_dt_ms": float(getattr(args, "max_raw_cloud_dt_ms", 20.0))},
        "transform": transform_meta,
        "num_images": len(per_image),
        "valid_ratio_summary": {
            "mean": float(valid_ratios.mean()) if valid_ratios.size else 0.0,
            "median": float(np.median(valid_ratios)) if valid_ratios.size else 0.0,
            "min": float(valid_ratios.min()) if valid_ratios.size else 0.0,
            "max": float(valid_ratios.max()) if valid_ratios.size else 0.0,
        },
        "per_image": per_image,
        "contract": {
            "coordinate_frame": "per-frame camera coordinates after T_camera_from_lidar",
            "pose_source": "not used for raw-frame projection",
            "supervision_geometry": "synchronized raw LiDAR frame",
        },
    }
    write_json(scene_dir / "depth_prior_report.json", report)
    return report


def _select_centered_sources(sequence: list[dict], target_index: int, fusion_window: int) -> list[dict]:
    radius = fusion_window // 2
    start = max(0, target_index - radius)
    end = min(len(sequence), target_index + radius + 1)
    return sequence[start:end]


def _transform_source_lidar_to_target_camera(
    points_lidar: np.ndarray,
    *,
    T_camera_from_lidar: np.ndarray,
    T_odom_from_source_camera: np.ndarray,
    T_odom_from_target_camera: np.ndarray,
) -> np.ndarray:
    points_lidar = np.asarray(points_lidar, dtype=np.float64)
    if points_lidar.ndim != 2 or points_lidar.shape[1] < 3:
        raise ValueError(f"Raw LiDAR cloud must be Nx3 or wider, got {points_lidar.shape}")
    points_lidar = points_lidar[:, :3]
    finite = np.isfinite(points_lidar).all(axis=1)
    points_lidar = points_lidar[finite]
    hom = np.concatenate([points_lidar, np.ones((points_lidar.shape[0], 1), dtype=np.float64)], axis=1)
    transform = np.linalg.inv(T_odom_from_target_camera) @ T_odom_from_source_camera @ T_camera_from_lidar
    return (transform @ hom.T).T[:, :3].astype(np.float32)


def _depth_stats(depth: np.ndarray, valid: np.ndarray) -> Dict[str, object]:
    values = depth[valid]
    if values.size == 0:
        return {
            "valid_depth_min": None,
            "valid_depth_median": None,
            "valid_depth_mean": None,
            "valid_depth_max": None,
            "depth_percentiles": {},
        }
    percentiles = np.percentile(values.astype(np.float64), [1, 5, 50, 95, 99])
    return {
        "valid_depth_min": float(values.min()),
        "valid_depth_median": float(np.median(values)),
        "valid_depth_mean": float(values.mean()),
        "valid_depth_max": float(values.max()),
        "depth_percentiles": {
            "p01": float(percentiles[0]),
            "p05": float(percentiles[1]),
            "p50": float(percentiles[2]),
            "p95": float(percentiles[3]),
            "p99": float(percentiles[4]),
        },
    }


def _project_local_fused_depth(
    source_rows: list[dict],
    *,
    prepared_scene_dir: Path,
    T_camera_from_lidar: np.ndarray,
    target_pose: np.ndarray,
    poses_by_frame: Dict[str, np.ndarray],
    camera_params: Tuple[int, int, float, float, float, float],
    chunk_size: int,
    min_depth: float,
    max_depth: float,
    conflict_depth_threshold_m: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    width, height, fx, fy, cx, cy = camera_params
    zbuffer = _empty_zbuffer_for_camera_params(width, height)
    point_count_flat = np.zeros(width * height, dtype=np.uint32)
    source_count_flat = np.zeros(width * height, dtype=np.uint16)
    all_pixel_indices = []
    all_depths = []
    per_source = []
    total_input = 0
    total_front = 0
    total_depth_range = 0
    total_projected = 0
    min_depth = max(float(min_depth), 0.0)
    max_depth = float(max_depth)

    for source_row in source_rows:
        source_frame_id = source_row["frame_id"]
        if source_frame_id not in poses_by_frame:
            raise ValueError(f"Missing camera pose for source frame '{source_frame_id}'.")
        raw_cloud_id = source_row["raw_cloud_id"]
        raw_path = prepared_scene_dir / "lidar" / "raw_frames" / f"{raw_cloud_id}.npz"
        source_pose = poses_by_frame[source_frame_id]
        points_target_camera = _transform_source_lidar_to_target_camera(
            load_raw_lidar_xyz(raw_path),
            T_camera_from_lidar=T_camera_from_lidar,
            T_odom_from_source_camera=source_pose,
            T_odom_from_target_camera=target_pose,
        )
        input_points = int(points_target_camera.shape[0])
        total_input += input_points
        source_front = 0
        source_depth_range = 0
        source_projected = 0
        source_unique_pixels = set()
        for start in range(0, input_points, max(1, int(chunk_size))):
            chunk = points_target_camera[start : start + max(1, int(chunk_size))].astype(np.float64, copy=False)
            z = chunk[:, 2]
            front = z > 1e-6
            if not np.any(front):
                continue
            chunk = chunk[front]
            z = z[front]
            source_front += int(z.shape[0])
            in_depth_range = z > min_depth
            if max_depth > 0.0:
                in_depth_range = in_depth_range & (z <= max_depth)
            if not np.any(in_depth_range):
                continue
            chunk = chunk[in_depth_range]
            z = z[in_depth_range]
            source_depth_range += int(z.shape[0])
            u = np.rint((fx * chunk[:, 0]) / z + cx).astype(np.int64)
            v = np.rint((fy * chunk[:, 1]) / z + cy).astype(np.int64)
            inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
            if not np.any(inside):
                continue
            u = u[inside]
            v = v[inside]
            z_inside = z[inside].astype(np.float32)
            pixel_indices = (v * width + u).astype(np.int64)
            source_projected += int(pixel_indices.shape[0])
            np.minimum.at(zbuffer, (v, u), z_inside)
            np.add.at(point_count_flat, pixel_indices, 1)
            unique_source_pixels = np.unique(pixel_indices)
            source_count_flat[unique_source_pixels] += 1
            source_unique_pixels.update(int(pixel) for pixel in unique_source_pixels.tolist())
            all_pixel_indices.append(pixel_indices)
            all_depths.append(z_inside)
        total_front += source_front
        total_depth_range += source_depth_range
        total_projected += source_projected
        per_source.append(
            {
                "frame_id": source_frame_id,
                "raw_cloud_id": raw_cloud_id,
                "raw_cloud_path": str(raw_path),
                "image_timestamp_ns": int(source_row["image_timestamp_ns"]),
                "raw_cloud_timestamp_ns": int(source_row["raw_cloud_timestamp_ns"]),
                "image_to_raw_cloud_dt_ns": int(source_row["image_to_raw_cloud_dt_ns"]),
                "image_to_raw_cloud_dt_ms": float(int(source_row["image_to_raw_cloud_dt_ns"]) / 1e6),
                "input_points": input_points,
                "front_points": source_front,
                "depth_range_points": source_depth_range,
                "projected_points": source_projected,
                "unique_valid_pixels": len(source_unique_pixels),
            }
        )

    valid = np.isfinite(zbuffer)
    depth = np.zeros_like(zbuffer, dtype=np.float32)
    depth[valid] = zbuffer[valid]
    mask = valid.astype(np.uint8)
    confidence = source_count_flat.reshape((height, width)).astype(np.float32)
    valid_count = int(np.count_nonzero(valid))
    pixel_count = int(width * height)

    collision_pixels = int(np.count_nonzero(point_count_flat > 1))
    conflict_pixels = 0
    conflict_projected_points = 0
    if all_pixel_indices:
        pixel_indices_all = np.concatenate(all_pixel_indices)
        depths_all = np.concatenate(all_depths).astype(np.float32)
        order = np.argsort(pixel_indices_all, kind="stable")
        sorted_pixels = pixel_indices_all[order]
        sorted_depths = depths_all[order]
        unique_pixels, starts, counts = np.unique(sorted_pixels, return_index=True, return_counts=True)
        for start, count in zip(starts, counts):
            if count <= 1:
                continue
            pixel_depths = sorted_depths[start : start + count]
            if float(pixel_depths.max() - pixel_depths.min()) > conflict_depth_threshold_m:
                conflict_pixels += 1
                conflict_projected_points += int(count)

    stats = {
        "input_points": total_input,
        "front_points": total_front,
        "depth_range_points": total_depth_range,
        "projected_points": total_projected,
        "valid_pixels": valid_count,
        "pixel_count": pixel_count,
        "valid_ratio": float(valid_count / max(pixel_count, 1)),
        "per_source": per_source,
        "zbuffer_collision_pixels": collision_pixels,
        "zbuffer_collision_ratio": float(collision_pixels / max(valid_count, 1)),
        "zbuffer_conflict_pixels": conflict_pixels,
        "zbuffer_conflict_projected_points": conflict_projected_points,
        "zbuffer_conflict_ratio": float(conflict_pixels / max(collision_pixels, 1)),
        "conflict_depth_threshold_m": float(conflict_depth_threshold_m),
        "outlier_depth_pixels": int(np.count_nonzero(depth[valid] > np.percentile(depth[valid], 99) * 1.5)) if valid_count else 0,
    }
    stats.update(_depth_stats(depth, valid))
    return depth, mask, confidence, stats


def _write_depth_overlay(image_path: Path, depth: np.ndarray, mask: np.ndarray, output_path: Path) -> None:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
    depth_values = depth[mask.astype(bool)]
    if depth_values.size == 0:
        overlay = Image.new("RGB", rgb.size, color=(0, 0, 0))
    else:
        lo = float(np.percentile(depth_values, 5))
        hi = float(np.percentile(depth_values, 95))
        scale = max(hi - lo, 1e-6)
        normalized = np.clip((depth - lo) / scale, 0.0, 1.0)
        heat = np.zeros((*depth.shape, 3), dtype=np.uint8)
        heat[..., 0] = (normalized * 255).astype(np.uint8)
        heat[..., 2] = ((1.0 - normalized) * 255).astype(np.uint8)
        heat[mask == 0] = 0
        overlay = Image.fromarray(heat, mode="RGB")
    blended = Image.blend(rgb, overlay.resize(rgb.size), alpha=0.45)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    blended.save(output_path)


def export_local_fused_depth_priors(
    args: argparse.Namespace,
    *,
    scene_dir: Path,
    depths_dir: Path,
    masks_dir: Path,
    confidence_dir: Path | None,
    extrinsics,
    intrinsics,
) -> Dict[str, object]:
    fusion_window = int(getattr(args, "fusion_window", 5) or 5)
    if fusion_window <= 0 or fusion_window % 2 != 1:
        raise ValueError("fusion-window must be a positive odd integer.")
    if getattr(args, "fusion_mode", "centered") != "centered":
        raise ValueError("Only fusion-mode='centered' is currently supported.")

    prepared_scene_dir = infer_prepared_scene_dir(scene_dir, getattr(args, "prepared_scene_dir", None))
    associations_path = resolve_associations_path(scene_dir, prepared_scene_dir)
    association_sequence = load_associations_sequence(associations_path)
    association_index = {row["frame_id"]: idx for idx, row in enumerate(association_sequence)}
    if len(association_index) != len(association_sequence):
        raise ValueError(f"Duplicate frame_id in frame associations: {associations_path}")
    poses_by_frame = load_camera_poses_by_frame(prepared_scene_dir)
    transform, transform_meta = load_camera_from_lidar_transform(prepared_scene_dir)
    width, height, fx, fy, cx, cy, camera_meta = load_rectified_camera_params(prepared_scene_dir)
    min_depth = float(getattr(args, "min_depth", 0.0) or 0.0)
    max_depth = float(getattr(args, "max_depth", 0.0) or 0.0)
    chunk_size = int(getattr(args, "chunk_size", 1_000_000) or 1_000_000)
    conflict_threshold = 1.0

    per_image = []
    overlay_remaining = max(0, int(getattr(args, "overlay_count", 0) or 0))
    overlay_dir = scene_dir / f"depth_prior_overlays_lidar_fused_{fusion_window}f"
    for image_id in sorted(extrinsics):
        extr = extrinsics[image_id]
        intr = intrinsics[extr.camera_id]
        intr_width, intr_height, *_ = intrinsics_to_params(intr)
        if (intr_width, intr_height) != (width, height):
            raise ValueError(
                "Rectified camera dimensions do not match COLMAP intrinsics for "
                f"{extr.name}: camera_rectified={width}x{height}, colmap={intr_width}x{intr_height}"
            )
        frame_id = Path(extr.name).stem
        if frame_id not in association_index:
            raise ValueError(f"Missing raw LiDAR association for image/frame_id '{frame_id}'.")
        if frame_id not in poses_by_frame:
            raise ValueError(f"Missing target camera pose for image/frame_id '{frame_id}'.")

        target_index = association_index[frame_id]
        source_rows = _select_centered_sources(association_sequence, target_index, fusion_window)
        target_ts = int(association_sequence[target_index]["image_timestamp_ns"])
        depth, mask, confidence, stats = _project_local_fused_depth(
            source_rows,
            prepared_scene_dir=prepared_scene_dir,
            T_camera_from_lidar=transform,
            target_pose=poses_by_frame[frame_id],
            poses_by_frame=poses_by_frame,
            camera_params=(width, height, fx, fy, cx, cy),
            chunk_size=chunk_size,
            min_depth=min_depth,
            max_depth=max_depth,
            conflict_depth_threshold_m=conflict_threshold,
        )
        source_timestamps = [int(row["image_timestamp_ns"]) for row in source_rows]
        source_offsets = [int(ts - target_ts) for ts in source_timestamps]
        source_dts = [int(row["image_to_raw_cloud_dt_ns"]) for row in source_rows]
        depth_path = depths_dir / f"{frame_id}.npy"
        mask_path = masks_dir / f"{frame_id}.npy"
        np.save(depth_path, depth.astype(np.float32, copy=False))
        np.save(mask_path, mask.astype(np.uint8, copy=False))
        confidence_path = None
        if confidence_dir is not None:
            confidence_path = confidence_dir / f"{frame_id}.npy"
            np.save(confidence_path, confidence.astype(np.float32, copy=False))
        stats.update(
            {
                "image_name": extr.name,
                "image_stem": frame_id,
                "target_frame_id": frame_id,
                "actual_source_count": len(source_rows),
                "source_frame_ids": [row["frame_id"] for row in source_rows],
                "source_raw_cloud_ids": [row["raw_cloud_id"] for row in source_rows],
                "source_time_offsets_ns": source_offsets,
                "source_time_offsets_ms": [float(offset / 1e6) for offset in source_offsets],
                "fusion_time_window_ns": int(max(source_timestamps) - min(source_timestamps)) if source_timestamps else 0,
                "fusion_time_window_ms": float((max(source_timestamps) - min(source_timestamps)) / 1e6) if source_timestamps else 0.0,
                "image_to_cloud_dt_summary": {
                    "mean_ms": float(np.mean(source_dts) / 1e6) if source_dts else 0.0,
                    "median_ms": float(np.median(source_dts) / 1e6) if source_dts else 0.0,
                    "min_ms": float(np.min(source_dts) / 1e6) if source_dts else 0.0,
                    "max_ms": float(np.max(source_dts) / 1e6) if source_dts else 0.0,
                },
                "depth_path": str(depth_path),
                "mask_path": str(mask_path),
                "confidence_path": str(confidence_path) if confidence_path is not None else None,
            }
        )
        if overlay_remaining > 0:
            _write_depth_overlay(scene_dir / "images" / extr.name, depth, mask, overlay_dir / f"{frame_id}.png")
            overlay_remaining -= 1
        per_image.append(stats)

    valid_ratios = np.asarray([item["valid_ratio"] for item in per_image], dtype=np.float64)
    report_path = scene_dir / f"depth_prior_report_local_fused_{fusion_window}f.json"
    report = {
        "scene_dir": str(scene_dir),
        "depths_dir": str(depths_dir),
        "masks_dir": str(masks_dir),
        "confidence_dir": str(confidence_dir) if confidence_dir is not None else None,
        "method": "local-fused",
        "fusion_window": fusion_window,
        "fusion_mode": "centered",
        "prepared_scene_dir": str(prepared_scene_dir),
        "associations_path": str(associations_path),
        "convention": "metric",
        "unit": "meter",
        "z_buffer_rule": "nearest",
        "depth_format": "float32_npy",
        "mask_format": "uint8_npy",
        "confidence_format": "float32_npy_source_count" if confidence_dir is not None else None,
        "depth_filter": {
            "min_depth": min_depth,
            "max_depth": max_depth,
            "max_depth_disabled_when_leq_zero": True,
        },
        "transform": transform_meta,
        "rectified_camera": camera_meta,
        "num_images": len(per_image),
        "valid_ratio_summary": {
            "mean": float(valid_ratios.mean()) if valid_ratios.size else 0.0,
            "median": float(np.median(valid_ratios)) if valid_ratios.size else 0.0,
            "min": float(valid_ratios.min()) if valid_ratios.size else 0.0,
            "max": float(valid_ratios.max()) if valid_ratios.size else 0.0,
        },
        "per_image": per_image,
        "contract": {
            "coordinate_frame": "target rectified camera coordinates",
            "pose_source": "prepared poses/camera_poses.csv T_odom_from_camera",
            "source_transform": "inverse(T_odom_from_target_cam) @ T_odom_from_source_cam @ T_camera_from_lidar",
            "supervision_geometry": "local multi-frame raw LiDAR fusion",
        },
    }
    write_json(report_path, report)
    return report


def export_depth_priors(args: argparse.Namespace) -> Dict[str, object]:
    scene_dir = args.scene_dir.expanduser().resolve()
    method = str(getattr(args, "method", "raw-frame") or "raw-frame")
    if method == "raw-frame":
        depths_dir = (args.output_depths_dir or scene_dir / "depths").expanduser().resolve()
    elif method == "local-fused":
        fusion_window = int(getattr(args, "fusion_window", 5) or 5)
        if fusion_window <= 0 or fusion_window % 2 != 1:
            raise ValueError("fusion-window must be a positive odd integer.")
        depths_dir = (args.output_depths_dir or scene_dir / f"depths_lidar_fused_{fusion_window}f").expanduser().resolve()
    else:
        raise ValueError(f"Unsupported depth prior method: {method}")
    if not scene_dir.exists():
        raise FileNotFoundError(f"Missing scene dir: {scene_dir}")
    if depths_dir.exists() and any(depths_dir.glob("*.npy")) and not args.overwrite:
        raise FileExistsError(f"Depth files already exist under {depths_dir}; pass --overwrite to replace them.")

    extrinsics, intrinsics = load_colmap_model(scene_dir)
    duplicate_stems = find_duplicate_image_stems(extrinsics)
    if duplicate_stems:
        examples = ", ".join(f"{stem}: {names}" for stem, names in list(duplicate_stems.items())[:5])
        raise ValueError(f"Image stems are not unique; depth .npy outputs would overwrite: {examples}")
    validate_image_dimensions(scene_dir, extrinsics, intrinsics)
    depths_dir.mkdir(parents=True, exist_ok=True)

    if method == "raw-frame":
        return export_raw_frame_depth_priors(
            args,
            scene_dir=scene_dir,
            depths_dir=depths_dir,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
        )

    fusion_window = int(getattr(args, "fusion_window", 5) or 5)
    masks_dir = (getattr(args, "output_masks_dir", None) or scene_dir / f"masks_lidar_fused_{fusion_window}f").expanduser().resolve()
    if masks_dir.exists() and any(masks_dir.glob("*.npy")) and not args.overwrite:
        raise FileExistsError(f"Mask files already exist under {masks_dir}; pass --overwrite to replace them.")
    masks_dir.mkdir(parents=True, exist_ok=True)
    confidence_dir = None
    if bool(getattr(args, "write_confidence", False)) or getattr(args, "output_confidence_dir", None) is not None:
        confidence_dir = (
            getattr(args, "output_confidence_dir", None) or scene_dir / f"confidence_lidar_fused_{fusion_window}f"
        ).expanduser().resolve()
        if confidence_dir.exists() and any(confidence_dir.glob("*.npy")) and not args.overwrite:
            raise FileExistsError(f"Confidence files already exist under {confidence_dir}; pass --overwrite to replace them.")
        confidence_dir.mkdir(parents=True, exist_ok=True)

    return export_local_fused_depth_priors(
        args,
        scene_dir=scene_dir,
        depths_dir=depths_dir,
        masks_dir=masks_dir,
        confidence_dir=confidence_dir,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
    )


def main(argv=None) -> int:
    report = export_depth_priors(parse_args(argv))
    print(f"[INFO] depth priors written: {report['depths_dir']}")
    print(f"[INFO] images={report['num_images']} mean_valid_ratio={report['valid_ratio_summary']['mean']:.6f}")
    if report["method"] == "raw-frame":
        report_path = Path(report["scene_dir"]) / "depth_prior_report.json"
    else:
        report_path = Path(report["scene_dir"]) / f"depth_prior_report_local_fused_{report['fusion_window']}f.json"
    print(f"[INFO] report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
