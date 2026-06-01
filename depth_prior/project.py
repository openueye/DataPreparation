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


def export_depth_priors(args: argparse.Namespace) -> Dict[str, object]:
    scene_dir = args.scene_dir.expanduser().resolve()
    depths_dir = (args.output_depths_dir or scene_dir / "depths").expanduser().resolve()
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

    return export_raw_frame_depth_priors(
        args,
        scene_dir=scene_dir,
        depths_dir=depths_dir,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
    )


def main(argv=None) -> int:
    report = export_depth_priors(parse_args(argv))
    print(f"[INFO] depth priors written: {report['depths_dir']}")
    print(f"[INFO] images={report['num_images']} mean_valid_ratio={report['valid_ratio_summary']['mean']:.6f}")
    print(f"[INFO] report={Path(report['scene_dir']) / 'depth_prior_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
