from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

try:
    from data_preparation.shared.io import write_json
    from data_preparation.shared.pointcloud import read_ply_xyz_rgb
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from data_preparation.shared.io import write_json
    from data_preparation.shared.pointcloud import read_ply_xyz_rgb

BASELINES_ROOT = Path(__file__).resolve().parents[2]
BASELINE_02_ROOT = BASELINES_ROOT / "02baseline"
if str(BASELINE_02_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_02_ROOT))

from common.camera_math import qvec2rotmat  # noqa: E402
from common.colmap_io import (  # noqa: E402
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project a dense LiDAR/global map into an existing COLMAP-compatible scene as metric .npy depth priors."
    )
    parser.add_argument("--scene-dir", required=True, type=Path, help="COLMAP-compatible scene root.")
    parser.add_argument("--point-cloud", required=True, type=Path, help="Dense point cloud used for projection.")
    parser.add_argument(
        "--source-frame",
        default="colmap",
        help="Frame of --point-cloud. Use 'colmap' only when points are already in COLMAP/SfM world coordinates.",
    )
    parser.add_argument(
        "--transform-json",
        type=Path,
        default=None,
        help="JSON file containing a 4x4 transform from source frame to COLMAP/SfM world frame.",
    )
    parser.add_argument("--output-depths-dir", type=Path, default=None, help="Output directory. Defaults to <scene-dir>/depths.")
    parser.add_argument("--chunk-size", type=int, default=1_000_000, help="Point projection chunk size.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .npy depth files.")
    return parser.parse_args(argv)


def load_colmap_model(scene_dir: Path):
    sparse0 = scene_dir / "sparse" / "0"
    if (sparse0 / "images.bin").exists() and (sparse0 / "cameras.bin").exists():
        return read_extrinsics_binary(str(sparse0 / "images.bin")), read_intrinsics_binary(str(sparse0 / "cameras.bin"))
    if (sparse0 / "images.txt").exists() and (sparse0 / "cameras.txt").exists():
        return read_extrinsics_text(str(sparse0 / "images.txt")), read_intrinsics_text(str(sparse0 / "cameras.txt"))
    raise FileNotFoundError(f"Missing COLMAP cameras/images under {sparse0}")


def load_points(path: Path) -> Tuple[np.ndarray, Dict[str, object]]:
    suffix = path.suffix.lower()
    if suffix == ".ply":
        xyz, _ = read_ply_xyz_rgb(path)
    elif suffix == ".npy":
        xyz = np.load(path)
    elif suffix == ".npz":
        with np.load(path) as data:
            if "xyz" not in data:
                raise ValueError(f"NPZ point cloud must contain key 'xyz': {path}")
            xyz = data["xyz"]
    else:
        raise ValueError(f"Unsupported point cloud format for depth prior projection: {path}")
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError(f"Point cloud must be Nx3 or wider, got {xyz.shape}: {path}")
    xyz = xyz[:, :3]
    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    return xyz, {
        "source_point_cloud": str(path),
        "source_points_total": int(len(finite)),
        "source_points_finite": int(xyz.shape[0]),
    }


def load_source_to_colmap_transform(path: Path) -> Tuple[np.ndarray, Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    matrix = payload.get("T_colmap_from_source") or payload.get("transform") or payload.get("matrix")
    if matrix is None:
        raise ValueError(
            f"Transform JSON must contain 'T_colmap_from_source' as a 4x4 row-major matrix: {path}"
        )
    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Transform must be 4x4, got {transform.shape}: {path}")
    return transform, {
        "transform_path": str(path),
        "transform_key": "T_colmap_from_source" if "T_colmap_from_source" in payload else "transform",
        "T_colmap_from_source": transform.tolist(),
        "transform_note": payload.get("note"),
    }


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    hom = np.concatenate([points.astype(np.float64), np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    transformed = (transform @ hom.T).T[:, :3]
    return transformed.astype(np.float32)


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


def project_depth_for_image(points_world: np.ndarray, extr, intr, *, chunk_size: int) -> Tuple[np.ndarray, Dict[str, object]]:
    width, height, fx, fy, cx, cy = intrinsics_to_params(intr)
    zbuffer = np.full((height, width), np.inf, dtype=np.float32)
    rotation_cw = qvec2rotmat(extr.qvec).astype(np.float64)
    translation_cw = np.asarray(extr.tvec, dtype=np.float64)
    input_points = int(points_world.shape[0])
    front_points = 0
    projected_points = 0

    for start in range(0, input_points, max(1, int(chunk_size))):
        chunk = points_world[start : start + max(1, int(chunk_size))].astype(np.float64, copy=False)
        points_camera = (rotation_cw @ chunk.T).T + translation_cw
        z = points_camera[:, 2]
        front = z > 1e-6
        if not np.any(front):
            continue
        points_camera = points_camera[front]
        z = z[front]
        front_points += int(z.shape[0])
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
    return depth, {
        "image_name": extr.name,
        "image_stem": Path(extr.name).stem,
        "width": width,
        "height": height,
        "input_points": input_points,
        "front_points": front_points,
        "projected_points": projected_points,
        "valid_pixels": valid_count,
        "pixel_count": pixel_count,
        "valid_ratio": float(valid_count / max(pixel_count, 1)),
    }


def export_depth_priors(args: argparse.Namespace) -> Dict[str, object]:
    scene_dir = args.scene_dir.expanduser().resolve()
    point_cloud = args.point_cloud.expanduser().resolve()
    depths_dir = (args.output_depths_dir or scene_dir / "depths").expanduser().resolve()
    if not scene_dir.exists():
        raise FileNotFoundError(f"Missing scene dir: {scene_dir}")
    if not point_cloud.exists():
        raise FileNotFoundError(f"Missing point cloud: {point_cloud}")
    if depths_dir.exists() and any(depths_dir.glob("*.npy")) and not args.overwrite:
        raise FileExistsError(f"Depth files already exist under {depths_dir}; pass --overwrite to replace them.")

    extrinsics, intrinsics = load_colmap_model(scene_dir)
    points_world, point_meta = load_points(point_cloud)
    source_frame = str(args.source_frame)
    transform_meta = {
        "source_frame": source_frame,
        "target_frame": "COLMAP/SfM world",
        "transform_required": source_frame not in {"colmap", "sfm", "colmap_world", "sfm_world"},
        "transform_applied": False,
    }
    if transform_meta["transform_required"]:
        if args.transform_json is None:
            raise ValueError(
                "--transform-json is required when --source-frame is not already COLMAP/SfM world."
            )
        transform, loaded_transform_meta = load_source_to_colmap_transform(args.transform_json.expanduser().resolve())
        points_world = apply_transform(points_world, transform)
        transform_meta.update(loaded_transform_meta)
        transform_meta["transform_applied"] = True
    elif args.transform_json is not None:
        transform, loaded_transform_meta = load_source_to_colmap_transform(args.transform_json.expanduser().resolve())
        points_world = apply_transform(points_world, transform)
        transform_meta.update(loaded_transform_meta)
        transform_meta["transform_applied"] = True
    depths_dir.mkdir(parents=True, exist_ok=True)

    per_image = []
    for image_id in sorted(extrinsics):
        extr = extrinsics[image_id]
        intr = intrinsics[extr.camera_id]
        depth, stats = project_depth_for_image(points_world, extr, intr, chunk_size=args.chunk_size)
        np.save(depths_dir / f"{stats['image_stem']}.npy", depth.astype(np.float32, copy=False))
        stats["depth_path"] = str(depths_dir / f"{stats['image_stem']}.npy")
        per_image.append(stats)

    valid_ratios = np.asarray([item["valid_ratio"] for item in per_image], dtype=np.float64)
    report = {
        "scene_dir": str(scene_dir),
        "depths_dir": str(depths_dir),
        **point_meta,
        "convention": "metric",
        "unit": "meter",
        "z_buffer_rule": "nearest",
        "depth_format": "float32_npy",
        "source_frame": source_frame,
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
            "coordinate_frame": "COLMAP/SfM world frame",
            "pose_source": "scene sparse/0 images/cameras",
            "initialization_geometry": "sparse/0/points3D remains independent from this supervision point cloud",
            "supervision_geometry": "projected dense source point cloud",
        },
    }
    write_json(scene_dir / "depth_prior_report.json", report)
    return report


def main(argv=None) -> int:
    report = export_depth_priors(parse_args(argv))
    print(f"[INFO] depth priors written: {report['depths_dir']}")
    print(f"[INFO] images={report['num_images']} mean_valid_ratio={report['valid_ratio_summary']['mean']:.6f}")
    print(f"[INFO] report={Path(report['scene_dir']) / 'depth_prior_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
