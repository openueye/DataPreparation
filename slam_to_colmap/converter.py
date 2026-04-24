from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from data_preparation.shared.camera_models import load_camera_json, pinhole_from_k_like
    from data_preparation.shared.colmap_io import write_cameras_text, write_images_text, write_points3d_text
    from data_preparation.shared.io import find_image_path, write_json
    from data_preparation.shared.pointcloud import read_ply_xyz_rgb
    from data_preparation.shared.poses import load_pose_rows_ordered, rotmat_to_qvec_colmap, world_from_camera_from_row
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from data_preparation.shared.camera_models import load_camera_json, pinhole_from_k_like
    from data_preparation.shared.colmap_io import write_cameras_text, write_images_text, write_points3d_text
    from data_preparation.shared.io import find_image_path, write_json
    from data_preparation.shared.pointcloud import read_ply_xyz_rgb
    from data_preparation.shared.poses import load_pose_rows_ordered, rotmat_to_qvec_colmap, world_from_camera_from_row


def find_image_name(images_dir: Path, frame_id: str) -> str:
    return find_image_path(images_dir, frame_id).name


def link_or_copy_images(source_images: Path, output_images: Path, *, copy_images: bool = False) -> None:
    if output_images.exists() or output_images.is_symlink():
        return
    output_images.parent.mkdir(parents=True, exist_ok=True)
    if copy_images:
        shutil.copytree(source_images, output_images)
    else:
        os.symlink(source_images, output_images, target_is_directory=True)


def read_ply_points(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    return read_ply_xyz_rgb(path)


def _camera_model_name(camera_data: Dict[str, object]) -> str:
    return str(camera_data.get("camera_model") or "").strip().upper()


def _is_declared_undistorted(camera_data: Dict[str, object]) -> bool:
    return bool(
        camera_data.get("undistorted")
        or camera_data.get("rectified")
        or str(camera_data.get("image_model") or "").strip().lower() in {"pinhole", "undistorted_pinhole"}
    )


def validate_colmap_camera_compatibility(
    camera_data: Dict[str, object],
    *,
    allow_pinhole_approximation: bool = False,
) -> Dict[str, object]:
    """Guard against exporting distorted images as a high-quality PINHOLE scene."""

    model = _camera_model_name(camera_data)
    native_pinhole = model in {"PINHOLE", "SIMPLE_PINHOLE"}
    declared_undistorted = _is_declared_undistorted(camera_data)
    compatible = native_pinhole or declared_undistorted

    if compatible:
        return {
            "compatible": True,
            "mode": "undistorted_pinhole",
            "source_camera_model": camera_data.get("camera_model"),
            "allow_pinhole_approximation": bool(allow_pinhole_approximation),
            "note": "Images are declared compatible with a COLMAP PINHOLE/SIMPLE_PINHOLE loader.",
        }

    note = (
        f"Source camera model '{camera_data.get('camera_model')}' is not an undistorted pinhole model. "
        "Exporting raw distorted images as COLMAP PINHOLE will blur 3DGS training results."
    )
    if not allow_pinhole_approximation:
        raise ValueError(
            note
            + " Rectify/undistort the images first, or rerun with --allow-pinhole-approximation "
            "only for smoke tests and legacy compatibility."
        )

    return {
        "compatible": False,
        "mode": "unsafe_k_like_pinhole_approximation",
        "source_camera_model": camera_data.get("camera_model"),
        "allow_pinhole_approximation": True,
        "note": note,
    }


def deterministic_subsample(xyz: np.ndarray, rgb: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or xyz.shape[0] <= max_points:
        return xyz, rgb
    indices = np.linspace(0, xyz.shape[0] - 1, num=max_points, dtype=np.int64)
    return xyz[indices], rgb[indices]


def build_image_records(pose_rows, images_dir: Path):
    records = []
    for image_id, row in enumerate(pose_rows, start=1):
        world_from_camera = world_from_camera_from_row(row)
        rotation_wc = world_from_camera[:3, :3]
        translation_wc = world_from_camera[:3, 3]
        rotation_cw = rotation_wc.T
        translation_cw = -rotation_cw @ translation_wc
        records.append(
            {
                "image_id": image_id,
                "qvec": rotmat_to_qvec_colmap(rotation_cw),
                "tvec": translation_cw,
                "image_name": find_image_name(images_dir, row["frame_id"]),
            }
        )
    return records


def convert_scene(
    scene_dir: Path,
    output_dir: Path,
    *,
    points_ply: Optional[Path] = None,
    max_points: int = 300_000,
    copy_images: bool = False,
    allow_pinhole_approximation: bool = False,
) -> Dict[str, object]:
    scene_dir = scene_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    images_dir = scene_dir / "images"
    sparse_dir = output_dir / "sparse" / "0"
    output_images = output_dir / "images"
    if points_ply is None:
        colorized = scene_dir / "lidar" / "global_map_colorized.ply"
        points_ply = colorized if colorized.exists() else scene_dir / "lidar" / "global_map.ply"
    points_ply = points_ply.expanduser().resolve()

    for required in (images_dir, scene_dir / "poses" / "poses.csv", scene_dir / "intrinsics" / "camera.json", points_ply):
        if not required.exists():
            raise FileNotFoundError(f"Missing required input: {required}")

    camera_data = load_camera_json(scene_dir / "intrinsics" / "camera.json")
    camera_compatibility = validate_colmap_camera_compatibility(
        camera_data,
        allow_pinhole_approximation=allow_pinhole_approximation,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    link_or_copy_images(images_dir, output_images, copy_images=copy_images)

    width, height, fx, fy, cx, cy = pinhole_from_k_like(camera_data)
    write_cameras_text(sparse_dir / "cameras.txt", width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)

    pose_rows = load_pose_rows_ordered(scene_dir / "poses" / "poses.csv")
    image_records = build_image_records(pose_rows, images_dir)
    write_images_text(sparse_dir / "images.txt", image_records)

    xyz, rgb = read_ply_points(points_ply)
    source_points = int(xyz.shape[0])
    xyz, rgb = deterministic_subsample(xyz, rgb, max_points=max_points)
    write_points3d_text(sparse_dir / "points3D.txt", xyz, rgb)

    metadata = {
        "source_scene": str(scene_dir),
        "output_scene": str(output_dir),
        "format": "COLMAP-compatible text model",
        "camera_model": "PINHOLE",
        "source_camera_model": camera_data.get("camera_model"),
        "camera_compatibility": camera_compatibility,
        "projection_model_note": camera_compatibility["note"],
        "pose_direction": "poses.csv T_world_from_camera converted to COLMAP world-to-camera qvec/tvec",
        "points_source": str(points_ply),
        "source_points": source_points,
        "written_points": int(xyz.shape[0]),
        "num_images": len(image_records),
        "copy_images": bool(copy_images),
        "baseline_type": "SLAM-pose/LiDAR-points COLMAP-compatible scene, not a pure RGB-only COLMAP SfM baseline.",
    }
    write_json(output_dir / "slam_to_colmap_report.json", metadata)
    return metadata
