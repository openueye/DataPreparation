from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

try:
    from data_preparation.shared.camera_models import pinhole_from_k_like
    from data_preparation.shared.colmap_io import write_cameras_text, write_images_text
    from data_preparation.shared.io import find_image_path, load_json, write_json
    from data_preparation.shared.pointcloud import read_ply_xyz_rgb, write_downsampled_colmap_points_with_ply
    from data_preparation.shared.poses import load_pose_rows_ordered, matrix_from_pose_row, rotmat_to_qvec_colmap, world_from_camera_from_row
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from data_preparation.shared.camera_models import pinhole_from_k_like
    from data_preparation.shared.colmap_io import write_cameras_text, write_images_text
    from data_preparation.shared.io import find_image_path, load_json, write_json
    from data_preparation.shared.pointcloud import read_ply_xyz_rgb, write_downsampled_colmap_points_with_ply
    from data_preparation.shared.poses import load_pose_rows_ordered, matrix_from_pose_row, rotmat_to_qvec_colmap, world_from_camera_from_row


def find_image_name(images_dir: Path, frame_id: str) -> str:
    return find_image_path(images_dir, frame_id).name


PURE_HEADERSTAMP_SUFFIX = "_pure_headerstamp"


def link_or_copy_images(source_images: Path, output_images: Path, *, copy_images: bool = False) -> None:
    if output_images.exists() or output_images.is_symlink():
        return
    output_images.parent.mkdir(parents=True, exist_ok=True)
    if copy_images:
        shutil.copytree(source_images, output_images)
    else:
        os.symlink(source_images, output_images, target_is_directory=True)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def ensure_pure_headerstamp_paths(scene_dir: Path, points_ply: Optional[Path] = None) -> Dict[str, Path]:
    paths = {
        "scene_dir": scene_dir,
        "images_rectified_dir": scene_dir / "images_rectified",
        "camera_poses_csv": scene_dir / "poses" / "camera_poses.csv",
        "camera_rectified_json": scene_dir / "calib" / "camera_rectified.json",
        "frame_associations_csv": scene_dir / "associations" / "frame_associations.csv",
        "slam_frames_manifest_csv": scene_dir / "lidar" / "slam_frames_manifest.csv",
        "lidar_slam_ply": points_ply or scene_dir / "lidar" / "global_map_slam_odom.ply",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required pure-headerstamp input paths:\n" + "\n".join(missing))
    return paths


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def select_reliable_slam_matched_rows(
    association_rows: Iterable[Mapping[str, str]],
    slam_rows: Iterable[Mapping[str, str]],
) -> List[Dict[str, str]]:
    association_list = [dict(row) for row in association_rows]
    slam_list = [dict(row) for row in slam_rows]
    if not association_list:
        raise ValueError("frame_associations.csv is empty.")
    if not slam_list:
        raise ValueError("slam_frames_manifest.csv is empty.")

    image_timestamps = np.asarray([int(row["image_timestamp_ns"]) for row in association_list], dtype=np.int64)
    slam_timestamps = np.asarray([int(row["timestamp_ns"]) for row in slam_list], dtype=np.int64)
    image_count = int(image_timestamps.shape[0])
    slam_count = int(slam_timestamps.shape[0])
    if slam_count > image_count:
        raise ValueError(
            f"Cannot build a one-to-one SLAM-to-image subset because slam_count={slam_count} > image_count={image_count}."
        )

    previous = np.abs(image_timestamps - slam_timestamps[0]).astype(np.float64)
    parents: List[np.ndarray] = [np.full(image_count, -1, dtype=np.int32)]
    inf = float("inf")

    for slam_index in range(1, slam_count):
        prefix_best = np.full(image_count, inf, dtype=np.float64)
        prefix_parent = np.full(image_count, -1, dtype=np.int32)
        best_cost = inf
        best_index = -1
        for image_index in range(image_count):
            if previous[image_index] < best_cost:
                best_cost = float(previous[image_index])
                best_index = image_index
            prefix_best[image_index] = best_cost
            prefix_parent[image_index] = best_index

        current = np.full(image_count, inf, dtype=np.float64)
        parent = np.full(image_count, -1, dtype=np.int32)
        for image_index in range(slam_index, image_count):
            previous_image_index = int(prefix_parent[image_index - 1])
            if previous_image_index < 0:
                continue
            current[image_index] = prefix_best[image_index - 1] + abs(
                int(image_timestamps[image_index]) - int(slam_timestamps[slam_index])
            )
            parent[image_index] = previous_image_index
        previous = current
        parents.append(parent)

    end_image_index = int(np.argmin(previous))
    if not np.isfinite(previous[end_image_index]):
        raise ValueError("Failed to compute a one-to-one SLAM-to-image assignment.")

    matched_image_indices = [end_image_index]
    current_image_index = end_image_index
    for slam_index in range(slam_count - 1, 0, -1):
        current_image_index = int(parents[slam_index][current_image_index])
        if current_image_index < 0:
            raise ValueError("Backtracking failed while reconstructing the SLAM-to-image assignment.")
        matched_image_indices.append(current_image_index)
    matched_image_indices.reverse()

    selected_rows: List[Dict[str, str]] = []
    for slam_index, image_index in enumerate(matched_image_indices):
        row = dict(association_list[image_index])
        row["matched_slam_cloud_id"] = slam_list[slam_index]["slam_cloud_id"]
        row["matched_slam_timestamp_ns"] = slam_list[slam_index]["timestamp_ns"]
        row["matched_image_to_slam_dt_ns"] = str(
            abs(int(row["image_timestamp_ns"]) - int(slam_list[slam_index]["timestamp_ns"]))
        )
        selected_rows.append(row)
    return selected_rows


def filter_pose_rows_by_frame_ids(
    pose_rows: Iterable[Mapping[str, str]],
    allowed_frame_ids: Iterable[str],
) -> List[Dict[str, str]]:
    allowed = {frame_id for frame_id in allowed_frame_ids}
    filtered_rows: List[Dict[str, str]] = []
    seen = set()
    for row in pose_rows:
        frame_id = row["frame_id"]
        if frame_id in allowed:
            filtered_rows.append(dict(row))
            seen.add(frame_id)
    missing = sorted(allowed - seen)
    if missing:
        raise KeyError(
            "Some SLAM-matched frames are missing from poses/camera_poses.csv: "
            + ", ".join(missing[:10])
        )
    return filtered_rows


def project_so3(rotation: np.ndarray) -> np.ndarray:
    u_mat, _, vt_mat = np.linalg.svd(rotation)
    fixed = u_mat @ vt_mat
    if np.linalg.det(fixed) < 0:
        u_mat[:, -1] *= -1.0
        fixed = u_mat @ vt_mat
    return fixed


def rotation_matrix_to_quaternion_wxyz(rotation: np.ndarray) -> Tuple[float, float, float, float]:
    return tuple(float(value) for value in rotmat_to_qvec_colmap(project_so3(rotation)))


def to_colmap_pose(world_from_camera: np.ndarray) -> Tuple[float, float, float, float, float, float, float]:
    camera_from_world = np.linalg.inv(world_from_camera)
    qw, qx, qy, qz = rotation_matrix_to_quaternion_wxyz(camera_from_world[:3, :3])
    tx, ty, tz = camera_from_world[:3, 3].tolist()
    return (qw, qx, qy, qz, float(tx), float(ty), float(tz))


def link_or_copy_selected_images(
    source_dir: Path,
    target_dir: Path,
    image_names: Iterable[str],
    copy_images: bool,
) -> Dict[str, Any]:
    target_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    mode = "copy" if copy_images else "symlink"
    for image_name in image_names:
        source_path = source_dir / image_name
        if not source_path.is_file():
            raise FileNotFoundError(f"Selected rectified image not found: {source_path}")
        target_path = target_dir / image_name
        if target_path.exists() or target_path.is_symlink():
            target_path.unlink()
        if copy_images:
            shutil.copy2(source_path, target_path)
        else:
            target_path.symlink_to(source_path.resolve())
        count += 1
    return {"mode": mode, "count": count}


def write_pure_headerstamp_cameras_txt(
    path: Path,
    *,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# Camera list with one line of data per camera:",
                "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
                f"1 PINHOLE {width} {height} {fx:.8f} {fy:.8f} {cx:.8f} {cy:.8f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_pure_headerstamp_images_txt(path: Path, image_records: Iterable[Mapping[str, Any]]) -> None:
    lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
    ]
    for record in image_records:
        qvec = record["qvec"]
        tvec = record["tvec"]
        lines.append(
            f"{record['image_id']} "
            f"{qvec[0]:.12f} {qvec[1]:.12f} {qvec[2]:.12f} {qvec[3]:.12f} "
            f"{tvec[0]:.12f} {tvec[1]:.12f} {tvec[2]:.12f} "
            f"1 {record['image_name']}"
        )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_pure_headerstamp_image_records(pose_rows: Iterable[Mapping[str, str]], images_dir: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    missing_images: List[str] = []
    for image_id, row in enumerate(pose_rows, start=1):
        image_name = f"{row['frame_id']}.jpg"
        if not (images_dir / image_name).is_file():
            missing_images.append(image_name)
            continue
        world_from_camera = matrix_from_pose_row(dict(row), "T_odom_from_camera")
        qw, qx, qy, qz, tx, ty, tz = to_colmap_pose(world_from_camera)
        records.append(
            {
                "image_id": image_id,
                "qvec": np.asarray([qw, qx, qy, qz], dtype=np.float64),
                "tvec": np.asarray([tx, ty, tz], dtype=np.float64),
                "image_name": image_name,
            }
        )
    if missing_images:
        raise FileNotFoundError(
            f"Missing {len(missing_images)} rectified images referenced by camera_poses.csv: "
            + ", ".join(missing_images[:10])
        )
    return records


def copy_support_files(source_paths: Mapping[str, Path], output_dir: Path) -> None:
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_paths["camera_poses_csv"], metadata_dir / "camera_poses.csv")
    shutil.copy2(source_paths["frame_associations_csv"], metadata_dir / "frame_associations.csv")
    shutil.copy2(source_paths["slam_frames_manifest_csv"], metadata_dir / "slam_frames_manifest.csv")
    shutil.copy2(source_paths["camera_rectified_json"], metadata_dir / "camera_rectified.json")
    shutil.copy2(source_paths["lidar_slam_ply"], output_dir / "global_map_slam_odom.ply")


def infer_scene_name(scene_dir: Path) -> str:
    name = scene_dir.name
    if name.endswith(PURE_HEADERSTAMP_SUFFIX):
        return name[: -len(PURE_HEADERSTAMP_SUFFIX)]
    return name


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


def convert_pure_headerstamp_scene(
    scene_dir: Path,
    output_dir: Path,
    *,
    points_ply: Optional[Path] = None,
    max_points: int = 3_000_000,
    copy_images: bool = False,
    overwrite: bool = False,
    images_subdir: str = "images",
) -> Dict[str, object]:
    scene_dir = scene_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    source_paths = ensure_pure_headerstamp_paths(scene_dir, points_ply.expanduser().resolve() if points_ply else None)
    prepare_output_dir(output_dir, overwrite=overwrite)

    sparse_dir = output_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / images_subdir

    rectified_camera = load_json(source_paths["camera_rectified_json"])
    association_rows = read_csv_rows(source_paths["frame_associations_csv"])
    slam_manifest_rows = read_csv_rows(source_paths["slam_frames_manifest_csv"])
    slam_matched_rows = select_reliable_slam_matched_rows(association_rows, slam_manifest_rows)
    slam_frame_ids = [row["frame_id"] for row in slam_matched_rows]
    pose_rows = filter_pose_rows_by_frame_ids(read_csv_rows(source_paths["camera_poses_csv"]), slam_frame_ids)
    selected_image_names = [f"{frame_id}.jpg" for frame_id in slam_frame_ids]

    images_meta = link_or_copy_selected_images(
        source_paths["images_rectified_dir"],
        images_dir,
        selected_image_names,
        copy_images,
    )
    width, height, fx, fy, cx, cy = pinhole_from_k_like(rectified_camera)
    write_pure_headerstamp_cameras_txt(
        sparse_dir / "cameras.txt",
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )

    image_records = build_pure_headerstamp_image_records(pose_rows, images_dir)
    write_pure_headerstamp_images_txt(sparse_dir / "images.txt", image_records)
    xyz, rgb = read_ply_points(source_paths["lidar_slam_ply"])
    points_meta = write_downsampled_colmap_points_with_ply(
        sparse_dir / "points3D.txt",
        sparse_dir / "points3D.ply",
        xyz,
        rgb,
        max_points=max_points,
    )
    copy_support_files(source_paths, output_dir)

    metadata = {
        "scene_name": infer_scene_name(scene_dir),
        "source_scene_dir": str(scene_dir),
        "colmap_export_dir": str(output_dir),
        "images": images_meta,
        "camera": {"camera_id": 1, "model": "PINHOLE", "width": int(width), "height": int(height)},
        "poses": {"frame_count": len(image_records)},
        "points3D": points_meta,
        "slam_matched_frame_count": len(slam_frame_ids),
        "format": "COLMAP-compatible text model",
        "pose_direction": "poses/camera_poses.csv T_odom_from_camera converted to COLMAP world-to-camera qvec/tvec",
        "notes": [
            "World frame follows the odom frame from poses/camera_poses.csv.",
            "points3D.txt is populated from lidar/global_map_slam_odom.ply.",
            "sparse/0/points3D.ply contains the same voxel-downsampled points for CloudCompare inspection.",
            "Lidar points do not have COLMAP observation tracks, so TRACK[] is left empty.",
        ],
        "baseline_type": "SLAM-pose/LiDAR-points COLMAP-compatible scene.",
    }
    write_json(output_dir / "manifest.json", metadata)
    return metadata


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
    max_points: int = 3_000_000,
    copy_images: bool = False,
    allow_pinhole_approximation: bool = False,
    overwrite: bool = False,
    images_subdir: str = "images",
) -> Dict[str, object]:
    """Export a pure-headerstamp Odin scene as a COLMAP text model."""
    if (scene_dir / "images_rectified").exists():
        return convert_pure_headerstamp_scene(
            scene_dir,
            output_dir,
            points_ply=points_ply,
            max_points=max_points,
            copy_images=copy_images,
            overwrite=overwrite,
            images_subdir=images_subdir,
        )

    # Legacy layout fallback retained for older processed scenes.
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

    camera_data = load_json(scene_dir / "intrinsics" / "camera.json")
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
    points_meta = write_downsampled_colmap_points_with_ply(
        sparse_dir / "points3D.txt",
        sparse_dir / "points3D.ply",
        xyz,
        rgb,
        max_points=max_points,
    )

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
        "points3D": points_meta,
        "source_points": int(points_meta["source_points"]),
        "written_points": int(points_meta["written_points"]),
        "num_images": len(image_records),
        "copy_images": bool(copy_images),
        "baseline_type": "SLAM-pose/LiDAR-points COLMAP-compatible scene, not a pure RGB-only COLMAP SfM baseline.",
    }
    write_json(output_dir / "slam_to_colmap_report.json", metadata)
    return metadata
