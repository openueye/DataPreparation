#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from data_preparation.rosbag_to_3dgs.db import require_topic_id
    from data_preparation.rosbag_to_3dgs.messages import parse_compressed_image, parse_odometry, parse_pointcloud2
    from data_preparation.shared.calibration import parse_camera_lidar_calibration
    from data_preparation.shared.io import write_json
    from data_preparation.shared.pointcloud import decode_ros2_xyz_points, write_ascii_ply
    from data_preparation.shared.poses import matrix_to_quaternion_xyzw, pose_to_matrix
    from data_preparation.shared.timing import nearest_indices
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from data_preparation.rosbag_to_3dgs.db import require_topic_id
    from data_preparation.rosbag_to_3dgs.messages import parse_compressed_image, parse_odometry, parse_pointcloud2
    from data_preparation.shared.calibration import parse_camera_lidar_calibration
    from data_preparation.shared.io import write_json
    from data_preparation.shared.pointcloud import decode_ros2_xyz_points, write_ascii_ply
    from data_preparation.shared.poses import matrix_to_quaternion_xyzw, pose_to_matrix
    from data_preparation.shared.timing import nearest_indices

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
DEFAULT_BAG_ROOT = REPO_ROOT / "03_Datasets" / "001_rosbags"
DEFAULT_CALIBRATION = DEFAULT_BAG_ROOT / "cam_in_ex.txt"


def parse_calibration(path: Path) -> Dict[str, Any]:
    return parse_camera_lidar_calibration(path, strict_matrix=True)


def require_non_empty_rows(rows: List[Tuple[int, bytes]], label: str, db_path: Path) -> None:
    if rows:
        return
    raise ValueError(f"No messages found for {label} in {db_path.name}. Check topic selection and bag health.")


def decode_xyz_points(pointcloud: Dict[str, Any]) -> np.ndarray:
    return decode_ros2_xyz_points(pointcloud)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a ROS bag into a 3DGS-ready LiDAR scene directory.")
    parser.add_argument(
        "--bag-dir",
        type=Path,
        required=True,
        help="Bag folder containing metadata.yaml and a .db3 file.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Target dataset directory.")
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
        help="Path to cam_in_ex.txt",
    )
    parser.add_argument("--image-topic", default="/odin1/image/compressed")
    parser.add_argument("--odom-topic", default="/odin1/odometry")
    parser.add_argument("--cloud-topic", default="/odin1/cloud_raw")
    parser.add_argument("--limit-images", type=int, default=None, help="Optional cap for exported images.")
    parser.add_argument(
        "--extrinsic-direction",
        choices=["camera_from_lidar", "lidar_from_camera"],
        default="camera_from_lidar",
        help="How to interpret the calibration matrix in cam_in_ex.txt.",
    )
    parser.add_argument(
        "--export-cloud-frames",
        action="store_true",
        help="Export matched raw LiDAR frames as .npy and build a sparse global_map.ply.",
    )
    parser.add_argument(
        "--keep-nonoverlap",
        action="store_true",
        help="Keep images outside the odometry/cloud overlap window. By default they are dropped.",
    )
    parser.add_argument(
        "--cloud-stride",
        type=int,
        default=8,
        help="Subsample factor used when accumulating a lightweight global map.",
    )
    args = parser.parse_args()

    bag_dir = args.bag_dir.resolve()
    output_dir = args.output_dir.resolve()
    db_path = next(iter(sorted(bag_dir.glob("*.db3"))), None)
    if db_path is None:
        raise FileNotFoundError(f"No .db3 file found under {bag_dir}")

    calibration = parse_calibration(args.calibration.resolve())
    t_camera_from_lidar = calibration["T_camera_from_lidar"]
    if args.extrinsic_direction == "camera_from_lidar":
        t_lidar_from_camera = np.linalg.inv(t_camera_from_lidar)
    else:
        t_lidar_from_camera = t_camera_from_lidar
        t_camera_from_lidar = np.linalg.inv(t_lidar_from_camera)

    connection = sqlite3.connect(str(db_path))
    topics = {
        name: topic_id
        for topic_id, name in connection.execute("SELECT id, name FROM topics")
    }
    image_topic_id = require_topic_id(topics, args.image_topic, db_path)
    odom_topic_id = require_topic_id(topics, args.odom_topic, db_path)
    cloud_topic_id = require_topic_id(topics, args.cloud_topic, db_path)

    image_rows = connection.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp", (image_topic_id,)
    ).fetchall()
    odom_rows = connection.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp", (odom_topic_id,)
    ).fetchall()
    cloud_rows = connection.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp", (cloud_topic_id,)
    ).fetchall()
    connection.close()

    require_non_empty_rows(image_rows, f"image topic '{args.image_topic}'", db_path)
    require_non_empty_rows(odom_rows, f"odometry topic '{args.odom_topic}'", db_path)
    require_non_empty_rows(cloud_rows, f"cloud topic '{args.cloud_topic}'", db_path)

    if args.limit_images is not None:
        image_rows = image_rows[: args.limit_images]
        if not image_rows:
            raise ValueError("--limit-images filtered out all frames; choose a larger value.")

    odom_timestamps = np.array([row[0] for row in odom_rows], dtype=np.int64)
    cloud_timestamps = np.array([row[0] for row in cloud_rows], dtype=np.int64)
    if not args.keep_nonoverlap:
        overlap_start = odom_timestamps[0]
        overlap_end = odom_timestamps[-1]
        if args.export_cloud_frames:
            overlap_start = max(overlap_start, int(cloud_timestamps[0]))
            overlap_end = min(overlap_end, int(cloud_timestamps[-1]))
        image_rows = [row for row in image_rows if overlap_start <= row[0] <= overlap_end]
        if not image_rows:
            raise ValueError(
                "No images remain after overlap filtering. Re-run with --keep-nonoverlap or inspect timestamp coverage."
            )
    image_timestamps = np.array([row[0] for row in image_rows], dtype=np.int64)

    odom_match_idx = nearest_indices(image_timestamps, odom_timestamps)
    cloud_match_idx = nearest_indices(image_timestamps, cloud_timestamps)

    images_dir = output_dir / "images"
    poses_dir = output_dir / "poses"
    intrinsics_dir = output_dir / "intrinsics"
    lidar_dir = output_dir / "lidar"
    lidar_frames_dir = lidar_dir / "frames"
    transforms_dir = output_dir / "transforms"
    metadata_dir = output_dir / "metadata"
    for path in (images_dir, poses_dir, intrinsics_dir, transforms_dir, metadata_dir):
        path.mkdir(parents=True, exist_ok=True)
    if args.export_cloud_frames:
        lidar_frames_dir.mkdir(parents=True, exist_ok=True)

    pose_rows_out: List[List[Any]] = []
    assoc_rows_out: List[List[Any]] = []
    global_map_chunks: List[np.ndarray] = []

    for image_index, (image_timestamp, image_blob) in enumerate(image_rows):
        image_msg = parse_compressed_image(image_blob)
        odom_timestamp, odom_blob = odom_rows[int(odom_match_idx[image_index])]
        odom_msg = parse_odometry(odom_blob)
        t_world_from_base = pose_to_matrix(odom_msg["position"], odom_msg["orientation_xyzw"])
        t_world_from_camera = t_world_from_base @ t_lidar_from_camera
        q_world_from_camera = matrix_to_quaternion_xyzw(t_world_from_camera[:3, :3])

        file_stem = f"{image_index:06d}"
        image_extension = ".jpg" if image_msg["format"].lower().startswith("jp") else ".png"
        image_path = images_dir / f"{file_stem}{image_extension}"
        image_path.write_bytes(image_msg["data"])

        pose_rows_out.append(
            [
                file_stem,
                image_timestamp,
                *t_world_from_camera[:3, 3].tolist(),
                *q_world_from_camera.tolist(),
                *t_world_from_camera.reshape(-1).tolist(),
            ]
        )
        assoc_row = [
            file_stem,
            image_timestamp,
            odom_timestamp,
            int(abs(image_timestamp - odom_timestamp)),
        ]

        if args.export_cloud_frames:
            cloud_timestamp, cloud_blob = cloud_rows[int(cloud_match_idx[image_index])]
            cloud_msg = parse_pointcloud2(cloud_blob)
            xyz = decode_xyz_points(cloud_msg)
            cloud_path = lidar_frames_dir / f"{file_stem}.npy"
            np.save(cloud_path, xyz)
            assoc_row.extend([cloud_timestamp, int(abs(image_timestamp - cloud_timestamp)), str(cloud_path.relative_to(output_dir))])

            sampled = xyz[:: max(args.cloud_stride, 1)]
            if len(sampled):
                hom = np.concatenate([sampled, np.ones((len(sampled), 1), dtype=np.float64)], axis=1)
                world_points = (t_world_from_base @ hom.T).T[:, :3]
                global_map_chunks.append(world_points)
        assoc_rows_out.append(assoc_row)

    camera_json = {
        "camera_name": calibration["camera_name"],
        "camera_model": calibration["camera_params"].get("cam_model"),
        "image_model": "raw_distorted",
        "undistorted": False,
        "rectified": False,
        "colmap_training_ready": False,
        "requires_rectification_for_colmap": calibration["camera_params"].get("cam_model") not in {"PINHOLE", "SIMPLE_PINHOLE"},
        "width": calibration["camera_params"].get("image_width"),
        "height": calibration["camera_params"].get("image_height"),
        "K_like": calibration["K_like"],
        "distortion": calibration["distortion"],
        "notes": [
            "The source camera model is FishPoly.",
            "Undistort to pinhole before feeding a COLMAP-oriented 3DGS baseline.",
            "Quaternion values in poses.csv are derived from T_world_from_camera.",
        ],
    }
    write_json(intrinsics_dir / "camera.json", camera_json)

    tf_payload = {
        "world_frame": "odom",
        "base_frame": "odin1_base_link",
        "imu_frame": "imu_link",
        "recorded_image_frame_id": "",
        "cloud_frame": args.cloud_topic,
        "calibration_matrix_name": calibration["matrix_name"],
        "extrinsic_direction_used": args.extrinsic_direction,
        "T_camera_from_lidar": t_camera_from_lidar.tolist(),
        "T_lidar_from_camera": t_lidar_from_camera.tolist(),
        "assumption": "cloud_raw points are already expressed in odin1_base_link or a rigidly aligned lidar/base frame.",
    }
    write_json(transforms_dir / "tf_chain.json", tf_payload)

    with (poses_dir / "poses.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame_id",
                "image_timestamp_ns",
                "tx",
                "ty",
                "tz",
                "qx",
                "qy",
                "qz",
                "qw",
                "T_world_from_camera_rowmajor_00",
                "T_world_from_camera_rowmajor_01",
                "T_world_from_camera_rowmajor_02",
                "T_world_from_camera_rowmajor_03",
                "T_world_from_camera_rowmajor_10",
                "T_world_from_camera_rowmajor_11",
                "T_world_from_camera_rowmajor_12",
                "T_world_from_camera_rowmajor_13",
                "T_world_from_camera_rowmajor_20",
                "T_world_from_camera_rowmajor_21",
                "T_world_from_camera_rowmajor_22",
                "T_world_from_camera_rowmajor_23",
                "T_world_from_camera_rowmajor_30",
                "T_world_from_camera_rowmajor_31",
                "T_world_from_camera_rowmajor_32",
                "T_world_from_camera_rowmajor_33",
            ]
        )
        writer.writerows(pose_rows_out)

    with (metadata_dir / "associations.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = ["frame_id", "image_timestamp_ns", "odom_timestamp_ns", "image_to_odom_dt_ns"]
        if args.export_cloud_frames:
            header.extend(["cloud_timestamp_ns", "image_to_cloud_dt_ns", "cloud_path"])
        writer.writerow(header)
        writer.writerows(assoc_rows_out)

    global_map_points = 0
    if args.export_cloud_frames and global_map_chunks:
        global_map = np.concatenate(global_map_chunks, axis=0)
        global_map_points = int(len(global_map))
        write_ascii_ply(lidar_dir / "global_map.ply", global_map)

    scene_meta = {
        "scene_name": output_dir.name,
        "source_type": "rosbag",
        "init_mode": "lidar",
        "raw_source": str(bag_dir),
        "generated_by": "rosbag_to_3dgs",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bag_db_path": str(db_path),
        "calibration_path": str(args.calibration.resolve()),
        "image_topic": args.image_topic,
        "odom_topic": args.odom_topic,
        "cloud_topic": args.cloud_topic,
        "extrinsic_direction": args.extrinsic_direction,
        "cloud_stride": int(args.cloud_stride),
        "export_cloud_frames": bool(args.export_cloud_frames),
        "keep_nonoverlap": bool(args.keep_nonoverlap),
        "limit_images": args.limit_images,
        "num_exported_images": len(pose_rows_out),
        "num_associations": len(assoc_rows_out),
        "global_map_points": global_map_points,
    }
    write_json(output_dir / "scene_meta.json", scene_meta)


if __name__ == "__main__":
    main()
