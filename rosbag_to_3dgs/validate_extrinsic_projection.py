#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from data_preparation.rosbag_to_3dgs.db import require_topic_id
    from data_preparation.rosbag_to_3dgs.messages import decode_compressed_image, parse_pointcloud2
    from data_preparation.shared.calibration import parse_camera_lidar_calibration
    from data_preparation.shared.io import require_cv2, write_json
    from data_preparation.shared.pointcloud import decode_ros2_xyz_points
    from data_preparation.shared.timing import nearest_indices
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from data_preparation.rosbag_to_3dgs.db import require_topic_id
    from data_preparation.rosbag_to_3dgs.messages import decode_compressed_image, parse_pointcloud2
    from data_preparation.shared.calibration import parse_camera_lidar_calibration
    from data_preparation.shared.io import require_cv2, write_json
    from data_preparation.shared.pointcloud import decode_ros2_xyz_points
    from data_preparation.shared.timing import nearest_indices

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
DEFAULT_BAG_ROOT = REPO_ROOT / "03_Datasets" / "001_rosbags"
DEFAULT_CALIBRATION = DEFAULT_BAG_ROOT / "cam_in_ex.txt"


def parse_compressed_image(blob: bytes) -> Dict[str, Any]:
    cv2_module = require_cv2("rosbag projection validation")
    return decode_compressed_image(blob, mode=cv2_module.IMREAD_COLOR)


def parse_calibration(path: Path) -> Dict[str, Any]:
    calibration = parse_camera_lidar_calibration(path, strict_matrix=True)
    calibration["K_like"] = np.asarray(calibration["K_like"], dtype=np.float64)
    return calibration


def decode_xyz_points(pointcloud: Dict[str, Any]) -> np.ndarray:
    return decode_ros2_xyz_points(pointcloud)


def build_projection_overlay(
    image: np.ndarray,
    xyz_lidar: np.ndarray,
    t_camera_from_lidar: np.ndarray,
    k_matrix: np.ndarray,
    max_points: int,
) -> Tuple[np.ndarray, Dict[str, int]]:
    cv2_module = require_cv2()
    if len(xyz_lidar) == 0:
        return image.copy(), {"input_points": 0, "front_points": 0, "projected_points": 0}

    if len(xyz_lidar) > max_points:
        stride = max(1, len(xyz_lidar) // max_points)
        xyz_lidar = xyz_lidar[::stride]

    hom = np.concatenate([xyz_lidar, np.ones((len(xyz_lidar), 1), dtype=np.float64)], axis=1)
    xyz_camera = (t_camera_from_lidar @ hom.T).T[:, :3]
    positive = xyz_camera[:, 2] > 1e-4
    xyz_camera = xyz_camera[positive]

    if len(xyz_camera) == 0:
        return image.copy(), {"input_points": int(len(hom)), "front_points": 0, "projected_points": 0}

    projected = (k_matrix @ xyz_camera.T).T
    uv = projected[:, :2] / projected[:, 2:3]
    h, w = image.shape[:2]
    inside = (
        (uv[:, 0] >= 0)
        & (uv[:, 0] < w)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < h)
    )
    uv_inside = uv[inside]
    z_inside = xyz_camera[inside, 2]

    overlay = image.copy()
    if len(uv_inside):
        z_min = float(np.percentile(z_inside, 5))
        z_max = float(np.percentile(z_inside, 95))
        z_span = max(z_max - z_min, 1e-6)
        norm = np.clip((z_inside - z_min) / z_span, 0.0, 1.0)
        colors = cv2_module.applyColorMap((255.0 * (1.0 - norm)).astype(np.uint8), cv2_module.COLORMAP_TURBO)
        for (u, v), color in zip(uv_inside.astype(np.int32), colors[:, 0, :]):
            cv2_module.circle(overlay, (int(u), int(v)), 1, tuple(int(c) for c in color.tolist()), -1)

    stats = {
        "input_points": int(len(hom)),
        "front_points": int(len(xyz_camera)),
        "projected_points": int(len(uv_inside)),
    }
    return overlay, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create approximate LiDAR-to-image projection overlays for extrinsic sanity checking."
    )
    parser.add_argument("--bag-dir", type=Path, required=True, help="Bag folder containing a .db3 file.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where overlays will be written.")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION, help="Path to cam_in_ex.txt")
    parser.add_argument("--image-topic", default="/odin1/image/compressed")
    parser.add_argument("--cloud-topic", default="/odin1/cloud_raw")
    parser.add_argument(
        "--extrinsic-direction",
        choices=["camera_from_lidar", "lidar_from_camera"],
        default="camera_from_lidar",
        help="How to interpret the calibration matrix in cam_in_ex.txt.",
    )
    parser.add_argument("--num-samples", type=int, default=8, help="How many overlay frames to export.")
    parser.add_argument("--frame-stride", type=int, default=20, help="Stride applied after overlap filtering.")
    parser.add_argument("--max-points", type=int, default=12000, help="Maximum points drawn per overlay.")
    args = parser.parse_args()

    bag_dir = args.bag_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    db_path = next(iter(sorted(bag_dir.glob("*.db3"))), None)
    if db_path is None:
        raise FileNotFoundError(f"No .db3 file found under {bag_dir}")

    calibration = parse_calibration(args.calibration.resolve())
    t_camera_from_lidar = calibration["T_camera_from_lidar"]
    if args.extrinsic_direction == "lidar_from_camera":
        t_camera_from_lidar = np.linalg.inv(t_camera_from_lidar)
    k_matrix = calibration["K_like"]

    connection = sqlite3.connect(str(db_path))
    topics = {name: topic_id for topic_id, name in connection.execute("SELECT id, name FROM topics")}
    image_topic_id = require_topic_id(topics, args.image_topic, db_path)
    cloud_topic_id = require_topic_id(topics, args.cloud_topic, db_path)

    image_rows = connection.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp",
        (image_topic_id,),
    ).fetchall()
    cloud_rows = connection.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp",
        (cloud_topic_id,),
    ).fetchall()
    connection.close()

    if not image_rows:
        raise ValueError(f"No messages found for image topic '{args.image_topic}' in {db_path.name}.")
    if not cloud_rows:
        raise ValueError(f"No messages found for cloud topic '{args.cloud_topic}' in {db_path.name}.")

    image_timestamps = np.array([row[0] for row in image_rows], dtype=np.int64)
    cloud_timestamps = np.array([row[0] for row in cloud_rows], dtype=np.int64)
    overlap_start = max(int(image_timestamps[0]), int(cloud_timestamps[0]))
    overlap_end = min(int(image_timestamps[-1]), int(cloud_timestamps[-1]))
    overlap_indices = np.where((image_timestamps >= overlap_start) & (image_timestamps <= overlap_end))[0]
    if len(overlap_indices) == 0:
        raise ValueError("No overlapping image/cloud timestamps found for extrinsic validation.")
    sampled_indices = overlap_indices[:: max(args.frame_stride, 1)][: args.num_samples]
    if len(sampled_indices) == 0:
        raise ValueError("No validation samples selected. Check --num-samples and --frame-stride.")
    cloud_match_idx = nearest_indices(image_timestamps[sampled_indices], cloud_timestamps)

    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "bag_dir": str(bag_dir),
        "calibration": str(args.calibration.resolve()),
        "extrinsic_direction": args.extrinsic_direction,
        "camera_model_note": "Projection uses the K-like matrix as an approximate pinhole sanity check; FishPoly distortion is ignored.",
        "samples": [],
    }

    for out_idx, image_idx in enumerate(sampled_indices):
        image_timestamp, image_blob = image_rows[int(image_idx)]
        cloud_timestamp, cloud_blob = cloud_rows[int(cloud_match_idx[out_idx])]

        image_msg = parse_compressed_image(image_blob)
        cloud_msg = parse_pointcloud2(cloud_blob)
        xyz_lidar = decode_xyz_points(cloud_msg)
        overlay, stats = build_projection_overlay(
            image_msg["image"],
            xyz_lidar,
            t_camera_from_lidar,
            k_matrix,
            args.max_points,
        )

        overlay_name = f"{out_idx:03d}_{image_timestamp}.png"
        overlay_path = overlay_dir / overlay_name
        require_cv2().imwrite(str(overlay_path), overlay)

        summary["samples"].append(
            {
                "overlay": str(overlay_path.relative_to(output_dir)),
                "image_timestamp_ns": int(image_timestamp),
                "cloud_timestamp_ns": int(cloud_timestamp),
                "dt_ms": float(abs(image_timestamp - cloud_timestamp) / 1e6),
                "image_shape": [int(v) for v in image_msg["image"].shape],
                "cloud_frame_id": cloud_msg["frame_id"],
                **stats,
            }
        )

    projected_counts = [sample["projected_points"] for sample in summary["samples"]]
    summary["aggregate"] = {
        "num_samples": len(summary["samples"]),
        "mean_projected_points": float(np.mean(projected_counts)) if projected_counts else 0.0,
        "min_projected_points": int(min(projected_counts)) if projected_counts else 0,
        "max_projected_points": int(max(projected_counts)) if projected_counts else 0,
    }

    write_json(output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
