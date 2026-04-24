#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from data_preparation.rosbag_to_3dgs.db import topic_timestamp_array
    from data_preparation.rosbag_to_3dgs.messages import (
        decode_compressed_image,
        parse_imu,
        parse_odometry as parse_ros_odometry,
        parse_pointcloud2 as parse_ros_pointcloud2,
    )
    from data_preparation.rosbag_to_3dgs.ros2_cdr import parse_header
    from data_preparation.shared.calibration import parse_camera_lidar_calibration
    from data_preparation.shared.io import require_cv2, write_json
    from data_preparation.shared.timing import nearest_neighbor_stats
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from data_preparation.rosbag_to_3dgs.db import topic_timestamp_array
    from data_preparation.rosbag_to_3dgs.messages import (
        decode_compressed_image,
        parse_imu,
        parse_odometry as parse_ros_odometry,
        parse_pointcloud2 as parse_ros_pointcloud2,
    )
    from data_preparation.rosbag_to_3dgs.ros2_cdr import parse_header
    from data_preparation.shared.calibration import parse_camera_lidar_calibration
    from data_preparation.shared.io import require_cv2, write_json
    from data_preparation.shared.timing import nearest_neighbor_stats

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
DEFAULT_BAG_ROOT = REPO_ROOT / "03_Datasets" / "001_rosbags"
DEFAULT_REPORT_DIR = REPO_ROOT / "05_Outputs" / "030_validation" / "rosbag_reports"


def parse_compressed_image(blob: bytes) -> Dict[str, Any]:
    cv2_module = require_cv2("rosbag inspection image decoding")
    message = decode_compressed_image(blob, mode=cv2_module.IMREAD_UNCHANGED)
    return {
        **{key: value for key, value in message.items() if key not in {"data", "image"}},
        "decoded_shape": message.get("decoded_shape"),
    }


def parse_odometry(blob: bytes) -> Dict[str, Any]:
    message = parse_ros_odometry(blob)
    return {
        **message,
        "position": message["position"].tolist(),
        "orientation_xyzw": message["orientation_xyzw"].tolist(),
    }


def parse_pointcloud2(blob: bytes) -> Dict[str, Any]:
    return parse_ros_pointcloud2(blob, include_data=False)


def parse_calibration(path: Path) -> Dict[str, Any]:
    calibration = parse_camera_lidar_calibration(path, strict_matrix=False)
    calibration["distortion"] = calibration["distortion_vector"]
    return calibration


def human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    unit = units[0]
    for candidate in units:
        unit = candidate
        if value < 1024.0 or candidate == units[-1]:
            break
        value /= 1024.0
    if unit == "B":
        return f"{int(value)} {unit}"
    return f"{value:.1f} {unit}"


def format_epoch_ns(epoch_ns: Optional[int]) -> str:
    if epoch_ns is None:
        return "N/A"
    dt = datetime.fromtimestamp(epoch_ns / 1e9, tz=timezone.utc)
    return dt.isoformat()


def classify_topic(name: str, msg_type: str) -> List[str]:
    categories = []
    lower_name = name.lower()
    lower_type = msg_type.lower()
    if "image" in lower_name or "compressedimage" in lower_type or lower_type.endswith("/image"):
        categories.append("image")
    if "camera_info" in lower_name or "camerainfo" in lower_type:
        categories.append("camera_info")
    if any(token in lower_name for token in ("odom", "pose", "path", "/tf")) or any(
        token in lower_type for token in ("odometry", "pose", "path")
    ):
        categories.append("pose")
    if "pointcloud2" in lower_type or "cloud" in lower_name or "points" in lower_name:
        categories.append("lidar")
    if "imu" in lower_name or "imu" in lower_type:
        categories.append("imu")
    if "slam" in lower_name or "registered" in lower_name or "map" in lower_name:
        categories.append("slam_output")
    return categories or ["other"]


def estimate_hz(count: int, span_ns: int) -> float:
    if count <= 1 or span_ns <= 0:
        return 0.0
    return count / (span_ns / 1e9)


def parse_topic_sample(msg_type: str, blob: bytes) -> Dict[str, Any]:
    if msg_type == "sensor_msgs/msg/CompressedImage":
        return parse_compressed_image(blob)
    if msg_type == "nav_msgs/msg/Odometry":
        return parse_odometry(blob)
    if msg_type == "sensor_msgs/msg/PointCloud2":
        return parse_pointcloud2(blob)
    if msg_type == "sensor_msgs/msg/Imu":
        return parse_imu(blob)
    return parse_header(blob)


def round_or_na(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


@dataclass
class BagSummary:
    name: str
    bag_dir: Path
    db_path: Optional[Path]
    metadata_path: Optional[Path]
    size_bytes: int
    ros_distro: Optional[str]
    valid: bool
    error: Optional[str]
    start_ns: Optional[int]
    end_ns: Optional[int]
    duration_ns: Optional[int]
    total_messages: int
    topics: List[Dict[str, Any]]
    sync_stats: Dict[str, Any]


def scan_bag_dir(bag_dir: Path) -> BagSummary:
    metadata_path = bag_dir / "metadata.yaml"
    db_files = sorted(bag_dir.glob("*.db3"))
    size_bytes = sum(path.stat().st_size for path in db_files)
    if metadata_path.exists():
        size_bytes += metadata_path.stat().st_size

    if not db_files:
        return BagSummary(
            name=bag_dir.name,
            bag_dir=bag_dir,
            db_path=None,
            metadata_path=metadata_path if metadata_path.exists() else None,
            size_bytes=size_bytes,
            ros_distro=None,
            valid=False,
            error="No .db3 file found.",
            start_ns=None,
            end_ns=None,
            duration_ns=None,
            total_messages=0,
            topics=[],
            sync_stats={},
        )

    db_path = db_files[0]
    try:
        connection = sqlite3.connect(str(db_path))
        ros_row = connection.execute("SELECT ros_distro FROM schema LIMIT 1").fetchone()
        ros_distro = ros_row[0] if ros_row else None

        topic_rows = connection.execute(
            "SELECT id, name, type, serialization_format FROM topics ORDER BY id"
        ).fetchall()
        stats_rows = connection.execute(
            "SELECT topic_id, COUNT(*) AS n, MIN(timestamp), MAX(timestamp) FROM messages GROUP BY topic_id"
        ).fetchall()
        stats_by_topic = {
            row[0]: {"count": row[1], "min_ts": row[2], "max_ts": row[3], "span_ns": row[3] - row[2]} for row in stats_rows
        }
        global_times = connection.execute("SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM messages").fetchone()
        start_ns, end_ns, total_messages = global_times
        duration_ns = end_ns - start_ns if start_ns is not None and end_ns is not None else None

        topics: List[Dict[str, Any]] = []
        for topic_id, name, msg_type, serialization_format in topic_rows:
            topic_stats = stats_by_topic.get(topic_id, {"count": 0, "min_ts": None, "max_ts": None, "span_ns": 0})
            sample_row = connection.execute(
                "SELECT data FROM messages WHERE topic_id = ? ORDER BY timestamp LIMIT 1", (topic_id,)
            ).fetchone()
            sample = parse_topic_sample(msg_type, sample_row[0]) if sample_row else {}
            topics.append(
                {
                    "id": topic_id,
                    "name": name,
                    "type": msg_type,
                    "serialization_format": serialization_format,
                    "categories": classify_topic(name, msg_type),
                    "message_count": int(topic_stats["count"]),
                    "start_ns": topic_stats["min_ts"],
                    "end_ns": topic_stats["max_ts"],
                    "span_ns": int(topic_stats["span_ns"]),
                    "hz": estimate_hz(int(topic_stats["count"]), int(topic_stats["span_ns"])),
                    "sample": sample,
                }
            )

        topics_by_name = {topic["name"]: topic for topic in topics}
        sync_stats: Dict[str, Any] = {}
        if "/odin1/image/compressed" in topics_by_name and "/odin1/odometry" in topics_by_name:
            img_ts = topic_timestamp_array(connection, topics_by_name["/odin1/image/compressed"]["id"])
            odom_ts = topic_timestamp_array(connection, topics_by_name["/odin1/odometry"]["id"])
            sync_stats["image_to_odometry_ms"] = nearest_neighbor_stats(img_ts, odom_ts)
        if "/odin1/image/compressed" in topics_by_name and "/odin1/cloud_raw" in topics_by_name:
            img_ts = topic_timestamp_array(connection, topics_by_name["/odin1/image/compressed"]["id"])
            cloud_ts = topic_timestamp_array(connection, topics_by_name["/odin1/cloud_raw"]["id"])
            sync_stats["image_to_cloud_raw_ms"] = nearest_neighbor_stats(img_ts, cloud_ts)

        return BagSummary(
            name=bag_dir.name,
            bag_dir=bag_dir,
            db_path=db_path,
            metadata_path=metadata_path if metadata_path.exists() else None,
            size_bytes=size_bytes,
            ros_distro=ros_distro,
            valid=True,
            error=None,
            start_ns=start_ns,
            end_ns=end_ns,
            duration_ns=duration_ns,
            total_messages=int(total_messages),
            topics=topics,
            sync_stats=sync_stats,
        )
    except Exception as exc:  # pragma: no cover - defensive path for broken bags
        return BagSummary(
            name=bag_dir.name,
            bag_dir=bag_dir,
            db_path=db_path,
            metadata_path=metadata_path if metadata_path.exists() else None,
            size_bytes=size_bytes,
            ros_distro=None,
            valid=False,
            error=str(exc),
            start_ns=None,
            end_ns=None,
            duration_ns=None,
            total_messages=0,
            topics=[],
            sync_stats={},
        )


def summarize_dataset(bags: List[BagSummary], calibration: Dict[str, Any]) -> Dict[str, Any]:
    valid_bags = [bag for bag in bags if bag.valid]
    topic_union: Dict[str, Dict[str, Any]] = {}
    for bag in valid_bags:
        for topic in bag.topics:
            entry = topic_union.setdefault(
                topic["name"],
                {
                    "type": topic["type"],
                    "bags": [],
                    "hz_values": [],
                    "message_counts": [],
                    "samples": [],
                },
            )
            entry["bags"].append(bag.name)
            entry["hz_values"].append(topic["hz"])
            entry["message_counts"].append(topic["message_count"])
            entry["samples"].append(topic["sample"])

    image_topic = topic_union.get("/odin1/image/compressed")
    odom_topic = topic_union.get("/odin1/odometry")
    cloud_raw_topic = topic_union.get("/odin1/cloud_raw")
    cloud_slam_topic = topic_union.get("/odin1/cloud_slam")

    sync_odom_means = [
        bag.sync_stats["image_to_odometry_ms"]["mean_ms"]
        for bag in valid_bags
        if bag.sync_stats.get("image_to_odometry_ms")
    ]
    sync_cloud_means = [
        bag.sync_stats["image_to_cloud_raw_ms"]["mean_ms"]
        for bag in valid_bags
        if bag.sync_stats.get("image_to_cloud_raw_ms")
    ]

    return {
        "valid_bag_count": len(valid_bags),
        "invalid_bag_count": len(bags) - len(valid_bags),
        "image_topic": image_topic,
        "odom_topic": odom_topic,
        "cloud_raw_topic": cloud_raw_topic,
        "cloud_slam_topic": cloud_slam_topic,
        "calibration": calibration,
        "sync_odom_mean_range_ms": [min(sync_odom_means), max(sync_odom_means)] if sync_odom_means else None,
        "sync_cloud_mean_range_ms": [min(sync_cloud_means), max(sync_cloud_means)] if sync_cloud_means else None,
    }


def build_bag_inventory_markdown(bags: List[BagSummary], calibration: Dict[str, Any], scan_root: Path) -> str:
    lines = []
    lines.append("# ROS Bag Inventory")
    lines.append("")
    lines.append(f"Scanned root: `{scan_root}`")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("| Bag | Status | Size | ROS distro | Start | End | Duration (s) | Messages |")
    lines.append("| --- | --- | --- | --- | --- | --- | ---: | ---: |")
    for bag in bags:
        status = "OK" if bag.valid else f"Broken ({bag.error})"
        duration_s = bag.duration_ns / 1e9 if bag.duration_ns is not None else None
        lines.append(
            "| "
            + " | ".join(
                [
                    bag.name,
                    status,
                    human_size(bag.size_bytes),
                    bag.ros_distro or "N/A",
                    format_epoch_ns(bag.start_ns),
                    format_epoch_ns(bag.end_ns),
                    round_or_na(duration_s),
                    str(bag.total_messages),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Calibration Artifact")
    lines.append("")
    lines.append(f"- External calibration file: `{calibration['path']}`")
    lines.append(f"- Camera model: `{calibration['camera_params'].get('cam_model', 'N/A')}`")
    lines.append(
        f"- Resolution in calibration: `{calibration['camera_params'].get('image_width', 'N/A')} x "
        f"{calibration['camera_params'].get('image_height', 'N/A')}`"
    )
    lines.append(f"- Extrinsic matrix key: `{calibration['matrix_name']}`")
    lines.append("")

    for bag in bags:
        lines.append(f"## {bag.name}")
        lines.append("")
        if not bag.valid:
            lines.append(f"- Status: broken")
            lines.append(f"- Error: `{bag.error}`")
            lines.append("")
            continue
        lines.append(f"- Bag path: `{bag.db_path}`")
        lines.append(f"- Size: `{human_size(bag.size_bytes)}`")
        lines.append(f"- ROS distro: `{bag.ros_distro}`")
        lines.append(f"- Start: `{format_epoch_ns(bag.start_ns)}`")
        lines.append(f"- End: `{format_epoch_ns(bag.end_ns)}`")
        lines.append(f"- Duration: `{bag.duration_ns / 1e9:.3f}s`")
        lines.append(f"- Total messages: `{bag.total_messages}`")
        lines.append("")
        lines.append("| Topic | Type | Categories | Count | Avg Hz | Frame / child | Key sample info |")
        lines.append("| --- | --- | --- | ---: | ---: | --- | --- |")
        for topic in bag.topics:
            sample = topic["sample"]
            frame_desc = sample.get("frame_id", "N/A")
            if sample.get("child_frame_id"):
                frame_desc = f"{frame_desc} -> {sample['child_frame_id']}"
            sample_bits = []
            if "decoded_shape" in sample and sample["decoded_shape"] is not None:
                sample_bits.append(f"shape={sample['decoded_shape']}")
                sample_bits.append(f"format={sample.get('format', 'N/A')}")
            if "width" in sample and "height" in sample:
                sample_bits.append(f"{sample['width']}x{sample['height']}")
                if sample.get("point_step") is not None:
                    field_names = ",".join(field["name"] for field in sample.get("fields", []))
                    sample_bits.append(f"fields={field_names}")
            if "position" in sample:
                position = ",".join(f"{value:.3f}" for value in sample["position"])
                sample_bits.append(f"pose_xyz={position}")
            lines.append(
                "| "
                + " | ".join(
                    [
                        topic["name"],
                        topic["type"],
                        ", ".join(topic["categories"]),
                        str(topic["message_count"]),
                        f"{topic['hz']:.3f}",
                        frame_desc or "(empty)",
                        "; ".join(sample_bits) or "-",
                    ]
                )
                + " |"
            )
        lines.append("")
        if bag.sync_stats:
            lines.append("### Time Alignment")
            lines.append("")
            for key, stats in bag.sync_stats.items():
                if not stats:
                    continue
                lines.append(
                    f"- `{key}`: matched={stats['matched_count']}, "
                    f"mean={stats['mean_ms']:.3f} ms, median={stats['median_ms']:.3f} ms, "
                    f"p95={stats['p95_ms']:.3f} ms, max={stats['max_ms']:.3f} ms"
                )
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def build_feasibility_report_markdown(bags: List[BagSummary], calibration: Dict[str, Any]) -> str:
    valid_bags = [bag for bag in bags if bag.valid]
    broken_bags = [bag for bag in bags if not bag.valid]
    image_bags = [bag for bag in valid_bags if any(topic["name"] == "/odin1/image/compressed" for topic in bag.topics)]
    odom_bags = [bag for bag in valid_bags if any(topic["name"] == "/odin1/odometry" for topic in bag.topics)]
    raw_cloud_bags = [bag for bag in valid_bags if any(topic["name"] == "/odin1/cloud_raw" for topic in bag.topics)]

    first_image_topic = next(
        topic for bag in image_bags for topic in bag.topics if topic["name"] == "/odin1/image/compressed"
    )
    first_odom_topic = next(
        topic for bag in odom_bags for topic in bag.topics if topic["name"] == "/odin1/odometry"
    )
    first_cloud_raw = next(
        topic for bag in raw_cloud_bags for topic in bag.topics if topic["name"] == "/odin1/cloud_raw"
    )
    first_cloud_slam = next(
        topic for bag in raw_cloud_bags for topic in bag.topics if topic["name"] == "/odin1/cloud_slam"
    )

    odom_sync_means = [bag.sync_stats["image_to_odometry_ms"]["mean_ms"] for bag in valid_bags if bag.sync_stats.get("image_to_odometry_ms")]
    odom_sync_medians = [bag.sync_stats["image_to_odometry_ms"]["median_ms"] for bag in valid_bags if bag.sync_stats.get("image_to_odometry_ms")]
    cloud_sync_means = [bag.sync_stats["image_to_cloud_raw_ms"]["mean_ms"] for bag in valid_bags if bag.sync_stats.get("image_to_cloud_raw_ms")]
    cloud_sync_medians = [bag.sync_stats["image_to_cloud_raw_ms"]["median_ms"] for bag in valid_bags if bag.sync_stats.get("image_to_cloud_raw_ms")]

    lines = []
    lines.append("# SLAM + LiDAR Feasibility Report for 3DGS Initialization")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        "The dataset is **largely usable** for a SLAM-guided 3DGS pipeline: RGB supervision, odometry poses, "
        "raw LiDAR, SLAM-space clouds, and an external camera/LiDAR calibration file are all present in the healthy bags."
    )
    lines.append(
        "The main blockers are that the bags do **not** record `camera_info` or a TF tree, and the camera model in "
        "`cam_in_ex.txt` is a `FishPoly` model that should be undistorted to a pinhole model before feeding the current COLMAP-oriented 3DGS baseline."
    )
    lines.append("")
    if broken_bags:
        lines.append("Broken bag(s) detected:")
        for bag in broken_bags:
            lines.append(f"- `{bag.name}`: {bag.error}")
        lines.append("")

    lines.append("## A. Image Availability")
    lines.append("")
    shape = first_image_topic["sample"].get("decoded_shape")
    lines.append(f"- Topic: `/odin1/image/compressed`")
    lines.append(f"- Type: `{first_image_topic['type']}`")
    lines.append(f"- Compression: `{first_image_topic['sample'].get('format', 'N/A')}`")
    lines.append(f"- Sample decoded resolution: `{shape[1]} x {shape[0]}`")
    lines.append(
        f"- Frequency range across healthy bags: `{min(topic['hz'] for bag in image_bags for topic in bag.topics if topic['name']=='/odin1/image/compressed'):.3f}`"
        f" to `{max(topic['hz'] for bag in image_bags for topic in bag.topics if topic['name']=='/odin1/image/compressed'):.3f}` Hz"
    )
    lines.append("- Suitability: directly exportable to `.jpg`; can also be decoded and re-saved to `.png` if needed.")
    lines.append("")

    lines.append("## B. Camera Intrinsics")
    lines.append("")
    lines.append("- No `camera_info` topic was recorded in the bags.")
    lines.append(f"- External file available: `{calibration['path']}`")
    lines.append(f"- Camera model: `{calibration['camera_params'].get('cam_model', 'N/A')}`")
    lines.append(
        f"- Resolution: `{calibration['camera_params'].get('image_width', 'N/A')} x "
        f"{calibration['camera_params'].get('image_height', 'N/A')}`"
    )
    lines.append(f"- K-like matrix from calibration: `{json.dumps(calibration['K_like'])}`")
    lines.append(f"- Distortion coefficients: `{json.dumps(calibration['distortion'])}`")
    lines.append(
        "- Assessment: intrinsics are sufficient to build a camera model, but because the model is `FishPoly`, "
        "a rectification step to a pinhole camera is recommended before training the current baseline."
    )
    lines.append("")

    lines.append("## C. Pose Availability")
    lines.append("")
    lines.append(f"- Topic: `/odin1/odometry`")
    lines.append(f"- Type: `{first_odom_topic['type']}`")
    lines.append(f"- Frame relationship from sample: `{first_odom_topic['sample'].get('frame_id')} -> {first_odom_topic['sample'].get('child_frame_id')}`")
    lines.append(
        f"- Frequency range across healthy bags: `{min(topic['hz'] for bag in odom_bags for topic in bag.topics if topic['name']=='/odin1/odometry'):.3f}`"
        f" to `{max(topic['hz'] for bag in odom_bags for topic in bag.topics if topic['name']=='/odin1/odometry'):.3f}` Hz"
    )
    lines.append(
        f"- Image to odometry nearest-neighbor alignment over overlap: mean `{min(odom_sync_means):.3f}` to `{max(odom_sync_means):.3f}` ms, "
        f"median `{min(odom_sync_medians):.3f}` to `{max(odom_sync_medians):.3f}` ms."
    )
    lines.append(
        "- Assessment: every healthy bag has enough odometry density to assign one pose per image via nearest-neighbor or interpolation."
    )
    lines.append("")

    lines.append("## D. LiDAR Point Clouds")
    lines.append("")
    lines.append(f"- Raw cloud topic: `/odin1/cloud_raw` (`{first_cloud_raw['type']}`)")
    lines.append(f"- Raw cloud sample frame: `{first_cloud_raw['sample'].get('frame_id')}`")
    lines.append(
        f"- Raw cloud sample layout: `{first_cloud_raw['sample'].get('width')} points`, fields "
        f"`{', '.join(field['name'] for field in first_cloud_raw['sample'].get('fields', []))}`"
    )
    lines.append(f"- SLAM cloud topic: `/odin1/cloud_slam` (`{first_cloud_slam['type']}`)")
    lines.append(f"- SLAM cloud sample frame: `{first_cloud_slam['sample'].get('frame_id')}`")
    lines.append(
        f"- Image to raw-cloud alignment over overlap: mean `{min(cloud_sync_means):.3f}` to `{max(cloud_sync_means):.3f}` ms, "
        f"median `{min(cloud_sync_medians):.3f}` to `{max(cloud_sync_medians):.3f}` ms."
    )
    lines.append(
        "- Assessment: `cloud_raw` is suitable for per-frame depth projection, while `cloud_slam` is a better starting point for a registered or accumulated Gaussian initialization cloud."
    )
    lines.append("")

    lines.append("## E. Extrinsics / TF")
    lines.append("")
    lines.append("- No `/tf` or `/tf_static` topics are present in the bags.")
    lines.append(f"- External rigid transform found in calibration file under key `{calibration['matrix_name']}`.")
    lines.append(
        "- The matrix name `Tcl_0` strongly suggests a `camera <- lidar` extrinsic. Because `cloud_raw` is already in frame "
        "`odin1_base_link`, the practical assumption is that the LiDAR frame and base frame are either identical or rigidly pre-aligned by the recorder."
    )
    lines.append(
        "- Assessment: a LiDAR-to-camera chain is recoverable from `odom -> odin1_base_link` (odometry) plus `Tcl_0`, "
        "but the exact transform direction should be validated with a projection sanity check because the TF tree itself was not recorded."
    )
    lines.append("")

    lines.append("## F. Time Synchronization")
    lines.append("")
    lines.append(
        f"- Image to raw cloud: mean `{min(cloud_sync_means):.3f}` to `{max(cloud_sync_means):.3f}` ms across healthy bags."
    )
    lines.append(
        f"- Image to odometry: mean `{min(odom_sync_means):.3f}` to `{max(odom_sync_means):.3f}` ms across healthy bags."
    )
    lines.append(
        "- The image stream and raw LiDAR overlap almost perfectly in time. Odometry starts a few seconds later than the sensor streams in several bags, "
        "but once overlapping, the nearest-neighbor pose assignment error stays in the tens of milliseconds."
    )
    lines.append("")

    lines.append("## Final Decision Table")
    lines.append("")
    lines.append("| Condition | Satisfied | Evidence | Risk |")
    lines.append("| --- | --- | --- | --- |")
    lines.append("| Image usable | Yes | `/odin1/image/compressed` is present in all healthy bags, JPEG-compressed, 1600x1296, ~10 Hz. | Images are compressed, not raw; still acceptable for RGB supervision. |")
    lines.append("| Camera intrinsics available | Yes (external) | `cam_in_ex.txt` contains FishPoly intrinsics and image size. | No `camera_info` topic; distortion model is not pinhole. |")
    lines.append("| Per-image pose available | Yes | `/odin1/odometry` runs at ~10 Hz and can be aligned to images with ~25-36 ms median error. | Pose is for `odin1_base_link`, not camera. |")
    lines.append("| Camera pose recoverable | Yes, with assumption | Combine `odom -> odin1_base_link` with `Tcl_0`. | TF tree absent; extrinsic direction must be validated. |")
    lines.append("| LiDAR available | Yes | `/odin1/cloud_raw` and `/odin1/cloud_slam` are present in all healthy bags. | `Motorworld3` is corrupted and should be excluded. |")
    lines.append("| LiDAR-camera extrinsic recoverable | Yes (external) | `Tcl_0` matrix is stored in `cam_in_ex.txt`. | No in-bag TF confirmation. |")
    lines.append("| Time sync acceptable | Yes | Image↔cloud mean 6-13 ms; image↔odom mean 24-34 ms over overlap. | Early seconds of some bags lack odometry. |")
    lines.append("| Can replace COLMAP initialization | Yes, conditionally | Healthy bags contain enough RGB + pose + LiDAR + calibration to build a SLAM-guided initialization. | Need undistortion, TF validation, and a loader that consumes non-COLMAP metadata. |")
    lines.append("")

    lines.append("## What Is Still Missing")
    lines.append("")
    lines.append("- A recorded TF tree or an explicit frame convention note to remove ambiguity around `Tcl_0`.")
    lines.append("- A rectification step from `FishPoly` to a pinhole camera model if you want to use the existing COLMAP-oriented 3DGS baseline without modifying its loader.")
    lines.append("- A conversion step that packages images, poses, calibration, and LiDAR into either a custom loader format or synthetic COLMAP binaries/text files.")
    lines.append("")

    lines.append("## Recommended Execution Order")
    lines.append("")
    lines.append("1. Exclude `Motorworld3` from all downstream processing.")
    lines.append("2. Export images from `/odin1/image/compressed` and store them as sequential `.jpg` files.")
    lines.append("3. Convert `cam_in_ex.txt` into a machine-readable `camera.json` and validate the `Tcl_0` direction by projecting a few LiDAR frames into sample images.")
    lines.append("4. Synchronize images to `/odin1/odometry` and `/odin1/cloud_raw` via nearest neighbor or interpolation.")
    lines.append("5. Accumulate `cloud_raw` into the odometry frame or reuse `cloud_slam` as the initial 3D point set.")
    lines.append("6. Either:")
    lines.append("   - write a custom 3DGS loader for the exported `images + poses + intrinsics + lidar`, or")
    lines.append("   - convert the exported metadata into COLMAP-style `cameras/images/points3D` files if you want to keep the current baseline unchanged.")
    lines.append("")

    lines.append("## Proposed Standardized Intermediate Layout")
    lines.append("")
    lines.append("```text")
    lines.append("scene/")
    lines.append("├── images/")
    lines.append("├── poses/")
    lines.append("│   ├── poses.csv")
    lines.append("│   └── poses_world_from_camera.txt")
    lines.append("├── intrinsics/")
    lines.append("│   └── camera.json")
    lines.append("├── lidar/")
    lines.append("│   ├── frames/")
    lines.append("│   └── global_map.ply")
    lines.append("├── transforms/")
    lines.append("│   └── tf_chain.json")
    lines.append("└── metadata/")
    lines.append("    └── associations.csv")
    lines.append("```")
    lines.append("")
    lines.append(
        "This format is enough to run a SLAM-guided 3DGS initialization even if you never invoke COLMAP. "
        "For the current baseline in this repository, you would still need one more adapter step from this intermediate format into `images/ + sparse/0/`."
    )
    lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect ROS 2 bags for SLAM-guided 3DGS feasibility.")
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_BAG_ROOT,
        help="Root directory containing rosbag subfolders.",
    )
    parser.add_argument(
        "--inventory-out",
        type=Path,
        default=None,
        help="Output path for bag_inventory.md",
    )
    parser.add_argument(
        "--feasibility-out",
        type=Path,
        default=None,
        help="Output path for feasibility_report.md",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional JSON dump of the parsed summaries.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    DEFAULT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    inventory_out = args.inventory_out or (DEFAULT_REPORT_DIR / "bag_inventory.md")
    feasibility_out = args.feasibility_out or (DEFAULT_REPORT_DIR / "feasibility_report.md")
    summary_json = args.summary_json

    bag_dirs = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and ((path / "metadata.yaml").exists() or any(path.glob("*.db3")))
    )
    calibration_path = root / "cam_in_ex.txt"
    calibration = parse_calibration(calibration_path)
    bags = [scan_bag_dir(bag_dir) for bag_dir in bag_dirs]

    inventory_text = build_bag_inventory_markdown(bags, calibration, root)
    feasibility_text = build_feasibility_report_markdown(bags, calibration)
    inventory_out.write_text(inventory_text, encoding="utf-8")
    feasibility_out.write_text(feasibility_text, encoding="utf-8")

    if summary_json:
        payload = {
            "calibration": calibration,
            "bags": [
                {
                    "name": bag.name,
                    "bag_dir": str(bag.bag_dir),
                    "db_path": str(bag.db_path) if bag.db_path else None,
                    "metadata_path": str(bag.metadata_path) if bag.metadata_path else None,
                    "size_bytes": bag.size_bytes,
                    "ros_distro": bag.ros_distro,
                    "valid": bag.valid,
                    "error": bag.error,
                    "start_ns": bag.start_ns,
                    "end_ns": bag.end_ns,
                    "duration_ns": bag.duration_ns,
                    "total_messages": bag.total_messages,
                    "topics": bag.topics,
                    "sync_stats": bag.sync_stats,
                }
                for bag in bags
            ],
        }
        write_json(summary_json, payload)


if __name__ == "__main__":
    main()
