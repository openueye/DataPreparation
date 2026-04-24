#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from data_preparation.rosbag_to_3dgs.db import require_topic_id
    from data_preparation.rosbag_to_3dgs.messages import parse_compressed_image
    from data_preparation.shared.calibration import parse_camera_lidar_calibration
    from data_preparation.shared.io import write_json
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from data_preparation.rosbag_to_3dgs.db import require_topic_id
    from data_preparation.rosbag_to_3dgs.messages import parse_compressed_image
    from data_preparation.shared.calibration import parse_camera_lidar_calibration
    from data_preparation.shared.io import write_json


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is non-empty: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _db_path_from_bag_dir(bag_dir: Path) -> Path:
    db_path = next(iter(sorted(bag_dir.glob("*.db3"))), None)
    if db_path is None:
        raise FileNotFoundError(f"No .db3 file found under {bag_dir}")
    return db_path


def _load_image_rows(db_path: Path, image_topic: str) -> List[Tuple[int, bytes]]:
    connection = sqlite3.connect(str(db_path))
    try:
        topics = {name: topic_id for topic_id, name in connection.execute("SELECT id, name FROM topics")}
        image_topic_id = require_topic_id(topics, image_topic, db_path)
        rows = connection.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp",
            (image_topic_id,),
        ).fetchall()
    finally:
        connection.close()
    if not rows:
        raise ValueError(f"No messages found for image topic '{image_topic}' in {db_path.name}.")
    return rows


def _camera_json_from_calibration(calibration: Dict[str, Any]) -> Dict[str, Any]:
    camera_model = calibration["camera_params"].get("cam_model")
    return {
        "camera_name": calibration["camera_name"],
        "camera_model": camera_model,
        "image_model": "raw_distorted",
        "undistorted": False,
        "rectified": False,
        "colmap_training_ready": False,
        "requires_rectification_for_colmap": camera_model not in {"PINHOLE", "SIMPLE_PINHOLE"},
        "width": calibration["camera_params"].get("image_width"),
        "height": calibration["camera_params"].get("image_height"),
        "K_like": calibration["K_like"],
        "distortion": calibration["distortion"],
        "notes": [
            "Image-only ROS bag extraction for pure visual COLMAP SfM.",
            "No SLAM odometry or LiDAR data are exported in this scene.",
            "FishPoly images must be rectified before running a pinhole COLMAP baseline.",
        ],
    }


def extract_images(
    bag_dir: Path,
    output_dir: Path,
    calibration_path: Path,
    *,
    image_topic: str = "/odin1/image/compressed",
    limit_images: Optional[int] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    bag_dir = bag_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    calibration_path = calibration_path.expanduser().resolve()
    db_path = _db_path_from_bag_dir(bag_dir)
    rows = _load_image_rows(db_path, image_topic)
    if limit_images is not None:
        rows = rows[: max(limit_images, 0)]
    if not rows:
        raise ValueError("--limit-images filtered out all frames; choose a larger value.")

    _prepare_output_dir(output_dir, overwrite=overwrite)
    images_dir = output_dir / "images"
    intrinsics_dir = output_dir / "intrinsics"
    metadata_dir = output_dir / "metadata"
    images_dir.mkdir(parents=True, exist_ok=True)
    intrinsics_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows: List[List[Any]] = []
    for image_index, (timestamp, blob) in enumerate(rows):
        image_msg = parse_compressed_image(blob)
        file_stem = f"{image_index:06d}"
        image_extension = ".jpg" if image_msg["format"].lower().startswith("jp") else ".png"
        image_path = images_dir / f"{file_stem}{image_extension}"
        image_path.write_bytes(image_msg["data"])
        metadata_rows.append([file_stem, timestamp, image_msg["format"], image_msg["data_len"], image_path.name])

    calibration = parse_camera_lidar_calibration(calibration_path, strict_matrix=False)
    camera_json = _camera_json_from_calibration(calibration)
    write_json(intrinsics_dir / "camera.json", camera_json)

    with (metadata_dir / "images.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_id", "image_timestamp_ns", "compressed_format", "data_len", "image_name"])
        writer.writerows(metadata_rows)

    scene_meta = {
        "scene_name": output_dir.name,
        "source_type": "rosbag_image_only",
        "init_mode": "sfm",
        "raw_source": str(bag_dir),
        "generated_by": "rosbag_to_colmap.extract_rosbag_images",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bag_db_path": str(db_path),
        "calibration_path": str(calibration_path),
        "image_topic": image_topic,
        "num_exported_images": len(metadata_rows),
        "uses_odometry": False,
        "uses_lidar": False,
    }
    write_json(output_dir / "scene_meta.json", scene_meta)

    report = {
        "source_bag": str(bag_dir),
        "output_scene": str(output_dir),
        "image_topic": image_topic,
        "num_images": len(metadata_rows),
        "camera_model": camera_json.get("camera_model"),
        "camera_json": str(intrinsics_dir / "camera.json"),
        "metadata_csv": str(metadata_dir / "images.csv"),
    }
    write_json(output_dir / "rosbag_image_extraction_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract image-only ROS bag frames for a pure visual COLMAP SfM baseline.")
    parser.add_argument("--bag-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--image-topic", default="/odin1/image/compressed")
    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = extract_images(
        args.bag_dir,
        args.output_dir,
        args.calibration,
        image_topic=args.image_topic,
        limit_images=args.limit_images,
        overwrite=args.overwrite,
    )
    print(f"[INFO] Extracted ROS bag images: {report['num_images']}")
    print(f"[INFO] Image-only scene written: {report['output_scene']}")


if __name__ == "__main__":
    main()
