#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sqlite3
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


BASE_OFFSET = 4
DISTORTION_KEYS = ("k2", "k3", "k4", "k5", "k6", "k7", "p1", "p2")
POINTFIELD_DTYPE = {
    1: np.int8,
    2: np.uint8,
    3: np.int16,
    4: np.uint16,
    5: np.int32,
    6: np.uint32,
    7: np.float32,
    8: np.float64,
}

# Official fixed extrinsic from the Odin1 wiki "5.2 外参描述" section:
# T^imu_lidar maps lidar-frame coordinates into the imu frame.
OFFICIAL_T_IMU_FROM_LIDAR = np.asarray(
    [
        [1.0, 0.0, 0.0, -0.02663],
        [0.0, 1.0, 0.0, 0.03447],
        [0.0, 0.0, 1.0, 0.02174],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class OdomSample:
    timestamp_ns: int
    frame_id: str
    child_frame_id: str
    position: np.ndarray
    quaternion_xyzw: np.ndarray

    @property
    def world_from_base(self) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = quaternion_xyzw_to_matrix(self.quaternion_xyzw)
        matrix[:3, 3] = self.position
        return matrix


@dataclass(frozen=True)
class CameraCalibration:
    path: str
    matrix_name: str
    camera_name: str
    camera_params: Dict[str, Any]
    k_like: List[List[float]]
    distortion: Dict[str, Any]
    t_camera_from_lidar: np.ndarray


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    images_dir: Path
    images_rectified_dir: Path
    lidar_raw_dir: Path
    lidar_slam_dir: Path
    poses_dir: Path
    imu_dir: Path
    calib_dir: Path
    reports_dir: Path
    associations_dir: Path


@dataclass(frozen=True)
class BagStreams:
    image_rows: List[Tuple[int, bytes]]
    odom_rows: List[Tuple[int, bytes]]
    imu_rows: List[Tuple[int, bytes]]
    raw_cloud_rows: List[Tuple[int, bytes]]
    slam_cloud_rows: List[Tuple[int, bytes]]


def thesis_root_from_here() -> Path:
    current = Path(__file__).resolve()
    for candidate in (current, *current.parents):
        if candidate.name == "Thesis":
            return candidate
    raise ValueError(f"Could not infer Thesis root from: {current}")


def parse_args() -> argparse.Namespace:
    thesis_root = thesis_root_from_here()
    parser = argparse.ArgumentParser(
        description=(
            "Organize an Odin1 ROS bag directory into the prepared pure-headerstamp scene layout "
            "used by the SFM, hybrid, and SLAM data-preparation routes."
        )
    )
    parser.add_argument("--scene", default="Downtown1")
    parser.add_argument("--bag-dir", type=Path, default=thesis_root / "03_Datasets" / "001_rosbags" / "Downtown1")
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--image-topic", default="/odin1/image/compressed")
    parser.add_argument("--odom-topic", default="/odin1/odometry")
    parser.add_argument("--imu-topic", default="/odin1/imu")
    parser.add_argument("--cloud-raw-topic", default="/odin1/cloud_raw")
    parser.add_argument("--cloud-slam-topic", default="/odin1/cloud_slam")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--imu-window-ms", type=float, default=50.0)
    parser.add_argument("--raw-map-stride", type=int, default=8)
    parser.add_argument("--slam-map-stride", type=int, default=8)
    parser.add_argument("--validation-samples", type=int, default=8)
    parser.add_argument("--validation-frame-stride", type=int, default=80)
    parser.add_argument("--validation-max-points", type=int, default=16000)
    parser.add_argument("--odom-max-dt-ms", type=float, default=50.0)
    parser.add_argument("--cloud-max-dt-ms", type=float, default=20.0)
    return parser.parse_args()


def resolve_calibration_path(bag_dir: Path, explicit: Optional[Path]) -> Path:
    if explicit:
        return explicit.expanduser().resolve()
    candidates = [
        bag_dir / "cam_in_ex.txt",
        bag_dir.parent / "cam_in_ex.txt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find cam_in_ex.txt. Checked:\n" + "\n".join(str(path) for path in candidates)
    )


def align(offset: int, alignment: int) -> int:
    return BASE_OFFSET + (((offset - BASE_OFFSET) + (alignment - 1)) & ~(alignment - 1))


def read_u8(blob: bytes, offset: int) -> Tuple[int, int]:
    return struct.unpack_from("<B", blob, offset)[0], offset + 1


def read_bool(blob: bytes, offset: int) -> Tuple[bool, int]:
    return struct.unpack_from("<?", blob, offset)[0], offset + 1


def read_u32(blob: bytes, offset: int) -> Tuple[int, int]:
    offset = align(offset, 4)
    return struct.unpack_from("<I", blob, offset)[0], offset + 4


def read_i32(blob: bytes, offset: int) -> Tuple[int, int]:
    offset = align(offset, 4)
    return struct.unpack_from("<i", blob, offset)[0], offset + 4


def read_f64(blob: bytes, offset: int) -> Tuple[float, int]:
    offset = align(offset, 8)
    return struct.unpack_from("<d", blob, offset)[0], offset + 8


def read_string(blob: bytes, offset: int) -> Tuple[str, int]:
    offset = align(offset, 4)
    length = struct.unpack_from("<I", blob, offset)[0]
    offset += 4
    raw = blob[offset : offset + length]
    offset += length
    return (raw[:-1].decode("utf-8") if length else ""), offset


def parse_header(blob: bytes) -> Dict[str, Any]:
    offset = BASE_OFFSET
    sec, offset = read_i32(blob, offset)
    nsec, offset = read_u32(blob, offset)
    frame_id, offset = read_string(blob, offset)
    return {"stamp_sec": sec, "stamp_nsec": nsec, "frame_id": frame_id, "offset": offset}


def parse_compressed_image(blob: bytes) -> Dict[str, Any]:
    header = parse_header(blob)
    offset = header["offset"]
    image_format, offset = read_string(blob, offset)
    data_len, offset = read_u32(blob, offset)
    data = blob[offset : offset + data_len]
    return {**header, "format": image_format, "data": data, "data_len": data_len}


def decode_compressed_image(blob: bytes) -> Dict[str, Any]:
    import cv2

    message = parse_compressed_image(blob)
    image = cv2.imdecode(np.frombuffer(message["data"], dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode compressed image.")
    return {**message, "image": image}


def parse_odometry(blob: bytes) -> Dict[str, Any]:
    header = parse_header(blob)
    offset = header["offset"]
    child_frame_id, offset = read_string(blob, offset)
    position = []
    for _ in range(3):
        value, offset = read_f64(blob, offset)
        position.append(value)
    orientation = []
    for _ in range(4):
        value, offset = read_f64(blob, offset)
        orientation.append(value)
    return {
        **header,
        "child_frame_id": child_frame_id,
        "position": np.asarray(position, dtype=np.float64),
        "orientation_xyzw": np.asarray(orientation, dtype=np.float64),
    }


def parse_imu(blob: bytes) -> Dict[str, Any]:
    header = parse_header(blob)
    offset = header["offset"]

    def read_vec(count: int) -> np.ndarray:
        nonlocal offset
        values = []
        for _ in range(count):
            value, offset = read_f64(blob, offset)
            values.append(value)
        return np.asarray(values, dtype=np.float64)

    return {
        "timestamp_ns": int(header["stamp_sec"]) * 1_000_000_000 + int(header["stamp_nsec"]),
        "frame_id": header["frame_id"],
        "orientation_xyzw": read_vec(4),
        "orientation_covariance": read_vec(9),
        "angular_velocity": read_vec(3),
        "angular_velocity_covariance": read_vec(9),
        "linear_acceleration": read_vec(3),
        "linear_acceleration_covariance": read_vec(9),
    }


def parse_pointcloud2(blob: bytes) -> Dict[str, Any]:
    header = parse_header(blob)
    offset = header["offset"]
    height, offset = read_u32(blob, offset)
    width, offset = read_u32(blob, offset)
    field_count, offset = read_u32(blob, offset)
    fields = []
    for _ in range(field_count):
        name, offset = read_string(blob, offset)
        field_offset, offset = read_u32(blob, offset)
        datatype, offset = read_u8(blob, offset)
        count, offset = read_u32(blob, offset)
        fields.append({"name": name, "offset": field_offset, "datatype": datatype, "count": count})
    is_bigendian, offset = read_bool(blob, offset)
    point_step, offset = read_u32(blob, offset)
    row_step, offset = read_u32(blob, offset)
    data_len, offset = read_u32(blob, offset)
    data = blob[offset : offset + data_len]
    return {
        **header,
        "height": height,
        "width": width,
        "fields": fields,
        "is_bigendian": is_bigendian,
        "point_step": point_step,
        "row_step": row_step,
        "data_len": data_len,
        "data": data,
    }


def quaternion_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n == 0.0:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = q / n
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion_xyzw(matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (matrix[2, 1] - matrix[1, 2]) * s
        y = (matrix[0, 2] - matrix[2, 0]) * s
        z = (matrix[1, 0] - matrix[0, 1]) * s
    else:
        diag = np.diag(matrix)
        idx = int(np.argmax(diag))
        if idx == 0:
            s = 2.0 * np.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 1e-12))
            w = (matrix[2, 1] - matrix[1, 2]) / s
            x = 0.25 * s
            y = (matrix[0, 1] + matrix[1, 0]) / s
            z = (matrix[0, 2] + matrix[2, 0]) / s
        elif idx == 1:
            s = 2.0 * np.sqrt(max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 1e-12))
            w = (matrix[0, 2] - matrix[2, 0]) / s
            x = (matrix[0, 1] + matrix[1, 0]) / s
            y = 0.25 * s
            z = (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 1e-12))
            w = (matrix[1, 0] - matrix[0, 1]) / s
            x = (matrix[0, 2] + matrix[2, 0]) / s
            y = (matrix[1, 2] + matrix[2, 1]) / s
            z = 0.25 * s
    quaternion = np.asarray([x, y, z, w], dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if norm == 0.0:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quaternion / norm


def slerp_quaternion_xyzw(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    q0_norm = np.linalg.norm(q0)
    q1_norm = np.linalg.norm(q1)
    if q0_norm == 0.0 or q1_norm == 0.0:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    q0 = q0 / q0_norm
    q1 = q1 / q1_norm
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        blended = q0 + alpha * (q1 - q0)
        norm = np.linalg.norm(blended)
        if norm == 0.0:
            return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        return blended / norm
    theta_0 = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    sin_theta_0 = float(np.sin(theta_0))
    if sin_theta_0 < 1e-8:
        return q0
    theta = theta_0 * alpha
    sin_theta = float(np.sin(theta))
    s0 = float(np.sin(theta_0 - theta) / sin_theta_0)
    s1 = float(sin_theta / sin_theta_0)
    return s0 * q0 + s1 * q1


def nearest_indices(reference_ts: np.ndarray, target_ts: np.ndarray) -> np.ndarray:
    if len(target_ts) == 0:
        raise ValueError("Cannot match timestamps against an empty target array.")
    indices = np.searchsorted(target_ts, reference_ts)
    left = np.clip(indices - 1, 0, len(target_ts) - 1)
    right = np.clip(indices, 0, len(target_ts) - 1)
    left_diff = np.abs(reference_ts - target_ts[left])
    right_diff = np.abs(reference_ts - target_ts[right])
    return np.where(left_diff <= right_diff, left, right)


def nearest_neighbor_stats(reference_ts: np.ndarray, target_ts: np.ndarray) -> Optional[Dict[str, float]]:
    if len(reference_ts) == 0 or len(target_ts) == 0:
        return None
    overlap = reference_ts[(reference_ts >= target_ts[0]) & (reference_ts <= target_ts[-1])]
    if len(overlap) == 0:
        return None
    matches = nearest_indices(overlap, target_ts)
    deltas_ms = np.abs(overlap - target_ts[matches]) / 1e6
    return {
        "matched_count": int(len(overlap)),
        "mean_ms": float(deltas_ms.mean()),
        "median_ms": float(np.median(deltas_ms)),
        "p95_ms": float(np.percentile(deltas_ms, 95)),
        "max_ms": float(deltas_ms.max()),
    }


def parse_camera_lidar_calibration(path: Path) -> CameraCalibration:
    matrix_name = None
    in_matrix_block = False
    matrix_values: List[float] = []
    camera_name = None
    camera_params: Dict[str, Any] = {}

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.endswith(": ["):
            matrix_name = line[:-3]
            in_matrix_block = True
            matrix_values = []
            continue
        if in_matrix_block:
            if line == "]":
                in_matrix_block = False
                continue
            cleaned = line.rstrip(",")
            matrix_values.extend(float(part.strip()) for part in cleaned.split(",") if part.strip())
            continue
        if line.endswith(":") and not raw_line.startswith(" "):
            camera_name = line[:-1]
            continue
        if camera_name and ":" in line:
            key, value = [part.strip() for part in line.split(":", 1)]
            camera_params[key] = _parse_scalar(value)

    if len(matrix_values) != 16:
        raise ValueError(f"Expected one 4x4 calibration matrix in {path}, got {len(matrix_values)} values.")
    matrix = np.asarray(matrix_values, dtype=np.float64).reshape(4, 4)
    k_like = [
        [camera_params["A11"], camera_params["A12"], camera_params["u0"]],
        [0.0, camera_params["A22"], camera_params["v0"]],
        [0.0, 0.0, 1.0],
    ]
    distortion = {key: camera_params.get(key) for key in DISTORTION_KEYS}
    return CameraCalibration(
        path=str(path),
        matrix_name=str(matrix_name),
        camera_name=str(camera_name),
        camera_params=camera_params,
        k_like=k_like,
        distortion=distortion,
        t_camera_from_lidar=matrix,
    )


def _parse_scalar(value: str) -> Any:
    value = value.split("#", 1)[0].strip()
    if value in {"", "."}:
        return value
    try:
        if value.lower() in {"true", "false"}:
            return value.lower() == "true"
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        return value


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is non-empty: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.12g}"
    text = str(value)
    if any(ch in text for ch in [":", "#", "[", "]", "{", "}", ","]) or text.strip() != text:
        return f'"{text}"'
    return text


def write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    lines: List[str] = []

    def emit(key: str, value: Any, level: int) -> None:
        prefix = "  " * level
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            for child_key, child_value in value.items():
                emit(str(child_key), child_value, level + 1)
            return
        if isinstance(value, (list, tuple)):
            if value and all(not isinstance(item, (dict, list, tuple)) for item in value):
                lines.append(f"{prefix}{key}: [{', '.join(_yaml_scalar(item) for item in value)}]")
            else:
                lines.append(f"{prefix}{key}:")
                for item in value:
                    if isinstance(item, dict):
                        lines.append(f"{prefix}  -")
                        for child_key, child_value in item.items():
                            emit(str(child_key), child_value, level + 2)
                    else:
                        lines.append(f"{prefix}  - {_yaml_scalar(item)}")
            return
        lines.append(f"{prefix}{key}: {_yaml_scalar(value)}")

    for top_key, top_value in payload.items():
        emit(str(top_key), top_value, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_ascii_ply(path: Path, points: np.ndarray, colors: Optional[np.ndarray] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if colors is not None and len(colors) != len(points):
        raise ValueError("PLY color array length must match points length.")
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        if colors is not None:
            handle.write("property uchar red\n")
            handle.write("property uchar green\n")
            handle.write("property uchar blue\n")
        handle.write("end_header\n")
        if colors is None:
            for point in points:
                handle.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
        else:
            colors_uint8 = np.asarray(colors, dtype=np.uint8)
            for point, color in zip(points, colors_uint8):
                handle.write(
                    f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                    f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
                )


def build_fishpoly_rectification_maps(camera_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    k_mat = np.asarray(camera_data["K_like"], dtype=np.float64)
    src_width = int(camera_data["width"])
    src_height = int(camera_data["height"])
    a11 = float(k_mat[0, 0])
    a12 = float(k_mat[0, 1])
    a22 = float(k_mat[1, 1])
    u0 = float(k_mat[0, 2])
    v0 = float(k_mat[1, 2])
    distortion = camera_data["distortion"]
    k2 = float(distortion["k2"])
    k3 = float(distortion["k3"])
    k4 = float(distortion["k4"])
    k5 = float(distortion["k5"])
    k6 = float(distortion["k6"])
    k7 = float(distortion["k7"])

    output_width = src_width
    output_height = src_height
    fx = output_width / 2.0
    fy = output_height / 2.0
    cx = output_width / 2.0
    cy = output_height / 2.0

    u, v = np.meshgrid(np.arange(output_width, dtype=np.float64), np.arange(output_height, dtype=np.float64))
    x = (u - cx) / fx
    y = (v - cy) / fy
    r = np.sqrt(x * x + y * y)
    theta = np.arctan(r)
    theta_d = theta * (
        1.0
        + k2 * theta**2
        + k3 * theta**3
        + k4 * theta**4
        + k5 * theta**5
        + k6 * theta**6
        + k7 * theta**7
    )
    scale = np.divide(theta_d, r, out=np.ones_like(theta_d), where=r > 1e-12)
    xd = scale * x
    yd = scale * y
    map_x = a11 * xd + a12 * yd + u0
    map_y = a22 * yd + v0
    pinhole = {"width": output_width, "height": output_height, "fx": fx, "fy": fy, "cx": cx, "cy": cy}
    return map_x.astype(np.float32), map_y.astype(np.float32), pinhole


def build_sampler(map_x: np.ndarray, map_y: np.ndarray, source_width: int, source_height: int) -> Dict[str, np.ndarray]:
    valid = (map_x >= 0.0) & (map_x <= source_width - 1) & (map_y >= 0.0) & (map_y <= source_height - 1)
    x0 = np.floor(np.clip(map_x, 0, source_width - 1)).astype(np.int32)
    y0 = np.floor(np.clip(map_y, 0, source_height - 1)).astype(np.int32)
    x1 = np.minimum(x0 + 1, source_width - 1)
    y1 = np.minimum(y0 + 1, source_height - 1)
    wx = (np.clip(map_x, 0, source_width - 1) - x0).astype(np.float32)
    wy = (np.clip(map_y, 0, source_height - 1) - y0).astype(np.float32)
    return {"valid": valid, "x0": x0, "x1": x1, "y0": y0, "y1": y1, "wx": wx, "wy": wy}


def remap_bilinear(image: np.ndarray, sampler: Dict[str, np.ndarray]) -> np.ndarray:
    arr = image.astype(np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    x0 = sampler["x0"]
    x1 = sampler["x1"]
    y0 = sampler["y0"]
    y1 = sampler["y1"]
    wx = sampler["wx"][:, :, None]
    wy = sampler["wy"][:, :, None]
    top = arr[y0, x0] * (1.0 - wx) + arr[y0, x1] * wx
    bottom = arr[y1, x0] * (1.0 - wx) + arr[y1, x1] * wx
    out = top * (1.0 - wy) + bottom * wy
    out[~sampler["valid"]] = 0.0
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out[:, :, 0] if image.ndim == 2 else out


def _dtype_for_field(field: Dict[str, Any], endian: str) -> Tuple[np.dtype, int]:
    base = POINTFIELD_DTYPE.get(int(field["datatype"]))
    if base is None:
        raise ValueError(f"Unsupported PointField datatype: {field['datatype']}")
    dtype = np.dtype(base).newbyteorder(endian)
    return dtype, int(field.get("count", 1))


def decode_structured_fields(pointcloud: Dict[str, Any], field_names: Iterable[str]) -> Dict[str, np.ndarray]:
    selected = []
    endian = ">" if pointcloud.get("is_bigendian") else "<"
    for field in pointcloud["fields"]:
        if field["name"] not in field_names:
            continue
        dtype, count = _dtype_for_field(field, endian)
        if count == 1:
            selected.append((field["name"], dtype, int(field["offset"])))
        else:
            selected.append((field["name"], dtype, int(field["offset"]), count))
    dtype_spec = {
        "names": [entry[0] for entry in selected],
        "formats": [entry[1] if len(entry) == 3 else (entry[1], (entry[3],)) for entry in selected],
        "offsets": [entry[2] for entry in selected],
        "itemsize": int(pointcloud["point_step"]),
    }
    structured = np.frombuffer(
        pointcloud["data"],
        dtype=np.dtype(dtype_spec),
        count=int(pointcloud["width"]) * int(pointcloud["height"]),
    )
    return {name: np.asarray(structured[name]) for name in structured.dtype.names}


def decode_raw_cloud(pointcloud: Dict[str, Any]) -> Dict[str, np.ndarray]:
    fields = decode_structured_fields(pointcloud, ("x", "y", "z", "intensity", "confidence", "offset_time"))
    xyz = np.stack([fields["x"], fields["y"], fields["z"]], axis=1).astype(np.float32, copy=False)
    finite = np.isfinite(xyz).all(axis=1)
    payload = {"xyz": xyz[finite]}
    for key in ("intensity", "confidence", "offset_time"):
        values = fields.get(key)
        if values is not None:
            payload[key] = np.asarray(values)[finite]
    return payload


def decode_rgb_field(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype == np.float32:
        packed = values.view(np.uint32)
    elif values.dtype == np.float64:
        packed = values.astype(np.float32).view(np.uint32)
    else:
        packed = values.astype(np.uint32, copy=False)
    rgb = np.empty((len(packed), 3), dtype=np.uint8)
    rgb[:, 0] = (packed >> 16) & 0xFF
    rgb[:, 1] = (packed >> 8) & 0xFF
    rgb[:, 2] = packed & 0xFF
    return rgb


def decode_slam_cloud(pointcloud: Dict[str, Any]) -> Dict[str, np.ndarray]:
    fields = decode_structured_fields(pointcloud, ("x", "y", "z", "rgb"))
    xyz = np.stack([fields["x"], fields["y"], fields["z"]], axis=1).astype(np.float32, copy=False)
    finite = np.isfinite(xyz).all(axis=1)
    payload = {"xyz": xyz[finite]}
    if "rgb" in fields:
        payload["rgb"] = decode_rgb_field(fields["rgb"])[finite]
    return payload


def load_rows(connection: sqlite3.Connection, topic_id: int) -> List[Tuple[int, bytes]]:
    return connection.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp",
        (topic_id,),
    ).fetchall()


def parse_odom_rows(rows: List[Tuple[int, bytes]]) -> List[OdomSample]:
    samples = []
    for timestamp_ns, blob in rows:
        message = parse_odometry(blob)
        samples.append(
            OdomSample(
                timestamp_ns=int(timestamp_ns),
                frame_id=str(message["frame_id"]),
                child_frame_id=str(message["child_frame_id"]),
                position=np.asarray(message["position"], dtype=np.float64),
                quaternion_xyzw=np.asarray(message["orientation_xyzw"], dtype=np.float64),
            )
        )
    return samples


def interpolate_pose(timestamp_ns: int, odom_samples: List[OdomSample]) -> Tuple[np.ndarray, Dict[str, Any]]:
    timestamps = np.asarray([sample.timestamp_ns for sample in odom_samples], dtype=np.int64)
    if timestamp_ns <= int(timestamps[0]):
        sample = odom_samples[0]
        return sample.world_from_base, {"prev_timestamp_ns": sample.timestamp_ns, "next_timestamp_ns": sample.timestamp_ns, "alpha": 0.0}
    if timestamp_ns >= int(timestamps[-1]):
        sample = odom_samples[-1]
        return sample.world_from_base, {"prev_timestamp_ns": sample.timestamp_ns, "next_timestamp_ns": sample.timestamp_ns, "alpha": 0.0}
    right = int(np.searchsorted(timestamps, timestamp_ns, side="right"))
    left = max(right - 1, 0)
    prev_sample = odom_samples[left]
    next_sample = odom_samples[right]
    span = max(next_sample.timestamp_ns - prev_sample.timestamp_ns, 1)
    alpha = float((timestamp_ns - prev_sample.timestamp_ns) / span)
    position = (1.0 - alpha) * prev_sample.position + alpha * next_sample.position
    quaternion = slerp_quaternion_xyzw(prev_sample.quaternion_xyzw, next_sample.quaternion_xyzw, alpha)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_xyzw_to_matrix(quaternion)
    matrix[:3, 3] = position
    return matrix, {"prev_timestamp_ns": prev_sample.timestamp_ns, "next_timestamp_ns": next_sample.timestamp_ns, "alpha": alpha}


def build_projection_overlay(
    image: np.ndarray,
    xyz_lidar: np.ndarray,
    t_camera_from_lidar: np.ndarray,
    k_matrix: np.ndarray,
    max_points: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    import cv2

    if len(xyz_lidar) == 0:
        return image.copy(), {"input_points": 0, "front_points": 0, "projected_points": 0, "front_ratio": 0.0, "inside_ratio": 0.0}
    sampled = xyz_lidar
    if len(sampled) > max_points:
        stride = max(1, len(sampled) // max_points)
        sampled = sampled[::stride]
    hom = np.concatenate([sampled.astype(np.float64), np.ones((len(sampled), 1), dtype=np.float64)], axis=1)
    xyz_camera = (t_camera_from_lidar @ hom.T).T[:, :3]
    positive = xyz_camera[:, 2] > 1e-4
    xyz_front = xyz_camera[positive]
    if len(xyz_front) == 0:
        return image.copy(), {"input_points": int(len(sampled)), "front_points": 0, "projected_points": 0, "front_ratio": 0.0, "inside_ratio": 0.0}
    projected = (k_matrix @ xyz_front.T).T
    uv = projected[:, :2] / projected[:, 2:3]
    h, w = image.shape[:2]
    inside = (uv[:, 0] >= 0.0) & (uv[:, 0] < w) & (uv[:, 1] >= 0.0) & (uv[:, 1] < h)
    uv_inside = uv[inside]
    depth_inside = xyz_front[inside, 2]
    overlay = image.copy()
    if len(uv_inside):
        z_min = float(np.percentile(depth_inside, 5))
        z_max = float(np.percentile(depth_inside, 95))
        z_span = max(z_max - z_min, 1e-6)
        norm = np.clip((depth_inside - z_min) / z_span, 0.0, 1.0)
        colors = cv2.applyColorMap((255.0 * (1.0 - norm)).astype(np.uint8), cv2.COLORMAP_TURBO)
        for (u, v), color in zip(uv_inside.astype(np.int32), colors[:, 0, :]):
            cv2.circle(overlay, (int(u), int(v)), 1, tuple(int(c) for c in color.tolist()), -1)
    front_ratio = float(len(xyz_front) / max(len(sampled), 1))
    inside_ratio = float(len(uv_inside) / max(len(xyz_front), 1))
    return overlay, {
        "input_points": int(len(sampled)),
        "front_points": int(len(xyz_front)),
        "projected_points": int(len(uv_inside)),
        "front_ratio": front_ratio,
        "inside_ratio": inside_ratio,
    }


def matrix_columns(prefix: str) -> List[str]:
    return [f"{prefix}_{row}{col}" for row in range(4) for col in range(4)]


def matrix_values(matrix: np.ndarray) -> List[float]:
    return [float(value) for value in matrix.reshape(-1).tolist()]


def write_csv(path: Path, header: List[str], rows: Iterable[Iterable[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_dict_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def bbox_stats(points: np.ndarray) -> Dict[str, Any]:
    if len(points) == 0:
        zeros = [0.0, 0.0, 0.0]
        return {"count": 0, "min": zeros, "max": zeros, "extent": zeros, "centroid": zeros}
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    centroid = np.mean(points, axis=0)
    return {
        "count": int(len(points)),
        "min": minimum.tolist(),
        "max": maximum.tolist(),
        "extent": (maximum - minimum).tolist(),
        "centroid": centroid.tolist(),
    }


def rotation_angle(prev_rotation: np.ndarray, next_rotation: np.ndarray) -> float:
    relative = prev_rotation.T @ next_rotation
    trace = float(np.trace(relative))
    return float(np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0)))


def np_stats(values: np.ndarray) -> Dict[str, float]:
    if len(values) == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def write_summary_markdown(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# Odin1 Dataset Organization Summary",
        "",
        f"- Scene: `{summary['scene_name']}`",
        f"- Raw images exported: `{summary['num_images']}`",
        f"- Rectified images exported: `{summary['num_rectified_images']}`",
        f"- Raw cloud frames exported: `{summary['num_raw_cloud_frames']}`",
        f"- SLAM cloud frames exported: `{summary['num_slam_cloud_frames']}`",
        f"- IMU samples exported: `{summary['num_imu_samples']}`",
        f"- Selected extrinsic interpretation: `{summary['extrinsic_validation']['selected_direction']}`",
        f"- Image↔odom median dt: `{summary['timing']['image_to_odom_ms']['median']:.3f} ms`",
        f"- Image↔raw-cloud median dt: `{summary['timing']['image_to_cloud_raw_ms']['median']:.3f} ms`",
        "",
        "## Validation",
        "",
        f"- Odom outlier frames (`>{summary['thresholds']['odom_dt_ms']} ms`): `{summary['validation']['num_odom_dt_outliers']}`",
        f"- Raw-cloud outlier frames (`>{summary['thresholds']['cloud_dt_ms']} ms`): `{summary['validation']['num_cloud_dt_outliers']}`",
        f"- IMU / odom angular-speed correlation: `{summary['validation']['imu_vs_odom_angular_speed_correlation']:.4f}`",
        f"- Raw-vs-slam centroid delta (m): `{summary['validation']['raw_vs_slam_centroid_delta_m']:.4f}`",
        "",
        "## Outputs",
        "",
        "- `images/`: raw JPEG frames",
        "- `images_rectified/`: FishPoly -> pinhole rectified frames",
        "- `lidar/raw_frames/`: per-frame raw cloud `.npz` with `xyz`, `intensity`, `confidence`, `offset_time`",
        "- `lidar/slam_frames/`: per-frame SLAM cloud `.npz`",
        "- `poses/`: odometry and per-image camera/base poses",
        "- `imu/imu.csv`: decoded IMU stream",
        "- `calib/`: raw + rectified camera metadata and selected TF chain",
        "- `reports/extrinsic_validation/`: both extrinsic-direction overlay candidates",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_output_paths(output_dir: Path) -> OutputPaths:
    return OutputPaths(
        root=output_dir,
        images_dir=output_dir / "images",
        images_rectified_dir=output_dir / "images_rectified",
        lidar_raw_dir=output_dir / "lidar" / "raw_frames",
        lidar_slam_dir=output_dir / "lidar" / "slam_frames",
        poses_dir=output_dir / "poses",
        imu_dir=output_dir / "imu",
        calib_dir=output_dir / "calib",
        reports_dir=output_dir / "reports",
        associations_dir=output_dir / "associations",
    )


def ensure_output_subdirs(paths: OutputPaths) -> None:
    for path in (
        paths.images_dir,
        paths.images_rectified_dir,
        paths.lidar_raw_dir,
        paths.lidar_slam_dir,
        paths.poses_dir,
        paths.imu_dir,
        paths.calib_dir,
        paths.reports_dir,
        paths.associations_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def build_camera_models(calibration: CameraCalibration, calib_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, np.ndarray]]:
    raw_camera = {
        "camera_name": calibration.camera_name,
        "camera_model": calibration.camera_params["cam_model"],
        "image_model": "raw_distorted",
        "undistorted": False,
        "rectified": False,
        "width": int(calibration.camera_params["image_width"]),
        "height": int(calibration.camera_params["image_height"]),
        "K_like": calibration.k_like,
        "distortion": calibration.distortion,
        "matrix_name": calibration.matrix_name,
        "T_camera_from_lidar": calibration.t_camera_from_lidar.tolist(),
    }
    write_json(calib_dir / "camera_raw.json", raw_camera)

    map_x, map_y, pinhole = build_fishpoly_rectification_maps(raw_camera)
    sampler = build_sampler(map_x, map_y, int(raw_camera["width"]), int(raw_camera["height"]))
    rectified_camera = {
        "camera_name": raw_camera["camera_name"],
        "camera_model": "PINHOLE",
        "image_model": "undistorted_pinhole",
        "undistorted": True,
        "rectified": True,
        "width": int(pinhole["width"]),
        "height": int(pinhole["height"]),
        "K_like": [
            [float(pinhole["fx"]), 0.0, float(pinhole["cx"])],
            [0.0, float(pinhole["fy"]), float(pinhole["cy"])],
            [0.0, 0.0, 1.0],
        ],
        "source_camera_model": raw_camera["camera_model"],
        "source_matrix_name": raw_camera["matrix_name"],
    }
    write_json(calib_dir / "camera_rectified.json", rectified_camera)
    write_yaml(calib_dir / "camera.yaml", {"raw_camera": raw_camera, "rectified_camera": rectified_camera})
    return raw_camera, rectified_camera, sampler


def load_bag_streams(db_path: Path, args: argparse.Namespace) -> BagStreams:
    connection = sqlite3.connect(str(db_path))
    topics = {name: topic_id for topic_id, name in connection.execute("SELECT id, name FROM topics")}

    def require_topic(name: str) -> int:
        if name not in topics:
            available = ", ".join(sorted(topics))
            raise KeyError(f"Missing topic {name!r} in {db_path.name}. Available: {available}")
        return int(topics[name])

    streams = BagStreams(
        image_rows=load_rows(connection, require_topic(args.image_topic)),
        odom_rows=load_rows(connection, require_topic(args.odom_topic)),
        imu_rows=load_rows(connection, require_topic(args.imu_topic)),
        raw_cloud_rows=load_rows(connection, require_topic(args.cloud_raw_topic)),
        slam_cloud_rows=load_rows(connection, require_topic(args.cloud_slam_topic)),
    )
    connection.close()
    return streams


def normalize_stream_timestamps_to_message_headers(streams: BagStreams) -> BagStreams:
    """Rebuild stream row timestamps from decoded message headers.

    The original pure-Python organizer reads `messages.timestamp` from the bag
    storage table. This helper keeps the same binary payloads but replaces the
    row timestamps with the message-level `header.stamp` values, matching the
    ROS runtime path more closely.
    """

    image_rows = []
    for _, blob in streams.image_rows:
        msg = parse_compressed_image(blob)
        timestamp_ns = int(msg["stamp_sec"]) * 1_000_000_000 + int(msg["stamp_nsec"])
        image_rows.append((timestamp_ns, blob))

    odom_rows = []
    for _, blob in streams.odom_rows:
        msg = parse_odometry(blob)
        timestamp_ns = int(msg["stamp_sec"]) * 1_000_000_000 + int(msg["stamp_nsec"])
        odom_rows.append((timestamp_ns, blob))

    raw_cloud_rows = []
    for _, blob in streams.raw_cloud_rows:
        msg = parse_pointcloud2(blob)
        timestamp_ns = int(msg["stamp_sec"]) * 1_000_000_000 + int(msg["stamp_nsec"])
        raw_cloud_rows.append((timestamp_ns, blob))

    slam_cloud_rows = []
    for _, blob in streams.slam_cloud_rows:
        msg = parse_pointcloud2(blob)
        timestamp_ns = int(msg["stamp_sec"]) * 1_000_000_000 + int(msg["stamp_nsec"])
        slam_cloud_rows.append((timestamp_ns, blob))

    imu_rows = []
    for _, blob in streams.imu_rows:
        msg = parse_imu(blob)
        imu_rows.append((int(msg["timestamp_ns"]), blob))

    return BagStreams(
        image_rows=image_rows,
        odom_rows=odom_rows,
        imu_rows=imu_rows,
        raw_cloud_rows=raw_cloud_rows,
        slam_cloud_rows=slam_cloud_rows,
    )


def export_images(image_rows: List[Tuple[int, bytes]], sampler: Dict[str, np.ndarray], paths: OutputPaths) -> None:
    from PIL import Image

    for image_index, (_, blob) in enumerate(image_rows):
        message = parse_compressed_image(blob)
        ext = ".jpg" if message["format"].lower().startswith("jp") else ".png"
        (paths.images_dir / f"{image_index:06d}{ext}").write_bytes(message["data"])
        decoded = decode_compressed_image(blob)
        rectified = remap_bilinear(decoded["image"], sampler)
        Image.fromarray(rectified[:, :, ::-1]).save(paths.images_rectified_dir / f"{image_index:06d}.jpg", quality=95)


def export_raw_clouds(
    raw_cloud_rows: List[Tuple[int, bytes]],
    odom_samples: List[OdomSample],
    paths: OutputPaths,
    raw_map_stride: int,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    raw_map_chunks: List[np.ndarray] = []
    raw_frame_rows: List[Dict[str, Any]] = []
    for raw_index, (timestamp_ns, blob) in enumerate(raw_cloud_rows):
        cloud_msg = parse_pointcloud2(blob)
        raw_payload = decode_raw_cloud(cloud_msg)
        pose_matrix, _ = interpolate_pose(int(timestamp_ns), odom_samples)
        frame_path = paths.lidar_raw_dir / f"{raw_index:06d}.npz"
        np.savez_compressed(
            frame_path,
            xyz=raw_payload["xyz"],
            intensity=raw_payload.get("intensity", np.zeros((len(raw_payload["xyz"]),), dtype=np.float32)),
            confidence=raw_payload.get("confidence", np.zeros((len(raw_payload["xyz"]),), dtype=np.float32)),
            offset_time=raw_payload.get("offset_time", np.zeros((len(raw_payload["xyz"]),), dtype=np.float32)),
            world_from_base=pose_matrix.astype(np.float64),
        )
        raw_frame_rows.append(
            {
                "raw_cloud_id": f"{raw_index:06d}",
                "timestamp_ns": int(timestamp_ns),
                "frame_id": cloud_msg["frame_id"],
                "num_points": int(len(raw_payload["xyz"])),
                "path": str(frame_path.relative_to(paths.root)),
            }
        )
        sampled = raw_payload["xyz"][:: max(raw_map_stride, 1)]
        if len(sampled):
            hom = np.concatenate([sampled.astype(np.float64), np.ones((len(sampled), 1), dtype=np.float64)], axis=1)
            raw_map_chunks.append((pose_matrix @ hom.T).T[:, :3])
    raw_global_map = np.concatenate(raw_map_chunks, axis=0) if raw_map_chunks else np.zeros((0, 3), dtype=np.float64)
    return raw_frame_rows, raw_global_map


def export_slam_clouds(
    slam_cloud_rows: List[Tuple[int, bytes]],
    paths: OutputPaths,
    slam_map_stride: int,
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
    slam_map_chunks: List[np.ndarray] = []
    slam_map_color_chunks: List[np.ndarray] = []
    slam_frame_rows: List[Dict[str, Any]] = []
    for slam_index, (timestamp_ns, blob) in enumerate(slam_cloud_rows):
        cloud_msg = parse_pointcloud2(blob)
        slam_payload = decode_slam_cloud(cloud_msg)
        frame_path = paths.lidar_slam_dir / f"{slam_index:06d}.npz"
        np.savez_compressed(
            frame_path,
            xyz=slam_payload["xyz"],
            rgb=slam_payload.get("rgb", np.zeros((len(slam_payload["xyz"]), 3), dtype=np.uint8)),
        )
        slam_frame_rows.append(
            {
                "slam_cloud_id": f"{slam_index:06d}",
                "timestamp_ns": int(timestamp_ns),
                "frame_id": cloud_msg["frame_id"],
                "num_points": int(len(slam_payload["xyz"])),
                "path": str(frame_path.relative_to(paths.root)),
            }
        )
        stride = max(slam_map_stride, 1)
        sampled = slam_payload["xyz"][::stride]
        if len(sampled):
            slam_map_chunks.append(sampled.astype(np.float64))
            sampled_rgb = slam_payload.get("rgb")
            if sampled_rgb is None:
                sampled_rgb = np.zeros((len(slam_payload["xyz"]), 3), dtype=np.uint8)
            slam_map_color_chunks.append(np.asarray(sampled_rgb[::stride], dtype=np.uint8))
    slam_global_map = np.concatenate(slam_map_chunks, axis=0) if slam_map_chunks else np.zeros((0, 3), dtype=np.float64)
    slam_global_map_rgb = (
        np.concatenate(slam_map_color_chunks, axis=0) if slam_map_color_chunks else np.zeros((0, 3), dtype=np.uint8)
    )
    return slam_frame_rows, slam_global_map, slam_global_map_rgb


def validate_extrinsic_direction(
    calibration: CameraCalibration,
    image_rows: List[Tuple[int, bytes]],
    raw_cloud_rows: List[Tuple[int, bytes]],
    image_timestamps: np.ndarray,
    raw_cloud_timestamps: np.ndarray,
    rectified_camera: Dict[str, Any],
    sampler: Dict[str, np.ndarray],
    paths: OutputPaths,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    import cv2

    overlap_start = max(int(image_timestamps[0]), int(raw_cloud_timestamps[0]))
    overlap_end = min(int(image_timestamps[-1]), int(raw_cloud_timestamps[-1]))
    overlap_indices = np.where((image_timestamps >= overlap_start) & (image_timestamps <= overlap_end))[0]
    sample_indices = overlap_indices[:: max(args.validation_frame_stride, 1)][: args.validation_samples]
    if len(sample_indices) == 0:
        sample_indices = overlap_indices[: min(len(overlap_indices), args.validation_samples)]
    raw_match_sample_indices = nearest_indices(image_timestamps[sample_indices], raw_cloud_timestamps)
    rectified_k = np.asarray(rectified_camera["K_like"], dtype=np.float64)

    candidates = {
        "camera_from_lidar": np.asarray(calibration.t_camera_from_lidar, dtype=np.float64),
        "lidar_from_camera": np.linalg.inv(np.asarray(calibration.t_camera_from_lidar, dtype=np.float64)),
    }
    extrinsic_summary: Dict[str, Any] = {"candidates": {}}
    candidate_scores: Dict[str, float] = {}
    for candidate_name, transform in candidates.items():
        candidate_dir = paths.reports_dir / "extrinsic_validation" / candidate_name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        samples = []
        for sample_rank, image_index in enumerate(sample_indices):
            image_timestamp_ns, image_blob = image_rows[int(image_index)]
            raw_timestamp_ns, raw_blob = raw_cloud_rows[int(raw_match_sample_indices[sample_rank])]
            decoded_image = decode_compressed_image(image_blob)
            rectified = remap_bilinear(decoded_image["image"], sampler)
            raw_payload = decode_raw_cloud(parse_pointcloud2(raw_blob))
            overlay, stats = build_projection_overlay(
                rectified,
                raw_payload["xyz"],
                transform,
                rectified_k,
                args.validation_max_points,
            )
            overlay_path = candidate_dir / f"{sample_rank:03d}_{int(image_timestamp_ns)}.png"
            cv2.imwrite(str(overlay_path), overlay)
            samples.append(
                {
                    "sample_index": int(sample_rank),
                    "image_index": int(image_index),
                    "image_timestamp_ns": int(image_timestamp_ns),
                    "cloud_timestamp_ns": int(raw_timestamp_ns),
                    "dt_ms": float(abs(int(image_timestamp_ns) - int(raw_timestamp_ns)) / 1e6),
                    "overlay_path": str(overlay_path.relative_to(paths.root)),
                    **stats,
                }
            )
        projected_ratios = np.asarray(
            [sample["projected_points"] / max(sample["input_points"], 1) for sample in samples],
            dtype=np.float64,
        )
        inside_ratios = np.asarray([sample["inside_ratio"] for sample in samples], dtype=np.float64)
        front_ratios = np.asarray([sample["front_ratio"] for sample in samples], dtype=np.float64)
        score = float(projected_ratios.mean() * 0.6 + inside_ratios.mean() * 0.3 + front_ratios.mean() * 0.1)
        candidate_scores[candidate_name] = score
        extrinsic_summary["candidates"][candidate_name] = {
            "transform": transform.tolist(),
            "score": score,
            "aggregate": {
                "mean_projected_ratio": float(projected_ratios.mean()) if len(projected_ratios) else 0.0,
                "mean_inside_ratio": float(inside_ratios.mean()) if len(inside_ratios) else 0.0,
                "mean_front_ratio": float(front_ratios.mean()) if len(front_ratios) else 0.0,
            },
            "samples": samples,
        }
    extrinsic_summary["selected_direction"] = max(candidate_scores.items(), key=lambda item: item[1])[0]
    write_json(paths.reports_dir / "extrinsic_validation" / "summary.json", extrinsic_summary)
    return extrinsic_summary


def build_tf_chain(calibration: CameraCalibration, extrinsic_summary: Dict[str, Any]) -> Dict[str, Any]:
    selected_direction = extrinsic_summary["selected_direction"]
    if selected_direction == "camera_from_lidar":
        t_camera_from_lidar = np.asarray(calibration.t_camera_from_lidar, dtype=np.float64)
        t_lidar_from_camera = np.linalg.inv(t_camera_from_lidar)
    else:
        t_lidar_from_camera = np.asarray(calibration.t_camera_from_lidar, dtype=np.float64)
        t_camera_from_lidar = np.linalg.inv(t_lidar_from_camera)

    t_imu_from_lidar = OFFICIAL_T_IMU_FROM_LIDAR.copy()
    t_lidar_from_imu = np.linalg.inv(t_imu_from_lidar)
    t_camera_from_imu = t_camera_from_lidar @ t_lidar_from_imu
    t_imu_from_camera = np.linalg.inv(t_camera_from_imu)
    return {
        "world_frame": "odom",
        "base_frame": "odin1_base_link",
        "imu_frame": "imu_link",
        "camera_frame": "camera",
        "lidar_frame_assumption": "odin1_base_link",
        "calibration_matrix_name": calibration.matrix_name,
        "selected_direction": selected_direction,
        "T_base_from_lidar": np.eye(4, dtype=np.float64).tolist(),
        "T_imu_from_lidar": t_imu_from_lidar.tolist(),
        "T_lidar_from_imu": t_lidar_from_imu.tolist(),
        "T_camera_from_lidar": t_camera_from_lidar.tolist(),
        "T_lidar_from_camera": t_lidar_from_camera.tolist(),
        "T_camera_from_imu": t_camera_from_imu.tolist(),
        "T_imu_from_camera": t_imu_from_camera.tolist(),
        "official_imu_lidar_basis": "Odin1 wiki section 5.2 外参描述: T^imu_lidar is fixed and maps lidar coordinates into the imu frame.",
        "assumption": "cloud_raw is treated as already expressed in odin1_base_link; adjust T_base_from_lidar only if projection validation reveals a fixed bias. The official IMU-LiDAR fixed transform is retained separately and is not collapsed into the base-lidar assumption.",
    }


def export_pose_and_association_tables(
    image_rows: List[Tuple[int, bytes]],
    odom_samples: List[OdomSample],
    imu_samples: List[Dict[str, Any]],
    odom_timestamps: np.ndarray,
    raw_cloud_timestamps: np.ndarray,
    slam_cloud_timestamps: np.ndarray,
    imu_timestamps: np.ndarray,
    image_to_odom_idx: np.ndarray,
    image_to_raw_idx: np.ndarray,
    image_to_slam_idx: np.ndarray,
    tf_chain: Dict[str, Any],
    paths: OutputPaths,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[float], List[float]]:
    t_lidar_from_camera = np.asarray(tf_chain["T_lidar_from_camera"], dtype=np.float64)
    base_pose_rows: List[Dict[str, Any]] = []
    camera_pose_rows: List[Dict[str, Any]] = []
    association_rows: List[Dict[str, Any]] = []
    odom_dt_values_ms: List[float] = []
    cloud_dt_values_ms: List[float] = []

    for image_index, (image_timestamp_ns, _) in enumerate(image_rows):
        frame_id = f"{image_index:06d}"
        odom_interp, interp_meta = interpolate_pose(int(image_timestamp_ns), odom_samples)
        odom_nearest_ts = int(odom_timestamps[int(image_to_odom_idx[image_index])])
        raw_nearest_ts = int(raw_cloud_timestamps[int(image_to_raw_idx[image_index])])
        slam_nearest_ts = int(slam_cloud_timestamps[int(image_to_slam_idx[image_index])])
        odom_dt_ms = abs(int(image_timestamp_ns) - odom_nearest_ts) / 1e6
        cloud_dt_ms = abs(int(image_timestamp_ns) - raw_nearest_ts) / 1e6
        odom_dt_values_ms.append(odom_dt_ms)
        cloud_dt_values_ms.append(cloud_dt_ms)

        odom_quaternion = matrix_to_quaternion_xyzw(odom_interp[:3, :3])
        world_from_camera = odom_interp @ t_lidar_from_camera
        camera_quaternion = matrix_to_quaternion_xyzw(world_from_camera[:3, :3])
        imu_window_start_ns = int(image_timestamp_ns - args.imu_window_ms * 1e6)
        imu_window_end_ns = int(image_timestamp_ns + args.imu_window_ms * 1e6)
        imu_left = int(np.searchsorted(imu_timestamps, imu_window_start_ns, side="left"))
        imu_right = int(np.searchsorted(imu_timestamps, imu_window_end_ns, side="right"))

        base_pose_rows.append(
            {
                "frame_id": frame_id,
                "image_timestamp_ns": int(image_timestamp_ns),
                "odom_prev_timestamp_ns": interp_meta["prev_timestamp_ns"],
                "odom_next_timestamp_ns": interp_meta["next_timestamp_ns"],
                "odom_interp_alpha": interp_meta["alpha"],
                "tx": float(odom_interp[0, 3]),
                "ty": float(odom_interp[1, 3]),
                "tz": float(odom_interp[2, 3]),
                "qx": float(odom_quaternion[0]),
                "qy": float(odom_quaternion[1]),
                "qz": float(odom_quaternion[2]),
                "qw": float(odom_quaternion[3]),
                **{name: value for name, value in zip(matrix_columns("T_odom_from_base"), matrix_values(odom_interp))},
            }
        )
        camera_pose_rows.append(
            {
                "frame_id": frame_id,
                "image_timestamp_ns": int(image_timestamp_ns),
                "odom_prev_timestamp_ns": interp_meta["prev_timestamp_ns"],
                "odom_next_timestamp_ns": interp_meta["next_timestamp_ns"],
                "odom_interp_alpha": interp_meta["alpha"],
                "tx": float(world_from_camera[0, 3]),
                "ty": float(world_from_camera[1, 3]),
                "tz": float(world_from_camera[2, 3]),
                "qx": float(camera_quaternion[0]),
                "qy": float(camera_quaternion[1]),
                "qz": float(camera_quaternion[2]),
                "qw": float(camera_quaternion[3]),
                **{
                    name: value
                    for name, value in zip(matrix_columns("T_odom_from_camera"), matrix_values(world_from_camera))
                },
            }
        )
        association_rows.append(
            {
                "frame_id": frame_id,
                "image_timestamp_ns": int(image_timestamp_ns),
                "image_path": f"images/{frame_id}.jpg",
                "rectified_image_path": f"images_rectified/{frame_id}.jpg",
                "raw_cloud_id": f"{int(image_to_raw_idx[image_index]):06d}",
                "raw_cloud_timestamp_ns": raw_nearest_ts,
                "image_to_raw_cloud_dt_ns": abs(int(image_timestamp_ns) - raw_nearest_ts),
                "raw_cloud_dt_outlier": cloud_dt_ms > float(args.cloud_max_dt_ms),
                "slam_cloud_id": f"{int(image_to_slam_idx[image_index]):06d}",
                "slam_cloud_timestamp_ns": slam_nearest_ts,
                "image_to_slam_cloud_dt_ns": abs(int(image_timestamp_ns) - slam_nearest_ts),
                "nearest_odom_id": f"{int(image_to_odom_idx[image_index]):06d}",
                "nearest_odom_timestamp_ns": odom_nearest_ts,
                "image_to_nearest_odom_dt_ns": abs(int(image_timestamp_ns) - odom_nearest_ts),
                "odom_prev_timestamp_ns": interp_meta["prev_timestamp_ns"],
                "odom_next_timestamp_ns": interp_meta["next_timestamp_ns"],
                "odom_interp_alpha": interp_meta["alpha"],
                "odom_dt_outlier": odom_dt_ms > float(args.odom_max_dt_ms),
                "imu_window_start_ns": imu_window_start_ns,
                "imu_window_end_ns": imu_window_end_ns,
                "imu_window_count": max(0, imu_right - imu_left),
            }
        )

    return base_pose_rows, camera_pose_rows, association_rows, odom_dt_values_ms, cloud_dt_values_ms


def compute_imu_odom_correlation(
    odom_samples: List[OdomSample],
    imu_samples: List[Dict[str, Any]],
    imu_timestamps: np.ndarray,
) -> float:
    imu_pairs = []
    for prev_sample, next_sample in zip(odom_samples[:-1], odom_samples[1:]):
        dt = max((next_sample.timestamp_ns - prev_sample.timestamp_ns) / 1e9, 1e-9)
        angular_speed = rotation_angle(
            quaternion_xyzw_to_matrix(prev_sample.quaternion_xyzw),
            quaternion_xyzw_to_matrix(next_sample.quaternion_xyzw),
        ) / dt
        left = int(np.searchsorted(imu_timestamps, prev_sample.timestamp_ns, side="left"))
        right = int(np.searchsorted(imu_timestamps, next_sample.timestamp_ns, side="right"))
        if right <= left:
            continue
        imu_norm = float(np.mean([np.linalg.norm(sample["angular_velocity"]) for sample in imu_samples[left:right]]))
        imu_pairs.append([imu_norm, angular_speed])
    if not imu_pairs:
        return 0.0
    imu_pair_arr = np.asarray(imu_pairs, dtype=np.float64)
    if np.allclose(imu_pair_arr[:, 0].std(), 0.0) or np.allclose(imu_pair_arr[:, 1].std(), 0.0):
        return 0.0
    return float(np.corrcoef(imu_pair_arr[:, 0], imu_pair_arr[:, 1])[0, 1])


def build_scene_summary(
    args: argparse.Namespace,
    bag_dir: Path,
    db_path: Path,
    calibration_path: Path,
    paths: OutputPaths,
    streams: BagStreams,
    image_timestamps: np.ndarray,
    odom_timestamps: np.ndarray,
    raw_cloud_timestamps: np.ndarray,
    slam_cloud_timestamps: np.ndarray,
    odom_dt_values_ms: List[float],
    cloud_dt_values_ms: List[float],
    extrinsic_summary: Dict[str, Any],
    raw_global_map: np.ndarray,
    slam_global_map: np.ndarray,
    imu_corr: float,
) -> Dict[str, Any]:
    raw_bbox = bbox_stats(raw_global_map)
    slam_bbox = bbox_stats(slam_global_map)
    centroid_delta = float(
        np.linalg.norm(
            np.asarray(raw_bbox["centroid"], dtype=np.float64) - np.asarray(slam_bbox["centroid"], dtype=np.float64)
        )
    )
    return {
        "scene_name": args.scene,
        "bag_dir": str(bag_dir),
        "bag_db_path": str(db_path),
        "output_dir": str(paths.root),
        "calibration_path": str(calibration_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "data_preparation.shared.organize_odin_scene",
        "num_images": len(streams.image_rows),
        "num_rectified_images": len(streams.image_rows),
        "num_raw_cloud_frames": len(streams.raw_cloud_rows),
        "num_slam_cloud_frames": len(streams.slam_cloud_rows),
        "num_imu_samples": len(streams.imu_rows),
        "num_odometry_samples": len(streams.odom_rows),
        "timing": {
            "image_to_odom_ms": np_stats(np.asarray(odom_dt_values_ms, dtype=np.float64)),
            "image_to_cloud_raw_ms": np_stats(np.asarray(cloud_dt_values_ms, dtype=np.float64)),
            "nearest_neighbor_image_to_odom_ms": nearest_neighbor_stats(image_timestamps, odom_timestamps),
            "nearest_neighbor_image_to_cloud_raw_ms": nearest_neighbor_stats(image_timestamps, raw_cloud_timestamps),
            "nearest_neighbor_image_to_cloud_slam_ms": nearest_neighbor_stats(image_timestamps, slam_cloud_timestamps),
        },
        "thresholds": {
            "odom_dt_ms": float(args.odom_max_dt_ms),
            "cloud_dt_ms": float(args.cloud_max_dt_ms),
            "imu_window_ms": float(args.imu_window_ms),
        },
        "extrinsic_validation": extrinsic_summary,
        "maps": {"raw_global_map": raw_bbox, "slam_global_map": slam_bbox},
        "validation": {
            "num_odom_dt_outliers": int(sum(value > float(args.odom_max_dt_ms) for value in odom_dt_values_ms)),
            "num_cloud_dt_outliers": int(sum(value > float(args.cloud_max_dt_ms) for value in cloud_dt_values_ms)),
            "imu_vs_odom_angular_speed_correlation": imu_corr,
            "raw_vs_slam_centroid_delta_m": centroid_delta,
        },
    }


def main() -> None:
    args = parse_args()
    bag_dir = args.bag_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else thesis_root_from_here() / "04_ProcessedData" / "rosbag_prepared" / f"{args.scene}_pure_headerstamp"
    )
    calibration_path = resolve_calibration_path(bag_dir, args.calibration)
    _prepare_output_dir(output_dir, overwrite=args.overwrite)

    db_path = next(iter(sorted(bag_dir.glob("*.db3"))), None)
    if db_path is None:
        raise FileNotFoundError(f"No .db3 file found under {bag_dir}")

    paths = build_output_paths(output_dir)
    ensure_output_subdirs(paths)

    calibration = parse_camera_lidar_calibration(calibration_path)
    raw_camera, rectified_camera, sampler = build_camera_models(calibration, paths.calib_dir)
    streams = normalize_stream_timestamps_to_message_headers(load_bag_streams(db_path, args))

    odom_samples = parse_odom_rows(streams.odom_rows)
    imu_samples = [parse_imu(blob) for _, blob in streams.imu_rows]
    odom_timestamps = np.asarray([sample.timestamp_ns for sample in odom_samples], dtype=np.int64)
    image_timestamps = np.asarray([int(ts) for ts, _ in streams.image_rows], dtype=np.int64)
    raw_cloud_timestamps = np.asarray([int(ts) for ts, _ in streams.raw_cloud_rows], dtype=np.int64)
    slam_cloud_timestamps = np.asarray([int(ts) for ts, _ in streams.slam_cloud_rows], dtype=np.int64)
    imu_timestamps = np.asarray([sample["timestamp_ns"] for sample in imu_samples], dtype=np.int64)

    image_to_odom_idx = nearest_indices(image_timestamps, odom_timestamps)
    image_to_raw_idx = nearest_indices(image_timestamps, raw_cloud_timestamps)
    image_to_slam_idx = nearest_indices(image_timestamps, slam_cloud_timestamps)

    export_images(streams.image_rows, sampler, paths)

    raw_frame_rows, raw_global_map = export_raw_clouds(
        streams.raw_cloud_rows,
        odom_samples,
        paths,
        args.raw_map_stride,
    )
    slam_frame_rows, slam_global_map, slam_global_map_rgb = export_slam_clouds(
        streams.slam_cloud_rows,
        paths,
        args.slam_map_stride,
    )
    write_ascii_ply(paths.root / "lidar" / "global_map_raw_odom.ply", raw_global_map)
    write_ascii_ply(paths.root / "lidar" / "global_map_slam_odom.ply", slam_global_map, slam_global_map_rgb)

    extrinsic_summary = validate_extrinsic_direction(
        calibration,
        streams.image_rows,
        streams.raw_cloud_rows,
        image_timestamps,
        raw_cloud_timestamps,
        rectified_camera,
        sampler,
        paths,
        args,
    )
    selected_direction = extrinsic_summary["selected_direction"]
    tf_chain = build_tf_chain(calibration, extrinsic_summary)
    write_json(paths.calib_dir / "tf_chain.json", tf_chain)

    write_csv(
        paths.imu_dir / "imu.csv",
        ["imu_id", "timestamp_ns", "frame_id", "qx", "qy", "qz", "qw", "wx", "wy", "wz", "ax", "ay", "az"],
        [[f"{idx:06d}", sample["timestamp_ns"], sample["frame_id"], *sample["orientation_xyzw"].tolist(), *sample["angular_velocity"].tolist(), *sample["linear_acceleration"].tolist()] for idx, sample in enumerate(imu_samples)],
    )
    write_csv(
        paths.poses_dir / "odometry.csv",
        ["odom_id", "timestamp_ns", "frame_id", "child_frame_id", "tx", "ty", "tz", "qx", "qy", "qz", "qw", *matrix_columns("T_odom_from_base")],
        [[f"{idx:06d}", sample.timestamp_ns, sample.frame_id, sample.child_frame_id, *sample.position.tolist(), *sample.quaternion_xyzw.tolist(), *matrix_values(sample.world_from_base)] for idx, sample in enumerate(odom_samples)],
    )

    base_pose_rows, camera_pose_rows, association_rows, odom_dt_values_ms, cloud_dt_values_ms = export_pose_and_association_tables(
        streams.image_rows,
        odom_samples,
        imu_samples,
        odom_timestamps,
        raw_cloud_timestamps,
        slam_cloud_timestamps,
        imu_timestamps,
        image_to_odom_idx,
        image_to_raw_idx,
        image_to_slam_idx,
        tf_chain,
        paths,
        args,
    )

    write_dict_csv(
        paths.poses_dir / "base_poses_for_images.csv",
        [
            "frame_id", "image_timestamp_ns", "odom_prev_timestamp_ns", "odom_next_timestamp_ns",
            "odom_interp_alpha", "tx", "ty", "tz", "qx", "qy", "qz", "qw", *matrix_columns("T_odom_from_base"),
        ],
        base_pose_rows,
    )
    write_dict_csv(
        paths.poses_dir / "camera_poses.csv",
        [
            "frame_id", "image_timestamp_ns", "odom_prev_timestamp_ns", "odom_next_timestamp_ns",
            "odom_interp_alpha", "tx", "ty", "tz", "qx", "qy", "qz", "qw", *matrix_columns("T_odom_from_camera"),
        ],
        camera_pose_rows,
    )
    write_dict_csv(
        paths.associations_dir / "frame_associations.csv",
        [
            "frame_id", "image_timestamp_ns", "image_path", "rectified_image_path", "raw_cloud_id",
            "raw_cloud_timestamp_ns", "image_to_raw_cloud_dt_ns", "raw_cloud_dt_outlier", "slam_cloud_id",
            "slam_cloud_timestamp_ns", "image_to_slam_cloud_dt_ns", "nearest_odom_id",
            "nearest_odom_timestamp_ns", "image_to_nearest_odom_dt_ns", "odom_prev_timestamp_ns",
            "odom_next_timestamp_ns", "odom_interp_alpha", "odom_dt_outlier", "imu_window_start_ns",
            "imu_window_end_ns", "imu_window_count",
        ],
        association_rows,
    )
    write_dict_csv(
        paths.root / "lidar" / "raw_frames_manifest.csv",
        ["raw_cloud_id", "timestamp_ns", "frame_id", "num_points", "path"],
        raw_frame_rows,
    )
    write_dict_csv(
        paths.root / "lidar" / "slam_frames_manifest.csv",
        ["slam_cloud_id", "timestamp_ns", "frame_id", "num_points", "path"],
        slam_frame_rows,
    )

    imu_corr = compute_imu_odom_correlation(odom_samples, imu_samples, imu_timestamps)
    summary = build_scene_summary(
        args,
        bag_dir,
        db_path,
        calibration_path,
        paths,
        streams,
        image_timestamps,
        odom_timestamps,
        raw_cloud_timestamps,
        slam_cloud_timestamps,
        odom_dt_values_ms,
        cloud_dt_values_ms,
        extrinsic_summary,
        raw_global_map,
        slam_global_map,
        imu_corr,
    )
    write_json(paths.root / "scene_meta.json", summary)
    write_json(paths.reports_dir / "organization_summary.json", summary)
    write_summary_markdown(paths.reports_dir / "organization_summary.md", summary)

    print(f"[INFO] Organized scene written to {paths.root}")
    print(
        f"[INFO] images={len(streams.image_rows)} raw_clouds={len(streams.raw_cloud_rows)} "
        f"slam_clouds={len(streams.slam_cloud_rows)} imu={len(streams.imu_rows)}"
    )
    print(f"[INFO] selected_extrinsic={selected_direction} summary={paths.reports_dir / 'organization_summary.md'}")


if __name__ == "__main__":
    main()
