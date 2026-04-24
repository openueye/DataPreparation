from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def decode_ros2_xyz_points(pointcloud: Dict[str, Any]) -> np.ndarray:
    fields = {field["name"]: field for field in pointcloud["fields"]}
    if not {"x", "y", "z"} <= set(fields):
        raise ValueError("Point cloud does not contain x/y/z fields.")
    point_step = pointcloud["point_step"]
    endian = ">" if pointcloud.get("is_bigendian") else "<"
    dtype = np.dtype(
        {
            "names": ["x", "y", "z"],
            "formats": [f"{endian}f4", f"{endian}f4", f"{endian}f4"],
            "offsets": [fields["x"]["offset"], fields["y"]["offset"], fields["z"]["offset"]],
            "itemsize": point_step,
        }
    )
    structured = np.frombuffer(pointcloud["data"], dtype=dtype, count=pointcloud["width"] * pointcloud["height"])
    xyz = np.stack([structured["x"], structured["y"], structured["z"]], axis=1)
    finite = np.isfinite(xyz).all(axis=1)
    return xyz[finite]


def write_ascii_ply(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("end_header\n")
        for point in points:
            handle.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")


def read_ply_xyz(path: Path) -> np.ndarray:
    from plyfile import PlyData

    ply = PlyData.read(path)
    vertices = ply["vertex"]
    return np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T.astype(np.float32)


def read_ply_xyz_rgb(path: Path, *, default_rgb: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    from plyfile import PlyData

    ply = PlyData.read(path)
    vertices = ply["vertex"]
    xyz = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T.astype(np.float32)
    names = vertices.data.dtype.names
    if all(name in names for name in ("red", "green", "blue")):
        rgb = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T.astype(np.uint8)
    else:
        rgb = np.full((xyz.shape[0], 3), default_rgb, dtype=np.uint8)
    finite = np.isfinite(xyz).all(axis=1)
    return xyz[finite], rgb[finite]


def write_colorized_ply(path: Path, xyz: np.ndarray, rgb_u8: np.ndarray) -> None:
    from plyfile import PlyData, PlyElement

    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    elements = np.empty(xyz.shape[0], dtype=dtype)
    elements["x"] = xyz[:, 0]
    elements["y"] = xyz[:, 1]
    elements["z"] = xyz[:, 2]
    elements["nx"] = 0.0
    elements["ny"] = 0.0
    elements["nz"] = 0.0
    elements["red"] = rgb_u8[:, 0]
    elements["green"] = rgb_u8[:, 1]
    elements["blue"] = rgb_u8[:, 2]
    PlyData([PlyElement.describe(elements, "vertex")], text=False).write(path)
