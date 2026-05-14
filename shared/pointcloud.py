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


def _voxelize_average(xyz: np.ndarray, rgb: np.ndarray, voxel_size: float) -> Tuple[np.ndarray, np.ndarray]:
    if voxel_size <= 0.0:
        raise ValueError("voxel_size must be positive.")
    origin = np.min(xyz, axis=0)
    keys = np.floor((xyz - origin) / voxel_size).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    count = int(inverse.max()) + 1 if inverse.size else 0
    sums_xyz = np.zeros((count, 3), dtype=np.float64)
    sums_rgb = np.zeros((count, 3), dtype=np.float64)
    counts = np.bincount(inverse, minlength=count).astype(np.float64)
    np.add.at(sums_xyz, inverse, xyz.astype(np.float64, copy=False))
    np.add.at(sums_rgb, inverse, rgb.astype(np.float64, copy=False))
    down_xyz = (sums_xyz / counts[:, None]).astype(np.float32)
    down_rgb = np.rint(sums_rgb / counts[:, None]).clip(0, 255).astype(np.uint8)
    return down_xyz, down_rgb


def voxel_downsample_to_max_points(
    xyz: np.ndarray,
    rgb: np.ndarray,
    max_points: int,
    *,
    search_iterations: int = 12,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    xyz = np.asarray(xyz)
    rgb = np.asarray(rgb)
    if xyz.shape[0] != rgb.shape[0]:
        raise ValueError(f"xyz/rgb point count mismatch: {xyz.shape[0]} != {rgb.shape[0]}")
    input_points = int(xyz.shape[0])
    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    rgb = np.clip(rgb[finite], 0, 255).astype(np.uint8)
    source_points = int(finite.sum())
    metadata: Dict[str, object] = {
        "method": "voxel_grid",
        "target_max_points": int(max_points),
        "input_points": input_points,
        "source_points": source_points,
        "finite_points": source_points,
        "voxel_size": None,
    }
    if source_points == 0 or max_points <= 0 or source_points <= max_points:
        metadata["written_points"] = source_points
        metadata["downsampled"] = False
        return xyz.astype(np.float32, copy=False), rgb, metadata

    bounds = np.ptp(xyz.astype(np.float64, copy=False), axis=0)
    diagonal = float(np.linalg.norm(bounds))
    if diagonal <= 0.0:
        xyz_down = xyz[:1].astype(np.float32, copy=False)
        rgb_down = rgb[:1]
        metadata.update({"written_points": int(xyz_down.shape[0]), "downsampled": True, "voxel_size": 0.0})
        return xyz_down, rgb_down, metadata

    high = max(diagonal / (float(max_points) ** (1.0 / 3.0)), np.finfo(np.float64).eps)
    xyz_high, rgb_high = _voxelize_average(xyz, rgb, high)
    while xyz_high.shape[0] > max_points:
        high *= 2.0
        xyz_high, rgb_high = _voxelize_average(xyz, rgb, high)

    best_xyz, best_rgb, best_size = xyz_high, rgb_high, high
    low = 0.0
    for _ in range(search_iterations):
        mid = (low + high) / 2.0
        xyz_mid, rgb_mid = _voxelize_average(xyz, rgb, mid)
        if xyz_mid.shape[0] > max_points:
            low = mid
        else:
            high = mid
            best_xyz, best_rgb, best_size = xyz_mid, rgb_mid, mid

    metadata.update(
        {
            "written_points": int(best_xyz.shape[0]),
            "downsampled": True,
            "voxel_size": float(best_size),
        }
    )
    return best_xyz, best_rgb, metadata


def write_downsampled_colmap_points_with_ply(
    points3d_txt: Path,
    points3d_ply: Path,
    xyz: np.ndarray,
    rgb: np.ndarray,
    *,
    max_points: int,
) -> Dict[str, object]:
    from data_preparation.shared.colmap_io import write_points3d_text

    down_xyz, down_rgb, metadata = voxel_downsample_to_max_points(xyz, rgb, max_points=max_points)
    write_points3d_text(points3d_txt, down_xyz, down_rgb)
    write_colorized_ply(points3d_ply, down_xyz, down_rgb)
    metadata.update(
        {
            "points3D_txt": str(points3d_txt),
            "points3D_ply": str(points3d_ply),
        }
    )
    return metadata
