from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


def load_pose_rows(path: Path) -> Dict[str, dict]:
    with path.open(newline="") as handle:
        return {row["frame_id"]: row for row in csv.DictReader(handle)}


def load_pose_rows_ordered(path: Path) -> List[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return sorted(rows, key=lambda row: row["frame_id"])


def world_from_camera_from_row(row: dict) -> np.ndarray:
    if "T_world_from_camera_rowmajor_00" in row:
        values = [
            float(row[f"T_world_from_camera_rowmajor_{idx // 4}{idx % 4}"])
            for idx in range(16)
        ]
        return np.asarray(values, dtype=np.float64).reshape(4, 4)

    qx = float(row["qx"])
    qy = float(row["qy"])
    qz = float(row["qz"])
    qw = float(row["qw"])
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_xyzw_to_matrix(np.asarray([qx, qy, qz, qw], dtype=np.float64))
    matrix[:3, 3] = np.asarray([float(row["tx"]), float(row["ty"]), float(row["tz"])], dtype=np.float64)
    return matrix


def camera_from_world(world_from_camera: np.ndarray) -> np.ndarray:
    return np.linalg.inv(world_from_camera)


def pose_to_matrix(position: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_xyzw_to_matrix(quaternion_xyzw)
    matrix[:3, 3] = position
    return matrix


def quaternion_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
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


def rotmat_to_qvec_colmap(rotation: np.ndarray) -> np.ndarray:
    rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz = rotation.flat
    k_mat = (
        np.asarray(
            [
                [rxx - ryy - rzz, 0, 0, 0],
                [ryx + rxy, ryy - rxx - rzz, 0, 0],
                [rzx + rxz, rzy + ryz, rzz - rxx - ryy, 0],
                [ryz - rzy, rzx - rxz, rxy - ryx, rxx + ryy + rzz],
            ],
            dtype=np.float64,
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(k_mat)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec / np.linalg.norm(qvec)


def frame_ids(rows: Iterable[dict]) -> List[str]:
    return [row["frame_id"] for row in rows]
