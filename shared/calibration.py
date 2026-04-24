from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np


DISTORTION_KEYS = ("k2", "k3", "k4", "k5", "k6", "k7", "p1", "p2")


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


def parse_camera_lidar_calibration(path: Path, *, strict_matrix: bool = True) -> Dict[str, Any]:
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

    matrix = None
    if len(matrix_values) == 16:
        matrix = np.asarray(matrix_values, dtype=np.float64).reshape(4, 4)
    elif strict_matrix:
        raise ValueError(f"Expected one 4x4 calibration matrix in {path}, got {len(matrix_values)} values.")

    k_like = None
    if {"A11", "A12", "A22", "u0", "v0"} <= set(camera_params):
        k_like = [
            [camera_params["A11"], camera_params["A12"], camera_params["u0"]],
            [0.0, camera_params["A22"], camera_params["v0"]],
            [0.0, 0.0, 1.0],
        ]

    distortion_dict = {key: camera_params.get(key) for key in DISTORTION_KEYS}
    return {
        "path": str(path),
        "matrix_name": matrix_name,
        "matrix": matrix.tolist() if matrix is not None else None,
        "T_camera_from_lidar": matrix,
        "camera_name": camera_name,
        "camera_params": camera_params,
        "K_like": k_like,
        "distortion": distortion_dict,
        "distortion_vector": [distortion_dict[key] for key in DISTORTION_KEYS],
    }
