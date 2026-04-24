from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def load_camera_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def pinhole_from_k_like(camera_data: Dict[str, object]) -> Tuple[int, int, float, float, float, float]:
    k_mat = np.asarray(camera_data["K_like"], dtype=np.float64)
    width = int(camera_data["width"])
    height = int(camera_data["height"])
    fx = float(k_mat[0, 0])
    fy = float(k_mat[1, 1])
    cx = float(k_mat[0, 2])
    cy = float(k_mat[1, 2])
    return width, height, fx, fy, cx, cy


def fishpoly_coefficients(camera_data: Dict[str, object]) -> Tuple[float, float, float, float, float, float]:
    distortion = camera_data.get("distortion") or {}
    if isinstance(distortion, dict):
        return tuple(float(distortion[key]) for key in ("k2", "k3", "k4", "k5", "k6", "k7"))
    values = list(distortion)
    if len(values) < 6:
        raise ValueError("FishPoly distortion must provide k2..k7.")
    return tuple(float(value) for value in values[:6])


def default_fishpoly_rectified_pinhole(camera_data: Dict[str, object]) -> Tuple[int, int, float, float, float, float]:
    """Match the upstream Odin fish2pinhole node's default output camera."""

    width = int(camera_data["width"])
    height = int(camera_data["height"])
    return width, height, width / 2.0, height / 2.0, width / 2.0, height / 2.0


def build_fishpoly_rectification_maps(
    camera_data: Dict[str, object],
    *,
    output_width: int | None = None,
    output_height: int | None = None,
    fx: float | None = None,
    fy: float | None = None,
    cx: float | None = None,
    cy: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Return source-pixel maps for rectifying FishPoly images to pinhole.

    The model follows ManifoldTechLtd/Odin-Nav-Stack fish2pinhole:
    theta = atan(r)
    theta_d = theta * (1 + k2*theta^2 + ... + k7*theta^7)
    u_d = A11*x_d + A12*y_d + u0
    v_d = A22*y_d + v0
    """

    src_width, src_height, a11, a22, u0, v0 = pinhole_from_k_like(camera_data)
    k2, k3, k4, k5, k6, k7 = fishpoly_coefficients(camera_data)
    k_mat = np.asarray(camera_data["K_like"], dtype=np.float64)
    a12 = float(k_mat[0, 1])

    if output_width is None:
        output_width = src_width
    if output_height is None:
        output_height = src_height
    if fx is None:
        fx = output_width / 2.0
    if fy is None:
        fy = output_height / 2.0
    if cx is None:
        cx = output_width / 2.0
    if cy is None:
        cy = output_height / 2.0

    u, v = np.meshgrid(
        np.arange(output_width, dtype=np.float64),
        np.arange(output_height, dtype=np.float64),
    )
    x = (u - float(cx)) / float(fx)
    y = (v - float(cy)) / float(fy)
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

    pinhole = {
        "width": int(output_width),
        "height": int(output_height),
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
    }
    return map_x.astype(np.float32), map_y.astype(np.float32), pinhole


def project_pinhole_k_like(points_camera: np.ndarray, k_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    depth = points_camera[:, 2]
    in_front = depth > 1e-6
    uv = np.full((points_camera.shape[0], 2), np.nan, dtype=np.float64)
    front = points_camera[in_front]
    front_depth = front[:, 2]
    uv[in_front, 0] = (k_mat[0, 0] * front[:, 0] + k_mat[0, 1] * front[:, 1]) / front_depth + k_mat[0, 2]
    uv[in_front, 1] = (k_mat[1, 0] * front[:, 0] + k_mat[1, 1] * front[:, 1]) / front_depth + k_mat[1, 2]
    return uv, in_front


def project_world_to_pinhole(
    points_world: np.ndarray,
    world_from_camera: np.ndarray,
    k_mat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rotation_wc = world_from_camera[:3, :3]
    translation_wc = world_from_camera[:3, 3]
    points_camera = (points_world.astype(np.float64) - translation_wc) @ rotation_wc
    uv, in_front = project_pinhole_k_like(points_camera, k_mat)
    return uv, points_camera[:, 2], in_front
