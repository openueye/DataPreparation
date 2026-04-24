from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def write_cameras_text(path: Path, *, width: int, height: int, fx: float, fy: float, cx: float, cy: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Camera list with one line of data per camera:\n")
        handle.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        handle.write("# Number of cameras: 1\n")
        handle.write(f"1 PINHOLE {width} {height} {fx:.12g} {fy:.12g} {cx:.12g} {cy:.12g}\n")


def write_images_text(path: Path, image_records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = list(image_records)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Image list with two lines of data per image:\n")
        handle.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        handle.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        handle.write(f"# Number of images: {len(records)}, mean observations per image: 0\n")
        for record in records:
            qvec = record["qvec"]
            tvec = record["tvec"]
            handle.write(
                f"{record['image_id']} "
                f"{qvec[0]:.17g} {qvec[1]:.17g} {qvec[2]:.17g} {qvec[3]:.17g} "
                f"{tvec[0]:.17g} {tvec[1]:.17g} {tvec[2]:.17g} "
                f"1 {record['image_name']}\n"
            )
            handle.write("\n")


def write_points3d_text(path: Path, xyz: np.ndarray, rgb: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rgb is None:
        rgb = np.full((xyz.shape[0], 3), 128, dtype=np.uint8)
    rgb = np.asarray(rgb)
    if rgb.dtype.kind == "f":
        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    else:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    with path.open("w", encoding="utf-8") as handle:
        handle.write("# 3D point list with one line of data per point:\n")
        handle.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        handle.write(f"# Number of points: {xyz.shape[0]}, mean track length: 0\n")
        for idx, (point, color) in enumerate(zip(xyz, rgb), start=1):
            handle.write(
                f"{idx} {point[0]:.17g} {point[1]:.17g} {point[2]:.17g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} 0\n"
            )
