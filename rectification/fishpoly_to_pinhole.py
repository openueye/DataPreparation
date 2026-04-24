#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image

try:
    from data_preparation.shared.camera_models import build_fishpoly_rectification_maps, load_camera_json
    from data_preparation.shared.io import IMAGE_EXTENSIONS, write_json
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from data_preparation.shared.camera_models import build_fishpoly_rectification_maps, load_camera_json
    from data_preparation.shared.io import IMAGE_EXTENSIONS, write_json


def _link_or_copy(source: Path, target: Path, *, copy: bool) -> None:
    if not source.exists():
        return
    if target.exists() or target.is_symlink():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
    else:
        os.symlink(source, target, target_is_directory=source.is_dir())


def _image_files(images_dir: Path):
    return sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is non-empty: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _build_sampler(map_x: np.ndarray, map_y: np.ndarray, source_width: int, source_height: int) -> Dict[str, np.ndarray]:
    valid = (map_x >= 0.0) & (map_x <= source_width - 1) & (map_y >= 0.0) & (map_y <= source_height - 1)
    x0 = np.floor(np.clip(map_x, 0, source_width - 1)).astype(np.int32)
    y0 = np.floor(np.clip(map_y, 0, source_height - 1)).astype(np.int32)
    x1 = np.minimum(x0 + 1, source_width - 1)
    y1 = np.minimum(y0 + 1, source_height - 1)
    wx = (np.clip(map_x, 0, source_width - 1) - x0).astype(np.float32)
    wy = (np.clip(map_y, 0, source_height - 1) - y0).astype(np.float32)
    return {"valid": valid, "x0": x0, "x1": x1, "y0": y0, "y1": y1, "wx": wx, "wy": wy}


def _remap_bilinear(image: np.ndarray, sampler: Dict[str, np.ndarray]) -> np.ndarray:
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


def rectify_scene(
    scene_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
    copy_links: bool = False,
    limit_images: Optional[int] = None,
    jpeg_quality: int = 95,
    fx: Optional[float] = None,
    fy: Optional[float] = None,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
) -> Dict[str, object]:
    scene_dir = scene_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    images_dir = scene_dir / "images"
    camera_path = scene_dir / "intrinsics" / "camera.json"
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")
    if not camera_path.exists():
        raise FileNotFoundError(f"Missing camera metadata: {camera_path}")

    camera_data = load_camera_json(camera_path)
    if str(camera_data.get("camera_model")) != "FishPoly":
        raise ValueError(f"Expected FishPoly camera_model, got {camera_data.get('camera_model')!r}.")

    _prepare_output_dir(output_dir, overwrite=overwrite)
    output_images = output_dir / "images"
    output_intrinsics = output_dir / "intrinsics"
    output_images.mkdir(parents=True, exist_ok=True)
    output_intrinsics.mkdir(parents=True, exist_ok=True)

    map_x, map_y, pinhole = build_fishpoly_rectification_maps(camera_data, fx=fx, fy=fy, cx=cx, cy=cy)
    sampler = _build_sampler(map_x, map_y, int(camera_data["width"]), int(camera_data["height"]))

    source_images = _image_files(images_dir)
    if limit_images is not None:
        source_images = source_images[: max(limit_images, 0)]

    for index, image_path in enumerate(source_images, start=1):
        image = Image.open(image_path).convert("RGB")
        if image.size != (int(camera_data["width"]), int(camera_data["height"])):
            raise ValueError(f"Image size mismatch for {image_path}: got {image.size}, expected {(camera_data['width'], camera_data['height'])}.")
        rectified = _remap_bilinear(np.asarray(image), sampler)
        output_path = output_images / image_path.name
        Image.fromarray(rectified).save(output_path, quality=int(jpeg_quality))
        if index % 100 == 0:
            print(f"[INFO] rectified {index}/{len(source_images)} images")

    for name in ("poses", "lidar", "transforms", "metadata"):
        _link_or_copy(scene_dir / name, output_dir / name, copy=copy_links)
    for name in ("scene_meta.json",):
        _link_or_copy(scene_dir / name, output_dir / name, copy=copy_links)

    rectified_camera = {
        "camera_name": camera_data.get("camera_name", "cam_0"),
        "camera_model": "PINHOLE",
        "image_model": "undistorted_pinhole",
        "undistorted": True,
        "rectified": True,
        "colmap_training_ready": True,
        "width": pinhole["width"],
        "height": pinhole["height"],
        "K_like": [
            [pinhole["fx"], 0.0, pinhole["cx"]],
            [0.0, pinhole["fy"], pinhole["cy"]],
            [0.0, 0.0, 1.0],
        ],
        "distortion": {"k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0},
        "source_camera": camera_data,
        "rectification": {
            "model": "FishPoly_to_PINHOLE",
            "implementation_reference": "ManifoldTechLtd/Odin-Nav-Stack ros_ws/src/fish2pinhole/src/fish2pinhole_node.cpp",
            "invalid_source_pixels": "black",
        },
    }
    write_json(output_intrinsics / "camera.json", rectified_camera)

    report = {
        "source_scene": str(scene_dir),
        "output_scene": str(output_dir),
        "source_camera_model": camera_data.get("camera_model"),
        "output_camera_model": "PINHOLE",
        "output_intrinsics": rectified_camera["K_like"],
        "num_images": len(source_images),
        "jpeg_quality": int(jpeg_quality),
        "colmap_training_ready": True,
    }
    write_json(output_dir / "fishpoly_rectification_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rectify a FishPoly SLAM/LiDAR scene into undistorted pinhole images.")
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--copy-links", action="store_true", help="Copy poses/lidar/transforms instead of symlinking them.")
    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = rectify_scene(
        args.scene_dir,
        args.output_dir,
        overwrite=args.overwrite,
        copy_links=args.copy_links,
        limit_images=args.limit_images,
        jpeg_quality=args.jpeg_quality,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
    )
    print(f"[INFO] Rectified FishPoly scene written: {report['output_scene']}")
    print(f"[INFO] images={report['num_images']} report={Path(report['output_scene']) / 'fishpoly_rectification_report.json'}")


if __name__ == "__main__":
    main()
