from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent)


def load_csv_rows(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_csv_rows_by_key(path: Path, key: str) -> Dict[str, dict]:
    if not path.exists():
        return {}
    return {row[key]: row for row in load_csv_rows(path)}


def find_image_path(images_dir: Path, frame_id: str, extensions: Iterable[str] = IMAGE_EXTENSIONS) -> Path:
    for ext in extensions:
        candidate = images_dir / f"{frame_id}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image found for frame {frame_id} under {images_dir}")


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def require_cv2(purpose: str):
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"OpenCV is required for {purpose}. Install opencv-python or opencv-contrib-python."
        ) from exc
    return cv2
