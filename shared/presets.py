from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class PrepPreset:
    name: str
    limit_images: Optional[int]
    projection_frames: Optional[List[str]]
    max_overlay_points: int
    export_max_points: int
    video_max_frames: int
    colorize_sample_points: int


PRESETS: Dict[str, PrepPreset] = {
    "smoke": PrepPreset(
        name="smoke",
        limit_images=200,
        projection_frames=["000000", "000500", "001000"],
        max_overlay_points=3000,
        export_max_points=5000,
        video_max_frames=120,
        colorize_sample_points=200_000,
    ),
    "full": PrepPreset(
        name="full",
        limit_images=None,
        projection_frames=None,
        max_overlay_points=12000,
        export_max_points=300_000,
        video_max_frames=0,
        colorize_sample_points=0,
    ),
    "debug": PrepPreset(
        name="debug",
        limit_images=500,
        projection_frames=["000000", "000250", "000500", "001000", "001500", "002000"],
        max_overlay_points=12000,
        export_max_points=50_000,
        video_max_frames=300,
        colorize_sample_points=500_000,
    ),
}


def get_preset(name: str) -> PrepPreset:
    try:
        return PRESETS[name]
    except KeyError as exc:
        valid = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown preset '{name}'. Expected one of: {valid}") from exc
