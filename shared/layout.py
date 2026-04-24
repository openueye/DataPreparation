from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _abs(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def _find_thesis_root(start: Path | str) -> Path:
    current = _abs(start)
    for candidate in (current, *current.parents):
        if candidate.name == "Thesis":
            return candidate
    raise ValueError(f"Could not infer Thesis root from: {start}")


@dataclass(frozen=True)
class DataPrepLayout:
    """Canonical data-preparation paths mirroring 3DGS_baseline01.common.paths."""

    thesis_root: Path

    @classmethod
    def from_repo_root(
        cls,
        repo_root: Optional[Path | str] = None,
        thesis_root: Optional[Path | str] = None,
    ) -> "DataPrepLayout":
        if thesis_root:
            return cls(_abs(thesis_root))
        if repo_root is None:
            repo_root = Path(__file__).resolve().parents[2]
        return cls(_find_thesis_root(repo_root))

    @property
    def datasets_root(self) -> Path:
        return self.thesis_root / "03_Datasets"

    @property
    def rosbag_root(self) -> Path:
        return self.datasets_root / "001_rosbags"

    @property
    def video_root(self) -> Path:
        return self.datasets_root / "003_videos"

    @property
    def processed_root(self) -> Path:
        return self.thesis_root / "04_ProcessedData"

    @property
    def outputs_root(self) -> Path:
        return self.thesis_root / "05_Outputs"

    @property
    def colmap_scenes_root(self) -> Path:
        return self.processed_root / "010_scenes_colmap"

    @property
    def lidar_scenes_root(self) -> Path:
        return self.processed_root / "011_scenes_lidar"

    @property
    def validation_root(self) -> Path:
        return self.outputs_root / "030_validation"

    def bag_dir(self, scene: str) -> Path:
        return self.rosbag_root / scene

    def calibration_file(self, override: Optional[Path | str] = None) -> Path:
        if override:
            return _abs(override)
        return self.rosbag_root / "cam_in_ex.txt"

    def lidar_scene_dir(self, scene: str) -> Path:
        return self.lidar_scenes_root / scene

    def rectified_lidar_scene_dir(self, scene: str) -> Path:
        return self.lidar_scenes_root / f"{scene}_pinhole_rectified"

    def lidar_scene_dir_for_preset(self, scene: str, preset: str) -> Path:
        if preset == "smoke":
            return self.lidar_scenes_root / f"{scene}_smoke"
        return self.lidar_scene_dir(scene)

    def colmap_scene_dir(self, scene: str) -> Path:
        return self.colmap_scenes_root / scene

    def colmap_compat_scene_dir(self, scene: str, suffix: str = "slam_compat") -> Path:
        return self.colmap_scenes_root / f"{scene}_{suffix}"

    def sfm_colmap_scene_dir(self, scene: str) -> Path:
        return self.colmap_scenes_root / f"{scene}_SFM"

    def validation_scene_dir(self, scene: str) -> Path:
        return self.validation_root / scene

    def validation_task_dir(self, scene: str, task: str) -> Path:
        return self.validation_scene_dir(scene) / task

    def resolve_video_path(self, scene: str, explicit: Optional[Path | str] = None) -> Path:
        if explicit:
            return _abs(explicit)
        direct = self.video_root / scene
        if direct.exists() and direct.is_file():
            return direct
        matches = sorted(path for path in self.video_root.glob(f"{scene}.*") if path.is_file())
        if matches:
            return matches[0]
        raise FileNotFoundError(
            f"Could not infer video for scene '{scene}' under {self.video_root}; pass --video-path."
        )
